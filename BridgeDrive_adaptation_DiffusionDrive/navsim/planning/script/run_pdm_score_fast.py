"""
Fast PDMS evaluation for BridgeDrive. Drop-in replacement for run_pdm_score.py (same hydra config,
same output CSV format, same `pdm_score()` methodology) ported from DiffusionDrive_HIDDEN_v2's
hidden/run_pdm_fast.py.

run_pdm_score.py bottleneck: a worker pool of N processes, each instantiate()-ing its OWN full
agent onto the GPU and doing batch-size-1 forward passes, while each process is mostly CPU-bound in
PDM scoring.

Here: ONE model in ONE process does real batched GPU inference; the resulting trajectories go to a
PERSISTENT CPU-only multiprocessing.Pool (one PDMSimulator/PDMScorer per worker) for scoring.
Feature build (next batch), GPU forward (this batch) and CPU scoring (previous batch) overlap via
a rolling loop with a one-batch prefetch.

BridgeDrive's head denoises all K anchors and the classifier picks one. Default mode
(POOL_EVAL=0) scores only that pick: one pdm_score() call per scene, identical to
run_pdm_score.py. POOL_EVAL=1 scores EVERY candidate (K calls per scene, ~K x slower) and reports
three rows: select (classifier pick = deployed output), best (oracle: max-PDMS candidate) and worst
(min-PDMS candidate). CAND_TOPK=N limits the pool to the top-N candidates by classifier logit
(select is always included); worst/best are then over that subset.

Usage:
    python navsim/planning/script/run_pdm_score_fast.py \\
        train_test_split=navtest agent=bridgedrive_agent_k80_beta10 \\
        agent.checkpoint_path=... experiment_name=...

Env knobs: SCORE_POOL_WORKERS (default 16), GPU_BATCH_SIZE (64), FEATURE_BUILD_THREADS (8).
No `worker=...` group: single GPU process + internal CPU pool.
"""

from typing import Any, Dict, List, Tuple
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
from multiprocessing import get_context
from concurrent.futures import ThreadPoolExecutor
import logging
import lzma
import os
import pickle
import traceback

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch

from nuplan.planning.script.builders.logging_builder import build_logger

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory, SensorConfig
from navsim.common.dataloader import SceneLoader, MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

SCORE_POOL_WORKERS = int(os.environ.get("SCORE_POOL_WORKERS", 16))
SCORE_POOL_MAXTASKSPERCHILD = 300  # periodic respawn - STRtree native-memory leak mitigation
GPU_BATCH_SIZE = int(os.environ.get("GPU_BATCH_SIZE", 64))
FEATURE_BUILD_THREADS = int(os.environ.get("FEATURE_BUILD_THREADS", 8))
POOL_EVAL = os.environ.get("POOL_EVAL", "0") == "1"
CAND_TOPK = int(os.environ.get("CAND_TOPK", 0))  # 0 = all candidates
METRIC_NAMES = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
]

# ---- persistent CPU worker pool ----
_worker_simulator: PDMSimulator = None
_worker_scorer: PDMScorer = None


def _init_score_worker(simulator_cfg: DictConfig, scorer_cfg: DictConfig) -> None:
    global _worker_simulator, _worker_scorer
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _worker_simulator = instantiate(simulator_cfg)
    _worker_scorer = instantiate(scorer_cfg)


def _score_one_scene_worker(args: Tuple[str, np.ndarray, np.ndarray]) -> Tuple[Dict[str, Any], bool]:
    """args = (metric_cache_path, poses (C,P,3), logits (C,)) where candidate 0 is the classifier pick.
    POOL_EVAL=0: only candidate 0 is passed -> returns asdict(PDMResults).
    POOL_EVAL=1: returns {select_*, best_*, worst_*} flat dict over all C candidates."""
    metric_cache_path, poses, logits = args
    try:
        with lzma.open(metric_cache_path, "rb") as f:
            metric_cache: MetricCache = pickle.load(f)
        sub = np.zeros((len(poses), len(METRIC_NAMES)), dtype=np.float64)
        for c in range(len(poses)):
            result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=Trajectory(poses[c].astype(np.float32)),
                future_sampling=_worker_simulator.proposal_sampling,
                simulator=_worker_simulator,
                scorer=_worker_scorer,
            )
            sub[c] = [getattr(result, n) for n in METRIC_NAMES]
        if not POOL_EVAL:
            return dict(zip(METRIC_NAMES, sub[0])), True
        picks = [("select", 0), ("best", int(sub[:, -1].argmax())), ("worst", int(sub[:, -1].argmin()))]
        row: Dict[str, Any] = {}
        for tag, i in picks:
            for j, n in enumerate(METRIC_NAMES):
                row[f"{tag}_{n}"] = sub[i, j]
        row["pool_size"] = len(poses)
        row["pool_score_mean"] = sub[:, -1].mean()
        return row, True
    except Exception:
        logger.warning("scoring failed for %s", metric_cache_path, exc_info=True)
        return {}, False


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)

    simulator_check: PDMSimulator = instantiate(cfg.simulator)
    scorer_check: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator_check.proposal_sampling == scorer_check.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"

    logger.info("Loading agent (single process, single GPU load)...")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent.to(device)
    agent.eval()

    token_scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    feature_scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=agent.get_sensor_config(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    scene_tokens = set(token_scene_loader.tokens)
    cache_tokens = set(metric_cache_loader.tokens)
    if scene_tokens - cache_tokens:
        logger.warning("Missing metric cache for %d tokens. Skipping these tokens.", len(scene_tokens - cache_tokens))
    if cache_tokens - scene_tokens:
        logger.warning("Unused metric cache for %d tokens. Skipping these tokens.", len(cache_tokens - scene_tokens))
    tokens_to_evaluate = sorted(scene_tokens & cache_tokens)
    logger.info(
        "Starting FAST pdm scoring of %d scenarios (GPU batch=%d, CPU score pool=%d workers)...",
        len(tokens_to_evaluate), GPU_BATCH_SIZE, SCORE_POOL_WORKERS,
    )

    feature_builders = agent.get_feature_builders()
    batches = [tokens_to_evaluate[i:i + GPU_BATCH_SIZE] for i in range(0, len(tokens_to_evaluate), GPU_BATCH_SIZE)]
    score_rows: List[Dict[str, Any]] = []

    def _build_one_feature(token: str):
        try:
            agent_input = feature_scene_loader.get_agent_input_from_token(token)
            feat: Dict[str, torch.Tensor] = {}
            for builder in feature_builders:
                feat.update(builder.compute_features(agent_input))
            return token, feat, None
        except Exception:
            return token, None, traceback.format_exc()

    def _build_batch(batch_tokens: List[str]):
        feats, cache_paths, valid_tokens, failed_tokens = [], [], [], []
        with ThreadPoolExecutor(max_workers=FEATURE_BUILD_THREADS) as texec:
            for token, feat, err in texec.map(_build_one_feature, batch_tokens):
                if err is not None:
                    logger.warning("failed building features for token %s:\n%s", token, err)
                    failed_tokens.append(token)
                    continue
                feats.append(feat)
                cache_paths.append(metric_cache_loader.metric_cache_paths[token])
                valid_tokens.append(token)
        return feats, cache_paths, valid_tokens, failed_tokens

    def _harvest(job_meta: List[str], async_result) -> None:
        for token, (row, ok) in zip(job_meta, async_result.get()):
            r: Dict[str, Any] = {"token": token, "valid": ok}
            if ok:
                r.update(row)
            score_rows.append(r)

    ctx = get_context("spawn")  # CUDA-context safety
    pool = ctx.Pool(
        processes=SCORE_POOL_WORKERS,
        initializer=_init_score_worker,
        initargs=(cfg.simulator, cfg.scorer),
        maxtasksperchild=SCORE_POOL_MAXTASKSPERCHILD,
    )
    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    pending_scores = None
    next_future = prefetch_executor.submit(_build_batch, batches[0]) if batches else None

    try:
        with torch.no_grad():
            for b_idx in range(len(batches)):
                logger.info("batch %d/%d (%d scenes)", b_idx + 1, len(batches), len(batches[b_idx]))
                feats, cache_paths, valid_tokens, failed_tokens = next_future.result()
                for token in failed_tokens:
                    score_rows.append({"token": token, "valid": False})

                # prefetch next batch's features while this batch runs on GPU / CPU pool
                if b_idx + 1 < len(batches):
                    next_future = prefetch_executor.submit(_build_batch, batches[b_idx + 1])

                job_meta: List[str] = []
                async_result = None
                if valid_tokens:
                    try:
                        keys = feats[0].keys()
                        features_batch = {k: torch.stack([f[k] for f in feats]).to(device) for k in keys}
                        out = agent.forward(features_batch, targets=None)
                        if POOL_EVAL:
                            pool_b = out["poses_all"].detach().float().cpu().numpy()  # (B, K, P, 3)
                            logits_b = out["poses_cls"].detach().float().cpu().numpy()  # (B, K)
                            jobs = []
                            for i in range(len(valid_tokens)):
                                order = np.argsort(-logits_b[i])  # order[0] = classifier pick
                                if CAND_TOPK > 0:
                                    order = order[:CAND_TOPK]
                                jobs.append((cache_paths[i], pool_b[i][order], logits_b[i][order]))
                        else:
                            poses_b = out["trajectory"].detach().float().cpu().numpy()  # (B, P, 3)
                            jobs = [(cache_paths[i], poses_b[i][None], None) for i in range(len(valid_tokens))]
                        job_meta = list(valid_tokens)
                        async_result = pool.map_async(_score_one_scene_worker, jobs)
                    except Exception:
                        logger.warning("GPU forward failed for batch %d:\n%s", b_idx + 1, traceback.format_exc())
                        for token in valid_tokens:
                            score_rows.append({"token": token, "valid": False})

                # harvest previous batch (had this batch's build + forward time to finish)
                if pending_scores is not None:
                    _harvest(*pending_scores)
                pending_scores = (job_meta, async_result) if async_result is not None else None

            if pending_scores is not None:
                _harvest(*pending_scores)
    finally:
        prefetch_executor.shutdown(wait=False)
        pool.close()
        pool.join()

    pdm_score_df = pd.DataFrame(score_rows)
    num_successful = int(pdm_score_df["valid"].sum())
    num_failed = len(pdm_score_df) - num_successful
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    csv_path = save_path / f"{timestamp}.csv"
    pdm_score_df.to_csv(csv_path)

    if POOL_EVAL:
        lines = ["", "Finished running FAST pool evaluation.",
                 f"    Successful: {num_successful}, failed: {num_failed}.", ""]
        for tag, label in [("select", "SELECT (classifier pick, deployed)"), ("best", "BEST (oracle max-PDMS)"),
                           ("worst", "WORST (min-PDMS candidate)")]:
            lines.append(f"    {label}: PDMS {average_row[f'{tag}_score']:.4f}")
            for n in METRIC_NAMES[:-1]:
                lines.append(f"        {n:32s}{average_row[f'{tag}_{n}']:.4f}")
            lines.append("")
        lines.append(f"    mean PDMS over whole pool: {average_row['pool_score_mean']:.4f} (avg pool size {average_row['pool_size']:.1f})")
        lines.append(f"    Results are stored in: {csv_path}.")
        logger.info("\n".join(lines))
    else:
        logger.info(
            f"""
        Finished running FAST evaluation.
            Number of successful scenarios: {num_successful}.
            Number of failed scenarios: {num_failed}.
            Final average score of valid results: {average_row['score']}.
            Results are stored in: {csv_path}.
        """
        )


if __name__ == "__main__":
    main()
