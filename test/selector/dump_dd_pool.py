"""DiffusionDrive P1 pool dump: public checkpoint, K=20 candidates per scene (Runbook section 1).

GPU: one model, batched forward over GPU_BATCH_SIZE scenes, seeded per batch.
CPU: persistent spawn pool, official navsim pdm_score() once per candidate and once for the human trajectory.
Shard i: DUMP_DIR/shard_XXXXX.npz with tokens (B,), traj (B,K,P,3), logit (B,K) raw final-step cls logits,
  pdm (B,K,7) official sub-scores in METRIC_NAMES order, human_pdm (B,7), ok (B,) PDM scoring succeeded.
Scenes whose feature build failed go to shard_XXXXX.failed.txt and are not in the shard.
Summary (params/metrics/tags) goes to RUN_DIR for test/selector/log_selector_run.py.

  DUMP_DIR=... RUN_DIR=... python test/selector/dump_dd_pool.py train_test_split=navtest agent=diffusiondrive_agent \
      agent.checkpoint_path=... +agent.config.bkb_path=... agent.config.plan_anchor_path=... experiment_name=dd_pool_dump
"""
import json, logging, lzma, os, pickle, traceback
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import hydra, numpy as np, torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.metric_cache import MetricCache

logger = logging.getLogger(__name__)

REPO_CONFIG = "/mnt/jimmys/public_repos/DiffusionDrive/navsim/planning/script/config/pdm_scoring"
METRIC_NAMES = [
    "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
    "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score",
]
GPU_BATCH_SIZE = int(os.environ.get("GPU_BATCH_SIZE", 32))
FEATURE_BUILD_THREADS = int(os.environ.get("FEATURE_BUILD_THREADS", 8))
SCORE_POOL_WORKERS = int(os.environ.get("SCORE_POOL_WORKERS", 16))
SEED = int(os.environ.get("SEED", 0))
EXPECTED_K = 20  # public DiffusionDrive plan anchors (kmeans_navsim_traj_20)
# v2 extras (Runbook section 1): pred_bev, pred_agents, gt_agents, gt_future, log_name. Off by default so v1 stays as dumped.
V2_KEYS = os.environ.get("V2_KEYS", "0") == "1"
N_BOX = 30  # TransFuser num_bounding_boxes
MAX_GT_AGENTS = 128

_sim = None
_scorer = None
_model_ts = None


def _init_worker(cfg_yaml: str) -> None:
    global _sim, _scorer, _model_ts
    cfg = OmegaConf.create(cfg_yaml)
    _sim = instantiate(cfg.simulator)
    _scorer = instantiate(cfg.scorer)
    _model_ts = instantiate(cfg.agent.config.trajectory_sampling)


def _score_scene(args):
    """(metric_cache_path, cand (K,P,3), human (P,3)) -> (pdm (K,7), human_pdm (7,), ok)."""
    metric_cache_path, cand, human = args
    try:
        with lzma.open(metric_cache_path, "rb") as f:
            metric_cache: MetricCache = pickle.load(f)

        def one(poses):
            r = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=Trajectory(poses.astype(np.float32), trajectory_sampling=_model_ts),
                future_sampling=_sim.proposal_sampling,
                simulator=_sim,
                scorer=_scorer,
            )
            return [getattr(r, m) for m in METRIC_NAMES]

        pdm = np.array([one(c) for c in cand], np.float32)
        return pdm, np.array(one(human), np.float32), True
    except Exception:
        logger.warning("PDM scoring failed for %s:\n%s", metric_cache_path, traceback.format_exc())
        return np.full((len(cand), 7), np.nan, np.float32), np.full(7, np.nan, np.float32), False


def _flush(pending, dump_dir: Path, totals: dict) -> None:
    bi, tokens, failed, traj, logit, jobs, v2 = pending
    results = [j.get() for j in jobs]
    pdm = np.stack([r[0] for r in results])
    human = np.stack([r[1] for r in results])
    ok = np.array([r[2] for r in results])
    np.savez(dump_dir / f"shard_{bi:05d}.npz", tokens=np.array(tokens), traj=traj, logit=logit,
             pdm=pdm, human_pdm=human, ok=ok, **(v2 or {}))
    if failed:
        (dump_dir / f"shard_{bi:05d}.failed.txt").write_text("\n".join(failed) + "\n")
    # model's deployed pick = argmax of final cls logits (same as TrajectoryHead.forward_test)
    pick = logit.argmax(axis=1)
    cls_score = pdm[np.arange(len(pick)), pick, -1]
    good = ok
    totals["n_scored"] += int(good.sum())
    totals["n_pdm_failed"] += int((~good).sum())
    totals["n_feature_failed"] += len(failed)
    totals["cls_sum"] += float(cls_score[good].sum())
    totals["oracle_sum"] += float(np.nanmax(pdm[good, :, -1], axis=1).sum())
    totals["human_sum"] += float(human[good, -1].sum())
    logger.info("shard %05d: %d scenes, %d pdm failed, %d feature failed", bi, len(tokens), int((~good).sum()), len(failed))


@hydra.main(config_path=REPO_CONFIG, config_name="default_run_pdm_score", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    dump_dir = Path(os.environ["DUMP_DIR"])
    run_dir = Path(os.environ["RUN_DIR"])
    dump_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval()
    device = "cuda"
    agent.to(device)
    num_poses = agent._config.trajectory_sampling.num_poses

    # Record the final-step candidates and cls logits from the decoder. forward_test returns only the argmax.
    holder = {}
    decoder = agent._transfuser_model._trajectory_head.diff_decoder
    orig_forward = decoder.forward

    def record(*args, **kwargs):
        reg_list, cls_list = orig_forward(*args, **kwargs)
        holder["reg"], holder["cls"] = reg_list[-1], cls_list[-1]
        return reg_list, cls_list

    decoder.forward = record

    token_loader = SceneLoader(
        sensor_blobs_path=None, data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter), sensor_config=SensorConfig.build_no_sensors(),
    )
    feat_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path), data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter), sensor_config=agent.get_sensor_config(),
    )
    mc_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    tokens = sorted(set(token_loader.tokens) & set(mc_loader.tokens))
    sub_n = int(os.environ.get("SUBSAMPLE", 0))  # same seeded subsample as dump_pool.py
    if sub_n and sub_n < len(tokens):
        tokens = sorted(np.random.RandomState(0).choice(tokens, size=sub_n, replace=False).tolist())
    logger.info("dumping %d scenes, K=%d expected", len(tokens), EXPECTED_K)
    builders = agent.get_feature_builders()

    def build_one(token):
        try:
            scene = feat_loader.get_scene_from_token(token)
            agent_input = feat_loader.get_agent_input_from_token(token)
            feat = {}
            for b in builders:
                feat.update(b.compute_features(agent_input))
            human = scene.get_future_trajectory(num_trajectory_frames=num_poses).poses
            extra = None
            if V2_KEYS:
                boxes = scene.frames[scene.scene_metadata.num_history_frames - 1].annotations.boxes
                gt_agents = np.zeros((MAX_GT_AGENTS, boxes.shape[1]), np.float32)
                n = min(len(boxes), MAX_GT_AGENTS)
                gt_agents[:n] = boxes[:n]
                extra = {"gt_agents": gt_agents, "n_gt_agents": n, "log_name": scene.scene_metadata.log_name}
            return token, feat, human, None, extra
        except Exception:
            return token, None, None, traceback.format_exc(), None

    pool = get_context("spawn").Pool(
        processes=SCORE_POOL_WORKERS, initializer=_init_worker,
        initargs=(OmegaConf.to_yaml(cfg, resolve=True),), maxtasksperchild=300,
    )
    totals = dict(n_scored=0, n_pdm_failed=0, n_feature_failed=0, cls_sum=0.0, oracle_sum=0.0, human_sum=0.0)
    batches = [tokens[i:i + GPU_BATCH_SIZE] for i in range(0, len(tokens), GPU_BATCH_SIZE)]
    pending = None
    with ThreadPoolExecutor(max_workers=FEATURE_BUILD_THREADS) as texec:
        for bi, batch_tokens in enumerate(batches):
            built = list(texec.map(build_one, batch_tokens))
            good = [b for b in built if b[3] is None]
            failed = [b[0] for b in built if b[3] is not None]
            for b in built:
                if b[3] is not None:
                    logger.warning("feature build failed for %s:\n%s", b[0], b[3])
            if not good:
                if failed:
                    (dump_dir / f"shard_{bi:05d}.failed.txt").write_text("\n".join(failed) + "\n")
                    totals["n_feature_failed"] += len(failed)
                continue
            feats = {k: torch.stack([b[1][k] for b in good]).to(device) for k in good[0][1]}
            torch.manual_seed(SEED + bi)
            with torch.no_grad():
                out = agent.forward(feats)
            traj = holder["reg"].float().cpu().numpy()
            logit = holder["cls"].float().cpu().numpy()
            assert traj.shape[1] == EXPECTED_K, f"expected K={EXPECTED_K}, got {traj.shape}"
            humans = np.stack([b[2] for b in good]).astype(np.float32)
            v2 = None
            if V2_KEYS:
                bev = out["bev_semantic_map"]
                assert bev.shape[1] == 7, f"expected 7 BEV classes on dim 1, got {bev.shape}"
                v2 = {
                    "pred_bev": bev.argmax(1).to(torch.uint8).cpu().numpy(),  # (B,H,W) argmax class, compact
                    "pred_agents": out["agent_states"].float().cpu().numpy().astype(np.float16),  # (B,N_BOX,5)
                    "pred_agent_prob": torch.sigmoid(out["agent_labels"].float()).cpu().numpy().astype(np.float16),
                    "gt_agents": np.stack([b[4]["gt_agents"] for b in good]),  # (B,MAX_GT_AGENTS,7) at t0
                    "n_gt_agents": np.array([b[4]["n_gt_agents"] for b in good]),
                    "gt_future": humans,  # (B,P,3) same poses as human_pdm
                    "log_name": np.array([b[4]["log_name"] for b in good]),
                }
            tks = [b[0] for b in good]
            jobs = [pool.apply_async(_score_scene, ((mc_loader.metric_cache_paths[t], traj[i], humans[i]),))
                    for i, t in enumerate(tks)]
            if pending is not None:
                _flush(pending, dump_dir, totals)
            pending = (bi, tks, failed, traj, logit, jobs, v2)
    if pending is not None:
        _flush(pending, dump_dir, totals)
    pool.close()
    pool.join()

    n = totals["n_scored"]
    metrics = {
        "n_scenes": len(tokens),
        "n_scored": n,
        "n_pdm_failed": totals["n_pdm_failed"],
        "n_feature_failed": totals["n_feature_failed"],
        "cls_score": totals["cls_sum"] / n,
        "pdm_oracle_score": totals["oracle_sum"] / n,
        "human_score": totals["human_sum"] / n,
    }
    params = {
        "pool_id": os.environ.get("POOL_ID", "dd_k20_public"), "model": "DiffusionDrive (hustvl)",
        "ckpt": str(cfg.agent.checkpoint_path), "K": EXPECTED_K, "split": os.environ.get("SPLIT", "navtest"), "subsample": int(os.environ.get("SUBSAMPLE", 0)),
        "seed": SEED, "v2_keys": int(V2_KEYS), "gpu_batch_size": GPU_BATCH_SIZE, "anchor_path": str(cfg.agent.config.plan_anchor_path),
    }
    tags = {"public": "1", "internal": "0", "smoke": os.environ.get("SMOKE", "0")}
    (run_dir / "params.json").write_text(json.dumps(params, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "tags.json").write_text(json.dumps(tags, indent=2))
    logger.info("done: %s", json.dumps(metrics))


if __name__ == "__main__":
    main()
