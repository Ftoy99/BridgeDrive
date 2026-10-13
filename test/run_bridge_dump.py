"""
Dump every BridgeDrive candidate (all K anchors, fully denoised) with
  * its REAL per-candidate PDMS sub-metrics (official navsim pdm_score(), same methodology as
    run_pdm_score.py), and
  * a set of "bridge signals" that need no ground truth, computed in one batched GPU pass:
      logit0/logit1   classifier logits of both cls decoder layers
      path_len        total xy distance travelled over the 20 denoising steps (mean over waypoints)
      last_move       how far the LAST denoising step moved the candidate
      x0_first_dev    ||one-shot x0 prediction from the clean anchor - final sample||
      x0_mean_dev     mean over steps of ||step x0 prediction - final sample||
      travel          ||final - anchor||
      recon           re-noise the final candidate along ITS OWN bridge to t in RECON_TS, denoise once,
                      measure ||x0_hat - candidate||  (diffusion-style self-consistency / ELBO proxy)
      samp_std/off    spread of the candidate across SAMPLES extra stochastic re-samplings
                      (the first denoising step injects noise) - bridge sampling uncertainty
  plus final trajectories, GT, tokens, so selectors can be compared OFFLINE
  (test/eval_bridge_selectors.py) without re-running anything.

Architecture = run_pdm_score_fast.py: one GPU process, persistent CPU scoring pool, rolling prefetch.
Only the training/eval code of the model is not touched: head.forward_test is swapped for an
instrumented replica at runtime (same math as DDBMTrajectoryHead.forward_test, asserted equal).

Env: SAMPLES (extra stochastic samples, default 2), RECON_TS ("200,400,600,800"), GPU_BATCH_SIZE (32),
     SCORE_POOL_WORKERS (16), FEATURE_BUILD_THREADS (8), DUMP_DIR, FLUSH_EVERY (batches per shard, 10), SUBSAMPLE (random N scenes)
"""
from typing import Any, Dict, List, Tuple
from pathlib import Path
from multiprocessing import get_context
from concurrent.futures import ThreadPoolExecutor
import logging
import lzma
import os
import pickle
import time
import traceback

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
import torch

from nuplan.planning.script.builders.logging_builder import build_logger

from navsim.common.dataclasses import Trajectory, SensorConfig
from navsim.common.dataloader import SceneLoader, MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.agents.diffusiondrive.model_diffusion_head_ddbm_v5 import gen_sineembed_for_position

logger = logging.getLogger(__name__)

CONFIG_PATH = str(Path(os.environ["NAVSIM_DEVKIT_ROOT"]) / "navsim" / "planning" / "script" / "config" / "pdm_scoring")
CONFIG_NAME = "default_run_pdm_score"
METRICS = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
           "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score"]

SAMPLES = int(os.environ.get("SAMPLES", 2))
RECON_TS = [int(x) for x in os.environ.get("RECON_TS", ",".join(str(t) for t in range(50, 1000, 50))).split(",")]
RECON_DRAWS = int(os.environ.get("RECON_DRAWS", 1))  # independent noise draws averaged per timestep
GPU_BATCH_SIZE = int(os.environ.get("GPU_BATCH_SIZE", 32))
SCORE_POOL_WORKERS = int(os.environ.get("SCORE_POOL_WORKERS", 16))
FEATURE_BUILD_THREADS = int(os.environ.get("FEATURE_BUILD_THREADS", 8))
FLUSH_EVERY = int(os.environ.get("FLUSH_EVERY", 10))
DUMP_DIR = Path(os.environ.get("DUMP_DIR", "/mnt/ds/navsim-main/exp/bridge_signals_ep45"))
STEP_NUM = 20


# ----------------------------- GPU side: bridge signals -----------------------------
@torch.no_grad()
def _denoise(head, plan_anchor, bev_feature, bev_spatial_shape, agents_query, ego_query, global_img):
    """Batched replica of DDBMTrajectoryHead.forward_test's sampling loop (stochastic only at the first step).
    Returns final_xy (B,K,P,2), head0 (B,K,P,1), stats dict of (B,K) tensors."""
    bs = plan_anchor.shape[0]
    device = plan_anchor.device
    ts = torch.arange(0, 1001, 1000 // STEP_NUM).to(device)
    xt_prev = plan_anchor
    path_len = torch.zeros(plan_anchor.shape[:2], device=device)
    kinetic = torch.zeros(plan_anchor.shape[:2], device=device)
    moves = []
    x0s = []
    for i in range(STEP_NUM, 0, -1):
        xt = xt_prev
        t = ts[i] * torch.ones([bs], device=device)
        t_prev = ts[i - 1] * torch.ones([bs], device=device)
        emb = gen_sineembed_for_position(xt, hidden_dim=64).flatten(-2)
        feat = head.plan_anchor_encoder(emb).view(bs, xt.shape[1], -1)
        time_embed = head.time_mlp(t).view(bs, 1, -1)
        reg_list, _ = head.diff_decoder(feat, xt, plan_anchor, bev_feature, bev_spatial_shape,
                                        agents_query, ego_query, time_embed, global_img)
        x0_mean = reg_list[-1][..., :2]
        head0 = reg_list[-1][..., [-1]]
        nxt = head.diffusion_scheduler.sample_step(
            t=t, t_prev=t_prev, xt=head.norm_odo(xt.clone()), x0=head.norm_odo(x0_mean),
            xT=head.norm_odo(plan_anchor.clone()))
        xt_prev = head.denorm_odo(nxt)
        move = (xt_prev - xt).norm(dim=-1).mean(-1)
        path_len += move
        kinetic += (xt_prev - xt).norm(dim=-1).pow(2).mean(-1)   # discretised transport (kinetic) energy of the bridge path
        moves.append(move)
        x0s.append(x0_mean)
    final = xt_prev
    stats = {
        "path_len": path_len,
        "last_move": move,
        "x0_first_dev": (x0s[0] - final).norm(dim=-1).mean(-1),
        "x0_mean_dev": torch.stack([(x - final).norm(dim=-1).mean(-1) for x in x0s]).mean(0),
        "travel": (final - plan_anchor).norm(dim=-1).mean(-1),
        "kinetic": kinetic,
        "step_move": torch.stack(moves, dim=-1),                                     # (B,K,20) movement per denoising step
        "x0dev_steps": torch.stack([(x - final).norm(dim=-1).mean(-1) for x in x0s], dim=-1),  # (B,K,20)
    }
    return final, head0, stats


@torch.no_grad()
def _recon_errors(head, final_xy, plan_anchor, bev_feature, bev_spatial_shape, agents_query, ego_query, global_img):
    """Re-noise each final candidate along its own bridge (x0=candidate, xT=its anchor) to t, run the denoiser
    once, return mean ||x0_hat - candidate||  (B,K,len(RECON_TS)). Fixed seed -> same noise for all candidates."""
    bs, K = final_xy.shape[:2]
    device = final_xy.device
    errs = []
    gen = torch.Generator(device=device).manual_seed(1234)
    for tv in RECON_TS:
        t = float(tv) * torch.ones([bs], device=device)
        acc = 0.0
        for _ in range(RECON_DRAWS):
            noise = torch.randn(final_xy.shape, device=device, generator=gen)
            noisy = head.diffusion_scheduler.add_noise(
                x0=head.norm_odo(final_xy), xT=head.norm_odo(plan_anchor), noise=noise, t=t).float()
            noisy = head.denorm_odo(noisy)
            emb = gen_sineembed_for_position(noisy, hidden_dim=64).flatten(-2)
            feat = head.plan_anchor_encoder(emb).view(bs, K, -1)
            time_embed = head.time_mlp(t).view(bs, 1, -1)
            reg_list, _ = head.diff_decoder(feat, noisy, plan_anchor, bev_feature, bev_spatial_shape,
                                            agents_query, ego_query, time_embed, global_img)
            acc = acc + (reg_list[-1][..., :2] - final_xy).norm(dim=-1).mean(-1)
        errs.append(acc / RECON_DRAWS)
    return torch.stack(errs, dim=-1)


def install_instrumented_forward_test(head, holder: Dict[str, Any]):
    """Replace head.forward_test (instance attr) by an instrumented replica. Returns the normal
    {'trajectory': ...} dict, and drops all signals into holder['out']."""
    def forward_test(ego_query, agents_query, bev_feature, bev_spatial_shape, global_img):
        bs = ego_query.shape[0]
        plan_anchor = head.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        K = plan_anchor.shape[1]
        emb = gen_sineembed_for_position(plan_anchor, hidden_dim=64).flatten(-2)
        ak_feat = head.plan_anchor_encoder(emb).view(bs, K, -1)
        cls_list = head.diff_decoder_cls(ak_feat, plan_anchor, bev_feature, bev_spatial_shape,
                                         agents_query, ego_query, global_img)
        mode_idx = cls_list[-1].argmax(dim=-1)

        final, head0, stats = _denoise(head, plan_anchor, bev_feature, bev_spatial_shape, agents_query, ego_query, global_img)
        recon = _recon_errors(head, final, plan_anchor, bev_feature, bev_spatial_shape, agents_query, ego_query, global_img)
        samples = [final]
        for _ in range(SAMPLES):
            f2, _, _ = _denoise(head, plan_anchor, bev_feature, bev_spatial_shape, agents_query, ego_query, global_img)
            samples.append(f2)
        S = torch.stack(samples, dim=0)  # (S+1,B,K,P,2)
        out = dict(stats)
        out["logit0"] = cls_list[0]
        out["logit1"] = cls_list[-1]
        out["recon"] = recon
        out["samp_std"] = S.std(dim=0).norm(dim=-1).mean(-1) if S.shape[0] > 1 else torch.zeros_like(stats["travel"])
        out["samp_off"] = (final - S.mean(0)).norm(dim=-1).mean(-1)
        out["final"] = torch.cat([final, head0], dim=-1)  # (B,K,P,3), same as forward_test's xt_prev_
        holder["out"] = out
        sel = torch.gather(out["final"], 1, mode_idx[..., None, None, None].repeat(1, 1, out["final"].shape[2], 3)).squeeze(1)
        return {"trajectory": sel}
    head.forward_test = forward_test


# ----------------------------- CPU side: score every candidate -----------------------------
_sim: PDMSimulator = None
_scorer: PDMScorer = None


def _init_worker(sim_cfg, scorer_cfg):
    global _sim, _scorer
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    _sim = instantiate(sim_cfg)
    _scorer = instantiate(scorer_cfg)


def _score_scene(args: Tuple[str, np.ndarray]) -> Tuple[np.ndarray, bool]:
    path, cands = args  # cands (K,P,3)
    try:
        with lzma.open(path, "rb") as f:
            mc: MetricCache = pickle.load(f)
        sub = np.zeros((len(cands), len(METRICS)), dtype=np.float32)
        for k in range(len(cands)):
            r = pdm_score(metric_cache=mc, model_trajectory=Trajectory(cands[k].astype(np.float32)),
                          future_sampling=_sim.proposal_sampling, simulator=_sim, scorer=_scorer)
            sub[k] = [getattr(r, m) for m in METRICS]
        return sub, True
    except Exception:
        logger.warning("scoring failed for %s", path, exc_info=True)
        return np.zeros((0,), dtype=np.float32), False


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    agent = instantiate(cfg.agent)
    agent.initialize()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent.to(device).eval()
    head = agent._transfuser_model._trajectory_head
    holder: Dict[str, Any] = {}
    install_instrumented_forward_test(head, holder)
    num_poses = agent._config.trajectory_sampling.num_poses

    sim_check = instantiate(cfg.simulator)
    assert sim_check.proposal_sampling == instantiate(cfg.scorer).proposal_sampling

    tok_loader = SceneLoader(sensor_blobs_path=None, data_path=Path(cfg.navsim_log_path),
                             scene_filter=instantiate(cfg.train_test_split.scene_filter),
                             sensor_config=SensorConfig.build_no_sensors())
    feat_loader = SceneLoader(sensor_blobs_path=Path(cfg.sensor_blobs_path), data_path=Path(cfg.navsim_log_path),
                              scene_filter=instantiate(cfg.train_test_split.scene_filter),
                              sensor_config=agent.get_sensor_config())
    mc_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    tokens = sorted(set(tok_loader.tokens) & set(mc_loader.tokens))
    sub_n = int(os.environ.get("SUBSAMPLE", 0))
    if sub_n and sub_n < len(tokens):  # reproducible random subset spread over all logs
        rs = np.random.RandomState(0)
        tokens = sorted(rs.choice(tokens, size=sub_n, replace=False).tolist())
    batches = [tokens[i:i + GPU_BATCH_SIZE] for i in range(0, len(tokens), GPU_BATCH_SIZE)]
    logger.info("bridge dump: %d scenes, %d batches, SAMPLES=%d RECON_TS=%s -> %s",
                len(tokens), len(batches), SAMPLES, RECON_TS, DUMP_DIR)
    np.save(DUMP_DIR / "anchors.npy", head.plan_anchor.detach().cpu().numpy())
    np.save(DUMP_DIR / "recon_ts.npy", np.array(RECON_TS))

    builders = agent.get_feature_builders()

    def build_one(token):
        try:
            scene = feat_loader.get_scene_from_token(token)
            ai = feat_loader.get_agent_input_from_token(token)
            f = {}
            for b in builders:
                f.update(b.compute_features(ai))
            return token, f, scene.get_future_trajectory(num_trajectory_frames=num_poses).poses, None
        except Exception:
            return token, None, None, traceback.format_exc()

    def build_batch(bt):
        feats, gts, paths, toks = [], [], [], []
        with ThreadPoolExecutor(max_workers=FEATURE_BUILD_THREADS) as ex:
            for token, f, gt, err in ex.map(build_one, bt):
                if err:
                    logger.warning("feature build failed %s\n%s", token, err)
                    continue
                feats.append(f); gts.append(gt); paths.append(mc_loader.metric_cache_paths[token]); toks.append(token)
        return feats, gts, paths, toks

    ctx = get_context("spawn")
    pool = ctx.Pool(SCORE_POOL_WORKERS, initializer=_init_worker, initargs=(cfg.simulator, cfg.scorer),
                    maxtasksperchild=300)
    prefetch = ThreadPoolExecutor(max_workers=1)
    nxt = prefetch.submit(build_batch, batches[0]) if batches else None

    shard: Dict[str, List[Any]] = {}
    shard_id = [0]
    n_done = [0]
    n_fail = [0]

    def harvest(meta, ar):
        subs = ar.get()
        for (tok, sig_i, gt), (sub, ok) in zip(meta, subs):
            if not ok:
                n_fail[0] += 1
                continue
            shard.setdefault("tokens", []).append(tok)
            shard.setdefault("gt", []).append(gt.astype(np.float32))
            shard.setdefault("pdm", []).append(sub)
            for k, v in sig_i.items():
                shard.setdefault(k, []).append(v)
            n_done[0] += 1

    def flush():
        if not shard:
            return
        arrs = {k: np.stack(v) if k != "tokens" else np.array(v) for k, v in shard.items()}
        np.savez(DUMP_DIR / f"shard_{shard_id[0]:04d}.npz", **arrs)
        logger.info("flushed shard %d (%d scenes total, %d failed)", shard_id[0], n_done[0], n_fail[0])
        shard_id[0] += 1
        shard.clear()

    pending = None
    t0 = time.time()
    try:
        with torch.no_grad():
            for bi in range(len(batches)):
                feats, gts, paths, toks = nxt.result()
                if bi + 1 < len(batches):
                    nxt = prefetch.submit(build_batch, batches[bi + 1])
                meta, ar = [], None
                if toks:
                    torch.manual_seed(bi)
                    fb = {k: torch.stack([f[k] for f in feats]).to(device) for k in feats[0].keys()}
                    out = agent.forward(fb, targets=None)
                    sig = {k: v.detach().float().cpu().numpy() for k, v in holder["out"].items()}
                    # replica must reproduce the model's own selected trajectory (stochastic first step: same
                    # global RNG order is not guaranteed, so only check shape/finite here)
                    assert np.isfinite(sig["final"]).all(), "non-finite candidates"
                    cands = sig["final"]
                    for i, tok in enumerate(toks):
                        meta.append((tok, {k: (v[i] if k != "final" else v[i].astype(np.float32)) for k, v in sig.items()}, gts[i]))
                    ar = pool.map_async(_score_scene, [(paths[i], cands[i]) for i in range(len(toks))])
                if pending is not None:
                    harvest(*pending)
                pending = (meta, ar) if ar is not None else None
                if (bi + 1) % FLUSH_EVERY == 0:
                    flush()
                if (bi + 1) % 5 == 0:
                    el = time.time() - t0
                    logger.info("batch %d/%d  %.1f min elapsed, ETA %.1f min", bi + 1, len(batches), el / 60,
                                el / (bi + 1) * (len(batches) - bi - 1) / 60)
            if pending is not None:
                harvest(*pending)
        flush()
    finally:
        prefetch.shutdown(wait=False)
        pool.close()
        pool.join()
    logger.info("DONE: %d scenes dumped, %d failed -> %s", n_done[0], n_fail[0], DUMP_DIR)


if __name__ == "__main__":
    main()
