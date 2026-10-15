"""Frozen selector pool dump = run_bridge_dump.py (all K candidates, real per-candidate PDM sub-scores, logits) plus what the
selector paper needs (second_brain/PhD/Selector_Runbook.md §1):
  human_pdm   (7,)        real sub-scores of the human trajectory (M5)            [scored as candidate K]
  pred_agents (30,5)/(30) predicted agent boxes + logits (model's own detection head)
  pred_bev    (128,256)   argmax of the predicted BEV semantic map, uint8
  tokens_kv   (65,256)    scene tokens the trajectory decoder attends to (64 BEV + status), fp16 -> selector-only heads
  tokens_ag   (30,256)    agent queries, fp16
  gt_agents   (30,5)/(30) GT vehicle boxes at t0 (TransfuserTargetBuilder rule)
  gt_future   (9,32,5)/(9,32) int types   GT boxes t0..t0+8 in the t0 ego frame, one slot per track (hidden_dualcls rule)
Sampler seeded per batch (torch.manual_seed(batch_idx)); SAMPLES/recon off. Same CLI as run_bridge_dump.py:
  DUMP_DIR=... [SUBSAMPLE=20000] python test/selector/dump_pool.py train_test_split=navtest agent=... agent.checkpoint_path=... \
      +agent.config.bkb_path=... agent.config.plan_anchor_path=... experiment_name=selector_dump [metric_cache_path=...]
Then: python test/selector/make_manifest.py DUMP_DIR POOL_ID --split navtest --model ... --ckpt ... --dump-script test/selector/dump_pool.py
"""
import logging, os, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List

import hydra, numpy as np, torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from nuplan.planning.script.builders.logging_builder import build_logger
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.enums import BoundingBoxIndex
from navsim.agents.diffusiondrive.transfuser_features import TransfuserTargetBuilder
from navsim.agents.diffusiondrive.model_diffusion_head_ddbm_v5 import gen_sineembed_for_position

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_bridge_dump as rbd  # _denoise, _init_worker, _score_scene, METRICS, CONFIG_*

logger = logging.getLogger(__name__)
GPU_BATCH_SIZE = int(os.environ.get("GPU_BATCH_SIZE", 32))
SCORE_POOL_WORKERS = int(os.environ.get("SCORE_POOL_WORKERS", 16))
FEATURE_BUILD_THREADS = int(os.environ.get("FEATURE_BUILD_THREADS", 8))
FLUSH_EVERY = int(os.environ.get("FLUSH_EVERY", 10))
DUMP_DIR = Path(os.environ["DUMP_DIR"])
FUT_T, FUT_N = 9, 32  # t0..t0+8 (= trajectory poses), slots per scene
TYPE_CODE = {"vehicle": 0, "pedestrian": 1, "bicycle": 2}


def install_forward_test(head, holder):
    """Replica of DDBMTrajectoryHead.forward_test (one denoise, seeded by the caller) that also keeps the scene tokens."""
    def forward_test(ego_query, agents_query, bev_feature, bev_spatial_shape, global_img):
        bs = ego_query.shape[0]
        plan_anchor = head.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        K = plan_anchor.shape[1]
        emb = gen_sineembed_for_position(plan_anchor, hidden_dim=64).flatten(-2)
        ak_feat = head.plan_anchor_encoder(emb).view(bs, K, -1)
        cls_list = head.diff_decoder_cls(ak_feat, plan_anchor, bev_feature, bev_spatial_shape, agents_query, ego_query, global_img)
        final, head0, stats = rbd._denoise(head, plan_anchor, bev_feature, bev_spatial_shape, agents_query, ego_query, global_img)
        out = {k: stats[k] for k in ("path_len", "last_move", "x0_first_dev", "x0_mean_dev", "travel", "kinetic")}
        out["logit0"], out["logit1"] = cls_list[0], cls_list[-1]
        out["final"] = torch.cat([final, head0], dim=-1)
        out["tokens_ag"] = agents_query.half()
        holder["out"] = out
        mode_idx = cls_list[-1].argmax(dim=-1)
        sel = torch.gather(out["final"], 1, mode_idx[..., None, None, None].repeat(1, 1, out["final"].shape[2], 3)).squeeze(1)
        return {"trajectory": sel}
    head.forward_test = forward_test


def future_boxes(scene, frame_idx):
    """GT boxes t0..t0+FUT_T-1 in the t0 ego frame, one slot per track token (hidden_dualcls._compute_future_boxes)."""
    boxes = np.zeros((FUT_T, FUT_N, 5), np.float32); types = np.full((FUT_T, FUT_N), 3, np.int8); valid = np.zeros((FUT_T, FUT_N), bool)
    slot: Dict[str, int] = {}
    p0 = scene.frames[frame_idx].ego_status.ego_pose; c0, s0 = np.cos(p0[2]), np.sin(p0[2])
    for k in range(FUT_T):
        if frame_idx + k >= len(scene.frames):
            break
        fr = scene.frames[frame_idx + k]; pk = fr.ego_status.ego_pose; ck, sk = np.cos(pk[2]), np.sin(pk[2])
        rows = []
        for box, name, tok in zip(fr.annotations.boxes, fr.annotations.names, fr.annotations.track_tokens):
            if name in ("czone_sign", "traffic_sign"):
                continue
            bx, by, bh = box[BoundingBoxIndex.X], box[BoundingBoxIndex.Y], box[BoundingBoxIndex.HEADING]
            gx, gy = pk[0] + ck * bx - sk * by, pk[1] + sk * bx + ck * by
            dx, dy = gx - p0[0], gy - p0[1]
            lx, ly = c0 * dx + s0 * dy, -s0 * dx + c0 * dy
            if not (-15.0 <= lx <= 75.0 and -25.0 <= ly <= 25.0):
                continue
            rows.append((lx * lx + ly * ly, tok, TYPE_CODE.get(name, 3), [lx, ly, bh + pk[2] - p0[2], box[BoundingBoxIndex.LENGTH], box[BoundingBoxIndex.WIDTH]]))
        rows.sort(key=lambda r: r[0])
        for _, tok, code, b in rows:
            if tok not in slot:
                if len(slot) >= FUT_N:
                    continue
                slot[tok] = len(slot)
            i = slot[tok]; boxes[k, i], types[k, i], valid[k, i] = b, code, True
    return boxes, types, valid


@hydra.main(config_path=rbd.CONFIG_PATH, config_name=rbd.CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    agent = instantiate(cfg.agent); agent.initialize()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent.to(device).eval()
    model = agent._transfuser_model
    head = model._trajectory_head
    holder: Dict[str, Any] = {}
    install_forward_test(head, holder)
    model._tf_decoder.register_forward_hook(lambda m, inp, out: holder.__setitem__("tokens_kv", inp[1].half()))
    tgt_builder = TransfuserTargetBuilder(agent._config)
    num_poses = agent._config.trajectory_sampling.num_poses
    assert num_poses + 1 == FUT_T
    sim_check = instantiate(cfg.simulator); assert sim_check.proposal_sampling == instantiate(cfg.scorer).proposal_sampling

    sf = instantiate(cfg.train_test_split.scene_filter)
    tok_loader = SceneLoader(sensor_blobs_path=None, data_path=Path(cfg.navsim_log_path), scene_filter=sf, sensor_config=SensorConfig.build_no_sensors())
    feat_loader = SceneLoader(sensor_blobs_path=Path(cfg.sensor_blobs_path), data_path=Path(cfg.navsim_log_path), scene_filter=sf, sensor_config=agent.get_sensor_config())
    mc_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    tokens = sorted(set(tok_loader.tokens) & set(mc_loader.tokens))
    sub_n = int(os.environ.get("SUBSAMPLE", 0))
    if sub_n and sub_n < len(tokens):
        tokens = sorted(np.random.RandomState(0).choice(tokens, size=sub_n, replace=False).tolist())
    batches = [tokens[i:i + GPU_BATCH_SIZE] for i in range(0, len(tokens), GPU_BATCH_SIZE)]
    logger.info("selector pool dump: %d scenes, %d batches -> %s", len(tokens), len(batches), DUMP_DIR)
    np.save(DUMP_DIR / "anchors.npy", head.plan_anchor.detach().cpu().numpy())
    builders = agent.get_feature_builders()

    def build_one(token):
        try:
            scene = feat_loader.get_scene_from_token(token)
            ai = feat_loader.get_agent_input_from_token(token)
            f = {}
            for b in builders:
                f.update(b.compute_features(ai))
            fi = scene.scene_metadata.num_history_frames - 1
            ga, gl = tgt_builder._compute_agent_targets(scene.frames[fi].annotations)
            extra = {"gt_agents": ga.numpy(), "gt_agents_valid": gl.numpy()}
            extra["gt_future"], extra["gt_future_type"], extra["gt_future_valid"] = future_boxes(scene, fi)
            return token, f, scene.get_future_trajectory(num_trajectory_frames=num_poses).poses, extra, None
        except Exception:
            return token, None, None, None, traceback.format_exc()

    def build_batch(bt):
        feats, gts, extras, paths, toks = [], [], [], [], []
        with ThreadPoolExecutor(max_workers=FEATURE_BUILD_THREADS) as ex:
            for token, f, gt, extra, err in ex.map(build_one, bt):
                if err:
                    logger.warning("feature build failed %s\n%s", token, err); continue
                feats.append(f); gts.append(gt); extras.append(extra); paths.append(mc_loader.metric_cache_paths[token]); toks.append(token)
        return feats, gts, extras, paths, toks

    ctx = get_context("spawn")
    pool = ctx.Pool(SCORE_POOL_WORKERS, initializer=rbd._init_worker, initargs=(cfg.simulator, cfg.scorer), maxtasksperchild=300)
    prefetch = ThreadPoolExecutor(max_workers=1)
    nxt = prefetch.submit(build_batch, batches[0]) if batches else None
    shard: Dict[str, List[Any]] = {}; shard_id = [0]; n_done = [0]; n_fail = [0]

    def harvest(meta, ar):
        for (tok, per, gt, extra), (sub, ok) in zip(meta, ar.get()):
            if not ok:
                n_fail[0] += 1; continue
            shard.setdefault("tokens", []).append(tok)
            shard.setdefault("gt", []).append(gt.astype(np.float32))
            shard.setdefault("pdm", []).append(sub[:-1]); shard.setdefault("human_pdm", []).append(sub[-1])  # human scored as candidate K
            for k, v in per.items():
                shard.setdefault(k, []).append(v)
            for k, v in extra.items():
                shard.setdefault(k, []).append(v)
            n_done[0] += 1

    def flush():
        if not shard:
            return
        np.savez(DUMP_DIR / f"shard_{shard_id[0]:04d}.npz", **{k: np.stack(v) if k != "tokens" else np.array(v) for k, v in shard.items()})
        logger.info("flushed shard %d (%d scenes total, %d failed)", shard_id[0], n_done[0], n_fail[0])
        shard_id[0] += 1; shard.clear()

    pending = None; t0 = time.time()
    try:
        with torch.no_grad():
            for bi in range(len(batches)):
                feats, gts, extras, paths, toks = nxt.result()
                if bi + 1 < len(batches):
                    nxt = prefetch.submit(build_batch, batches[bi + 1])
                meta, ar = [], None
                if toks:
                    torch.manual_seed(bi)  # seeded sampler: the dump is reproducible
                    fb = {k: torch.stack([f[k] for f in feats]).to(device) for k in feats[0].keys()}
                    out = agent.forward(fb, targets=None)
                    sig = {k: v.detach().cpu().numpy() for k, v in holder["out"].items()}
                    sig["tokens_kv"] = holder["tokens_kv"].detach().cpu().numpy()
                    sig["pred_agents"] = out["agent_states"].detach().float().cpu().numpy()
                    sig["pred_agents_logit"] = out["agent_labels"].detach().float().cpu().numpy()
                    sig["pred_bev"] = out["bev_semantic_map"].argmax(1).to(torch.uint8).cpu().numpy()
                    assert np.isfinite(sig["final"]).all(), "non-finite candidates"
                    cands = sig["final"].astype(np.float32)
                    for i, tok in enumerate(toks):
                        meta.append((tok, {k: v[i] for k, v in sig.items()}, gts[i], extras[i]))
                    ar = pool.map_async(rbd._score_scene, [(paths[i], np.concatenate([cands[i], gts[i][None].astype(np.float32)])) for i in range(len(toks))])
                if pending is not None:
                    harvest(*pending)
                pending = (meta, ar) if ar is not None else None
                if (bi + 1) % FLUSH_EVERY == 0:
                    flush()
                if (bi + 1) % 5 == 0:
                    el = time.time() - t0
                    logger.info("batch %d/%d  %.1f min elapsed, ETA %.1f min", bi + 1, len(batches), el / 60, el / (bi + 1) * (len(batches) - bi - 1) / 60)
            if pending is not None:
                harvest(*pending)
        flush()
    finally:
        prefetch.shutdown(wait=False); pool.close(); pool.join()
    logger.info("DONE: %d scenes dumped, %d failed -> %s", n_done[0], n_fail[0], DUMP_DIR)


if __name__ == "__main__":
    main()
