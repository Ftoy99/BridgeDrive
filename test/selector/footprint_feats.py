"""§5j (Amendment A1, Selector_Experiment_Plan.md §5j): footprint-sampled DD BEV probabilities.

Per scene (lag-0 cache, frozen DD backbone): softmax of DD's semantic head (7 classes, 128x256, §5i grid: row=floor(4x) forward,
col=floor(4y)+128). Sampled (nearest pixel) at the 4 corners + centre of the devkit box at each of the 8 waypoints of each of the
20 candidates and of the human future. Labels (train only): GT semantic BEV at the same points; box classes and out-of-window = no label.
Output: POOL_OUT/feats.npz. No navtest label is used here; navtest is read by the head script only.
  SPLIT=navtrain20k|navtest  bash test/selector/footprint_feats.sh
"""
import glob, gzip, logging, os, pickle
from pathlib import Path
import hydra, numpy as np, torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
REPO_CONFIG = "/mnt/jimmys/public_repos/DiffusionDrive/navsim/planning/script/config/pdm_scoring"
EXP = Path("/mnt/ds/navsim-main/exp")
SPLIT = os.environ["SPLIT"]
POOL = EXP / "selector/dd_k20_public_v2" / SPLIT.replace("navtrain20k", "navtrain")
CACHE = {"navtrain20k": EXP / "sensor_cache_dd_navtrain20k/lag0", "navtest": EXP / "sensor_cache_dd_navtest"}[SPLIT]
GTDIR = {"navtrain20k": EXP / "selector/diag_bev_vs_map/gtbev_navtrain20k", "navtest": EXP / "selector/diag_bev_vs_map/gtbev_navtest"}[SPLIT]
OUT = Path(os.environ.get("POOL_OUT", EXP / "selector/footprint_feats" / SPLIT))
L, W, FWD = 5.176, 2.297, 1.461  # devkit Pacifica box (§5i GEOM devkit)
PX, BH, BW = 4.0, 128, 256
ROAD, BOX = (1, 3), (4, 5, 6)


class CacheFeat(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with gzip.open(self.paths[i], "rb") as f:
            d = pickle.load(f)
        return d["camera_feature"], d["lidar_feature"]


def scene_paths(root, tokens):
    idx = {}
    for log in root.iterdir():
        if log.is_dir():
            for tok in log.iterdir():
                if tok.is_dir():
                    idx[tok.name] = tok / "transfuser_feature.gz"
    return [idx[t] for t in tokens]


def points(poses):
    """(...,3) ego (x,y,h) -> (...,5,2) world points: 4 corners then centre, and pixel (row, col, in_window)."""
    x, y, h = poses[..., 0], poses[..., 1], poses[..., 2]
    cx, sx = np.cos(h), np.sin(h)
    ox, oy = x + FWD * cx, y + FWD * sx
    pts = [(ox + a * L / 2 * cx - b * W / 2 * sx, oy + a * L / 2 * sx + b * W / 2 * cx) for a in (1, -1) for b in (1, -1)]
    pts.append((ox, oy))
    px = np.stack([p[0] for p in pts], -1)
    py = np.stack([p[1] for p in pts], -1)
    row = np.floor(PX * px).astype(np.int64)
    col = np.floor(PX * py).astype(np.int64) + BW // 2
    inside = (row >= 0) & (row < BH) & (col >= 0) & (col < BW)
    return np.clip(row, 0, BH - 1), np.clip(col, 0, BW - 1), inside


def load_gt(tokens):
    m = {}
    for f in sorted(glob.glob(str(GTDIR / "*.npz"))):
        d = np.load(f, allow_pickle=True)
        for t, g in zip(d["tokens"], d["gt_bev"]):
            m[str(t)] = g
    return m


@hydra.main(config_path=REPO_CONFIG, config_name="default_run_pdm_score", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    OUT.mkdir(parents=True, exist_ok=True)
    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval().to("cuda")
    backbone = agent._transfuser_model._backbone
    head = agent._transfuser_model._bev_semantic_head
    recs = [np.load(f, allow_pickle=True) for f in sorted(POOL.glob("shard_*.npz"))]
    cat = lambda k: np.concatenate([r[k] for r in recs])
    tokens, traj, pdm, logit, gt_future = cat("tokens"), cat("traj"), cat("pdm"), cat("logit"), cat("gt_future")
    gt = load_gt(tokens)
    have = np.array([str(t) in gt for t in tokens])
    logger.info("pool %d scenes, GT BEV for %d (dropped %d)", len(tokens), have.sum(), (~have).sum())
    tokens, traj, pdm, logit, gt_future = tokens[have], traj[have], pdm[have], logit[have], gt_future[have]
    gt_arr = np.stack([gt[str(t)] for t in tokens])
    del gt
    N, K, P = traj.shape[0], traj.shape[1], traj.shape[2]
    poses = np.concatenate([traj, gt_future[:, None]], axis=1)  # (N, K+1, P, 3); last row = human
    row, col, inside = points(poses)  # (N, K+1, P, 5)
    # labels: GT class at the point; box classes and out-of-window get 255 (no label)
    gtc = gt_arr[np.arange(N)[:, None, None, None], row, col].astype(np.uint8)
    gtc = np.where(inside & ~np.isin(gtc, BOX), gtc, 255)
    paths = scene_paths(CACHE, tokens.tolist())
    dl = DataLoader(CacheFeat(paths), batch_size=32, num_workers=8)
    feats = np.zeros((N, K + 1, P, 5, 7), dtype=np.float16)
    offset = 0
    with torch.no_grad():
        for cam, lid in dl:
            bev_up, _, _ = backbone(cam.cuda(), lid.cuda())
            prob = torch.softmax(head(bev_up), dim=1).permute(0, 2, 3, 1).cpu().numpy()  # (B,128,256,7)
            b = prob.shape[0]
            r, c = row[offset:offset + b], col[offset:offset + b]
            feats[offset:offset + b] = prob[np.arange(b)[:, None, None, None], r, c]
            offset += b
    assert offset == N and np.allclose(feats[:2].astype(np.float32).sum(-1), 1.0, atol=1e-2)
    am = feats.argmax(-1)
    gate_pt = np.isin(am, ROAD) | np.isin(am, BOX) | ~inside
    np.savez(OUT / "feats.npz", tokens=tokens, feats=feats, inside=inside, gtc=gtc, gate_pt=gate_pt,
             dac=pdm[..., 1] >= 0.5, pdm=pdm, logit=logit, poses=poses)
    logger.info("wrote %s (N=%d K=%d P=%d)", OUT / "feats.npz", N, K, P)


if __name__ == "__main__":
    main()
