"""§5f: frozen DD backbone on cached camera+LiDAR, pooled BEV per scene, for lags 0 (current), 1, 2, 3.
Output: POOL_OUT/sensor_pool.npz with tokens (N,), lag0..lag3 (N,512) = spatial mean of the DD BEV (8x8).
Inputs: sensor_cache_dd_navtest (current) and sensor_cache_dd_navtest_hist/lag{1,2,3}. Folder name == pool token.
  bash test/selector/pool_sensor_feats.sh
"""
import logging, os, pickle, gzip
from pathlib import Path
import hydra, numpy as np, torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
REPO_CONFIG = "/mnt/jimmys/public_repos/DiffusionDrive/navsim/planning/script/config/pdm_scoring"
EXP = Path(os.environ.get("NAVSIM_EXP_ROOT", "/mnt/ds/navsim-main/exp"))
POOL_TOKENS = Path(os.environ.get("POOL_TOKENS", EXP / "selector/dd_k20_public_v2/navtest"))
if os.environ.get("CACHE_ROOT"):  # §5h: one root with lag0..lag3 subdirs (navtrain-20k)
    _root = Path(os.environ["CACHE_ROOT"])
    DIRS = {l: _root / f"lag{l}" for l in range(4)}
else:
    DIRS = {0: EXP / "sensor_cache_dd_navtest", 1: EXP / "sensor_cache_dd_navtest_hist/lag1",
            2: EXP / "sensor_cache_dd_navtest_hist/lag2", 3: EXP / "sensor_cache_dd_navtest_hist/lag3"}
OUT = Path(os.environ.get("POOL_OUT", EXP / "selector/sensor_pool"))


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
    """Map token -> feature file (token folder is nested under log_name)."""
    idx = {}
    for log in root.iterdir():
        if not log.is_dir():
            continue
        for tok in log.iterdir():
            if tok.is_dir():
                idx[tok.name] = tok / "transfuser_feature.gz"
    return [idx[t] for t in tokens]


@hydra.main(config_path=REPO_CONFIG, config_name="default_run_pdm_score", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    OUT.mkdir(parents=True, exist_ok=True)
    agent = instantiate(cfg.agent)
    agent.initialize()  # loads public DD ckpt (checkpoint_path set in .sh)
    agent.eval().to("cuda")
    backbone = agent._transfuser_model._backbone
    tokens = []
    for f in sorted(POOL_TOKENS.glob("shard_*.npz")):
        tokens += np.load(f)["tokens"].tolist()
    logger.info("pooling %d scenes, lags %s", len(tokens), list(DIRS))
    res = {"tokens": np.array(tokens)}
    for lag, root in DIRS.items():
        paths = scene_paths(root, tokens)
        dl = DataLoader(CacheFeat(paths), batch_size=32, num_workers=8)
        out = []
        with torch.no_grad():
            for cam, lid in dl:
                _, bev, _ = backbone(cam.cuda(), lid.cuda())
                out.append(bev.mean((2, 3)).float().cpu().numpy())
        res[f"lag{lag}"] = np.concatenate(out)
        logger.info("lag %d pooled %s", lag, res[f"lag{lag}"].shape)
    np.savez(OUT / "sensor_pool.npz", **res)
    logger.info("wrote %s", OUT / "sensor_pool.npz")


if __name__ == "__main__":
    main()
