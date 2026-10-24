"""§5f S3 history cache: DD camera + LiDAR features at past frames (lag 1..3 at 2 Hz) for navtest.
Same builder, same per-scene layout as the devkit cache (log_name/initial_token/transfuser_feature.gz),
one cache dir per lag. Current frame is the existing sensor_cache_dd_navtest. Resumes (skips existing files).
  N_THREADS=20 python test/selector/cache_dd_history.py   (run via cache_dd_history.sh for env)
"""
import logging, os, traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.dataset import dump_feature_target_to_pickle
from navsim.agents.diffusiondrive.transfuser_features import TransfuserFeatureBuilder

logger = logging.getLogger(__name__)
REPO_CONFIG = "/mnt/jimmys/public_repos/DiffusionDrive/navsim/planning/script/config/pdm_scoring"
LAGS = [1, 2, 3]  # past frames at 2 Hz, fixed in advance (Experiment Plan §5f)
N_THREADS = int(os.environ.get("N_THREADS", 20))
CACHE_ROOT = Path(os.environ["CACHE_ROOT"])


def _lag_input(agent_input, lag):
    """Current-frame view of frame (current - lag): builders read cameras[-1] and lidars[-1]."""
    return replace(
        agent_input,
        cameras=[agent_input.cameras[-1 - lag]],
        lidars=[agent_input.lidars[-1 - lag]],
    )


@hydra.main(config_path=REPO_CONFIG, config_name="default_run_pdm_score", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    agent = instantiate(cfg.agent)
    builder = TransfuserFeatureBuilder(config=agent._config)
    # DD builder reads cam_l0/f0/r0 and lidar_pc; load them at every history step (index 0..3), not only current.
    hist = [0, 1, 2, 3]
    sensor_config = SensorConfig(
        cam_f0=hist, cam_l0=hist, cam_l1=False, cam_l2=False, cam_r0=hist, cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=hist,
    )
    loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path), data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter), sensor_config=sensor_config,
    )
    tokens = list(loader.tokens)
    logger.info("history cache: %d scenes, lags %s -> %s", len(tokens), LAGS, CACHE_ROOT)

    def work(token):
        try:
            scene = loader.get_scene_from_token(token)
            meta = scene.scene_metadata
            agent_input = scene.get_agent_input()
            for lag in LAGS:
                out = CACHE_ROOT / f"lag{lag}" / meta.log_name / meta.initial_token / (builder.get_unique_name() + ".gz")
                if out.exists():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                dump_feature_target_to_pickle(out, builder.compute_features(_lag_input(agent_input, lag)))
            return token, None
        except Exception:
            return token, traceback.format_exc()

    failed = []
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        for i, (token, err) in enumerate(ex.map(work, tokens), 1):
            if err:
                failed.append((token, err))
            if i % 1000 == 0:
                logger.info("%d / %d done, %d failed", i, len(tokens), len(failed))
    logger.info("done: %d scenes, %d failed", len(tokens), len(failed))
    if failed:
        (CACHE_ROOT / "failed.txt").write_text("\n".join(f"{t}\n{e}" for t, e in failed))


if __name__ == "__main__":
    main()
