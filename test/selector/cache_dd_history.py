"""§5f S3 history cache: DD camera + LiDAR features at past frames (lag 1..3 at 2 Hz) for navtest.
Same builder, same per-scene layout as the devkit cache (log_name/initial_token/transfuser_feature.gz),
one cache dir per lag. Current frame is the existing sensor_cache_dd_navtest. Resumes (skips existing files).
  N_THREADS=20 python test/selector/cache_dd_history.py   (run via cache_dd_history.sh for env)
"""
import json, logging, os, time, traceback
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
# past frames at 2 Hz, fixed in advance (Experiment Plan §5f). LAGS=0,1,2,3 env builds the current-frame cache too (§5h).
LAGS = [int(x) for x in os.environ.get("LAGS", "1,2,3").split(",")]
TOKENS_FILE = os.environ.get("TOKENS_FILE")  # optional token list (one per line): restricts the scene set (§5h navtrain-20k)
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
    if TOKENS_FILE:
        if TOKENS_FILE.endswith(".json"):  # E-sel-B fit list {tokens, logs}: processing order = list order, scenes cached in the list's log are skipped without loading
            fl = json.loads(Path(TOKENS_FILE).read_text())
            order, log_of = fl["tokens"], dict(zip(fl["tokens"], fl["logs"]))
            have = set(tokens)
            assert all(t in have for t in order), f"{sum(t not in have for t in order)} wanted tokens not in filter"
            tokens = order
        else:
            wanted = set(Path(TOKENS_FILE).read_text().split())
            tokens = [t for t in tokens if t in wanted]
            assert len(tokens) == len(wanted), f"{len(wanted) - len(tokens)} wanted tokens not in filter"
    logger.info("history cache: %d scenes, lags %s -> %s", len(tokens), LAGS, CACHE_ROOT)

    log_of = locals().get("log_of", {})
    names = builder.get_unique_name() + ".gz"

    def work(token):
        try:
            if log_of and all((CACHE_ROOT / f"lag{l}" / log_of[token] / token / names).exists() for l in LAGS):
                return token, None
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

    failed, t0, mark = [], time.time(), int(os.environ.get("MARK_AT", 0))
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        for i, (token, err) in enumerate(ex.map(work, tokens), 1):
            if err:
                failed.append((token, err))
            if i % 1000 == 0:
                eta = (time.time() - t0) / i * (len(tokens) - i)
                logger.info("%d / %d done, %d failed, ETA %.1f h (includes skipped scenes)", i, len(tokens), len(failed), eta / 3600)
            if mark and i == mark:
                logger.info("STAGE_B_CACHE_50K %d scenes", i)
    logger.info("done: %d scenes, %d failed", len(tokens), len(failed))
    print("STAGE_B_CACHE_DONE" if not failed else f"STAGE_B_CACHE_DONE with {len(failed)} failed", flush=True)
    if failed:
        (CACHE_ROOT / "failed.txt").write_text("\n".join(f"{t}\n{e}" for t, e in failed))


if __name__ == "__main__":
    main()
