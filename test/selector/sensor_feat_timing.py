"""§5f timing test: frozen DD backbone features (camera + LiDAR) for a few navtest scenes.
Measures feature-build time, GPU backbone time, and per-scene feature sizes. No dump, no scoring.
  N_TIME=10 bash test/selector/sensor_feat_timing.sh
"""
import logging, os, time
from pathlib import Path
import hydra, numpy as np, torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SensorConfig

logger = logging.getLogger(__name__)
REPO_CONFIG = "/mnt/jimmys/public_repos/DiffusionDrive/navsim/planning/script/config/pdm_scoring"
N_TIME = int(os.environ.get("N_TIME", 10))


@hydra.main(config_path=REPO_CONFIG, config_name="default_run_pdm_score", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval().to("cuda")
    backbone = agent._transfuser_model._backbone
    builders = agent.get_feature_builders()
    loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path), data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter), sensor_config=agent.get_sensor_config(),
    )
    tokens = loader.tokens[:N_TIME]
    logger.info("timing %d scenes", len(tokens))

    t_build, t_gpu, shapes = [], [], None
    with torch.no_grad():
        for token in tokens:
            t0 = time.perf_counter()
            agent_input = loader.get_agent_input_from_token(token)
            feat = {}
            for b in builders:
                feat.update(b.compute_features(agent_input))
            t1 = time.perf_counter()
            cam = feat["camera_feature"].unsqueeze(0).cuda()
            lid = feat["lidar_feature"].unsqueeze(0).cuda()
            bev_up, bev, _ = backbone(cam, lid)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            t_build.append(t1 - t0)
            t_gpu.append(t2 - t1)
            if shapes is None:
                shapes = dict(
                    camera=tuple(cam.shape), lidar=tuple(lid.shape),
                    bev_upscale=tuple(bev_up.shape), bev=tuple(bev.shape),
                    bev_fp32_mb=bev.numel() * 4 / 1e6, bev_up_fp32_mb=bev_up.numel() * 4 / 1e6,
                )

    # skip first (warm-up) timing
    tb, tg = np.array(t_build[1:]), np.array(t_gpu[1:])
    logger.info("shapes %s", shapes)
    logger.info("feature build per scene: median %.3f s, mean %.3f s", np.median(tb), tb.mean())
    logger.info("backbone GPU per scene: median %.4f s, mean %.4f s", np.median(tg), tg.mean())
    logger.info("GPU mem peak %.2f GB", torch.cuda.max_memory_allocated() / 1e9)


if __name__ == "__main__":
    main()
