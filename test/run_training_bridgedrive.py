from typing import Tuple
from pathlib import Path
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule

logger = logging.getLogger(__name__)


class CycleEndCheckpoint(pl.Callback):
    """Save a checkpoint at the end of every cosine warm-restart cycle (lowest-LR point of each cycle).
    With warm restarts the lowest-validation-loss checkpoints are not necessarily the cycle ends, so keep these too.
    end_epochs are 0-based epoch indices (e.g. T0=10, T_mult=2 -> 9, 29, 69, 149)."""

    def __init__(self, dirpath: str, end_epochs):
        self.dirpath = dirpath
        self.end_epochs = set(end_epochs)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch in self.end_epochs:
            os.makedirs(self.dirpath, exist_ok=True)
            trainer.save_checkpoint(os.path.join(self.dirpath, f"cycle-end-epoch{trainer.current_epoch:03d}.ckpt"))


class CompactProgressBar(TQDMProgressBar):
    """Single-line bar: show only a few metrics (all are still logged to the loggers)."""

    KEEP = ("train/loss_step", "train/trajectory_loss_step", "val/trajectory_loss_epoch")

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        short = {"train/loss_step": "loss", "train/trajectory_loss_step": "traj", "val/trajectory_loss_epoch": "val_traj"}
        return {short[k]: v for k, v in items.items() if k in self.KEEP}

# Copy of navsim/planning/script/run_training.py with checkpoint selection added (repo file untouched).
CONFIG_PATH = "../BridgeDrive_adaptation_DiffusionDrive/navsim/planning/script/config/training"
CONFIG_NAME = "default_training"


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


def _flatten_classmethod_properties() -> None:
    """Dynamo cannot trace `@classmethod @property` (e.g. StateSE2Index.HEADING in the head, the cause of
    "AssertionError: expected FunctionType found property"). Replace each with its resolved plain value."""
    import inspect
    import navsim.agents.diffusiondrive.model_diffusion_head_ddbm_v5 as head
    n = 0
    for _, cls in inspect.getmembers(head, inspect.isclass):
        if cls.__module__ != head.__name__:
            continue
        for name, raw in list(vars(cls).items()):
            if isinstance(raw, classmethod) and isinstance(raw.__func__, property):
                setattr(cls, name, getattr(cls, name))
                n += 1
    logger.info("flattened %d classmethod-property attributes for torch.compile", n)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    if os.environ.get("COMPILE", "1") == "1":
        _flatten_classmethod_properties()
        # in-place Module.compile() on the backbone: keeps state_dict keys identical (no "_orig_mod." prefix), so ckpts stay loadable by the repo's eval
        logger.info("torch.compile enabled (COMPILE=0 to disable)")
        # whole-model compile fails in dynamo (AssertionError: expected FunctionType found property)
        mode = os.environ.get("COMPILE_MODE", "default")
        agent._transfuser_model._backbone.compile(mode=mode)
        if os.environ.get("COMPILE_DECODER", "0") == "1":
            # EXPERIMENTAL: Hidden compiles diff_decoder too (hidden/train.py:485), but here dynamo currently dies with
            # "SyntaxError: invalid syntax (<eval_with_key>...)" while generating the graph (torch 2.5.1)
            agent._transfuser_model._trajectory_head.diff_decoder.compile(mode=mode)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    monitor = "val/trajectory_loss_epoch"
    callbacks = agent.get_training_callbacks() + [
        CompactProgressBar(refresh_rate=int(os.environ.get("PROGRESS_REFRESH", 1))),
        # top-K lowest val trajectory loss + last.ckpt (resume); filenames kept slash-free
        ModelCheckpoint(
            dirpath=os.path.join(str(cfg.output_dir), "checkpoints"),
            monitor=monitor,
            mode="min",
            save_top_k=int(os.environ.get("TOPK", 3)),
            save_last=True,
            filename="best-epoch{epoch:03d}-valtraj{val/trajectory_loss_epoch:.3f}",
            auto_insert_metric_name=False,
        )
    ]
    cycle_ends = [int(x) for x in os.environ.get("CYCLE_END_EPOCHS", "").split(",") if x.strip()]
    if cycle_ends:
        callbacks.append(CycleEndCheckpoint(os.path.join(str(cfg.output_dir), "checkpoints"), cycle_ends))
    patience = int(os.environ.get("PATIENCE", 0))  # 0 = no early stopping (paper: fixed epochs)
    if patience > 0:
        callbacks.append(EarlyStopping(monitor=monitor, patience=patience, mode="min", verbose=True))
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=callbacks)

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=cfg.agent.checkpoint_path
    )


if __name__ == "__main__":
    main()
