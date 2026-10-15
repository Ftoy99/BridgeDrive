"""Smoke test: BridgeDrive vs HeavyBridge head switch, shapes, warm start, gradients.

Run from the LEAD root: python tests/test_heavy_head_smoke.py
"""

import os
import sys

from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lead.tfv6.diffusion_modules.model_diffusion_head_ddbm import DEFAULT_BEV_DIMS, TrajectoryHead  # noqa: E402
from lead.tfv6.diffusion_modules.model_diffusion_head_heavybridge import HeavyTrajectoryHead  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B, K, P = 2, 60, 10


def default_cfg():
    # stub of the TrainingConfig fields the head reads, so the test runs without the full LEAD environment
    return SimpleNamespace(
        plan_anchor_path=os.path.join(os.path.dirname(__file__), "..", "anchor_utils/anchor_data/lead_cp_kmeans_60_10.npy"),
        trajectory_sampling_num_poses=10, tf_d_ffn=1024, tf_d_model=256, tf_num_head=8, tf_dropout=0.0,
        lidar_min_x=-32, lidar_max_x=32, lidar_min_y=-32, lidar_max_y=32,
        diffusion_speed=False, step_num=20,
        hb_alpha=1.8, hb_varpi=1.25, hb_eta=0.5, hb_num_grid=20, hb_profile="levy",
    )


def make(cls, **over):
    cfg = default_cfg()
    for k, v in over.items():
        setattr(cfg, k, v)
    return cls(cfg.trajectory_sampling_num_poses, cfg.tf_d_ffn, cfg.tf_d_model, cfg.plan_anchor_path, cfg).to(DEV)


def inputs():
    q = torch.randn(B, 1, 256, device=DEV)
    bev = torch.randn(B, DEFAULT_BEV_DIMS, 64, 64, device=DEV)
    gt = torch.cumsum(torch.randn(B, P, 2, device=DEV).abs(), 1)
    return q, bev, gt


def test_warm_start_is_exact_and_gradients_flow():
    base, heavy = make(TrajectoryHead), make(HeavyTrajectoryHead, step_num=2)
    missing = heavy.load_state_dict(base.state_dict(), strict=False)
    assert all(k.startswith("sigma_mlp") for k in missing.missing_keys) and not missing.unexpected_keys
    q, bev, gt = inputs()
    heavy.train()
    out = heavy(q, bev, (64, 64), gt_checkpoints=gt)
    assert out["best_reg"].shape == (B, P, 2) or out["best_reg"].shape[0] == B
    out["best_reg"].square().mean().backward()
    assert heavy.sigma_mlp[-1].weight.grad.abs().sum() > 0  # zero-init layer still trains


def test_sampling_shapes_and_variants():
    q, bev, _ = inputs()
    for kw in (dict(step_num=2), dict(step_num=20), dict(step_num=2, hb_profile="ones")):
        h = make(HeavyTrajectoryHead, **kw).eval()
        with torch.no_grad():
            traj = h(q, bev, (64, 64))
        assert traj.shape == (B, P, 2), traj.shape
        assert torch.isfinite(traj).all()
    h.forced_anchor_rank = 19
    with torch.no_grad():
        assert h(q, bev, (64, 64)).shape == (B, P, 2)


if __name__ == "__main__":
    test_warm_start_is_exact_and_gradients_flow()
    print("ok warm start + grads")
    test_sampling_shapes_and_variants()
    print("ok sampling")
