"""Smoke test: HeavyBridge NAVSIM head vs BridgeDrive head: warm start, gradients, sampling shapes, variants.

Run from this dir: PYTHONPATH=. python tests/test_heavy_head_smoke.py   (env with navsim deps, e.g. conda `hidden`)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from navsim.agents.diffusiondrive.model_diffusion_head_ddbm_v5 import TrajectoryHead  # noqa: E402
from navsim.agents.diffusiondrive.model_diffusion_head_heavybridge import HeavyBridgeTrajectoryHead  # noqa: E402
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B, P = 2, 8
ANCHOR = os.path.join(os.path.dirname(__file__), "..", "kmeans_navsim", "navsim_anchor_k80.npy")


def make(cls, **over):
    cfg = TransfuserConfig(plan_anchor_path=ANCHOR, beta_d=2.0, **over)
    return cls(cfg.trajectory_sampling.num_poses, cfg.tf_d_ffn, cfg.tf_d_model, cfg.plan_anchor_path, cfg).to(DEV)


def inputs(bs=B):
    q = torch.randn(bs, 1, 256, device=DEV)
    agents = torch.randn(bs, 30, 256, device=DEV)
    bev = torch.randn(bs, 256, 8, 8, device=DEV)
    traj = torch.cumsum(torch.rand(bs, P, 3, device=DEV), 1)
    return q, agents, bev, {"trajectory": traj, "agent_states": None}


def test_warm_start_and_grads():
    base, heavy = make(TrajectoryHead), make(HeavyBridgeTrajectoryHead)
    msg = heavy.load_state_dict(base.state_dict(), strict=False)
    assert msg.missing_keys and all(k.startswith("sigma_mlp") for k in msg.missing_keys) and not msg.unexpected_keys
    q, agents, bev, targets = inputs()
    heavy.train()
    out = heavy(q, agents, bev, (8, 8), None, targets=targets)
    out["trajectory_loss"].backward()
    assert heavy.sigma_mlp[-1].weight.grad.abs().sum() > 0  # zero-init layer still trains


def test_sampling():
    q, agents, bev, _ = inputs()
    for kw in (dict(), dict(hb_step_num=20), dict(hb_profile="ones")):
        h = make(HeavyBridgeTrajectoryHead, **kw).eval()
        with torch.no_grad():
            out = h(q, agents, bev, (8, 8), None)
        assert out["trajectory"].shape == (B, P, 3), out["trajectory"].shape
        assert out["poses_all"].shape[:2] == (B, 80) and out["poses_cls"].shape == (B, 80)
        assert torch.isfinite(out["trajectory"]).all()
    h.forced_anchor_rank = 79
    with torch.no_grad():
        assert h(q, agents, bev, (8, 8), None)["trajectory"].shape == (B, P, 3)


if __name__ == "__main__":
    test_warm_start_and_grads()
    print("ok warm start + grads")
    test_sampling()
    print("ok sampling")
