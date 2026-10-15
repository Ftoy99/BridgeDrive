"""Checks for navsim/agents/diffusiondrive/modules/heavy_bridge_sampler.py against numbers published in the HeavyBridge paper.

Run: python -m pytest tests/test_heavy_bridge_sampler.py   (or python tests/test_heavy_bridge_sampler.py)
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from navsim.agents.diffusiondrive.modules.heavy_bridge_sampler import HeavyBridgeSchedule  # noqa: E402

DEV = torch.device("cpu")


def test_alpha2_stable_scale_equals_gaussian_step_variance():
    sch = HeavyBridgeSchedule(alpha=1.8, varpi=1.25)
    np.testing.assert_allclose(sch._stable_scale(2.0), sch.u, rtol=1e-10)  # Alg. 1 Precompute line 5


def test_kappa_matches_paper():
    # Sec. 4.3: gentle (1.8, 5/4) -> 0.956, aggressive (1.7, 5/3) -> 0.0285 on the N=20 VP schedule
    assert abs(HeavyBridgeSchedule(1.8, 5 / 4).kappa - 0.956) < 5e-3
    assert abs(HeavyBridgeSchedule(1.7, 5 / 3).kappa - 0.0285) < 5e-4


def test_acceptance_rate_matches_paper():
    # Sec. 4.5: schedule mean of exp(-c_i kappa^(alpha/2)) = 0.265 at the gentle calibration
    sch = HeavyBridgeSchedule(1.8, 5 / 4)
    acc = np.exp(-sch.c * sch.kappa ** (sch.alpha / 2)).mean()
    assert abs(acc - 0.265) < 5e-3, acc


def test_profile_mean_hits_target_jump_energy():
    sch = HeavyBridgeSchedule(1.8, 5 / 4)
    g = torch.Generator().manual_seed(0)
    lam = sch.draw_profile(200_000, DEV, generator=g)
    assert abs(lam.mean().item() - 1.25) < 0.03, lam.mean().item()


def test_lambda_one_matches_bridgedrive_scheduler():
    from navsim.agents.diffusiondrive.model_diffusion_head_ddbm_v5 import DDBMScheduler

    sch = HeavyBridgeSchedule(1.8, 5 / 4)
    ref = DDBMScheduler(beta_d=2.0, beta_min=0.1, T=1.0)
    B = 3
    s2 = sch.profile_sigma2(sch.ones_profile(B, DEV))
    a, b, c = sch.coefficients(s2)
    for i in range(1, sch.N):  # i = 0 and i = N are 0/0 or boundary in the reference
        ra, rb, rc = ref.get_abc(torch.tensor(i / sch.N, dtype=torch.float64))
        np.testing.assert_allclose(a[:, i].numpy(), ra.item(), rtol=1e-6)
        np.testing.assert_allclose(b[:, i].numpy(), rb.item(), rtol=1e-6)
        np.testing.assert_allclose(c[:, i].numpy(), rc.item(), rtol=1e-6)


def test_lambda_one_sampler_matches_bridgedrive_sample_step():
    # BridgeDrive = first step fully stochastic (eta=1), later steps deterministic (eta=0)
    from navsim.agents.diffusiondrive.model_diffusion_head_ddbm_v5 import DDBMScheduler

    sch = HeavyBridgeSchedule(1.8, 5 / 4)
    ref = DDBMScheduler(beta_d=2.0, beta_min=0.1, T=1.0)
    B, shape = 4, (4, 5, 10, 2)
    y = torch.randn(shape, dtype=torch.float64)
    x0 = torch.randn(shape, dtype=torch.float64)
    s2 = sch.profile_sigma2(sch.ones_profile(B, DEV))
    coefs = sch.coefficients(s2)
    N = sch.N
    xi = torch.randn(shape, dtype=torch.float64)

    # step N -> N-1
    ours = sch.step(y, x0, y, coefs, s2, N, N - 1, eta=1.0, noise=xi)
    t = torch.full((B,), 1000.0, dtype=torch.float64)
    t_prev = torch.full((B,), 1000.0 * (N - 1) / N, dtype=torch.float64)
    torch.manual_seed(1)
    torch.randn_like(y)  # keep generators separate; reference draws its own noise
    a_p, b_p, c_p = ref.get_abc(t_prev.reshape(-1, 1, 1, 1) / 1000.0)
    expect = a_p * y + b_p * x0 + c_p * xi
    np.testing.assert_allclose(ours.numpy(), expect.numpy(), rtol=1e-5, atol=1e-6)  # reference computes its T terms in float32

    # step N-1 -> N-2, deterministic
    x1 = ours
    ours2 = sch.step(x1, x0, y, coefs, s2, N - 1, N - 2, eta=0.0)
    t1 = torch.full((B,), 1000.0 * (N - 1) / N, dtype=torch.float64)
    t2 = torch.full((B,), 1000.0 * (N - 2) / N, dtype=torch.float64)
    expect2 = ref.sample_step(t=t1, t_prev=t2, xt=x1, x0=x0, xT=y)
    np.testing.assert_allclose(ours2.numpy(), expect2.numpy(), rtol=1e-5, atol=1e-6)  # reference computes its T terms in float32


def test_profile_changes_deterministic_dynamics():
    # Sec. 4.4: per-step profile moves a, b, c, so it acts even at eta = 0 (opposite of Prop. 2)
    sch = HeavyBridgeSchedule(1.8, 5 / 4)
    lam = sch.draw_profile(2, DEV, generator=torch.Generator().manual_seed(3))
    a1, _, _ = sch.coefficients(sch.profile_sigma2(lam))
    a0, _, _ = sch.coefficients(sch.profile_sigma2(sch.ones_profile(2, DEV)))
    assert (a1 - a0).abs().max() > 1e-3
    assert torch.allclose(a1[:, -1], torch.ones(2, dtype=torch.float64))  # pinned at the anchor


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
