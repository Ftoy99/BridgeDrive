"""(Copy of BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/heavy_bridge_sampler.py, keep in sync.)

HeavyBridge: per-step tempered Levy bridge schedule and sampler.

Implements the profile machinery of "HeavyBridge: Tempered Levy Bridges for Anchored Diffusion Driving
Policies" (TMLR submission): Eq. (13) kappa calibration, Eq. (14) profile recursion, Eq. (15) profile
bridge coefficients, Eq. (16) DBIM step, Eq. (17) offline table and Kanter draw, tilt-rejection tempering.

Everything is scalar work on a (B, N+1) grid, negligible next to a denoiser call. Setting lambda == 1 recovers
the Gaussian bridge (BridgeDrive) exactly.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch

# 8-node Gauss-Legendre rule on [-1, 1]
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)

MAX_REJECTION_TRIES = 256


class HeavyBridgeSchedule:
    """Offline table (Eq. 17) plus profile draw, recursion and coefficients, for one VP schedule."""

    def __init__(
        self,
        alpha: float = 1.8,
        varpi: float = 1.25,
        beta_d: float = 2.0,
        beta_min: float = 0.1,
        num_grid: int = 20,
        T: float = 1.0,
    ):
        assert 0.0 < alpha < 2.0, "alpha in (0, 2); alpha -> 2 is the Gaussian bridge, use lambda=1 instead"
        assert varpi >= 1.0
        self.alpha, self.varpi = alpha, varpi
        self.beta_d, self.beta_min, self.N, self.T = beta_d, beta_min, num_grid, T
        beta = alpha / 2.0
        N = num_grid

        t = np.linspace(0.0, T, N + 1)
        B = self._B(t)
        self.t = t
        self.m = np.exp(-B / 2.0)  # m_{t_i}, i = 0..N
        self.sigma2 = 1.0 - self.m**2  # sigma^2_{t_i}
        # Gaussian step variance u_i (integral-free)
        self.u = self.sigma2[1:] - (self.m[1:] / self.m[:-1]) ** 2 * self.sigma2[:-1]
        # stable step scale s_i^alpha, 8-node Gauss-Legendre
        self.s_alpha = self._stable_scale(alpha)
        # mixing scales c_i
        self.c = (2.0 / self.u) ** beta * self.s_alpha
        # tempering kappa from target jump energy, Eq. (13): varpi = beta * cbar * kappa^(beta-1)
        self.kappa = (varpi / (beta * self.c.mean())) ** (1.0 / (beta - 1.0))
        # shortcut: constant kappa (no tempering ramp), add kappa_i = kappa * rho^(1-i/(N-1)) if R3k is needed

    def _B(self, t):
        return self.beta_d / 2.0 * np.asarray(t) ** 2 + self.beta_min * np.asarray(t)

    def _stable_scale(self, alpha: float) -> np.ndarray:
        out = np.zeros(self.N)
        for i in range(self.N):
            lo, hi = self.t[i], self.t[i + 1]
            r = 0.5 * (hi - lo) * _GL_X + 0.5 * (hi + lo)
            integrand = np.exp(alpha / 2.0 * (self._B(r) - self._B(hi))) * (self.beta_d * r + self.beta_min) ** (alpha / 2.0)
            out[i] = 0.5 * (hi - lo) * np.sum(_GL_W * integrand)
        return out

    # ---- tensors on device -------------------------------------------------------------------------------
    def to(self, device, dtype=torch.float64) -> "HeavyBridgeSchedule":
        self._dev = device
        self._dt = dtype
        self.m_t = torch.as_tensor(self.m, device=device, dtype=dtype)
        self.u_t = torch.as_tensor(self.u, device=device, dtype=dtype)
        self.c_t = torch.as_tensor(self.c, device=device, dtype=dtype)
        return self

    def _ensure(self, device):
        if getattr(self, "_dev", None) != device:
            self.to(device)

    # ---- profile -----------------------------------------------------------------------------------------
    def draw_profile(self, batch: int, device, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Tempered positive alpha/2-stable profile lambda_{0:N-1}, shape (batch, N). Kanter + tilt rejection."""
        self._ensure(device)
        beta = self.alpha / 2.0
        shape = (batch, self.N)

        def kanter(n_shape):
            U = torch.rand(n_shape, device=device, dtype=torch.float64, generator=generator) * math.pi
            W = -torch.log1p(-torch.rand(n_shape, device=device, dtype=torch.float64, generator=generator))
            return (
                torch.sin(beta * U) / torch.sin(U) ** (1.0 / beta)
                * (torch.sin((1.0 - beta) * U) / W) ** ((1.0 - beta) / beta)
            )

        scale = self.c_t ** (1.0 / beta)  # lambda_i = c_i^(2/alpha) X_i
        lam = kanter(shape) * scale
        done = torch.rand(shape, device=device, dtype=torch.float64, generator=generator) < torch.exp(-self.kappa * lam)
        for _ in range(MAX_REJECTION_TRIES - 1):
            if bool(done.all()):
                break
            new = kanter(shape) * scale
            acc = torch.rand(shape, device=device, dtype=torch.float64, generator=generator) < torch.exp(-self.kappa * new)
            lam = torch.where(done, lam, new)  # after the cap the latest proposal is kept as fallback
            done = done | acc
        return lam

    def ones_profile(self, batch: int, device) -> torch.Tensor:
        self._ensure(device)
        return torch.ones((batch, self.N), device=device, dtype=torch.float64)

    # ---- Eq. (14), (15) ----------------------------------------------------------------------------------
    def profile_sigma2(self, lam: torch.Tensor) -> torch.Tensor:
        """(sigma_i^lambda)^2 for i = 0..N, shape (batch, N+1). O(N) recursion, Eq. (14)."""
        self._ensure(lam.device)
        lam = lam.to(torch.float64)
        s2 = [torch.zeros(lam.shape[0], device=lam.device, dtype=torch.float64)]
        for i in range(1, self.N + 1):
            s2.append((self.m_t[i] / self.m_t[i - 1]) ** 2 * s2[-1] + lam[:, i - 1] * self.u_t[i - 1])
        return torch.stack(s2, dim=1)

    def coefficients(self, sigma2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """a, b, c for every grid point, each (batch, N+1), Eq. (15)."""
        m = self.m_t
        mT = m[self.N]
        gamma2 = (mT**2 * sigma2) / (m**2 * sigma2[:, self.N : self.N + 1])
        a = m * gamma2 / mT
        b = m * (1.0 - gamma2)
        c = (sigma2 * (1.0 - gamma2)).clamp_min(0.0).sqrt()
        return a, b, c

    def conditional_c2(self, sigma2: torch.Tensor, i: int, j: int) -> torch.Tensor:
        """(c_{j|i})^2 for j < i (Lemma 1 with profile variances), shape (batch,)."""
        m = self.m_t
        s_j, s_i = sigma2[:, j], sigma2[:, i]
        return (s_j * (1.0 - (m[i] ** 2 * s_j) / (m[j] ** 2 * s_i))).clamp_min(0.0)

    # ---- Eq. (16) ---------------------------------------------------------------------------------------
    def step(
        self,
        xt: torch.Tensor,
        x0_hat: torch.Tensor,
        y: torch.Tensor,
        coefs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        sigma2: torch.Tensor,
        i: int,
        j: int,
        eta: float,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One DBIM step from grid index i to j < i. xt, x0_hat, y are (B, ...); coefs from `coefficients`.

        First step (i == N) uses eps_hat := 0 (paper Sec. 3.2) so the step is prediction plus fresh noise only.
        """
        a, b, c = coefs
        d = xt.ndim - 1

        def bc(v):  # (B,) -> (B, 1, ..., 1), in xt dtype
            return v.reshape(-1, *([1] * d)).to(xt.dtype)

        if i == self.N:
            eps_hat = torch.zeros_like(xt)
        else:
            eps_hat = (xt - bc(a[:, i]) * y - bc(b[:, i]) * x0_hat) / bc(c[:, i])
        c_cond = self.conditional_c2(sigma2, i, j).sqrt()
        varsigma = eta * c_cond
        resid = (c[:, j] ** 2 - varsigma**2).clamp_min(0.0).sqrt()
        if noise is None:
            noise = torch.randn_like(xt)
        return bc(a[:, j]) * y + bc(b[:, j]) * x0_hat + bc(resid) * eps_hat + bc(varsigma) * noise
