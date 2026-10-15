"""HeavyBridge trajectory head: BridgeDrive head with a per-step tempered Levy bridge (see heavy_bridge_sampler.py).

Differences to TrajectoryHead (model_diffusion_head_ddbm.py), nothing else changes:
  * training: pick a grid index i, draw a mixing profile lambda, noise with the profile coefficients (Eq. 15);
  * denoiser also sees sigma_i^lambda through one zero-initialised MLP added to the time embedding (Sec. 4.6);
  * sampling: draw a profile, DBIM steps with the profile coefficients and churn eta (Eq. 16).
Zero-init of the sigma pathway means a BridgeDrive checkpoint loaded non-strictly starts exactly at its denoiser.

Config keys (TrainingConfig): hb_alpha, hb_varpi, hb_eta, hb_num_grid, hb_profile ('levy' | 'ones').
hb_profile='ones' freezes lambda == 1: the Gaussian bridge with the same sampler and sigma pathway (paper's R0).
"""

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from lead.tfv6.diffusion_modules.blocks import gen_sineembed_for_position
from lead.tfv6.diffusion_modules.heavy_bridge_sampler import HeavyBridgeSchedule
from lead.tfv6.diffusion_modules.model_diffusion_head_ddbm import (
    DEFAULT_BETA_D,
    DEFAULT_BETA_MIN,
    T_NORMALIZE,
    TrajectoryHead,
)


class HeavyTrajectoryHead(TrajectoryHead):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str, config):
        super().__init__(num_poses, d_ffn, d_model, plan_anchor_path, config)
        cfg = self.config
        self.hb_profile = cfg.hb_profile
        assert self.hb_profile in ("levy", "ones"), self.hb_profile
        self.hb_eta = cfg.hb_eta
        self.forced_anchor_rank: Optional[int] = None
        self.hb = HeavyBridgeSchedule(
            alpha=cfg.hb_alpha,
            varpi=cfg.hb_varpi,
            beta_d=DEFAULT_BETA_D,
            beta_min=DEFAULT_BETA_MIN,
            num_grid=cfg.hb_num_grid,
        )
        self.sigma_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Mish(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.sigma_mlp[-1].weight)
        nn.init.zeros_(self.sigma_mlp[-1].bias)

    def _profile(self, bs: int, device) -> torch.Tensor:
        if self.hb_profile == "ones":
            return self.hb.ones_profile(bs, device)
        return self.hb.draw_profile(bs, device)

    def _time_embed(self, t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """t in [0, 1000], sigma = sigma_i^lambda, both (bs,). Returns (bs, 1, d_model)."""
        te = self.time_mlp(t) + self.sigma_mlp(sigma.to(t.dtype).reshape(-1, 1))
        return te.view(te.shape[0], 1, -1)

    def forward_train(
        self,
        ego_query: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape: Tuple[int, int],
        global_img: Optional[torch.Tensor] = None,
        gt_checkpoints: torch.Tensor = None,
    ) -> Dict:
        bs = ego_query.shape[0]
        device = ego_query.device
        N = self.hb.N

        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        x_0 = gt_checkpoints.unsqueeze(1)[..., :2].repeat(1, plan_anchor.shape[1], 1, 1)
        odo_info_fut = self.norm_odo(x_0)
        odo_plan_anchor = self.norm_odo(plan_anchor)

        # profile, grid index, profile coefficients (Alg. 1 Train lines 3-5)
        sigma2 = self.hb.profile_sigma2(self._profile(bs, device))
        a, b, c = self.hb.coefficients(sigma2)
        idx = torch.randint(1, N + 1, (bs, 1), device=device)
        a_i, b_i, c_i = (v.gather(1, idx).reshape(bs, 1, 1, 1).float() for v in (a, b, c))
        sigma_i = sigma2.gather(1, idx).squeeze(1).sqrt().float()
        timesteps = idx.squeeze(1).float() / N * T_NORMALIZE

        noise = torch.randn(odo_info_fut.shape, device=device)
        noisy_traj_points = (a_i * odo_plan_anchor + b_i * odo_info_fut + c_i * noise).float()
        noisy_traj_points = self.denorm_odo(noisy_traj_points)
        if self.config.diffusion_speed:
            noisy_traj_points[..., -1, -1] = 0

        ego_fut_mode = noisy_traj_points.shape[1]
        ak_ego_fut_mode = plan_anchor.shape[1]

        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64).flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed).view(bs, ego_fut_mode, -1)
        time_embed = self._time_embed(timesteps, sigma_i)

        ak_traj_pos_embed = gen_sineembed_for_position(plan_anchor, hidden_dim=64).flatten(-2)
        ak_traj_feature = self.plan_anchor_encoder(ak_traj_pos_embed).view(bs, ak_ego_fut_mode, -1).to(dtype=torch.float32)

        poses_reg_list, _ = self.diff_decoder(
            traj_feature, noisy_traj_points, plan_anchor, bev_feature, bev_spatial_shape, ego_query, time_embed, global_img
        )
        poses_cls_list = self.diff_decoder_cls(ak_traj_feature, plan_anchor, bev_feature, bev_spatial_shape, ego_query)

        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses + self.config.speed_holder, 2)
        best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1)
        if self.config.diffusion_speed:
            best_reg[..., -1, -1] = 0

        return {"poses_reg_list": None, "poses_cls_list": poses_cls_list, "best_reg": best_reg}

    def forward_test(
        self,
        ego_query: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape: Tuple[int, int],
        global_img: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sampling. Set head.forced_anchor_rank = k to refine the k-th ranked anchor (0 = argmax): stress test."""
        device = ego_query.device
        bs = ego_query.shape[0]
        N = self.hb.N
        step_num = self.config.step_num
        assert N % step_num == 0, f"hb_num_grid={N} must be divisible by step_num={step_num}"
        grid = list(range(N, -1, -(N // step_num)))  # e.g. N=20, step_num=2 -> [20, 10, 0]

        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        ak_ego_fut_mode = plan_anchor.shape[1]
        ak_traj_pos_embed = gen_sineembed_for_position(plan_anchor, hidden_dim=64).flatten(-2)
        ak_traj_feature = self.plan_anchor_encoder(ak_traj_pos_embed).view(bs, ak_ego_fut_mode, -1)

        poses_cls_list = self.diff_decoder_cls(ak_traj_feature, plan_anchor, bev_feature, bev_spatial_shape, ego_query)
        logits = poses_cls_list[-1]
        if self.forced_anchor_rank is None:
            mode_idx = logits.argmax(dim=-1)
        else:
            mode_idx = logits.argsort(dim=-1, descending=True)[:, self.forced_anchor_rank]
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses + self.config.speed_holder, 2)

        sigma2 = self.hb.profile_sigma2(self._profile(bs, device))
        coefs = self.hb.coefficients(sigma2)
        y_norm = self.norm_odo(plan_anchor.clone())

        xt_prev = plan_anchor
        for i, j in zip(grid[:-1], grid[1:]):
            xt = xt_prev
            traj_pos_embed = gen_sineembed_for_position(xt, hidden_dim=64).flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed).view(bs, xt.shape[1], -1)
            t = torch.full((bs,), i / N * T_NORMALIZE, device=device)
            time_embed = self._time_embed(t, sigma2[:, i].sqrt())

            poses_reg_list, _ = self.diff_decoder(
                traj_feature, xt, plan_anchor, bev_feature, bev_spatial_shape, ego_query, time_embed, global_img
            )
            x0_mean = poses_reg_list[-1]
            x0_mean[..., -1, -1] = 0

            xt_prev_norm = self.hb.step(
                xt=self.norm_odo(xt.clone()), x0_hat=self.norm_odo(x0_mean), y=y_norm,
                coefs=coefs, sigma2=sigma2, i=i, j=j, eta=self.hb_eta,
            )
            xt_prev = self.denorm_odo(xt_prev_norm)
            xt_prev[..., -1, -1] = 0

        return torch.gather(xt_prev, 1, mode_idx).squeeze(1)
