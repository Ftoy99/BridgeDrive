"""HeavyBridge trajectory head for NAVSIM: BridgeDrive head (model_diffusion_head_ddbm_v5) with a per-step tempered
Levy bridge (modules/heavy_bridge_sampler.py). Select with `head: HeavyBridgeTrajectoryHead` in the agent config.

Differences to the BridgeDrive head, nothing else changes (losses, classifier, anchors, normalisation):
  * training: pick grid index i, draw a mixing profile lambda, noise with the profile coefficients (Eq. 15);
  * the denoiser also sees sigma_i^lambda through one zero-initialised MLP added to the time embedding (Sec. 4.6);
  * sampling: draw a profile, DBIM steps with the profile coefficients and churn eta (Eq. 16), hb_step_num strides.
A BridgeDrive checkpoint loaded non-strictly therefore starts exactly at its denoiser.

Config keys (TransfuserConfig): hb_alpha, hb_varpi, hb_eta, hb_num_grid, hb_step_num, hb_profile ('levy' | 'ones').
hb_profile='ones' freezes lambda == 1: the Gaussian bridge on the same code path and sampler (paper's R0).
`head.forced_anchor_rank = k` refines the k-th ranked anchor (0 = argmax) for the paper's forced-anchor stress test.
"""

from typing import Optional

import torch
from torch import nn

from navsim.agents.diffusiondrive.model_diffusion_head_ddbm_v5 import TrajectoryHead as BridgeTrajectoryHead
from navsim.agents.diffusiondrive.modules.blocks import gen_sineembed_for_position
from navsim.agents.diffusiondrive.modules.heavy_bridge_sampler import HeavyBridgeSchedule

T_NORMALIZE = 1000.0
BETA_MIN = 0.1  # fixed in BridgeDrive's DDBMScheduler call


class HeavyBridgeTrajectoryHead(BridgeTrajectoryHead):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str, config):
        super().__init__(num_poses, d_ffn, d_model, plan_anchor_path, config)
        assert config.hb_profile in ("levy", "ones"), config.hb_profile
        assert config.hb_num_grid % config.hb_step_num == 0, "hb_step_num must divide hb_num_grid"
        self.hb_profile = config.hb_profile
        self.hb_eta = config.hb_eta
        self.hb_step_num = config.hb_step_num
        self.forced_anchor_rank: Optional[int] = None
        self.hb = HeavyBridgeSchedule(
            alpha=config.hb_alpha,
            varpi=config.hb_varpi,
            beta_d=config.beta_d,  # must match the checkpoint being warm-started (K80 ckpts: 1.0)
            beta_min=BETA_MIN,
            num_grid=config.hb_num_grid,
        )
        self.sigma_mlp = nn.Sequential(nn.Linear(1, d_model), nn.Mish(), nn.Linear(d_model, d_model))
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

    def forward_train(self, ego_query, agents_query, bev_feature, bev_spatial_shape, targets=None, global_img=None):
        bs = ego_query.shape[0]
        device = ego_query.device
        N = self.hb.N

        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        x_0 = targets["trajectory"].unsqueeze(1)[..., :2].repeat(1, plan_anchor.shape[1], 1, 1)
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
        noisy_traj_points = self.denorm_odo((a_i * odo_plan_anchor + b_i * odo_info_fut + c_i * noise).float())

        ego_fut_mode = noisy_traj_points.shape[1]
        ak_ego_fut_mode = plan_anchor.shape[1]
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64).flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed).view(bs, ego_fut_mode, -1)
        time_embed = self._time_embed(timesteps, sigma_i)

        ak_traj_pos_embed = gen_sineembed_for_position(plan_anchor, hidden_dim=64).flatten(-2)
        ak_traj_feature = self.plan_anchor_encoder(ak_traj_pos_embed).view(bs, ak_ego_fut_mode, -1)

        poses_reg_list, _ = self.diff_decoder(
            traj_feature, noisy_traj_points, plan_anchor, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, global_img
        )
        poses_cls_list = self.diff_decoder_cls(ak_traj_feature, plan_anchor, bev_feature, bev_spatial_shape, agents_query, ego_query, global_img)

        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses, 3)
        best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1)

        ret_traj_loss = reg_traj_loss = cls_traj_loss = 0
        for poses_reg, poses_cls in zip(poses_reg_list, poses_cls_list):
            traj_cls_loss, traj_reg_loss = self.loss_computer(poses_reg, poses_cls, targets, plan_anchor, best_reg)
            cls_traj_loss += traj_cls_loss
            reg_traj_loss += traj_reg_loss
            ret_traj_loss += traj_cls_loss + traj_reg_loss

        return {
            "poses_reg_list": poses_reg_list, "poses_cls_list": poses_cls_list,
            "trajectory": best_reg, "trajectory_loss": ret_traj_loss,
            "trajectory_loss_dict": {"cls_traj_loss": cls_traj_loss, "reg_traj_loss": reg_traj_loss},
        }

    def forward_test(self, ego_query, agents_query, bev_feature, bev_spatial_shape, global_img):
        device = ego_query.device
        bs = ego_query.shape[0]
        N = self.hb.N
        grid = list(range(N, -1, -(N // self.hb_step_num)))  # N=20, hb_step_num=2 -> [20, 10, 0]

        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        ak_ego_fut_mode = plan_anchor.shape[1]
        ak_traj_pos_embed = gen_sineembed_for_position(plan_anchor, hidden_dim=64).flatten(-2)
        ak_traj_feature = self.plan_anchor_encoder(ak_traj_pos_embed).view(bs, ak_ego_fut_mode, -1)

        poses_cls_list = self.diff_decoder_cls(ak_traj_feature, plan_anchor, bev_feature, bev_spatial_shape, agents_query, ego_query, global_img)
        logits = poses_cls_list[-1]
        if self.forced_anchor_rank is None:
            mode_idx = logits.argmax(dim=-1)
        else:
            mode_idx = logits.argsort(dim=-1, descending=True)[:, self.forced_anchor_rank]
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses, 3)

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
                traj_feature, xt, plan_anchor, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, global_img
            )
            x0_mean = poses_reg_list[-1][..., :2]
            head0_mean = poses_reg_list[-1][..., [-1]]

            xt_prev_norm = self.hb.step(
                xt=self.norm_odo(xt.clone()), x0_hat=self.norm_odo(x0_mean), y=y_norm,
                coefs=coefs, sigma2=sigma2, i=i, j=j, eta=self.hb_eta,
            )
            xt_prev = self.denorm_odo(xt_prev_norm)

        xt_prev_ = torch.cat([xt_prev, head0_mean], dim=-1)
        x0_sample = torch.gather(xt_prev_, 1, mode_idx).squeeze(1)
        return {"trajectory": x0_sample, "poses_all": xt_prev_, "poses_cls": logits}
