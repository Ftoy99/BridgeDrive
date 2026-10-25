"""Helpers copied verbatim from sensor_rungs.py (§5f) so the §5h trainer uses identical feature code."""
import numpy as np
import phase_c as pc


def zs(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)


def within_r2(pred, y):
    pc_ = pred - pred.mean(1, keepdims=True)
    yc = y - y.mean(1, keepdims=True)
    return 1 - ((yc - pc_) ** 2).sum() / max((yc ** 2).sum(), 1e-12)


def traj_flat(t):
    x, y, h = t[..., 0] / 32.0, t[..., 1] / 32.0, t[..., 2]
    return np.stack([x, y, np.cos(h), np.sin(h)], -1).reshape(t.shape[0], t.shape[1], -1)


def cand_feats_min(D):
    """Per-candidate observable-at-inference features: predicted-BEV gate, predicted-agent distance, traj; privileged GT future."""
    traj = D["traj"][..., :3].astype(np.float64)
    N_, K_ = traj.shape[:2]
    out = {k: np.zeros((N_, K_)) for k in ("gate", "pa_dist", "gt_ade", "gt_fin")}
    pa = D["pred_agents"][..., :2].astype(np.float64)
    pprob = D["pred_agent_prob"].astype(np.float64)
    gt_f = D["gt_future"][..., :2].astype(np.float64)
    CH = 500
    for s0 in range(0, N_, CH):
        sl = slice(s0, min(s0 + CH, N_))
        t = traj[sl]
        out["gate"][sl] = pc.gate_pass(t, D["pred_bev"][sl]).astype(np.float64)
        d = np.linalg.norm(t[:, :, :, None, :2] - pa[sl][:, None, None], axis=-1)
        d = np.where(pprob[sl][:, None, None] > 0.5, d, 50.0)
        out["pa_dist"][sl] = np.minimum(d.min((2, 3)), 50.0)
        gd = np.linalg.norm(t[..., :2] - gt_f[sl][:, None], axis=-1)
        out["gt_ade"][sl] = gd.mean(-1)
        out["gt_fin"][sl] = gd[..., -1]
    out["traj"] = traj_flat(traj)
    return out
