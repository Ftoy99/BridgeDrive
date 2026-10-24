"""Step 2 of the headroom plan: which input predicts which candidate sub-score, within scene. CPU, sklearn HistGB.
Fit on navtrain (seed-subsampled 90%, per seed), read navtest once. Reuses phase_c.load_split / gate_pass / gt_agent_feats.
Feature sets (incremental, F2-F4 built on F2): F0 traj; F1 +DD logit (z over K); F2 +predicted BEV gate and predicted-agent min distance;
F3 +GT ego future (privileged); F4 +GT agents (privileged). Targets per candidate: NC, DAC, EP, TTC, Comf, PDMS.
  python test/selector/info_probe.py --out /mnt/ds/navsim-main/exp/selector/diag_probe --seeds 0,1,2
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase_c as pc

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--boot", type=int, default=2000)  # scene bootstrap for the pick-vs-DD CI
a = ap.parse_args()

TARGETS = {"NC": 0, "DAC": 1, "EP": 2, "TTC": 3, "Comf": 4, "PDMS": 6}
SETS = ["F0", "F1", "F2", "F3", "F4"]
CHUNK = 500


def zs(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)


def cand_feats(D):
    """Candidate-level arrays (N,K,...) per feature group, computed in scene chunks."""
    traj = D["traj"][..., :3].astype(np.float64)
    N, K = traj.shape[:2]
    out = {"gate": np.zeros((N, K)), "pa_dist": np.zeros((N, K)), "gt_ade": np.zeros((N, K)),
           "gt_fin": np.zeros((N, K)), "ga_dist": np.zeros((N, K)), "gt_ag": None}
    pa = D["pred_agents"][..., :2].astype(np.float64)
    pprob = D["pred_agent_prob"].astype(np.float64)
    gt_f = D["gt_future"][..., :2].astype(np.float64)
    ga = D["gt_agents"][..., :2].astype(np.float64)
    n_gt = D["n_gt_agents"]
    gt_ag = []
    for s in range(0, N, CHUNK):
        sl = slice(s, min(s + CHUNK, N))
        t = traj[sl]
        out["gate"][sl] = pc.gate_pass(t, D["pred_bev"][sl]).astype(np.float64)
        d = np.linalg.norm(t[:, :, :, None, :2] - pa[sl][:, None, None], axis=-1)  # (n,K,8,30)
        d = np.where(pprob[sl][:, None, None] > 0.5, d, 50.0)
        out["pa_dist"][sl] = np.minimum(d.min((2, 3)), 50.0)
        gd = np.linalg.norm(t[..., :2] - gt_f[sl][:, None], axis=-1)  # (n,K,8)
        out["gt_ade"][sl] = gd.mean(-1)
        out["gt_fin"][sl] = gd[..., -1]
        valid = np.arange(ga.shape[1])[None] < n_gt[sl][:, None]  # (n,128)
        dg = np.linalg.norm(t[:, :, :, None, :2] - ga[sl][:, None, None], axis=-1)  # (n,K,8,128)
        dg = np.where(valid[:, None, None], dg, 50.0)
        out["ga_dist"][sl] = np.minimum(dg.min((2, 3)), 50.0)
        gt_ag.append(pc.gt_agent_feats(D["gt_agents"][sl], n_gt[sl]).reshape(sl.stop - sl.start, -1))
    out["gt_ag"] = np.concatenate(gt_ag)
    out["traj"] = traj_flat(traj)
    return out


def traj_flat(t):
    x, y, h = t[..., 0] / 32.0, t[..., 1] / 32.0, t[..., 2]
    return np.stack([x, y, np.cos(h), np.sin(h)], -1).reshape(t.shape[0], t.shape[1], -1)


def build(D, c, logit, fset):
    N, K = logit.shape
    parts = [c["traj"]]
    if fset >= 1: parts.append(zs(logit)[..., None])
    if fset >= 2: parts.append(np.stack([c["gate"], c["pa_dist"]], -1))
    if fset == 3: parts.append(np.stack([c["gt_ade"], c["gt_fin"]], -1))
    if fset == 4: parts.append(np.stack([c["ga_dist"]], -1)); parts.append(np.broadcast_to(c["gt_ag"][:, None], (N, K, c["gt_ag"].shape[1])))
    return np.concatenate(parts, -1).astype(np.float32)


def within_r2(pred, y, mask=None):
    if mask is not None:
        pred, y = pred[mask], y[mask]
    pc_ = pred - pred.mean(1, keepdims=True); yc = y - y.mean(1, keepdims=True)
    return 1 - ((yc - pc_) ** 2).sum() / max((yc ** 2).sum(), 1e-12)


def fit_predict(Xtr, ytr, Xte):
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=0)
    m.fit(Xtr.reshape(-1, Xtr.shape[-1]), ytr.reshape(-1))
    return m.predict(Xte.reshape(-1, Xte.shape[-1])).reshape(Xte.shape[:2])


print("loading navtrain", flush=True)
TR = pc.load_split("navtrain")
print("loading navtest", flush=True)
TE = pc.load_split("navtest")
ctr, cte = cand_feats(TR), cand_feats(TE)
lg_tr, lg_te = TR["logit"].astype(np.float64), TE["logit"].astype(np.float64)
pdm_te = TE["pdm"].astype(np.float64)
score_te = pdm_te[..., 6]
ar = np.arange(len(score_te))
dd_pick = lg_te.argmax(1)
dd_best = score_te.argmax(1)
dd_sel = score_te[ar, dd_pick].mean()
print(f"sanity: DD cls pick navtest PDMS {dd_sel:.4f} (expect 0.8801)", flush=True)

regret = dd_pick != dd_best
rows, out, pick_scores = [], Path(a.out), {}
out.mkdir(parents=True, exist_ok=True)
seeds = [int(s) for s in a.seeds.split(",")]
for seed in seeds:
    rng = np.random.RandomState(seed)
    sub = np.sort(rng.permutation(len(lg_tr))[: int(0.9 * len(lg_tr))])
    for fs in SETS:
        Xtr = build(TR, {k: (v[sub] if v is not None else None) for k, v in ctr.items()}, lg_tr[sub], int(fs[1]))
        Xte = build(TE, cte, lg_te, int(fs[1]))
        pred_pdms = None
        for tname, col in TARGETS.items():
            ytr = TR["pdm"][sub][..., col].astype(np.float64)
            yte = pdm_te[..., col]
            pred = fit_predict(Xtr, ytr, Xte)
            r2_all = within_r2(pred, yte)
            r2_reg = within_r2(pred, yte, regret)
            rows.append({"seed": seed, "set": fs, "target": tname, "r2_all": r2_all, "r2_regret": r2_reg})
            if tname == "PDMS":
                pick = pred.argmax(1)
                pick_scores.setdefault(fs, []).append(score_te[ar, pick])
                rows.append({"seed": seed, "set": fs, "target": "PICK_PDMS", "r2_all": score_te[ar, pick].mean(), "r2_regret": np.nan})
                rows.append({"seed": seed, "set": fs, "target": "PICK_vs_DD", "r2_all": (score_te[ar, pick] - score_te[ar, dd_pick]).mean(), "r2_regret": np.nan})
                if fs == "F2":  # control: within-scene shuffled train labels, same features
                    yshuf = np.take_along_axis(ytr, np.argsort(rng.rand(*ytr.shape), 1), 1)
                    pc_ = fit_predict(Xtr, yshuf, Xte).argmax(1)
                    rows.append({"seed": seed, "set": "F2", "target": "CTRL_shuffled_PICK", "r2_all": score_te[ar, pc_].mean(), "r2_regret": np.nan})
        print(f"seed {seed} {fs} done", flush=True)

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv(out / "probe_rows.csv", index=False)
agg = df.groupby(["set", "target"]).agg(r2_all=("r2_all", "mean"), r2_all_sd=("r2_all", "std"),
                                         r2_regret=("r2_regret", "mean")).reset_index()
print(agg.to_string(index=False))

metrics = {"dd_cls_pdms": float(dd_sel), "dd_best_oracle": float(score_te.max(1).mean()), "n_scenes_test": int(len(score_te)),
           "n_regret_scenes": int(regret.sum())}
# paired scene-bootstrap CIs: DD headroom (oracle - DD cls) and, per set, pick - DD cls (per-scene pick PDMS averaged over seeds)
bs = np.random.RandomState(0).randint(0, len(score_te), (a.boot, len(score_te)))
hd = (score_te.max(1) - score_te[ar, dd_pick])
metrics["headroom_ci_lo"], metrics["headroom_ci_hi"] = (float(v) for v in np.percentile(hd[bs].mean(1), [2.5, 97.5]))
np.savez_compressed(out / "picks.npz", dd=score_te[ar, dd_pick], **{fs: np.stack(v) for fs, v in pick_scores.items()})
for fs, v in pick_scores.items():
    diff = np.mean(v, 0) - score_te[ar, dd_pick]
    metrics[f"{fs}_PICK_vs_DD_ci_lo"], metrics[f"{fs}_PICK_vs_DD_ci_hi"] = (float(x) for x in np.percentile(diff[bs].mean(1), [2.5, 97.5]))
for r in agg.itertuples():
    metrics[f"{r.set}_{r.target}_r2"] = float(r.r2_all); metrics[f"{r.set}_{r.target}_r2_regret"] = float(r.r2_regret)
    if r.target in ("PICK_PDMS", "PICK_vs_DD"): metrics[f"{r.set}_{r.target}"] = float(r.r2_all)
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
try:
    commit = subprocess.check_output(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    commit = "unknown"
(out / "params.json").write_text(json.dumps({"pool": "dd_k20_public_v2", "fit": "navtrain 90% per seed", "test": "navtest once",
                                             "seeds": seeds, "boot": a.boot, "model": "HistGradientBoostingRegressor(200,0.1)",
                                             "sets": "F0 traj; F1 +logit; F2 +gate+pred agent dist; F3 +GT ego (priv); F4 +GT agents (priv)",
                                             "script": "test/selector/info_probe.py", "git_commit": commit}))
(out / "tags.json").write_text(json.dumps({"diag": "2", "public": "1", "internal": "0", "smoke": "0"}))
