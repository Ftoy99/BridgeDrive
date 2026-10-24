"""§5f sensor-feature ladder on the DD public pool (Experiment Plan §5f). CPU, sklearn HistGB. Navtest, cross-fit by log_name.
Rungs (deployable): S0 DD cls (= logit argmax, no head). S1 = traj + DD logit z + DD predicted BEV gate + predicted-agent distance.
S2 = S1 + PCA32(DD BEV pooled, current camera+LiDAR). S3 = S2 + PCA16 of pooled BEV at lags 1,2,3 (history).
Diagnostic (privileged=1, not deployable): P2 = S1 + GT ego future (ADE, final displacement).
Heads: per sub-score HistGB regression (NC, DAC, EP, TTC, Comf) and PDMS direct. Primary pick = NAVSIM-form product
NC*DAC*(5EP+5TTC+2Comf)/12 from the sub-score heads. Secondary pick = argmax PDMS head.
Cross-fit: scenes split 2 ways by log_name. Seed s sets the log-to-fold assignment (3 seeds). PCA fit on the training fold only.
Not yet in this pass: pairwise pick head, inner early stopping (fixed max_iter=200), map-DAC rung P1.
  python test/selector/sensor_rungs.py --out /mnt/ds/navsim-main/exp/selector/sensor_rungs --seeds 0,1,2
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
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


ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--boot", type=int, default=20000)
ap.add_argument("--pool", default="/mnt/ds/navsim-main/exp/selector/sensor_pool/sensor_pool.npz")
a = ap.parse_args()

TARGETS = {"NC": 0, "DAC": 1, "EP": 2, "TTC": 3, "Comf": 4, "PDMS": 6}
RUNGS = ["S1", "S2", "S3", "P2"]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

print("loading navtest pool", flush=True)
D = pc.load_split("navtest")
# load_split does not return log_name: read it from the same shards with the same keep mask
_parts = [np.load(f) for f in sorted(Path("/mnt/ds/navsim-main/exp/selector/dd_k20_public_v2/navtest").glob("shard_*.npz"))]
_ok = np.concatenate([p_["ok"] for p_ in _parts]).astype(bool)
_fin = np.concatenate([np.isfinite(p_["pdm"]).all(axis=(1, 2)) for p_ in _parts])
_keep = _ok & _fin
D["log_name"] = np.concatenate([p_["log_name"] for p_ in _parts])[_keep]
assert len(D["log_name"]) == len(D["tokens"])
pool = np.load(a.pool)
idx = {t: i for i, t in enumerate(pool["tokens"].tolist())}
keep = np.array([t in idx for t in D["tokens"]])
D = {k: v[keep] for k, v in D.items()}
sp = np.array([idx[t] for t in D["tokens"]])
lag = {f"lag{l}": pool[f"lag{l}"][sp] for l in range(4)}
logs = D["log_name"]
N, K = D["logit"].shape
print(f"scenes {N}, K {K}, logs {len(set(logs))}", flush=True)

c = cand_feats_min(D)  # gate, pa_dist, gt_ade, gt_fin, traj
lg = D["logit"].astype(np.float64)
score = D["pdm"][..., 6].astype(np.float64)
ar = np.arange(N)
dd_pick = lg.argmax(1)
dd_sel = score[ar, dd_pick]
print(f"sanity: DD cls navtest PDMS {dd_sel.mean():.4f} (expect 0.8801)", flush=True)


def cand_matrix(rung, pca_models):
    """(N,K,F) features. PCA models fitted on tr_idx only."""
    parts = [c["traj"], zs(lg)[..., None], np.stack([c["gate"], c["pa_dist"]], -1)]
    if rung == "P2":
        parts.append(np.stack([c["gt_ade"], c["gt_fin"]], -1))
    if rung in ("S2", "S3"):
        z0 = pca_models["lag0"].transform(lag["lag0"])
        parts.append(np.broadcast_to(z0[:, None], (N, K, z0.shape[1])))
    if rung == "S3":
        for l in (1, 2, 3):
            zl = pca_models[f"lag{l}"].transform(lag[f"lag{l}"])
            parts.append(np.broadcast_to(zl[:, None], (N, K, zl.shape[1])))
    return np.concatenate(parts, -1).astype(np.float32)


def fit_pca(tr_idx):
    m = {"lag0": PCA(32, random_state=0).fit(lag["lag0"][tr_idx])}
    for l in (1, 2, 3):
        m[f"lag{l}"] = PCA(16, random_state=0).fit(lag[f"lag{l}"][tr_idx])
    return m


def fit_predict_heads(X, y_all, tr, te):
    preds = {}
    Xtr = X[tr].reshape(-1, X.shape[-1])
    Xte = X[te].reshape(-1, X.shape[-1])
    for tname, col in TARGETS.items():
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=0)
        m.fit(Xtr, y_all[tr][..., col].reshape(-1))
        preds[tname] = m.predict(Xte).reshape(len(te), K)
    return preds


def product_score(p):
    return p["NC"] * p["DAC"] * (5 * p["EP"] + 5 * p["TTC"] + 2 * p["Comf"]) / 12.0


def bootstrap_diff(diff, log_ids, n, rng):
    """Paired scene bootstrap (n resamples) and cluster-by-log bootstrap for mean diff."""
    m = len(diff)
    s_means = np.array([diff[rng.randint(0, m, m)].mean() for _ in range(n)])
    uniq = np.unique(log_ids)
    groups = {u: np.where(log_ids == u)[0] for u in uniq}
    c_means = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq))
        ii = np.concatenate([groups[u] for u in pick])
        c_means.append(diff[ii].mean())
    return (np.percentile(s_means, [2.5, 97.5]), np.percentile(c_means, [2.5, 97.5]))


rows, cand_out = [], {}
for seed in [int(s) for s in a.seeds.split(",")]:
    rng = np.random.RandomState(seed)
    uniq = np.unique(logs)
    fold_of_log = dict(zip(uniq, rng.permutation(len(uniq)) % 2))
    fold = np.array([fold_of_log[l] for l in logs])
    for rung in RUNGS:
        oof_pick = np.zeros(N, int)
        oof_pick_pdms = np.zeros(N, int)
        oof_sub = {t: np.zeros((N, K)) for t in TARGETS}
        for f in (0, 1):
            tr = np.where(fold != f)[0]
            te = np.where(fold == f)[0]
            pcam = fit_pca(tr)
            X = cand_matrix(rung, pcam)
            p = fit_predict_heads(X, D["pdm"].astype(np.float64), tr, te)
            for t in TARGETS:
                oof_sub[t][te] = p[t]
            oof_pick[te] = product_score(p).argmax(1)
            oof_pick_pdms[te] = p["PDMS"].argmax(1)
            print(f"seed {seed} {rung} fold {f} done", flush=True)
        pick_prod = score[ar, oof_pick]
        pick_head = score[ar, oof_pick_pdms]
        lo_s, lo_c = bootstrap_diff(pick_prod - dd_sel, logs, a.boot, np.random.RandomState(seed))
        rows.append(dict(seed=seed, rung=rung, pick_pdms=pick_prod.mean(), delta_vs_dd=(pick_prod - dd_sel).mean(),
                         ci_scene_lo=lo_s[0], ci_scene_hi=lo_s[1], ci_cluster_lo=lo_c[0], ci_cluster_hi=lo_c[1],
                         pdms_head_pick=pick_head.mean(), pdms_head_delta=(pick_head - dd_sel).mean(),
                         **{f"r2_{t}": within_r2(oof_sub[t], D["pdm"][..., col].astype(np.float64))
                            for t, col in TARGETS.items()}))
        print(rows[-1], flush=True)
        cand_out[f"{rung}_seed{seed}"] = oof_pick

rows.append(dict(seed=-1, rung="DD_cls", pick_pdms=dd_sel.mean(), delta_vs_dd=0.0))
rows.append(dict(seed=-1, rung="oracle", pick_pdms=score.max(1).mean(), delta_vs_dd=score.max(1).mean() - dd_sel.mean()))
df = pd.DataFrame(rows)
df.to_csv(out / "sensor_rungs_rows.csv", index=False)
np.savez(out / "oof_picks.npz", **cand_out)
print(df.to_string(index=False))
json.dump({"n_scenes": int(N), "dd_cls": float(dd_sel.mean()), "oracle": float(score.max(1).mean())},
          open(out / "summary.json", "w"), indent=2)
