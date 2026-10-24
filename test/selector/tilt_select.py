"""Soft tilt on label-free observable terms (Selector_Experiment_Plan.md §5e). Cache only, CPU.

  pick_k = argmax_k ( z_k + beta * R_k ),  z = DD cls logit,  R = DAC_map * (5*EP_proxy + 2*Comf) / 7
  beta = inf: argmax R, ties broken by z (reward-only with prior tie-break; collapse check).
beta is chosen by 2-fold cross-fit on navtest, folds by log_name (fold A picks beta, fold B scores, then swap).
Out-of-fold PDMS is reported with paired scene bootstrap (20k) vs DD cls and vs the hard map gate (0.9063), plus a
cluster bootstrap by log. Labels (cache pdm) are read only to score picks, never to build R or to choose beta.
  /home/harris/miniconda3/envs/hidden/bin/python test/selector/tilt_select.py --pool <pool> --dac <map_gate dir> --terms <observable_terms dir> --logs <navsim_logs/test> --out <dir> --split navtest
"""
import argparse, json, pickle, subprocess, zlib
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--dac", required=True)
ap.add_argument("--terms", required=True)
ap.add_argument("--logs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--boot", type=int, default=20000)
a = ap.parse_args()

PDMS = 6
GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, np.inf]
rng = np.random.default_rng(0)

# token -> log_name, from navsim logs (label-free metadata)
tok_log = {}
for f in Path(a.logs).glob("*.pkl"):
    for fr in pickle.load(open(f, "rb")):
        tok_log[fr["token"]] = fr["log_name"]

Z, R, S, DAC, EPX, COMF, LOG, HARD_OK = [], [], [], [], [], [], [], []
for sp in sorted(Path(a.pool).glob("shard_*.npz")):
    d = np.load(sp)
    dm = np.load(Path(a.dac) / f"dac_{sp.stem}.npz")
    tm = np.load(Path(a.terms) / f"terms_{sp.stem}.npz")
    assert (dm["tokens"] == d["tokens"]).all() and (tm["tokens"] == d["tokens"]).all(), f"token order {sp.name}"
    ok = d["ok"].astype(bool) & ~dm["err"] & ~tm["err"]
    dac = dm["dac_map"][ok].astype(np.float64)
    ep, comf = tm["ep"][ok], tm["comf"][ok]
    Z.append(d["logit"][ok].astype(np.float64)); S.append(d["pdm"][ok].astype(np.float64))
    DAC.append(dac); EPX.append(ep); COMF.append(comf)
    R.append(dac * (5 * ep + 2 * comf) / 7)
    LOG += [tok_log[t] for t in d["tokens"][ok]]
Z = np.concatenate(Z); S = np.concatenate(S); DAC = np.concatenate(DAC); R = np.concatenate(R)
EPX = np.concatenate(EPX); COMF = np.concatenate(COMF)
LOG = np.array(LOG)
N, K = Z.shape
ar = np.arange(N)
cls_pick = Z.argmax(1)
cls = S[ar, cls_pick, PDMS]


def pick_tilt(beta, idx):
    if np.isinf(beta):
        m = R[idx].max(1, keepdims=True)
        p = np.where(R[idx] >= m - 1e-12, Z[idx], -np.inf).argmax(1)
    else:
        p = (Z[idx] + beta * R[idx]).argmax(1)
    return p


def pdms_of(beta, idx):
    return S[idx, pick_tilt(beta, idx), PDMS]


# sanity 1: beta = 0 is DD cls
assert np.allclose(pdms_of(0.0, ar), cls), "beta=0 does not reproduce DD cls"

# hard map gate (no fitting)
hard_p = np.where(DAC >= 1, Z, -np.inf).argmax(1)
hard_p[~(DAC >= 1).any(1)] = cls_pick[~(DAC >= 1).any(1)]
hard = S[ar, hard_p, PDMS]

# cross-fit folds by log_name (crc32 is stable across runs)
fold = np.array([zlib.crc32(l.encode()) % 2 for l in LOG])
oof_pick_pdms = np.zeros(N)
chosen = {}
for fa in (0, 1):
    fit_idx, eval_idx = ar[fold == fa], ar[fold != fa]
    best = max(GRID, key=lambda b: (pdms_of(b, fit_idx).mean(), -b if np.isfinite(b) else -1e9))
    chosen[f"fold{fa}_beta"] = float(best)
    oof_pick_pdms[eval_idx] = pdms_of(best, eval_idx)

# diagnostic curve on all navtest (NOT used for selection; exposure recorded in §5e)
curve = {str(b): float(pdms_of(b, ar).mean()) for b in GRID}


def paired_ci(diff):
    means = np.empty(a.boot)
    for i in range(0, a.boot, 1000):
        idx = rng.integers(0, N, size=(min(1000, a.boot - i), N))
        means[i:i + len(idx)] = diff[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


logs_u = np.unique(LOG)
log_idx = {l: np.where(LOG == l)[0] for l in logs_u}


def cluster_ci(diff, n=2000):
    out = np.empty(n)
    for i in range(n):
        pick = rng.choice(logs_u, size=len(logs_u), replace=True)
        out[i] = diff[np.concatenate([log_idx[l] for l in pick])].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


d_cls = oof_pick_pdms - cls
d_hard = oof_pick_pdms - hard
lo_c, hi_c = paired_ci(d_cls)
lo_h, hi_h = paired_ci(d_hard)
clo, chi = cluster_ci(d_hard)
m = {
    "split": a.split, "n_scenes": int(N), "K": int(K), "n_logs": int(len(logs_u)),
    "cls_pdms": float(cls.mean()), "hard_gate_pdms": float(hard.mean()), "oracle_pdms": float(S[..., PDMS].max(1).mean()),
    "tilt_oof_pdms": float(oof_pick_pdms.mean()),
    "tilt_oof_delta_vs_cls": float(d_cls.mean()), "tilt_oof_delta_vs_cls_ci_lo": lo_c, "tilt_oof_delta_vs_cls_ci_hi": hi_c,
    "tilt_oof_delta_vs_hard": float(d_hard.mean()), "tilt_oof_delta_vs_hard_ci_lo": lo_h, "tilt_oof_delta_vs_hard_ci_hi": hi_h,
    "tilt_oof_delta_vs_hard_cluster_ci_lo": clo, "tilt_oof_delta_vs_hard_cluster_ci_hi": chi,
    "chosen_beta_fold0": chosen["fold0_beta"], "chosen_beta_fold1": chosen["fold1_beta"],
    "reward_only_beta_inf_pdms": curve[str(np.inf)],
    "diag_curve_not_used_for_selection": json.dumps(curve),
    "sanity_beta0_equals_cls": True,
    "ep_proxy_mean": float(EPX.mean()), "comf_mean": float(COMF.mean()),
}
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
try:
    commit = subprocess.check_output(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    commit = "unknown"
(out / "metrics.json").write_text(json.dumps(m, indent=2))
(out / "params.json").write_text(json.dumps({"pool": a.pool, "dac": a.dac, "terms": a.terms, "grid": [str(g) for g in GRID],
                                             "weights": "EP 5, Comf 2 (scorer weights), /7", "folds": "crc32(log_name) % 2, 2-fold cross-fit",
                                             "script": "test/selector/tilt_select.py", "git_commit": commit}))
(out / "tags.json").write_text(json.dumps({"diag": "tilt", "map": "1", "privileged": "1", "deployable": "0", "smoke": "0"}))
for k, v in m.items():
    print(f"{k:40s} {v:.4f}" if isinstance(v, float) else f"{k:40s} {v}")
