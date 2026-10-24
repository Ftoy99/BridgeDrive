"""A5 train-vs-test pool shift + A5b effective modes per scene, on the DD K=20 pool (dd_k20_public_v2). CPU only, streams shards.
Spec: second_brain/PhD/Selector_Experiment_Plan.md (A5, A5b). Writes metrics.json / params.json / tags.json / scenes_<split>.npz to --out.
  python test/selector/a5_shift.py --out /mnt/ds/navsim-main/exp/selector/diag_a5
"""
import argparse, glob, json, subprocess
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="/mnt/ds/navsim-main/exp/selector/dd_k20_public_v2")
ap.add_argument("--out", required=True)
ap.add_argument("--splits", default="navtrain,navtest")
ap.add_argument("--mode-thr", type=float, default=0.5)  # metres, A5b
a = ap.parse_args()

NC, DAC, EP, TTC, COMF, DDC, PDMS = range(7)  # pdm column order, see phase_c.py COLS


def pairwise_ade(tr):  # tr (n,K,8,2) -> (n,K,K) mean L2 over 8 waypoints
    return np.linalg.norm(tr[:, :, None] - tr[:, None], axis=-1).mean(-1)


def n_modes(ade, thr):  # leader clustering in candidate index order: new mode if > thr from every existing rep
    out = np.empty(len(ade), int)
    for i, D in enumerate(ade):
        reps = []
        for k in range(len(D)):
            if all(D[k, r] > thr for r in reps):
                reps.append(k)
        out[i] = len(reps)
    return out


def rank(x):
    return np.argsort(np.argsort(x, 1), 1).astype(float)


def corr(x, y):
    x = x - x.mean(1, keepdims=True); y = y - y.mean(1, keepdims=True)
    return (x * y).sum(1) / np.sqrt((x * x).sum(1) * (y * y).sum(1) + 1e-12)


def collect(split):
    cols = {k: [] for k in ["oracle", "pool_mean", "cls", "cls_dac_fail", "scene_any_dac_fail",
                            "cand_dac_fail_frac", "spread", "rho_logit_pdms", "modes", "tok"]}
    for f in sorted(glob.glob(f"{a.pool}/{split}/shard_*.npz")):
        d = np.load(f)
        ok = d["ok"]
        p = d["pdm"][ok].astype(np.float64)
        lg = d["logit"][ok].astype(np.float64)
        tr = d["traj"][ok][..., :2].astype(np.float64)
        s = p[..., PDMS]
        cls = lg.argmax(1)
        ar = np.arange(len(s))
        dacf = p[..., DAC] < 1
        ade = pairwise_ade(tr)
        K = s.shape[1]
        off = ade.sum((1, 2)) / (K * (K - 1))
        cols["oracle"].append(s.max(1))
        cols["pool_mean"].append(s.mean(1))
        cols["cls"].append(s[ar, cls])
        cols["cls_dac_fail"].append(dacf[ar, cls])
        cols["scene_any_dac_fail"].append(dacf.any(1))
        cols["cand_dac_fail_frac"].append(dacf.mean(1))
        cols["spread"].append(off)
        cols["rho_logit_pdms"].append(corr(rank(lg), rank(s)))
        cols["modes"].append(n_modes(ade, a.mode_thr))
        cols["tok"].append(d["tokens"][ok])
    return {k: np.concatenate(v) for k, v in cols.items()}


def summ(x):
    x = x.astype(float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x)), len(x)


out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
splits = a.splits.split(",")
data = {s: collect(s) for s in splits}
metrics = {}
for s, c in data.items():
    for k in ["oracle", "pool_mean", "cls", "cls_dac_fail", "scene_any_dac_fail", "cand_dac_fail_frac", "spread", "rho_logit_pdms"]:
        m, se, n = summ(c[k])
        metrics[f"{s}_{k}"] = m; metrics[f"{s}_{k}_se"] = se
    metrics[f"{s}_n_scenes"] = int(len(c["oracle"]))
    metrics[f"{s}_modes_median"] = float(np.median(c["modes"]))
    metrics[f"{s}_modes_mean"] = float(c["modes"].mean())
    metrics[f"{s}_modes_frac_lt8"] = float((c["modes"] < 8).mean())
    metrics[f"{s}_headroom"] = float(c["oracle"].mean() - c["cls"].mean())
    np.savez_compressed(out / f"scenes_{s}.npz", **{k: v for k, v in c.items() if k != "tok"}, tokens=c["tok"])

if {"navtrain", "navtest"} <= set(splits):
    tr_, te_ = data["navtrain"], data["navtest"]
    for k in ["oracle", "pool_mean", "cls", "cls_dac_fail", "scene_any_dac_fail", "cand_dac_fail_frac", "spread", "rho_logit_pdms"]:
        d_ = tr_[k].mean() - te_[k].mean()
        se = np.sqrt(tr_[k].var(ddof=1) / len(tr_[k]) + te_[k].var(ddof=1) / len(te_[k]))
        metrics[f"delta_train_minus_test_{k}"] = float(d_)
        metrics[f"delta_train_minus_test_{k}_ci_lo"] = float(d_ - 1.96 * se)
        metrics[f"delta_train_minus_test_{k}_ci_hi"] = float(d_ + 1.96 * se)
    metrics["gap_cls_minus_test_cls_vs_oracle_shift"] = float(metrics["delta_train_minus_test_cls"] - metrics["delta_train_minus_test_oracle"])

(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
try:
    commit = subprocess.check_output(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    commit = "unknown"
(out / "params.json").write_text(json.dumps({"pool": a.pool, "splits": a.splits, "mode_thr_m": a.mode_thr,
                                             "script": "test/selector/a5_shift.py", "git_commit": commit,
                                             "cls_rule": "argmax DD logit", "modes_rule": "leader clustering, index order",
                                             "rho_rule": "per-scene Spearman(logit, PDMS)"}))
(out / "tags.json").write_text(json.dumps({"diag": "A5", "public": "1", "internal": "0", "smoke": "0"}))
for k, v in metrics.items():
    print(f"{k:55s} {v:.5f}" if isinstance(v, float) else f"{k:55s} {v}")
