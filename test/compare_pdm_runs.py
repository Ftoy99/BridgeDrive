"""Paired bootstrap comparison of two (sets of) fast-PDM eval runs, the HeavyBridge paper's statistic.

For each forced-anchor rank k (rank_score_k columns written by run_pdm_score_fast.py) it resamples scenes with
replacement and reports mean(B - A) with a 95% CI and a one-sided p for B > A. Several CSVs per side (training seeds) are
averaged per scene before the bootstrap, as in the paper ("pooled over seeds"). Also counts catastrophes (PDMS < 0.5 at k).

  uv run --no-project --python 3.11 --with 'mlflow>=3.5' --with pandas python test/compare_pdm_runs.py \
      --a exp_dir_R0_s0 exp_dir_R0_s1 --b exp_dir_R3g_s0 exp_dir_R3g_s1 --name-a R0 --name-b R3g
Each path is a result CSV or an exp dir (newest *.csv below it). Without mlflow installed pass --no-mlflow.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

RANKS = (1, 2, 3, 5, 10, 20, 40, 80)


def paired_bootstrap(diff: np.ndarray, n_boot: int = 20000, seed: int = 0, chunk: int = 500):
    """diff = per-scene B - A. Returns (mean, ci_lo, ci_hi, p_one_sided for mean > 0)."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_boot)
    for i in range(0, n_boot, chunk):
        m = min(chunk, n_boot - i)
        means[i:i + m] = diff[rng.integers(0, n, size=(m, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi), float((means <= 0).mean())


def load_scores(paths):
    """Per-scene rank_score_k averaged over the given runs (seeds), indexed by token. Scenes must match across runs."""
    frames = []
    for p in paths:
        p = Path(p)
        csv = max(p.rglob("*.csv"), key=lambda f: f.stat().st_mtime) if p.is_dir() else p
        df = pd.read_csv(csv, index_col=0)
        df = df[(df["token"] != "average") & (df["valid"].astype(bool))].set_index("token")
        cols = [c for c in (f"rank_score_{k}" for k in RANKS) if c in df.columns]
        frames.append(df[cols])
    common = frames[0].index
    for f in frames[1:]:
        common = common.intersection(f.index)
    stack = np.stack([f.loc[common].to_numpy() for f in frames])
    return pd.DataFrame(stack.mean(axis=0), index=common, columns=frames[0].columns)


def compare(a_paths, b_paths, n_boot=20000):
    a, b = load_scores(a_paths), load_scores(b_paths)
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    rows = {}
    for c in a.columns:
        k = int(c.rsplit("_", 1)[1])
        m, lo, hi, p = paired_bootstrap((b[c] - a[c]).to_numpy(), n_boot=n_boot, seed=k)
        rows[k] = dict(a=100 * a[c].mean(), b=100 * b[c].mean(), delta=100 * m, lo=100 * lo, hi=100 * hi, p=p,
                       catas_a=int((a[c] < 0.5).sum()), catas_b=int((b[c] < 0.5).sum()))
    return len(common), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--experiment", default="navsim_compare")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    n, rows = compare(args.a, args.b, args.n_boot)
    print(f"{args.name_b} - {args.name_a} on {n} paired scenes (PDMS x100; CI 95% paired bootstrap, {args.n_boot} resamples)")
    print(f"{'k':>3} {args.name_a:>8} {args.name_b:>8} {'delta':>7} {'CI':>16} {'p(B>A)':>8} {'catas A/B':>10}")
    for k, r in rows.items():
        print(f"{k:>3} {r['a']:8.2f} {r['b']:8.2f} {r['delta']:+7.2f} [{r['lo']:+6.2f},{r['hi']:+6.2f}] {r['p']:8.4f} {r['catas_a']:>5}/{r['catas_b']:<4}")
    if args.no_mlflow:
        return
    import mlflow

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////mnt/ds/mlflow/mlflow.db"))
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"{args.name_b}_minus_{args.name_a}"):
        mlflow.log_params({"a": args.name_a, "b": args.name_b, "a_paths": ";".join(args.a)[:500], "b_paths": ";".join(args.b)[:500],
                           "n_boot": args.n_boot, "n_scenes": n, "seeds_a": len(args.a), "seeds_b": len(args.b)})
        for k, r in rows.items():
            mlflow.log_metrics({f"delta_k{k}": r["delta"], f"ci_lo_k{k}": r["lo"], f"ci_hi_k{k}": r["hi"], f"p_k{k}": r["p"],
                                f"pdms_a_k{k}": r["a"], f"pdms_b_k{k}": r["b"], f"catas_a_k{k}": r["catas_a"], f"catas_b_k{k}": r["catas_b"]})
    print("logged to MLflow experiment", args.experiment)


if __name__ == "__main__":
    main()
