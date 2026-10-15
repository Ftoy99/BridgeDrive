"""Unit checks for test/compare_pdm_runs.py (paired bootstrap, seed pooling, catastrophe counts). No mlflow needed."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "test"))
from compare_pdm_runs import compare, paired_bootstrap  # noqa: E402


def test_bootstrap_known_effect():
    rng = np.random.default_rng(0)
    d = rng.normal(0.2, 1.0, 12000)  # true effect +0.2, sd 1 -> SE 0.009
    m, lo, hi, p = paired_bootstrap(d, n_boot=4000, seed=1)
    assert abs(m - d.mean()) < 1e-12 and lo < m < hi and lo > 0.15 and hi < 0.25 and p < 1e-3


def test_bootstrap_null_is_not_significant():
    d = np.random.default_rng(1).normal(0.0, 1.0, 5000)
    m, lo, hi, p = paired_bootstrap(d, n_boot=4000, seed=2)
    assert lo < 0 < hi and 0.05 < p < 0.95


def _csv(tmp, name, scores):
    n = len(scores[1])
    df = pd.DataFrame({"token": [f"t{i}" for i in range(n)] + ["average"], "valid": [True] * n + [True]})
    for k in (1, 2, 20, 80):
        df[f"rank_score_{k}"] = list(scores[k]) + [float(np.mean(scores[k]))]
    p = tmp / f"{name}.csv"
    df.to_csv(p)
    return p


def test_compare_pools_seeds_and_counts_catastrophes(tmp_path):
    n = 400
    base = {k: np.full(n, 0.9) for k in (1, 2, 20, 80)}
    a = {**base, 80: np.r_[np.full(40, 0.1), np.full(n - 40, 0.9)]}  # 40 catastrophes at k=80
    b = {**base, 80: np.full(n, 0.9)}
    pa = [_csv(tmp_path, "a0", a), _csv(tmp_path, "a1", a)]
    pb = [_csv(tmp_path, "b0", b), _csv(tmp_path, "b1", b)]
    npair, rows = compare(pa, pb, n_boot=500)
    assert npair == n and rows[1]["delta"] == 0 and rows[80]["catas_a"] == 40 and rows[80]["catas_b"] == 0
    assert rows[80]["delta"] > 0 and rows[80]["lo"] > 0 and rows[80]["p"] < 0.01


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_bootstrap_known_effect(); test_bootstrap_null_is_not_significant()
    test_compare_pools_seeds_and_counts_catastrophes(Path(tempfile.mkdtemp()))
    print("ok")
