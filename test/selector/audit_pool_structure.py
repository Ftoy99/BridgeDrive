"""Pool-structure audit for a label-only pool (no model logits), e.g. the GTRS vocabulary pools. Streams shards.
  python test/selector/audit_pool_structure.py POOL_ID [--split navtrain] [--curve-scenes 4000] [--seed 0]
Reports: oracle and pool mean of the composed score, per-column pass rates (all candidates), forgiveness candidates
(scene-columns where every candidate passes a binary metric, the signature of the human-reference rule, arXiv 2609.33189),
all-gate-pass fraction, NC/DAC co-failure, near-tie counts at the top, support (oracle below thresholds), oracle vs pool size.
"""
import argparse, glob, json, os, subprocess, time
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("NAVSIM_EXP_ROOT", "/mnt/ds/navsim-main/exp")) / "selector"
ap = argparse.ArgumentParser()
ap.add_argument("pool_id"); ap.add_argument("--split", default="navtrain"); ap.add_argument("--curve-scenes", type=int, default=4000)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--phase", default="0")
a = ap.parse_args()
pool = ROOT / a.pool_id / a.split
man = json.loads((pool / "pool_manifest.json").read_text())
cols = json.loads((pool / "metrics.json").read_text())["columns"]
ci = {c: i for i, c in enumerate(cols)}
S = ci["pdm_score"]; NC, DAC = ci["no_at_fault_collisions"], ci["drivable_area_compliance"]
binary = [c for c in cols if c not in ("pdm_score", "ego_progress")]
rng = np.random.RandomState(a.seed)
acc = {"n": 0, "oracle": 0.0, "mean": 0.0, "allgate": 0.0, "nc_fail": 0.0, "dac_fail": 0.0, "both_fail": 0.0,
       "tie02": 0.0, "tie_exact": 0.0, "below": {t: 0 for t in (0.5, 0.8, 0.9)}}
pass_rate = {c: 0.0 for c in cols}; forgiven = {c: 0 for c in binary}
curve_sizes = [1, 4, 16, 64, 256, 1024, 4096, man["K"]]
curve = {n: [] for n in curve_sizes}
t0 = time.time()
for f in sorted(glob.glob(str(pool / "shard_*.npz"))):
    p = np.load(f)["pdm"]  # (n,V,M) float16
    sc = p[..., S].astype(np.float32); n = len(sc)
    best = sc.max(1)
    acc["n"] += n; acc["oracle"] += best.sum(); acc["mean"] += sc.mean(1).sum()
    for t in acc["below"]: acc["below"][t] += int((best < t).sum())
    for c in cols: pass_rate[c] += (p[..., ci[c]].astype(np.float32) >= 1).mean(1).sum() if c != "ego_progress" else p[..., ci[c]].astype(np.float32).mean(1).sum()
    for c in binary: forgiven[c] += int((p[..., ci[c]].astype(np.float32).min(1) >= 1).sum())
    ncf, dacf = p[..., NC].astype(np.float32) < 1, p[..., DAC].astype(np.float32) < 1
    acc["nc_fail"] += ncf.mean(1).sum(); acc["dac_fail"] += dacf.mean(1).sum(); acc["both_fail"] += (ncf & dacf).mean(1).sum()
    gates = np.ones(sc.shape, bool)
    for c in binary: gates &= p[..., ci[c]].astype(np.float32) >= 1
    acc["allgate"] += gates.mean(1).sum()
    acc["tie02"] += ((sc >= best[:, None] - 0.02).sum(1)).sum(); acc["tie_exact"] += ((sc >= best[:, None] - 1e-6).sum(1)).sum()
    if sum(len(v) for v in curve[1]) < a.curve_scenes:
        take = sc[rng.choice(n, size=min(n, 256), replace=False)]
        for m in curve_sizes:
            idx = np.stack([rng.permutation(man["K"])[:m] for _ in range(len(take))])
            curve[m].append(np.take_along_axis(take, idx, 1).max(1))
N = acc["n"]
m = {"n_scenes": N, "K": man["K"], "pdms_oracle": acc["oracle"] / N, "pool_mean": acc["mean"] / N,
     "all_gate_pass_frac": acc["allgate"] / N, "nc_fail_frac": acc["nc_fail"] / N, "dac_fail_frac": acc["dac_fail"] / N,
     "nc_dac_cofail_frac": acc["both_fail"] / N, "neartie_count_mean": acc["tie02"] / N, "exact_tie_count_mean": acc["tie_exact"] / N}
m["nc_dac_cofail_ratio_vs_indep"] = m["nc_dac_cofail_frac"] / max(m["nc_fail_frac"] * m["dac_fail_frac"], 1e-9)
for t, v in acc["below"].items(): m[f"M11_scenes_oracle_below_{t}_pct"] = 100 * v / N
for c in cols: m[f"pass_{c}"] = pass_rate[c] / N
for c in binary: m[f"FORG_allpass_{c}_pct"] = 100 * forgiven[c] / N
curve_out = {int(k): float(np.concatenate(v).mean()) for k, v in curve.items() if v}
run = f"sel_{a.pool_id}_{a.phase}_poolstructure_s{a.seed}"
out = ROOT / "runs" / run; out.mkdir(parents=True, exist_ok=True)
try: commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parents[2]).decode().strip()
except Exception: commit = "unknown"
(out / "params.json").write_text(json.dumps({"pool_id": a.pool_id, "cache_sha": man["cache_sha"], "split": a.split, "n_scenes": N,
    "selector_family": "none (pool structure)", "inputs": "labels", "labels": "navsim_v2_components", "loss": "none", "arch": "none",
    "seed": a.seed, "git_commit": commit, "calib_split_frac": 0, "columns": ",".join(cols), "curve_scenes": a.curve_scenes}))
(out / "metrics.json").write_text(json.dumps({k: float(v) for k, v in m.items()}))
(out / "tags.json").write_text(json.dumps({"phase": a.phase, "deployable": "0", "privileged": "0", "metric_version": "v2"}))
(out / "curves.json").write_text(json.dumps({"oracle_vs_pool_size": curve_out}, indent=1))
print(f"run dir {out}  ({time.time() - t0:.0f}s)"); print(json.dumps({k: round(float(v), 4) for k, v in m.items()}, indent=1)); print("oracle vs pool size", {k: round(v, 4) for k, v in curve_out.items()})
