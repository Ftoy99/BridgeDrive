"""Score the map gate (Selector_Experiment_Plan.md §5c) from map DAC (map_gate.py output) + pool logits.
Gate is fixed before labels are read: pick = argmax DD logit among map-DAC-pass candidates, plain DD cls if none pass.
Labels (cache pdm) are read only here, to score the pick and to check agreement of map DAC with cache DAC.
  python test/selector/map_gate_score.py --pool <pool/split> --dac <map_gate out dir> --out <run dir> --expect 0.9063
"""
import argparse, json, subprocess
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--dac", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--expect", type=float, required=True, help="cache DAC-gate row to reproduce (navtest 0.9063, navtrain-20k 0.9393)")
ap.add_argument("--boot", type=int, default=20000)
ap.add_argument("--split", required=True)
a = ap.parse_args()

NC, DAC, EP, TTC, COMF, DDC, PDMS = range(7)
L, S, MAPD, CACHED = [], [], [], []
for sp in sorted(Path(a.pool).glob("shard_*.npz")):
    d = np.load(sp)
    m = np.load(Path(a.dac) / f"dac_{sp.stem}.npz")
    assert (m["tokens"] == d["tokens"]).all(), f"token order mismatch {sp.name}"
    ok = d["ok"].astype(bool) & ~m["err"]
    L.append(d["logit"][ok].astype(np.float64)); S.append(d["pdm"][ok].astype(np.float64))
    MAPD.append(m["dac_map"][ok]); CACHED.append(d["pdm"][ok][..., DAC] >= 1)
L = np.concatenate(L); S = np.concatenate(S); MAPD = np.concatenate(MAPD); CACHED = np.concatenate(CACHED)
N, K = L.shape
ar = np.arange(N)
cls_pick = L.argmax(1)
cls = S[ar, cls_pick, PDMS]
rng = np.random.default_rng(0)


def pick_in(mask):
    p = np.where(mask, L, -np.inf).argmax(1)
    none = ~mask.any(1)
    p[none] = cls_pick[none]
    return p, none


def paired_ci(diff):
    means = np.empty(a.boot)
    for i in range(0, a.boot, 1000):
        idx = rng.integers(0, N, size=(min(1000, a.boot - i), N))
        means[i:i + len(idx)] = diff[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


p_map, none_map = pick_in(MAPD)
v_map = S[ar, p_map, PDMS]
p_cache, _ = pick_in(CACHED)  # control: same rule with cache DAC (labels), must equal the gate_ceiling row
v_cache = S[ar, p_cache, PDMS]
agree = float((MAPD == CACHED).mean())
lo, hi = paired_ci(v_map - cls)
m = {
    "split": a.split, "n_scenes": int(N), "K": int(K),
    "cls_pdms": float(cls.mean()), "oracle_pdms": float(S[..., PDMS].max(1).mean()),
    "mapgate_pdms": float(v_map.mean()), "mapgate_delta": float((v_map - cls).mean()),
    "mapgate_delta_ci_lo": lo, "mapgate_delta_ci_hi": hi,
    "mapgate_no_pass_pct": float(100 * none_map.mean()),
    "cachegate_control_pdms": float(v_cache.mean()),
    "map_vs_cache_dac_agreement": agree,
    "map_dac_pass_rate": float(MAPD.mean()), "cache_dac_pass_rate": float(CACHED.mean()),
    "expect_cache_row": a.expect,
    "sanity_abs_diff_vs_expect": float(abs(v_map.mean() - a.expect)),
    "sanity_pass": bool(abs(v_map.mean() - a.expect) <= 0.001 and agree >= 0.99),
}
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
try:
    commit = subprocess.check_output(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    commit = "unknown"
(out / "metrics.json").write_text(json.dumps(m, indent=2))
(out / "params.json").write_text(json.dumps({"pool": a.pool, "dac": a.dac, "split": a.split, "boot": a.boot,
                                             "gate_source": "map DAC (PDMScorer code, nuPlan map on disk), no labels",
                                             "script": "test/selector/map_gate_score.py", "git_commit": commit}))
(out / "tags.json").write_text(json.dumps({"diag": "mapgate", "public": "1", "map": "1", "privileged": "1", "deployable": "0", "smoke": "0"}))
for k, v in m.items():
    print(f"{k:40s} {v:.4f}" if isinstance(v, float) else f"{k:40s} {v}")
