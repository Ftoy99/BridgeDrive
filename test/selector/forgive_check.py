"""Gate A4.3: human-reference forgiveness on the DD v2 pool (navtest). CPU, cache only, no training.
Per binary gate (NC, DAC, DDC): where the human reference fails the metric, does the candidate set also pass it (forgiven)?
Also: approximate forgiveness-adjusted PDMS = gates with human-failed metrics forced to pass, times the soft part.
The approximation is cache-derived, NOT the official NAVSIM v2 EPDMS (v2 scorer is not on disk; PLAN Decisions 1).
  python test/selector/forgive_check.py --pool /mnt/ds/navsim-main/exp/selector/dd_k20_public_v2/navtest --out /mnt/ds/navsim-main/exp/selector/diag_forgive
"""
import argparse, json, subprocess
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

GATES = {"NC": 0, "DAC": 1, "DDC": 5}
rows = {k: [] for k in ["pdm", "human", "ok"]}
for f in sorted(Path(a.pool).glob("shard_*.npz")):
    d = np.load(f)
    rows["pdm"].append(d["pdm"].astype(np.float64))
    rows["human"].append(d["human_pdm"].astype(np.float64))
    rows["ok"].append(d["ok"])
pdm = np.concatenate(rows["pdm"]); hum = np.concatenate(rows["human"]); ok = np.concatenate(rows["ok"]).astype(bool)
pdm, hum = pdm[ok], hum[ok]
N, K = pdm.shape[:2]
ar = np.arange(N)

m = {"n_scenes": int(N), "K": int(K)}
forgiven_gate = {}
for name, col in GATES.items():
    h_fail = hum[:, col] < 1
    c_pass = pdm[..., col] >= 1
    all_pass = c_pass.all(1)
    m[f"{name}_human_fail_pct"] = 100 * h_fail.mean()
    m[f"{name}_forgiveness_signature_pct"] = 100 * (h_fail & all_pass).mean()  # human fails, every candidate passes
    m[f"{name}_cand_pass_given_human_fail"] = float(c_pass[h_fail].mean()) if h_fail.any() else float("nan")
    m[f"{name}_cand_pass_given_human_pass"] = float(c_pass[~h_fail].mean()) if (~h_fail).any() else float("nan")
    forgiven_gate[name] = h_fail  # scenes where human fails this gate

# approximate forgiveness-adjusted PDMS (cache-derived; NOT official EPDMS). Only NC/DAC gate the score; DDC is stored, not scored.
gate_pass = np.ones((N, K))
for name in ("NC", "DAC"):
    col = GATES[name]
    gate_pass *= np.where(forgiven_gate[name][:, None], 1.0, pdm[..., col])  # keep continuous v1 value unless forgiven
score = pdm[..., 6]
soft = (5 * pdm[..., 2] + 5 * pdm[..., 3] + 2 * pdm[..., 4]) / 12
gp_v1 = pdm[..., 0] * pdm[..., 1]  # v1 gates (NC, DAC) as scored
forg_score = gate_pass * soft
m["v1_check_max_abs"] = float(np.abs(gp_v1 * soft - score).max())  # should be ~0 (PDMS = gates * soft)
m["oracle_pdms_v1"] = float(score.max(1).mean())
m["oracle_pdms_forgiven_approx"] = float(forg_score.max(1).mean())
m["scenes_pdms_changed_by_forgiveness_pct"] = float(100 * ((gp_v1 * soft) != forg_score).any(1).mean())
m["mean_candidate_pdms_v1"] = float(score.mean())
m["mean_candidate_pdms_forgiven_approx"] = float(forg_score.mean())

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
try:
    commit = subprocess.check_output(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    commit = "unknown"
(out / "metrics.json").write_text(json.dumps(m, indent=2))
(out / "params.json").write_text(json.dumps({"pool": a.pool, "gate_A4_3": True, "forgiveness": "cache-derived: human-failed gate forced to pass",
                                             "script": "test/selector/forgive_check.py", "git_commit": commit}))
(out / "tags.json").write_text(json.dumps({"diag": "A4.3", "public": "1", "internal": "0", "smoke": "0"}))
for k, v in m.items():
    print(f"{k:45s} {v:.4f}" if isinstance(v, float) else f"{k:45s} {v}")
