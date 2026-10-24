"""Gate ceilings on a cached pool: argmax DD logit restricted to candidates that pass a binary gate (cache sub-scores as the gate).
Answers "what would a perfect gate in front of DD cls give" without training. DAC is a function of (candidate, ego pose, HD map),
so the DAC row is attainable with map access; NC/TTC rows need other agents' futures (unobservable). CPU, cache only.
  python test/selector/gate_ceiling.py --pool /mnt/ds/navsim-main/exp/selector/dd_k20_public_v2/navtest --out /mnt/ds/navsim-main/exp/selector/diag_gate_ceiling_navtest
"""
import argparse, json, subprocess
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--boot", type=int, default=20000)  # paired scene bootstrap resamples (runbook: 20k)
a = ap.parse_args()

NC, DAC, EP, TTC, COMF, DDC, PDMS = range(7)  # pdm column order, see phase_c.py COLS
S, L = [], []
for f in sorted(Path(a.pool).glob("shard_*.npz")):
    d = np.load(f)
    ok = d["ok"].astype(bool)
    S.append(d["pdm"][ok].astype(np.float64)); L.append(d["logit"][ok].astype(np.float64))
S = np.concatenate(S); L = np.concatenate(L)
N, K = S.shape[:2]; ar = np.arange(N)
cls_pick = L.argmax(1)
cls = S[ar, cls_pick, PDMS]
rng = np.random.default_rng(0)


def paired_ci(diff):
    means = np.empty(a.boot)
    for i in range(0, a.boot, 1000):
        idx = rng.integers(0, N, size=(min(1000, a.boot - i), N))
        means[i:i + len(idx)] = diff[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def filtered_pick(mask, score=L):
    """argmax `score` among candidates with mask True; plain argmax logit if no candidate passes."""
    p = np.where(mask, score, -np.inf).argmax(1)
    none = ~mask.any(1)
    p[none] = cls_pick[none]
    return p, none


gates = {
    "DAC": S[..., DAC] >= 1,
    "NC": S[..., NC] >= 1,
    "NC_DAC": (S[..., NC] >= 1) & (S[..., DAC] >= 1),
    "NC_DAC_TTC": (S[..., NC] >= 1) & (S[..., DAC] >= 1) & (S[..., TTC] >= 1),
}
m = {"n_scenes": int(N), "K": int(K), "cls_pdms": float(cls.mean()), "oracle_pdms": float(S[..., PDMS].max(1).mean())}
for name, mask in gates.items():
    p, none = filtered_pick(mask)
    v = S[ar, p, PDMS]
    lo, hi = paired_ci(v - cls)
    m[f"{name}_gate_logit_pdms"] = float(v.mean())
    m[f"{name}_gate_logit_delta"] = float((v - cls).mean())
    m[f"{name}_gate_logit_delta_ci_lo"] = lo
    m[f"{name}_gate_logit_delta_ci_hi"] = hi
    m[f"{name}_no_pass_pct"] = float(100 * none.mean())
    m[f"{name}_cls_already_passes_pct"] = float(100 * mask[ar, cls_pick].mean())
# control: inside the NC&DAC pass set rank by cache EP instead of the logit
p, _ = filtered_pick(gates["NC_DAC"], score=S[..., EP])
m["NC_DAC_gate_EP_pdms"] = float(S[ar, p, PDMS].mean())

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
try:
    commit = subprocess.check_output(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    commit = "unknown"
(out / "metrics.json").write_text(json.dumps(m, indent=2))
(out / "params.json").write_text(json.dumps({"pool": a.pool, "boot": a.boot, "gate_source": "cache sub-scores (perfect gate)",
                                             "script": "test/selector/gate_ceiling.py", "git_commit": commit}))
(out / "tags.json").write_text(json.dumps({"diag": "gate_ceiling", "public": "1", "internal": "0", "smoke": "0", "privileged": "1"}))
for k, v in m.items():
    print(f"{k:40s} {v:.4f}" if isinstance(v, float) else f"{k:40s} {v}")
