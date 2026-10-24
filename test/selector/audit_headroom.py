"""Headroom audit on a frozen selector pool (cache only, CPU, no training). Spec: second_brain/PhD/Selector_Runbook.md,
measurements M1 M3 M4 M7 M10 M11 T3 T5 OBS of Selector_Experiment_Plan.md v2.

  python test/selector/audit_headroom.py POOL_ID [--split navtest] [--seed 0] [--run-name ...]
Writes exp/selector/runs/<run_name>/{params,metrics,tags}.json + curves.json + slices.csv, then log with log_selector_run.py.
PDMS (NAVSIM v1.0 as in this cache) = NC*DAC*(5EP+5TTC+2Comf)/12; gates = NC, DAC; DDC is stored but not in the score. H = oracle - pick, per scene.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_bridge_selectors import load, topk_mask

ROOT = Path(os.environ.get("NAVSIM_EXP_ROOT", "/mnt/ds/navsim-main/exp")) / "selector"
COLS = ["NC", "DAC", "EP", "TTC", "Comf", "DDC", "PDMS"]
NEAR_TIE = 0.02

ap = argparse.ArgumentParser()
ap.add_argument("pool_id"); ap.add_argument("--split", default="navtest"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--phase", default="0"); ap.add_argument("--run-name", default=None)
a = ap.parse_args()
pool = ROOT / a.pool_id / a.split
man = json.loads((pool / "pool_manifest.json").read_text())
d = load(str(pool))
# DD public dump (dump_dd_pool.py): one final-step cls head, traj instead of final, no gt in v1 (Runbook 1)
if "logit1" not in d: d["logit1"] = d["logit"]
if "final" not in d: d["final"] = d["traj"]
if "gt" not in d and "gt_future" in d: d["gt"] = d["gt_future"]  # v2 dump: GT future under gt_future (Runbook 1)
has_gt = "gt" in d
pdm = d["pdm"].astype(np.float64)
score = pdm[..., 6]; N, K = score.shape; ar = np.arange(N)
gates = pdm[..., 0] * pdm[..., 1]
soft = (5 * pdm[..., 2] + 5 * pdm[..., 3] + 2 * pdm[..., 4]) / 12
assert np.abs(gates * soft - score).max() < 1e-4, "PDMS is not gates*soft: check metric version"
logit = d["logit1"].astype(np.float64)
cls = logit.argmax(1); best = score.argmax(1)
s_pick, s_best = score[ar, cls], score[ar, best]
H = s_best - s_pick
m = {"n_scenes": N, "K": K, "pdms_select": s_pick.mean(), "pdms_oracle": s_best.mean(), "pool_mean": score.mean(),
     "headroom": H.mean(), "regret_top1": H.mean(), "hit_best_pct": 100 * (s_pick >= s_best - 1e-6).mean(),
     "gate_fail_rate": (gates[ar, cls] < 1).mean(), "random_pick": score[ar, np.random.RandomState(a.seed).randint(0, K, N)].mean()}
for i, c in enumerate(COLS[:-1]):
    m[c] = pdm[ar, cls, i].mean(); m[f"oracle_{c}"] = pdm[ar, best, i].mean()
rng = np.random.RandomState(a.seed)

# ---- reproduce the csv anchors (sanity): cls layer0, ADE oracle, nearest anchor to GT
if has_gt:
    gt = d["gt"]; final = d["final"]
    if "logit0" in d: m["anchor_cls_layer0"] = score[ar, d["logit0"].argmax(1)].mean()
    ade = np.linalg.norm(final[..., :2] - gt[:, None, :, :2], axis=-1).mean(-1)
    m["anchor_ade_oracle"] = score[ar, ade.argmin(1)].mean()
    if (pool / "anchors.npy").exists():
        anc = np.load(pool / "anchors.npy")
        near = np.linalg.norm(gt[:, None, :, :2] - anc[None], axis=-1).mean(-1).argmin(1)
        m["anchor_nearest_anchor_gt"] = score[ar, near].mean()
        m["M10_agree_cls_nearest_anchor_pct"] = 100 * (cls == near).mean()
    m["M10_agree_cls_ade_oracle_pct"] = 100 * (cls == ade.argmin(1)).mean()
    m["M10_agree_cls_best_pct"] = 100 * (cls == best).mean()

# ---- M1 factor oracles (ties broken by classifier prior)
tie = 1e-6 * logit
m["M1_gates_only_oracle"] = score[ar, (gates + tie).argmax(1)].mean()
m["M1_soft_only_oracle"] = score[ar, (soft + tie).argmax(1)].mean()
m["M1_full_oracle"] = s_best.mean()

# ---- H decomposition of the deployed pick (mutually exclusive, in this order)
zero = H <= 1e-6
gate_fail = (~zero) & (gates[ar, cls] < 1) & (gates[ar, best] >= 1)
near = (~zero) & (~gate_fail) & (H < NEAR_TIE)
softd = (~zero) & (~gate_fail) & (~near)
Ht = H.sum()
for nm, mk in (("gate", gate_fail), ("neartie", near), ("soft", softd)):
    m[f"H_{nm}_share"] = H[mk].sum() / Ht; m[f"scenes_{nm}_pct"] = 100 * mk.mean()
m["scenes_zeroH_pct"] = 100 * zero.mean()
# concentration: fraction of scenes that hold 50% of H
Hs = np.sort(H)[::-1]; m["H_conc_scenes_for_50pct"] = float((np.cumsum(Hs) < 0.5 * Ht).sum() + 1) / N
# OBS: which sub-scores differ between pick and best
dif = pdm[ar, best, :6] - pdm[ar, cls, :6]
unobs = (dif[:, [0, 3]] > 1e-6).any(1)          # NC, TTC
obs = (dif[:, [1, 2, 4]] > 1e-6).any(1)         # DAC, EP, Comf (DDC not in score)
m["OBS_unobs_only_share"] = H[unobs & ~obs].sum() / Ht
m["OBS_obs_only_share"] = H[obs & ~unobs].sum() / Ht
m["OBS_mixed_share"] = H[obs & unobs].sum() / Ht
# gate-failure H by which gate the pick fails
for i, c in ((0, "NC"), (1, "DAC")):
    m[f"H_gate_{c}_share"] = H[gate_fail & (pdm[ar, cls, i] < 1)].sum() / Ht

# ---- M4 gate-forced oracle: pick's gates set to 1
m["M4_gate_forced_pick"] = soft[ar, cls].mean()
# ---- M3 prior-limited oracle, T3 rank of best, best-of-top-k
rank_of_best = (logit > logit[ar, best][:, None]).sum(1) + 1
m["T3_rank_of_best_median"] = float(np.median(rank_of_best)); m["T3_rank_of_best_mean"] = rank_of_best.mean()
curves = {"best_of_top_k": {}, "pool_size": {}, "rank_hist": np.bincount(rank_of_best, minlength=K + 1)[1:].tolist()}
for k in (1, 2, 3, 4, 5, 8, 10, 16, 20, 32, 40, 64, K):
    v = np.where(topk_mask(logit, min(k, K)), score, -1).max(1).mean(); curves["best_of_top_k"][k] = v
    if k in (1, 2, 4, 8): m[f"best_of_top_{k}"] = v
# ---- T5 pool-size curve: random subsets, selector = argmax logit inside subset
for n in (1, 2, 4, 8, 16, 32, 64, K):
    o, s = [], []
    for _ in range(10):
        idx = np.stack([rng.permutation(K)[:n] for _ in range(N)])
        sc = np.take_along_axis(score, idx, 1); lg = np.take_along_axis(logit, idx, 1)
        o.append(sc.max(1).mean()); s.append(sc[ar, lg.argmax(1)].mean())
    curves["pool_size"][n] = {"oracle": float(np.mean(o)), "selected": float(np.mean(s))}
# ---- M11 support gap (no human score in this cache: threshold version only)
for th in (0.5, 0.8, 0.9):
    m[f"M11_scenes_oracle_below_{th}_pct"] = 100 * (s_best < th).mean()
m["M11_all_candidates_gate_fail_pct"] = 100 * (gates.max(1) < 1).mean()
# ---- M7 slices
if has_gt:
    disp = np.linalg.norm(gt[:, -1, :2], axis=-1); lat = gt[:, -1, 1]
    sl = {"unsafe_base_plan": gates[ar, cls] < 1, "all_unsafe": gates.max(1) < 1,
          "slow(<10m)": disp < 10, "mid(10-30m)": (disp >= 10) & (disp < 30), "fast(>=30m)": disp >= 30,
          "left(y>2)": lat > 2, "straight": np.abs(lat) <= 2, "right(y<-2)": lat < -2}
    rows = [{"slice": k, "scenes_pct": 100 * v.mean(), "H_mean": H[v].mean() if v.any() else np.nan, "H_share": H[v].sum() / Ht,
             "pick": s_pick[v].mean() if v.any() else np.nan, "oracle": s_best[v].mean() if v.any() else np.nan} for k, v in sl.items()]
    slices = pd.DataFrame(rows)
else: slices, rows = pd.DataFrame(), []
for r in rows: m[f"M7_{r['slice']}_H_share"] = r["H_share"]

run = a.run_name or f"sel_{a.pool_id}_{a.phase}_audit_s{a.seed}"
out = ROOT / "runs" / run; out.mkdir(parents=True, exist_ok=True)
try: commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parents[2]).decode().strip()
except Exception: commit = "unknown"
params = {"pool_id": a.pool_id, "cache_sha": man["cache_sha"], "split": a.split, "n_scenes": N, "selector_family": "audit",
          "inputs": "cache", "labels": "real_pdms", "loss": "none", "arch": "none", "seed": a.seed, "git_commit": commit,
          "calib_split_frac": 0, "near_tie": NEAR_TIE, "forgiveness_rule": "absent in this NAVSIM v1 scorer (no human-reference logic in pdm_scorer)",
          "human_score": "not in cache (M5 deferred to dump_pool)"}
(out / "params.json").write_text(json.dumps(params)); (out / "metrics.json").write_text(json.dumps({k: float(v) for k, v in m.items()}))
(out / "tags.json").write_text(json.dumps({"phase": a.phase, "deployable": "1", "privileged": "0"}))
(out / "curves.json").write_text(json.dumps(curves, indent=1)); slices.to_csv(out / "slices.csv", index=False)
np.savez_compressed(out / "picks.npz", tokens=d["tokens"], pick_idx=cls, best_idx=best, H=H)
print(f"run dir {out}\n"); print(json.dumps({k: round(float(v), 4) for k, v in m.items()}, indent=1)); print(slices.to_string(index=False))
print("best_of_top_k", {k: round(v, 4) for k, v in curves["best_of_top_k"].items()})
print("pool_size", {k: (round(v["oracle"], 4), round(v["selected"], 4)) for k, v in curves["pool_size"].items()})
