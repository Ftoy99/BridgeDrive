"""§5j J1 (Amendment A1): HistGB per-pose off-road classifier on footprint-sampled DD BEV probabilities.
Train: navtrain-20k poses with GT-BEV labels. Threshold: navtrain inner 10% by log, candidate DAC precision >= 0.95, max recall.
Read: navtest, once, with the frozen threshold. Writes the run dir; no MLflow here.
  /home/harris/miniconda3/envs/hidden/bin/python test/selector/footprint_j1.py [--out DIR]
"""
import argparse, glob, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

EXP = Path("/mnt/ds/navsim-main/exp/selector")
FEAT = EXP / "footprint_feats"
PDMS_COL = 6
DD_CLS, MAP_GATE = 0.8801, 0.9063  # reference rows (Selector_Results.md)


def load(split):
    d = np.load(FEAT / split / "feats.npz")
    logs = np.concatenate([np.load(f, allow_pickle=True)["log_name"] for f in sorted((EXP / "dd_k20_public_v2" / ("navtrain" if split == "navtrain20k" else split)).glob("shard_*.npz"))])
    return {k: d[k] for k in d.files} | {"log": logs}


def pose_matrix(d):
    """(N,K+1,P,41): 35 probs + 5 in-window flags + 1 class-gate bit per pose. Last candidate row = human."""
    f = d["feats"].astype(np.float32).reshape(*d["feats"].shape[:3], 35)
    x = np.concatenate([f, d["inside"].astype(np.float32), d["gate_pt"].all(-1, keepdims=True).astype(np.float32)], -1)
    return x.reshape(*x.shape[:3], -1)  # (N,K+1,P,41)


def pose_labels(d):
    g = d["gtc"]
    has = (g != 255).any(-1)
    off = ((g != 255) & ~np.isin(g, [1, 3])).any(-1)
    return off, has  # (N,K+1,P)


def candidate_score(p_off):
    return p_off.max(-1)  # (N,K+1): candidate off-road score = worst pose


def pick(score, logit, pdm, thr):
    """filter-then-rank: DD logit inside the pass set (score < thr); fall back to all candidates if none pass."""
    pass_ = score[:, :20] < thr
    any_pass = pass_.any(1)
    mask = np.where(any_pass[:, None], pass_, True)
    z = np.where(mask, logit, -np.inf)
    idx = z.argmax(1)
    return pdm[np.arange(len(idx)), idx, PDMS_COL], pass_, idx


def dd_cls(logit, pdm):
    return pdm[np.arange(len(logit)), logit.argmax(1), PDMS_COL]


def reject_stats(pass_, dac):
    rej, fail = ~pass_, ~dac
    tp = (rej & fail).sum()
    return dict(precision=float(tp / max(rej.sum(), 1)), recall=float(tp / max(fail.sum(), 1)),
                pass_rate=float(pass_.mean()))


def paired_ci(diff, logs, n=20000, seed=0, chunk=1000):
    """Paired bootstrap of the mean Δ: scene resampling and cluster (by log) resampling, n resamples each."""
    rng = np.random.default_rng(seed)
    N = len(diff)
    scene = np.concatenate([diff[rng.integers(0, N, (chunk, N))].mean(1) for _ in range(n // chunk)])
    u, inv = np.unique(logs, return_inverse=True)
    sums, cnts = np.bincount(inv, diff, minlength=len(u)), np.bincount(inv, minlength=len(u)).astype(float)
    cl = []
    for _ in range(n // chunk):
        j = rng.integers(0, len(u), (chunk, len(u)))
        cl.append(sums[j].sum(1) / cnts[j].sum(1))
    cl = np.concatenate(cl)
    return [float(np.percentile(scene, 2.5)), float(np.percentile(scene, 97.5))], \
           [float(np.percentile(cl, 2.5)), float(np.percentile(cl, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(EXP / "runs" / "sel_dd_k20_public_v2_5j_J1_none_none_s0"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr, te = load("navtrain20k"), load("navtest")
    X_tr, (y_tr_off, has_tr) = pose_matrix(tr), pose_labels(tr)
    # inner split: 10% of logs (navtrain), used only for model selection and threshold
    logs = np.unique(tr["log"])
    rng = np.random.default_rng(0)
    inner_logs = set(rng.choice(logs, max(1, len(logs) // 10), replace=False))
    is_inner = np.array([l in inner_logs for l in tr["log"]])
    m = (~is_inner)[:, None, None] & has_tr
    m[:, 20] = False  # candidate poses only; the human future is never a training row
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
    clf.fit(X_tr[m], y_tr_off[m])  # (N,21,8) mask, candidates only
    p_tr = clf.predict_proba(X_tr.reshape(-1, X_tr.shape[-1]))[:, 1].reshape(X_tr.shape[:3])
    p_te = clf.predict_proba(pose_matrix(te).reshape(-1, 41))[:, 1].reshape(te["feats"].shape[:3])
    score_tr, score_te = candidate_score(p_tr), candidate_score(p_te)
    dac_tr, dac_te = tr["dac"], te["dac"]
    # threshold: inner split only
    best = None
    rows = []
    for thr in np.linspace(0.02, 0.98, 49):
        sel = is_inner
        s = reject_stats(score_tr[sel, :20] < thr, dac_tr[sel])
        rows.append(dict(thr=float(thr), **s))
        if s["precision"] >= 0.95 and (best is None or s["recall"] > best["recall"]):
            best = dict(thr=float(thr), **s)
    if best is None:
        best = max(rows, key=lambda r: r["precision"]) | {"fallback": True}
    thr = best["thr"]
    inner = {}
    pdm_in, _, _ = pick(score_tr[is_inner], tr["logit"][is_inner], tr["pdm"][is_inner], thr)
    inner["J1_pick"] = float(pdm_in.mean())
    inner["DD_cls"] = float(dd_cls(tr["logit"][is_inner], tr["pdm"][is_inner]).mean())
    inner["auc"] = float(roc_auc_score((~dac_tr[is_inner]).ravel(), score_tr[is_inner, :20].ravel()))
    # navtest: read once, frozen threshold
    pdm_te, pass_te, _ = pick(score_te, te["logit"], te["pdm"], thr)
    base = dd_cls(te["logit"], te["pdm"])
    diff = pdm_te - base
    ci_s, ci_c = paired_ci(diff, te["log"])
    stats = reject_stats(pass_te, dac_te)
    auc_te = float(roc_auc_score((~dac_te).ravel(), score_te[:, :20].ravel()))
    res = dict(threshold=thr, inner_threshold_table=best, navtrain_inner=inner,
               navtest=dict(J1_pick=float(pdm_te.mean()), DD_cls=float(base.mean()), delta=float(diff.mean()),
                            delta_CI_scene=ci_s, delta_CI_cluster=ci_c, reject=stats, auc_vs_map=auc_te,
                            recovered_share=float((pdm_te.mean() - DD_CLS) / (MAP_GATE - DD_CLS))),
               human_pass_navtest=float((score_te[:, 20] < thr).mean()))
    (out / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
