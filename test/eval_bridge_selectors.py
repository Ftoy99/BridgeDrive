"""
Offline comparison of candidate-selection rules on a bridge dump (test/run_bridge_dump.py).
Every selector picks one of the K=80 denoised candidates per scene; the score is the REAL PDMS of that
candidate (official pdm_score), so numbers are directly comparable to the deployed classifier pick.

Families
  base        classifier layer-0 / layer-1 / sum
  consensus   medoid of all / of classifier top-k / classifier-weighted medoid / kernel-density mode
  bridge      argmin of one bridge signal (path length, last-step move, one-shot-vs-final deviation,
              travel, re-noise reconstruction error, resampling spread ...), alone or within classifier top-k
  combo       z(logit) + lambda * z(signal), lambda chosen by 2-fold CV over scenes (honest)
  learned     gradient-boosted ranker over ALL features, 2-fold CV over scenes
  ceilings    oracle best, nearest-anchor-to-GT (the classifier's own target), ADE oracle, top-k oracle

Reported per selector: mean PDMS, delta vs deployed classifier +- paired SE, sub-metrics, how often it
picks a candidate that is (tied) best. Usage: python test/eval_bridge_selectors.py DUMP_DIR [--seed 0]
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["NC", "DAC", "EP", "TTC", "Comf", "DDC", "PDMS"]


def load(dump_dir):
    shards = sorted(glob.glob(str(Path(dump_dir) / "shard_*.npz")))
    assert shards, f"no shards in {dump_dir}"
    parts = [np.load(s, allow_pickle=False) for s in shards]
    keys = parts[0].files
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}


def zs(x):
    """z-score over the candidate axis (axis=1)."""
    m = x.mean(1, keepdims=True)
    s = x.std(1, keepdims=True)
    return (x - m) / (s + 1e-6)


def softmax(x, T=1.0):
    x = x / T
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def pairwise_dist(xy):
    """(N,K,P,2) -> (N,K,K) mean-over-waypoint L2 distance between candidates."""
    N, K = xy.shape[:2]
    D = np.empty((N, K, K), dtype=np.float32)
    for n in range(N):
        D[n] = np.linalg.norm(xy[n][:, None] - xy[n][None], axis=-1).mean(-1)
    return D


def topk_mask(logit, k):
    idx = np.argsort(-logit, axis=1)[:, :k]
    m = np.zeros_like(logit, dtype=bool)
    np.put_along_axis(m, idx, True, axis=1)
    return m


def argmin_in(x, mask=None):
    x = x.copy()
    if mask is not None:
        x[~mask] = np.inf
    return x.argmin(1)


def argmax_in(x, mask=None):
    x = x.copy()
    if mask is not None:
        x[~mask] = -np.inf
    return x.argmax(1)


def geometry(final):
    """Cheap geometric features of each final candidate (N,K,8,3)."""
    xy = final[..., :2]
    seg = np.linalg.norm(np.diff(np.concatenate([np.zeros_like(xy[:, :, :1]), xy], axis=2), axis=2), axis=-1)
    length = seg.sum(-1)
    d2 = np.diff(xy, n=2, axis=2)
    jerk = np.linalg.norm(d2, axis=-1).mean(-1)
    lateral = np.abs(xy[..., 1]).max(-1)
    return {"geo_len": length, "geo_jerk": jerk, "geo_lat": lateral, "geo_fx": xy[:, :, -1, 0], "geo_fy": xy[:, :, -1, 1],
            "geo_head": final[:, :, -1, 2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-learned", action="store_true")
    args = ap.parse_args()
    d = load(args.dump_dir)
    pdm = d["pdm"].astype(np.float64)  # (N,K,7)
    score = pdm[..., -1]
    N, K = score.shape
    final = d["final"]
    gt = d["gt"]
    logit1, logit0 = d["logit1"].astype(np.float64), d["logit0"].astype(np.float64)
    print(f"{N} scenes x {K} candidates | deployed-classifier PDMS {score[np.arange(N), logit1.argmax(1)].mean():.4f} | "
          f"oracle {score.max(1).mean():.4f} | pool mean {score.mean():.4f}\n", flush=True)

    rng = np.random.RandomState(args.seed)
    fold = rng.randint(0, 2, size=N)  # scene-level 2-fold split

    sig = {k: d[k].astype(np.float64) for k in
           ["path_len", "last_move", "x0_first_dev", "x0_mean_dev", "travel", "samp_std", "samp_off"]}
    recon = d["recon"].astype(np.float64)  # (N,K,R) = ||denoiser(re-noised candidate, t) - candidate||
    ts_path = Path(args.dump_dir) / "recon_ts.npy"
    rts = np.load(ts_path) if ts_path.exists() else np.array([200, 400, 600, 800])
    for r in range(recon.shape[-1]):
        if recon.shape[-1] <= 6 or int(rts[r]) % 200 == 0:   # keep the signal list readable when the grid is dense
            sig[f"recon_t{int(rts[r])}"] = recon[..., r]
    sig["recon_mean"] = recon.mean(-1)
    # "bridge reward" = negative denoising error of the bridge on the candidate, aggregated over noise levels
    for nm, m in (("early(t<=300)", rts <= 300), ("mid(350-650)", (rts > 300) & (rts < 700)), ("late(t>=700)", rts >= 700)):
        if m.any():
            sig[f"recon_{nm}"] = recon[..., m].mean(-1)
    sig["recon_max"] = recon.max(-1)
    # SNR-ish weighting: low-noise levels matter more for fine plausibility
    wlow = 1.0 / (rts / 1000.0 + 0.1)
    sig["recon_lownoise_weighted"] = (recon * wlow).sum(-1) / wlow.sum()
    for k in ("kinetic",):
        if k in d.files if hasattr(d, "files") else k in d:
            sig[k] = d[k].astype(np.float64)
    if "step_move" in d:
        sm = d["step_move"].astype(np.float64)
        sig["step_move_first"] = sm[..., 0]
        sig["step_move_last3"] = sm[..., -3:].mean(-1)
        sig["step_move_max"] = sm.max(-1)
    if "x0dev_steps" in d:
        xs = d["x0dev_steps"].astype(np.float64)
        sig["x0dev_last5"] = xs[..., -5:].mean(-1)
    sig.update({k: v.astype(np.float64) for k, v in geometry(final).items()})

    D = pairwise_dist(final[..., :2])
    ar = np.arange(N)
    picks = {}   # name -> (family, idx)
    cls = logit1.argmax(1)

    # ---- base
    picks["cls (deployed, layer1)"] = ("base", cls)
    picks["cls layer0"] = ("base", logit0.argmax(1))
    picks["cls layer0+layer1"] = ("base", (logit0 + logit1).argmax(1))

    # ---- consensus
    picks["medoid of all 80"] = ("consensus", D.mean(2).argmin(1))
    for k in (3, 5, 10, 20):
        m = topk_mask(logit1, k)
        Dk = np.where(m[:, None, :], D, 0.0).sum(2) / m.sum(1, keepdims=True)
        picks[f"medoid of cls top-{k}"] = ("consensus", argmin_in(Dk, m))
    for T in (0.5, 1.0, 2.0):
        p = softmax(logit1, T)
        picks[f"cls-weighted medoid (T={T})"] = ("consensus", np.einsum("nij,nj->ni", D, p).argmin(1))
        for sg in (1.0, 2.0, 4.0):
            dens = np.einsum("nij,nj->ni", np.exp(-(D.astype(np.float64) ** 2) / (2 * sg * sg)), p)
            picks[f"KDE mode (sigma={sg}m, T={T})"] = ("consensus", dens.argmax(1))

    # ---- bridge signals, alone and within top-k
    for name, s in sig.items():
        picks[f"min {name}"] = ("bridge", s.argmin(1))
        picks[f"max {name}"] = ("bridge", s.argmax(1))
        for k in (5, 10, 20):
            m = topk_mask(logit1, k)
            picks[f"min {name} in cls top-{k}"] = ("bridge-topk", argmin_in(s, m))
            picks[f"max {name} in cls top-{k}"] = ("bridge-topk", argmax_in(s, m))

    # ---- combos: z(logit) + lam*z(signal), lam picked by 2-fold CV (never on the scenes it is evaluated on)
    lams = np.array([-4, -2, -1, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1, 2, 4], dtype=np.float64)
    zl = zs(logit1)
    consensus_sig = {
        "consensus_mean_dist": D.mean(2),
        "consensus_kde2": -np.einsum("nij,nj->ni", np.exp(-(D.astype(np.float64) ** 2) / 8.0), softmax(logit1, 1.0)),
    }
    allsig = {**sig, **consensus_sig}
    chosen = {}
    for name, s in allsig.items():
        zsg = zs(s)
        cand = np.stack([(zl + l * zsg).argmax(1) for l in lams])  # (L,N)
        sc = np.stack([score[ar, c] for c in cand])                  # (L,N)
        out = np.zeros(N, dtype=np.int64)
        used = []
        for f in (0, 1):
            te, tr = fold == f, fold != f
            best = sc[:, tr].mean(1).argmax()
            out[te] = cand[best, te]
            used.append(lams[best])
        picks[f"cls + lam*z({name}) [CV]"] = ("combo", out)
        chosen[name] = used

    # ---- classifier + bridge reward, natural scale: argmax log p_cls - beta * (bridge denoising error)
    logp1 = np.log(softmax(logit1) + 1e-12)
    betas = np.array([0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32])
    for nm in [k for k in sig if k.startswith("recon_") or k in ("kinetic", "x0dev_last5")]:
        e = sig[nm]
        cand = np.stack([(logp1 - b * e).argmax(1) for b in betas])
        sc_b = np.stack([score[ar, c] for c in cand])
        out = np.zeros(N, dtype=np.int64)
        for f in (0, 1):
            te, tr = fold == f, fold != f
            out[te] = cand[sc_b[:, tr].mean(1).argmax(), te]
        picks[f"log p_cls - beta*{nm} [CV]"] = ("bridge-reward", out)

    # ---- learned ranker over all features
    if not args.no_learned:
        from sklearn.ensemble import HistGradientBoostingRegressor
        feats = {"logit0": logit0, "logit1": logit1, "p1": softmax(logit1), "z_logit1": zl, "rank_logit1": np.argsort(np.argsort(-logit1, 1), 1).astype(np.float64)}
        for name, s in allsig.items():
            feats[name] = s
            feats["z_" + name] = zs(s)
        feats["kde1"] = np.einsum("nij,nj->ni", np.exp(-(D.astype(np.float64) ** 2) / 2.0), softmax(logit1))
        names = list(feats)
        X = np.stack([feats[n] for n in names], axis=-1).reshape(N * K, -1).astype(np.float32)
        y = (score - score.mean(1, keepdims=True)).reshape(-1)  # scene-relative target
        out = np.zeros(N, dtype=np.int64)
        sfold = np.repeat(fold, K)
        for f in (0, 1):
            tr = sfold != f
            if tr.sum() < 200:
                continue
            reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
                                                min_samples_leaf=40, random_state=args.seed)
            reg.fit(X[tr], y[tr])
            te_scenes = np.where(fold == f)[0]
            pred = reg.predict(X.reshape(N, K, -1)[te_scenes].reshape(-1, X.shape[1])).reshape(len(te_scenes), K)
            out[te_scenes] = pred.argmax(1)
        picks["LEARNED GBDT ranker (all features) [CV]"] = ("learned", out)

    # ---- ceilings (use GT or PDM -> not deployable)
    anchors = np.load(Path(args.dump_dir) / "anchors.npy")  # (K,8,2) clean k-means anchors
    nearest_anchor = np.linalg.norm(gt[:, None, :, :2] - anchors[None], axis=-1).mean(-1).argmin(1)
    ade = np.linalg.norm(final[..., :2] - gt[:, None, :, :2], axis=-1).mean(-1)
    picks["[ceiling] oracle best PDMS"] = ("ceiling", score.argmax(1))
    picks["[ceiling] nearest anchor to GT (cls target)"] = ("ceiling", nearest_anchor)
    picks["[ceiling] ADE oracle (closest to GT)"] = ("ceiling", ade.argmin(1))
    picks["[reference] random candidate"] = ("ceiling", rng.randint(0, K, size=N))

    # ---- holdout: pick the best of ALL selectors on one half of the scenes, score it on the other half ----
    # Choosing the best of ~290 selectors on the same scenes you then report is optimistic (winner's curse).
    # Here every scene is scored by a selector chosen WITHOUT looking at that scene's half. Two directions
    # (A->B, B->A) so each scene is evaluated exactly once; the pooled number is the honest gain of
    # "run all selectors, take the winner".
    elig = [n for n, (fam, _) in picks.items() if fam != "ceiling"]
    Smat = np.stack([score[ar, picks[n][1]] for n in elig])  # (M, N)
    hs = np.random.RandomState(args.seed + 1).rand(N) < 0.5
    base_s = score[ar, cls]
    held = np.zeros(N)
    print(f"\n=== HOLDOUT ({hs.sum()} / {N - hs.sum()} scenes): winner chosen on one half, scored on the other ===")
    for tag, sel_set, ev_set in (("A->B", hs, ~hs), ("B->A", ~hs, hs)):
        m = Smat[:, sel_set].mean(1)
        order = np.argsort(-m)
        w = order[0]
        held[ev_set] = Smat[w, ev_set]
        d_sel = (Smat[w, sel_set] - base_s[sel_set]).mean()
        dh = Smat[w, ev_set] - base_s[ev_set]
        print(f"\n  [{tag}] top 10 on the selection half -> their score on the held-out half "
              f"(deployed classifier: sel {base_s[sel_set].mean():.4f}, held-out {base_s[ev_set].mean():.4f})")
        print(f"  {'selector':58s} {'sel-half':>9s} {'held-out':>9s} {'d_vs_cls(held-out)':>20s}")
        for r in order[:10]:
            dd = Smat[r, ev_set] - base_s[ev_set]
            print(f"  {elig[r]:58s} {Smat[r, sel_set].mean():9.4f} {Smat[r, ev_set].mean():9.4f} "
                  f"{dd.mean():+10.4f} +- {dd.std(ddof=1) / np.sqrt(ev_set.sum()):.4f}")
        print(f"  winner: {elig[w]} | gain on selection half {d_sel:+.4f} -> on held-out half {dh.mean():+.4f} "
              f"(shrinkage = winner's curse {d_sel - dh.mean():+.4f})")
    dh_all = held - base_s
    print(f"\n  POOLED honest result of 'pick the best selector on the other half': PDMS {held.mean():.4f} vs deployed {base_s.mean():.4f} "
          f"-> {dh_all.mean():+.4f} +- {dh_all.std(ddof=1) / np.sqrt(N):.4f}  "
          f"({'beyond 2 SE' if abs(dh_all.mean()) > 2 * dh_all.std(ddof=1) / np.sqrt(N) else 'within noise'})")

    # ---- table
    base = score[ar, cls]
    rows = []
    for name, (fam, idx) in picks.items():
        s = score[ar, idx]
        dlt = s - base
        sub = pdm[ar, idx].mean(0)
        avg = score.mean(1)
        mx = score.max(1)
        ok = (mx - avg) > 1e-6
        rows.append({"selector": name, "family": fam, "PDMS": s.mean(), "lift": (s / np.maximum(avg, 1e-3)).mean(),
                     "headroom%": 100 * ((s - avg)[ok] / (mx - avg)[ok]).mean(), "d_vs_cls": dlt.mean(),
                     "se": dlt.std(ddof=1) / np.sqrt(N), "hit_best%": 100 * (s >= score.max(1) - 1e-6).mean(),
                     **{m: sub[i] for i, m in enumerate(METRICS[:-1])}})
    df = pd.DataFrame(rows).sort_values("PDMS", ascending=False).reset_index(drop=True)
    out_csv = Path(args.dump_dir) / "selector_results.csv"
    df.to_csv(out_csv, index=False)

    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 52)
    fmt = df.copy()
    for c in ["PDMS", "lift", "headroom%", "d_vs_cls", "se"] + METRICS[:-1]:
        fmt[c] = fmt[c].map(lambda v: f"{v:.4f}")
    fmt["hit_best%"] = fmt["hit_best%"].map(lambda v: f"{v:.1f}")
    fmt["headroom%"] = df["headroom%"].map(lambda v: f"{v:.1f}")
    nshow = 45
    print(f"=== top {nshow} selectors by real PDMS (d_vs_cls is paired vs deployed classifier; |d| < 2*se is noise) ===")
    print(fmt.head(nshow).to_string(index=False))
    print("\n=== best selector per family ===")
    print(fmt.loc[df.groupby("family")["PDMS"].idxmax()].sort_values("PDMS", ascending=False).to_string(index=False))
    # ---- confidence * PDMS / pool-average, as a score of the deployed classifier ----
    sg = 1 / (1 + np.exp(-logit1))
    pm = softmax(logit1)
    pdm_pick = score[ar, cls]
    avg_pool = np.maximum(score.mean(1), 1e-3)
    mx = score.max(1)
    print("\n=== score metrics for the deployed classifier (per scene, then mean over scenes) ===")
    print(f"  PDMS of pick                                  {pdm_pick.mean():.4f}")
    print(f"  pool average PDMS (random pick)               {score.mean():.4f}")
    print(f"  selection lift   = PDMS_pick / avg            {(pdm_pick / avg_pool).mean():.3f}   (1.0 = no better than a random candidate)")
    ok = (mx - score.mean(1)) > 1e-6
    print(f"  headroom captured = (pick-avg)/(oracle-avg)   {100 * ((pdm_pick - score.mean(1))[ok] / (mx - score.mean(1))[ok]).mean():.1f}%")
    print(f"  expected PDMS under the classifier, sum_k softmax_k*PDMS_k   {(pm * score).sum(1).mean():.4f}   | lift {((pm * score).sum(1) / avg_pool).mean():.3f}")
    print(f"  expected PDMS under independent sigmoids (normalised)      {((sg * score).sum(1) / sg.sum(1)).mean():.4f}   | lift {(((sg * score).sum(1) / sg.sum(1)) / avg_pool).mean():.3f}")
    for nm, c in (("softmax", pm[ar, cls]), ("sigmoid", sg[ar, cls])):
        m = c * pdm_pick / avg_pool
        print(f"  confidence({nm}) * PDMS_pick / avg              mean {m.mean():.3f}   | corr(confidence, PDMS_pick) {np.corrcoef(c, pdm_pick)[0, 1]:+.3f}")
    print("  calibration: is the classifier's confidence predictive of success? (sigmoid confidence of the pick, 5 equal-count bins)")
    qs = np.quantile(sg[ar, cls], [0, .2, .4, .6, .8, 1.0])
    for a, b in zip(qs[:-1], qs[1:]):
        mk = (sg[ar, cls] >= a) & (sg[ar, cls] <= b)
        print(f"     conf in [{a:.2f},{b:.2f}]  n={mk.sum():4d}  mean PDMS of pick {pdm_pick[mk].mean():.3f}  | oracle {mx[mk].mean():.3f} | pool avg {score.mean(1)[mk].mean():.3f}")
    print("\n=== top-k oracle: best PDMS available inside the classifier's top-k (headroom for any re-ranker) ===")
    for k in (1, 2, 3, 5, 10, 20, 40, 80):
        m = topk_mask(logit1, k)
        print(f"  k={k:3d}: {np.where(m, score, -1).max(1).mean():.4f}")
    print("\nCV-chosen lambda per signal (fold0, fold1):", {k: v for k, v in list(chosen.items())[:6]}, "...")
    print(f"\nfull table: {out_csv}")


if __name__ == "__main__":
    main()
