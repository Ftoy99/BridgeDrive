"""Train the candidate ranker (gradient-boosted regressor) on one bridge dump and test it on another.
Typical: train on a NAVTRAIN dump, test on the full NAVTEST dump (clean split, no scene is in both).

    python test/train_ranker.py TRAIN_DUMP TEST_DUMP [--groups cls,geo,cons,free] [--save model.joblib]

Features are computed from the model's own outputs only (nothing from GT / simulator at inference).
Groups: cls (classifier logits) | geo (trajectory geometry) | cons (consensus among the 80) | free (signals tracked
in the normal denoising loop). 'recon' and 'samp' need the extra-compute dump and are optional.
Target = candidate's real PDMS minus the scene mean; pick = argmax prediction per scene."""
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from eval_bridge_selectors import load, zs, softmax, pairwise_dist, geometry


def features(d, groups):
    lg0, lg1 = d["logit0"].astype(np.float64), d["logit1"].astype(np.float64)
    out = []
    def add(n, x): out.append((n, x.astype(np.float32)))
    if "cls" in groups:
        for n, x in (("logit0", lg0), ("logit1", lg1), ("p1", softmax(lg1)), ("zlogit", zs(lg1)),
                     ("rank", np.argsort(np.argsort(-lg1, 1), 1).astype(float))):
            add(n, x)
    if "geo" in groups:
        for n, x in geometry(d["final"]).items():
            add(n, x); add("z" + n, zs(x))
    if "cons" in groups:
        D = pairwise_dist(d["final"][..., :2]).astype(np.float64)
        p1 = softmax(lg1)
        add("meandist", D.mean(2)); add("kde1", np.einsum("nij,nj->ni", np.exp(-D ** 2 / 2.0), p1))
        add("kde2", np.einsum("nij,nj->ni", np.exp(-D ** 2 / 8.0), p1))
    if "free" in groups:
        for k in ("path_len", "last_move", "x0_first_dev", "x0_mean_dev", "travel", "kinetic"):
            x = d[k].astype(np.float64); add(k, x); add("z" + k, zs(x))
        sm, xs = d["step_move"].astype(np.float64), d["x0dev_steps"].astype(np.float64)
        add("sm_first", sm[..., 0]); add("sm_last3", sm[..., -3:].mean(-1)); add("sm_max", sm.max(-1)); add("x0dev_last5", xs[..., -5:].mean(-1))
    if "recon" in groups:
        rec = d["recon"].astype(np.float64)
        for r in range(rec.shape[-1]): add(f"recon{r}", rec[..., r])
        add("recon_mean", rec.mean(-1)); add("zrecon_mean", zs(rec.mean(-1)))
    if "samp" in groups:
        for k in ("samp_std", "samp_off"):
            x = d[k].astype(np.float64); add(k, x); add("z" + k, zs(x))
    names = [n for n, _ in out]
    return np.stack([x for _, x in out], -1), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_dump"); ap.add_argument("test_dump")
    ap.add_argument("--groups", default="cls,geo,cons,free")
    ap.add_argument("--save", default=None)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--crossfit", action="store_true", help="use only TEST_DUMP: honest 2-fold-by-scene score, then fit on ALL scenes and save")
    args = ap.parse_args()
    groups = args.groups.split(",")
    from sklearn.ensemble import HistGradientBoostingRegressor
    if args.crossfit:
        d = load(args.test_dump)
        Xall, names = features(d, groups)
        s_all = d["pdm"][..., -1].astype(float); N, K, F = Xall.shape; ar = np.arange(N)
        y = (s_all - s_all.mean(1, keepdims=True)).reshape(-1)
        base = s_all[ar, d["logit1"].argmax(1)]
        fold = np.random.RandomState(0).randint(0, 2, N); pick = np.zeros(N, dtype=np.int64)
        print(f"crossfit on {N} scenes | {F} features: {groups}")
        for f in (0, 1):
            trm = np.repeat(fold != f, K)
            m = HistGradientBoostingRegressor(max_iter=args.iters, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, random_state=0)
            m.fit(Xall.reshape(-1, F)[trm], y[trm]); te_i = np.where(fold == f)[0]
            pick[te_i] = m.predict(Xall[te_i].reshape(-1, F)).reshape(len(te_i), K).argmax(1)
        s = s_all[ar, pick]; dl = s - base
        print(f"2-fold CV (each scene scored by a model that never saw it): deployed {base.mean():.4f} -> ranker {s.mean():.4f} | gain {dl.mean():+.4f} +- {dl.std(ddof=1) / np.sqrt(N):.4f} | oracle {s_all.max(1).mean():.4f}")
        sb, sr = d["pdm"][ar, d["logit1"].argmax(1)].mean(0), d["pdm"][ar, pick].mean(0)
        print("   " + "  ".join(f"{n}: {a:.4f}->{b:.4f}" for n, a, b in zip(["NC", "DAC", "EP", "TTC", "Comf", "DDC"], sb, sr)))
        m = HistGradientBoostingRegressor(max_iter=args.iters, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, random_state=0)
        m.fit(Xall.reshape(-1, F), y)
        if args.save:
            import joblib
            joblib.dump({"model": m, "names": names, "groups": groups, "trained_on": args.test_dump, "n_scenes": int(N)}, args.save)
            print("final model fit on ALL", N, "scenes, saved ->", args.save, "(do not evaluate it on these same scenes)")
        return
    tr, te = load(args.train_dump), load(args.test_dump)
    Xtr, names = features(tr, groups); Xte, _ = features(te, groups)
    s_tr, s_te = tr["pdm"][..., -1].astype(float), te["pdm"][..., -1].astype(float)
    Ntr, K, F = Xtr.shape; Nte = len(s_te); ar = np.arange(Nte)
    print(f"train {Ntr} scenes | test {Nte} scenes | {F} features: {groups}")
    print(f"train-scene classifier PDMS {s_tr[np.arange(Ntr), tr['logit1'].argmax(1)].mean():.4f} | test-scene classifier PDMS {s_te[ar, te['logit1'].argmax(1)].mean():.4f}")
    y = (s_tr - s_tr.mean(1, keepdims=True)).reshape(-1)
    t0 = time.time()
    m = HistGradientBoostingRegressor(max_iter=args.iters, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, random_state=0)
    m.fit(Xtr.reshape(-1, F), y)
    print(f"trained in {time.time() - t0:.0f}s")
    pred = m.predict(Xte.reshape(-1, F)).reshape(Nte, K)
    pick = pred.argmax(1)
    base = s_te[ar, te["logit1"].argmax(1)]
    s = s_te[ar, pick]; dl = s - base
    print(f"\nTEST  deployed classifier {base.mean():.4f} | ranker {s.mean():.4f} | gain {dl.mean():+.4f} +- {dl.std(ddof=1) / np.sqrt(Nte):.4f} | oracle {s_te.max(1).mean():.4f}")
    names6 = ["NC", "DAC", "EP", "TTC", "Comf", "DDC"]
    sub_b = te["pdm"][ar, te["logit1"].argmax(1)].mean(0); sub_r = te["pdm"][ar, pick].mean(0)
    print("      " + "  ".join(f"{n}: {b:.4f}->{r:.4f}" for n, b, r in zip(names6, sub_b, sub_r)))
    if args.save:
        import joblib
        joblib.dump({"model": m, "names": names, "groups": groups}, args.save)
        print("saved", args.save)


if __name__ == "__main__":
    main()
