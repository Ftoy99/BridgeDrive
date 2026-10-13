"""Which feature groups drive the gradient-boosted ranker's gain, and what do they cost at inference?
Same model/CV as eval_bridge_selectors.py (2-fold by scene, scene-relative PDMS target, argmax pick)."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from eval_bridge_selectors import load, zs, softmax, pairwise_dist, geometry
from sklearn.ensemble import HistGradientBoostingRegressor

dump = sys.argv[1]
d = load(dump)
score = d["pdm"][..., -1].astype(np.float64)
N, K = score.shape
ar = np.arange(N)
lg1, lg0 = d["logit1"].astype(np.float64), d["logit0"].astype(np.float64)
cls = lg1.argmax(1)
base = score[ar, cls]
fold = np.random.RandomState(0).randint(0, 2, N)

def add(g, name, x):
    g.append((name, x.astype(np.float32)))
G = {k: [] for k in ("cls", "geo", "cons", "free", "recon", "samp")}
for n, x in (("logit0", lg0), ("logit1", lg1), ("p1", softmax(lg1)), ("zlogit", zs(lg1)), ("rank", np.argsort(np.argsort(-lg1, 1), 1).astype(float))):
    add(G["cls"], n, x)
for n, x in geometry(d["final"]).items():
    add(G["geo"], n, x); add(G["geo"], "z" + n, zs(x))
D = pairwise_dist(d["final"][..., :2]).astype(np.float64)
p1 = softmax(lg1)
add(G["cons"], "meandist", D.mean(2)); add(G["cons"], "kde1", np.einsum("nij,nj->ni", np.exp(-D ** 2 / 2.0), p1))
add(G["cons"], "kde2", np.einsum("nij,nj->ni", np.exp(-D ** 2 / 8.0), p1))
for k in ("path_len", "last_move", "x0_first_dev", "x0_mean_dev", "travel", "kinetic"):
    add(G["free"], k, d[k].astype(np.float64)); add(G["free"], "z" + k, zs(d[k].astype(np.float64)))
sm = d["step_move"].astype(np.float64); xs = d["x0dev_steps"].astype(np.float64)
add(G["free"], "sm_first", sm[..., 0]); add(G["free"], "sm_last3", sm[..., -3:].mean(-1)); add(G["free"], "sm_max", sm.max(-1))
add(G["free"], "x0dev_last5", xs[..., -5:].mean(-1))
rec = d["recon"].astype(np.float64)
for r in range(rec.shape[-1]):
    add(G["recon"], f"recon{r}", rec[..., r])
add(G["recon"], "recon_mean", rec.mean(-1)); add(G["recon"], "zrecon_mean", zs(rec.mean(-1)))
for k in ("samp_std", "samp_off"):
    add(G["samp"], k, d[k].astype(np.float64)); add(G["samp"], "z" + k, zs(d[k].astype(np.float64)))

configs = [
    ("classifier logits only (re-calibration)", ["cls"], "free"),
    ("+ trajectory geometry", ["cls", "geo"], "free (cheap CPU)"),
    ("+ consensus among the 80", ["cls", "geo", "cons"], "free (cheap CPU)"),
    ("+ signals from the normal denoising loop", ["cls", "geo", "cons", "free"], "free (tracked inside the loop)"),
    ("+ re-noise reconstruction error (19 levels)", ["cls", "geo", "cons", "free", "recon"], "+19 decoder passes"),
    ("+ stochastic resampling spread (2 samples)", ["cls", "geo", "cons", "free", "samp"], "+2 extra full denoises"),
    ("ALL", ["cls", "geo", "cons", "free", "recon", "samp"], "both extras"),
]
print(f"{N} scenes | deployed {base.mean():.4f} | oracle {score.max(1).mean():.4f}\n")
print(f"{'features':46s} {'#feat':>5s} {'PDMS':>7s} {'gain':>8s} {'+-SE':>7s}   extra inference cost")
for name, groups, cost in configs:
    cols = [x for g in groups for _, x in G[g]]
    X = np.stack(cols, -1)
    y = (score - score.mean(1, keepdims=True)).reshape(-1)
    pick = np.zeros(N, dtype=np.int64)
    t0 = time.time()
    for f in (0, 1):
        tr = np.repeat(fold != f, K)
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, random_state=0)
        m.fit(X.reshape(N * K, -1)[tr], y[tr])
        te = np.where(fold == f)[0]
        pick[te] = m.predict(X[te].reshape(-1, X.shape[-1])).reshape(len(te), K).argmax(1)
    s = score[ar, pick]; dl = s - base
    print(f"{name:46s} {len(cols):5d} {s.mean():7.4f} {dl.mean():+8.4f} {dl.std(ddof=1)/np.sqrt(N):7.4f}   {cost}  [{time.time()-t0:.0f}s]", flush=True)
