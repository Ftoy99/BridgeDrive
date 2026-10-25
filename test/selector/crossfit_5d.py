"""§5d cross-fit protocol on navtest (Selector_Experiment_Plan §5d, pre-registered). Reuses the model code of phase_c.py.

Configs: a1 pairwise (traj + DD pred BEV + pred agents), a1l (a1 + DD logit, z over K), hgb_a1l (HistGB head on the a1l summary
vector: traj, z-logit, DD BEV gate, predicted-agent distance; HGB cannot take the raster, so the BEV enters as its gate).
2-fold cross-fit on navtest, folds by log_name (same fold assignment as sensor_rungs.py for a given seed).
Neural: 10% of training-fold logs held out as inner split; epoch chosen by inner-split pick PDMS; test fold scored once.
Output: one run dir per (config, seed) with out-of-fold picks, plus cmp_ dir vs DD cls.

  python test/selector/crossfit_5d.py --seeds 0,1,2
  SMOKE=1 EPOCHS=1 python test/selector/crossfit_5d.py --seeds 0 --boot 500   (7 shards, runs_smoke, tagged smoke=1)
"""
import argparse, glob, json, subprocess, time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

import phase_c as pc
from compare_pdm_runs import paired_bootstrap

EPOCHS = pc.EPOCHS
CONFIGS = {  # name: (selector family, scene, logit, loss, labels, inputs)
    "a1": dict(scene=True, logit=False, loss="pairwise", inputs="traj+pred_bev+pred_agents"),
    "a1l": dict(scene=True, logit=True, loss="pairwise", inputs="traj+pred_bev+pred_agents+dd_cls_logit(z over K)"),
    "hgb_a1l": dict(scene=None, logit=True, loss="mse", inputs="summary(traj, z-logit, DD BEV gate, pred-agent dist); HGB"),
}


def load_navtest():
    shards = sorted(glob.glob(str(pc.POOL_DIR / "navtest" / "shard_*.npz")))
    if pc.SMOKE:
        shards = shards[:7]
    parts = [np.load(s) for s in shards]
    d = pc.load_split("navtest")
    keep = np.concatenate([p["ok"] for p in parts]).astype(bool) & np.concatenate(
        [np.isfinite(p["pdm"]).all(axis=(1, 2)) for p in parts])
    d["log_name"] = np.concatenate([p["log_name"] for p in parts])[keep]
    assert len(d["log_name"]) == len(d["tokens"])
    return d


def pack(d, mu, sd):
    ag = np.concatenate([d["pred_agents"].astype(np.float32), d["pred_agent_prob"].astype(np.float32)[..., None]], -1)
    ag[..., :5] = (ag[..., :5] - mu) / sd
    lg = d["logit"].astype(np.float32)
    lg = (lg - lg.mean(1, keepdims=True)) / (lg.std(1, keepdims=True) + 1e-6)
    return dict(traj=torch.from_numpy(d["traj"]), lg=torch.from_numpy(lg),
                pdms=torch.from_numpy(d["pdm"][:, :, pc.PDMS_COL].copy()),
                bev=torch.from_numpy(d["pred_bev"]), ag=torch.from_numpy(ag))


def inner_split(tr, logs, seed, f):
    """10% of training-fold logs (at least one) as inner split, by log. Same split for every config on this (seed, fold)."""
    tr_logs = np.unique(logs[tr])
    val_logs = set(np.random.RandomState(seed * 10 + f + 100).permutation(tr_logs)[: max(1, len(tr_logs) // 10)])
    in_val = np.array([l in val_logs for l in logs[tr]])
    return tr[~in_val], tr[in_val]


def train_neural(D, fit, val, te, scene, logit, seed, dev, trlog, tag):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = pc.Selector(scene, False, False, logit, True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=pc.LR, weight_decay=pc.WD)
    best = dict(val=-1.0, state=None, epoch=-1)
    for ep in range(EPOCHS):
        model.train()
        order = fit[torch.randperm(len(fit), generator=g).numpy()]
        tot, nb = 0.0, 0
        for i in range(0, len(order), pc.BATCH):
            idx = order[i:i + pc.BATCH]
            s = model(**pc.make_inputs(D, idx, dev, scene, False, False, logit, True))
            loss = pc.loss_fn(s, D["pdms"][idx].to(dev), "pairwise")
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        vs = pc.score_all(model, D, val, dev, scene, False, False, logit, True)
        vpick = D["pdms"][torch.as_tensor(val)].numpy()[np.arange(len(val)), vs.argmax(1)].mean()
        trlog.append(dict(tag=tag, epoch=ep, train_loss=tot / nb, inner_val_pdms_select=float(vpick)))
        print(trlog[-1], flush=True)
        if vpick > best["val"]:
            best = dict(val=float(vpick), state={k: v.clone() for k, v in model.state_dict().items()}, epoch=ep)
    model.load_state_dict(best["state"])
    return pc.score_all(model, D, te, dev, scene, False, False, logit, True), best


def hgb_features(d):
    traj = d["traj"][..., :3].astype(np.float64)
    N, K = traj.shape[:2]
    gate = pc.gate_pass(traj, d["pred_bev"]).astype(np.float64)
    pa = d["pred_agents"][..., :2].astype(np.float64)
    pprob = d["pred_agent_prob"].astype(np.float64)
    pa_dist = np.zeros((N, K))
    for s0 in range(0, N, 500):
        sl = slice(s0, min(s0 + 500, N))
        dd = np.linalg.norm(traj[sl][:, :, :, None, :2] - pa[sl][:, None, None], axis=-1)
        dd = np.where(pprob[sl][:, None, None] > 0.5, dd, 50.0)
        pa_dist[sl] = np.minimum(dd.min((2, 3)), 50.0)
    lg = d["logit"].astype(np.float64)
    zl = (lg - lg.mean(1, keepdims=True)) / (lg.std(1, keepdims=True) + 1e-6)
    return np.concatenate([pc.traj_feats(torch.from_numpy(traj)).numpy(), zl[..., None], gate[..., None], pa_dist[..., None]], -1)


def hgb_fit_predict(X, y, tr, te):
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=0)
    m.fit(X[tr].reshape(-1, X.shape[-1]), y[tr].reshape(-1))
    return m.predict(X[te].reshape(-1, X.shape[-1])).reshape(len(te), X.shape[1])


def cluster_ci(diff, logs, n, seed):
    rng = np.random.RandomState(seed)
    uniq = np.unique(logs)
    groups = {u: np.where(logs == u)[0] for u in uniq}
    means = np.empty(n)
    for i in range(n):
        ii = np.concatenate([groups[u] for u in rng.choice(uniq, len(uniq))])
        means[i] = diff[ii].mean()
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--boot", type=int, default=20000)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    d = load_navtest()
    logs = d["log_name"]
    N, K = d["logit"].shape
    ar = np.arange(N)
    pdms = d["pdm"][..., pc.PDMS_COL].astype(np.float64)
    dd_pick = d["logit"].argmax(1)
    dd_sel = pdms[ar, dd_pick]
    print(f"scenes {N} K {K} logs {len(np.unique(logs))} device {dev} epochs {EPOCHS} smoke {int(pc.SMOKE)}", flush=True)
    print(f"sanity: DD cls navtest PDMS {dd_sel.mean():.4f} (expect 0.8801 full set)", flush=True)
    hgb_X = hgb_features(d)
    git = subprocess.run(["git", "-C", "/mnt/jimmys/BridgeDrive", "rev-parse", "--short", "HEAD"],
                                       capture_output=True, text=True).stdout.strip()
    cache_sha = json.loads((pc.POOL_DIR / "navtest" / "pool_manifest.json").read_text())["cache_sha"]
    tags = {"public": "1", "internal": "0", "smoke": str(int(pc.SMOKE)), "privileged": "0", "deployable": "1"}
    summary = []

    for seed in [int(s) for s in a.seeds.split(",")]:
        rng = np.random.RandomState(seed)
        uniq = np.unique(logs)
        fold_of_log = dict(zip(uniq, rng.permutation(len(uniq)) % 2))
        fold = np.array([fold_of_log[l] for l in logs])
        picks = {c: np.zeros(N, int) for c in CONFIGS}
        scores = {c: np.zeros((N, K), np.float32) for c in CONFIGS if c != "hgb_a1l"}
        trlog = {c: [] for c in CONFIGS}
        ep_sel = {c: [] for c in CONFIGS}
        for f in (0, 1):
            tr = np.where(fold != f)[0]
            te = np.where(fold == f)[0]
            ag_fit = np.concatenate([d["pred_agents"].astype(np.float32),
                                     d["pred_agent_prob"].astype(np.float32)[..., None]], -1)[tr][..., :5].reshape(-1, 5)
            D = pack(d, ag_fit.mean(0), ag_fit.std(0) + 1e-6)
            fit, val = inner_split(tr, logs, seed, f)
            for c, cfg in CONFIGS.items():
                if c == "hgb_a1l":
                    p = hgb_fit_predict(hgb_X, pdms, tr, te)
                    picks[c][te] = p.argmax(1)
                    continue
                s, best = train_neural(D, fit, val, te, cfg["scene"], cfg["logit"], seed, dev, trlog[c], f"{c}_s{seed}_f{f}")
                scores[c][te] = s
                picks[c][te] = s.argmax(1)
                ep_sel[c].append(best["epoch"])
                print(f"seed {seed} fold {f} {c} best epoch {best['epoch']}", flush=True)
            del D

        for c, cfg in CONFIGS.items():
            pk = picks[c]
            pick_pdms = pdms[ar, pk]
            diff = pick_pdms - dd_sel
            mean, lo, hi, p_le0 = paired_bootstrap(diff, n_boot=a.boot, seed=seed)
            c_lo, c_hi = cluster_ci(diff, logs, a.boot, seed)
            m = pc.metrics_of(d["pdm"], pk, scores.get(c))
            m.update(delta_vs_dd_cls=mean, delta_ci_lo=lo, delta_ci_hi=hi, delta_p_le0=p_le0,
                     delta_cluster_ci_lo=c_lo, delta_cluster_ci_hi=c_hi, dd_cls_pdms=dd_sel.mean(),
                     best_epoch_fold0=ep_sel[c][0] if ep_sel[c] else -1, best_epoch_fold1=ep_sel[c][1] if ep_sel[c] else -1,
                     wall_s=time.time() - t0)
            labels = "real_pdms"
            run_name = f"sel_dd_k20_public_v2_D5d_{c}_{labels}_{cfg['loss']}_s{seed}" + ("_smoke" if pc.SMOKE else "")
            params = dict(pool_id=pc.POOL_ID, split="navtest", n_scenes=N, K=K, cache_sha=cache_sha, git_commit=git,
                          seed=seed, public=1, internal=0, smoke=int(pc.SMOKE), selector_family=c, inputs=cfg["inputs"],
                          labels=labels, loss=cfg["loss"], protocol="crossfit_navtest_by_log_2fold", inner_split_frac=0.10,
                          epochs=EPOCHS if cfg["scene"] is not None else 0, batch=pc.BATCH, lr=pc.LR, wd=pc.WD,
                          arch=pc.ARCH if cfg["scene"] is not None else "HistGB max_iter 200 lr 0.1 on summary vector",
                          privileged=0, deployable=1, boot=a.boot, device=dev)
            out_dir = pc.RUNS / run_name
            out_dir.mkdir(parents=True, exist_ok=True)
            save = dict(tokens=d["tokens"], pick_idx=pk, pdms_pick=pick_pdms, fold=fold, log_name=logs)
            if c in scores:
                save["score"] = scores[c]
            np.savez(out_dir / "picks.npz", **save)
            (out_dir / "params.json").write_text(json.dumps(params, indent=2, default=float))
            (out_dir / "metrics.json").write_text(json.dumps(m, indent=2, default=float))
            (out_dir / "tags.json").write_text(json.dumps(tags, indent=2))
            (out_dir / "train_log.json").write_text(json.dumps(trlog[c], indent=2))
            cmp_dir = pc.RUNS / f"cmp_{run_name}_vs_ddcls"
            cmp_dir.mkdir(parents=True, exist_ok=True)
            (cmp_dir / "params.json").write_text(json.dumps(dict(a="dd_cls", b=run_name, split="navtest", n_scenes=N,
                                                                pool_id=pc.POOL_ID, cache_sha=cache_sha,
                                                                metric="pdms_v1_pick", n_boot=a.boot, seed=seed), indent=2))
            (cmp_dir / "metrics.json").write_text(json.dumps(dict(delta_vs_dd_cls=mean, delta_ci_lo=lo, delta_ci_hi=hi,
                                                                 delta_p_le0=p_le0, delta_cluster_ci_lo=c_lo,
                                                                 delta_cluster_ci_hi=c_hi, n_scenes=N), indent=2, default=float))
            (cmp_dir / "tags.json").write_text(json.dumps(tags, indent=2))
            print(f"RUN_DIR {out_dir}", flush=True)
            summary.append(dict(seed=seed, config=c, pdms=float(m["pdms_select"]), delta=float(mean), ci=[float(lo), float(hi)],
                                cluster_ci=[float(c_lo), float(c_hi)]))
            print(json.dumps(summary[-1]), flush=True)

    (pc.RUNS / "crossfit_5d_summary.json").write_text(json.dumps(summary, indent=2))
    print("CROSSFIT_5D_DONE", flush=True)


if __name__ == "__main__":
    main()
