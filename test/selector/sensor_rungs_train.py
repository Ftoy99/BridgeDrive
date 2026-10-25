"""§5h data-size test: §5f sensor-feature rungs, heads fit on navtrain-20k, scored once on navtest (Selector_Experiment_Plan §5h).
Same features and heads as sensor_rungs.py (cross-fit on navtest, 6k per fold). Only change: train = navtrain 20k pool, test = navtest.
Rungs: S1 (traj, z-logit, DD BEV gate, pred-agent dist), S2 (+ PCA32 current BEV), S3 (+ PCA16 lags 1-3), P2 (S1 + GT future, privileged diagnostic).
  python test/selector/sensor_rungs_train.py --train-pool sensor_pool_navtrain20k/sensor_pool.npz --out DIR
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase_c as pc
from compare_pdm_runs import paired_bootstrap
from sensor_rungs_helpers import cand_feats_min, zs, within_r2  # noqa: E402

TARGETS = {"NC": 0, "DAC": 1, "EP": 2, "TTC": 3, "Comf": 4, "PDMS": 6}
RUNGS = ["S1", "S2", "S3", "P2"]
EXP = Path("/mnt/ds/navsim-main/exp/selector")


def load_pool(split, pool_path):
    """Scenes of a split (phase_c.load_split filters) aligned to a sensor pool by token."""
    d = pc.load_split(split)
    pool = np.load(pool_path)
    idx = {t: i for i, t in enumerate(pool["tokens"].tolist())}
    keep = np.array([t in idx for t in d["tokens"]])
    d = {k: v[keep] for k, v in d.items()}
    sp = np.array([idx[t] for t in d["tokens"]])
    lag = {l: pool[f"lag{l}"][sp] for l in range(4)}
    return d, lag, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pool", required=True)
    ap.add_argument("--test-pool", default=str(EXP / "sensor_pool/sensor_pool.npz"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=20000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tr, lag_tr, _ = load_pool("navtrain", a.train_pool)
    te, lag_te, _ = load_pool("navtest", a.test_pool)
    # navtest log names: same keep mask as load_split (ok & finite pdm)
    import glob
    parts = [np.load(f) for f in sorted(glob.glob(str(pc.POOL_DIR / "navtest" / "shard_*.npz")))]
    keep_te = np.concatenate([p["ok"] for p in parts]).astype(bool) & np.concatenate(
        [np.isfinite(p["pdm"]).all(axis=(1, 2)) for p in parts])
    logs_te = np.concatenate([p["log_name"] for p in parts])[keep_te]
    assert len(logs_te) == len(te["tokens"])
    print(f"train navtrain {len(tr['tokens'])} | test navtest {len(te['tokens'])}", flush=True)

    Ntr, K = tr["logit"].shape
    Nte = te["logit"].shape[0]
    pca = {"lag0": PCA(32, random_state=0).fit(lag_tr[0]),
           **{f"lag{l}": PCA(16, random_state=0).fit(lag_tr[l]) for l in (1, 2, 3)}}

    def cand_matrix(c, D, lag, N, rung):
        lg = D["logit"].astype(np.float64)
        parts_ = [c["traj"], zs(lg)[..., None], np.stack([c["gate"], c["pa_dist"]], -1)]
        if rung == "P2":
            parts_.append(np.stack([c["gt_ade"], c["gt_fin"]], -1))
        if rung in ("S2", "S3"):
            z0 = pca["lag0"].transform(lag[0])
            parts_.append(np.broadcast_to(z0[:, None], (N, K, z0.shape[1])))
        if rung == "S3":
            for l in (1, 2, 3):
                zl = pca[f"lag{l}"].transform(lag[l])
                parts_.append(np.broadcast_to(zl[:, None], (N, K, zl.shape[1])))
        return np.concatenate(parts_, -1).astype(np.float32)

    def product(p):
        return p["NC"] * p["DAC"] * (5 * p["EP"] + 5 * p["TTC"] + 2 * p["Comf"]) / 12.0

    c_tr, c_te = cand_feats_min(tr), cand_feats_min(te)
    rows, sub_summary = [], {}
    ar = np.arange(Nte)
    pdms_te = te["pdm"][..., 6].astype(np.float64)
    dd_pick = te["logit"].argmax(1)
    dd_sel = pdms_te[ar, dd_pick]
    print(f"sanity: DD cls navtest PDMS {dd_sel.mean():.4f} (expect 0.8801)", flush=True)
    for rung in RUNGS:
        Xtr = cand_matrix(c_tr, tr, lag_tr, Ntr, rung)
        Xte = cand_matrix(c_te, te, lag_te, Nte, rung)
        pred = {}
        for tname, col in TARGETS.items():
            m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=0)
            m.fit(Xtr.reshape(-1, Xtr.shape[-1]), tr["pdm"][..., col].reshape(-1))
            pred[tname] = m.predict(Xte.reshape(-1, Xte.shape[-1])).reshape(Nte, K)
            print(f"{rung} head {tname} done", flush=True)
        pk_prod = product(pred).argmax(1)
        pk_head = pred["PDMS"].argmax(1)
        pick_prod = pdms_te[ar, pk_prod]
        pick_head = pdms_te[ar, pk_head]
        diff = pick_prod - dd_sel
        mean, lo, hi, p_le0 = paired_bootstrap(diff, n_boot=a.boot, seed=0)
        uniq = np.unique(logs_te)
        rng = np.random.RandomState(0)
        groups = {u: np.where(logs_te == u)[0] for u in uniq}
        cm = np.array([diff[np.concatenate([groups[u] for u in rng.choice(uniq, len(uniq))])].mean() for _ in range(a.boot)])
        clo, chi = np.percentile(cm, [2.5, 97.5])
        r2 = {f"r2_{t}": within_r2(pred[t], te["pdm"][..., col].astype(np.float64)) for t, col in TARGETS.items()}
        row = dict(rung=rung, n_train=Ntr, n_test=Nte, pick_pdms=pick_prod.mean(), delta_vs_dd_cls=mean,
                   delta_ci_lo=lo, delta_ci_hi=hi, delta_p_le0=p_le0, delta_cluster_ci_lo=clo, delta_cluster_ci_hi=chi,
                   pdms_head_pick=pick_head.mean(), pdms_head_delta=(pick_head - dd_sel).mean(),
                   privileged=int(rung == "P2"), **r2)
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        run_dir = out / f"rung_{rung}"
        run_dir.mkdir(parents=True, exist_ok=True)
        np.savez(run_dir / "picks.npz", tokens=te["tokens"], pick_idx=pk_prod, pdms_pick=pick_prod, log_name=logs_te)
        git = subprocess.run(["git", "-C", "/mnt/jimmys/BridgeDrive", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        params = dict(selector_family="sensor_rungs_datasize", rung=rung, train_split="navtrain", train_n=Ntr,
                      test_split="navtest", test_n=Nte, train_pool=a.train_pool, test_pool=a.test_pool, git_commit=git,
                      inputs={"S1": "traj+z_logit+bev_gate+pred_agent_dist", "S2": "S1+pca32_current_bev",
                              "S3": "S2+pca16_lags1-3", "P2": "S1+gt_future(privileged)"}[rung],
                      labels="real_pdms", loss="hgb_per_subscore", head="HistGB max_iter200 lr0.1", seed=0,
                      deployable=int(rung != "P2"), privileged=int(rung == "P2"), public=1, internal=0,
                      smoke=0, dd_in_sample_train=1, boot=a.boot)
        params_metrics = {k: float(v) for k, v in row.items() if isinstance(v, (int, float, np.floating)) and k not in ("n_train", "n_test")}
        params_metrics.update(n_train=Ntr, n_test=Nte)
        (run_dir / "params.json").write_text(json.dumps(params, indent=2, default=float))
        (run_dir / "metrics.json").write_text(json.dumps(params_metrics, indent=2, default=float))
        (run_dir / "tags.json").write_text(json.dumps({"public": "1", "internal": "0", "smoke": "0",
                                                       "privileged": str(int(rung == "P2")),
                                                       "deployable": str(int(rung != "P2")),
                                                       "dd_in_sample_train": "1"}, indent=2))
        print(f"RUN_DIR {run_dir}", flush=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2, default=float))
    print("SENSOR_RUNGS_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
