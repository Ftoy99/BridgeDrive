"""§5i BEV-vs-map check (Selector_Experiment_Plan.md §5i; pre-registered 2026-10-25, before any run).

Question: is the sensor-side drivable gate (filter-then-rank on DD's predicted BEV, F13) wrong because of our gate or because of perception?
  1. Gate variants on the same pred_bev: V0 (approx ego box, classes {1,3}), V1 (devkit box, {1,3}), V2 (devkit box, {1,3} + boxes {4,5,6}
     treated as pass). Human-pass rate, candidate pass rate, agreement with cache DAC, DAC-fail recall and precision.
  2. Road IoU vs the map raster (GT BEV `gt_bev.py`, same grid) per range band; box cells excluded (IoU A) or counted as road (IoU B).
  3. V2 (and V1) on the GT BEV: logic check, agreement with cache DAC.
  4. Filter-then-rank with the variant chosen on navtrain-20k, read once on navtest. Compare DD cls 0.8801, F13 0.8660, map gate 0.9063.
  V3 = V2 on the scorer's LQR-simulated poses (sim_poses.py), pre-registered as §5i measurement 1.

  /home/harris/miniconda3/envs/hidden/bin/python test/selector/bev_vs_map.py            # writes the run dir, no MLflow
Run dir: /mnt/ds/navsim-main/exp/selector/runs/sel_dd_k20_public_v2_5i_bevmap_<variant>_none_none_s0
"""
import glob, json, os, sys
from pathlib import Path

import numpy as np

POOL = Path("/mnt/ds/navsim-main/exp/selector/dd_k20_public_v2")
DIAG = Path("/mnt/ds/navsim-main/exp/selector/diag_bev_vs_map")
RUNS = Path("/mnt/ds/navsim-main/exp/selector/runs")
PDMS_COL, NC_COL, DAC_COL = 6, 0, 1
PX_PER_M, BEV_H, BEV_W = 4.0, 128, 256
ROAD = (1, 3)
BOX = (4, 5, 6)
# ego box: approx (phase_c) vs devkit Pacifica (Plan §5i code inspection)
GEOM = {"approx": (4.9, 2.0, 1.45), "devkit": (5.176, 2.297, 1.461)}
VARIANTS = {"V0": ("approx", False), "V1": ("devkit", False), "V2": ("devkit", True), "V3": ("devkit", True)}  # V3: sim poses (sim_poses.py)
SIM_DIR = {"navtest": DIAG / "sim_navtest", "navtrain": DIAG / "sim_navtrain"}
# range distance of each pixel from ego (row 0 = rear of window, ego at col 128)
_r = (np.arange(BEV_H) + 0.5) / PX_PER_M
_c = (np.arange(BEV_W) - BEV_W // 2 + 0.5) / PX_PER_M
DIST = np.hypot(_r[:, None], _c[None, :])
BANDS = [(0, 10), (10, 20), (20, 32.01)]


def load_pool(split, tokens=None):
    shards = sorted(glob.glob(str(POOL / split / "shard_*.npz")))
    keys = ["tokens", "traj", "logit", "pdm", "ok", "pred_bev", "gt_future", "log_name"]
    parts = [np.load(s) for s in shards]
    d = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    keep = d["ok"].astype(bool) & np.isfinite(d["pdm"]).all(axis=(1, 2))
    if tokens is not None:
        keep &= np.isin(d["tokens"], list(tokens))
    return {k: v[keep] for k, v in d.items()}


def load_gt(split_dir, tokens):
    """token -> GT BEV raster, restricted to `tokens`. Accepts shard files (navtest) or grouped files (navtrain)."""
    files = sorted(glob.glob(str(DIAG / split_dir / "*.npz")))
    out = {}
    want = set(tokens)
    for f in files:
        d = np.load(f)
        for t, b in zip(d["tokens"], d["gt_bev"]):
            if t in want:
                out[str(t)] = b
    return out


def gate(poses, bev, geom, box_pass):
    """poses (N,K,P,3) ego frame (x, y, heading) -> (N,K) bool: all ego-box corners at all poses on road (or out of window / box if box_pass)."""
    L, W, fwd = GEOM[geom]
    N, K = poses.shape[:2]
    x, y, h = poses[..., 0], poses[..., 1], poses[..., 2]
    cx, sx = np.cos(h), np.sin(h)
    ox, oy = x + fwd * cx, y + fwd * sx
    cols = []
    for a in (1, -1):
        for b in (1, -1):
            cols.append((ox + a * L / 2 * cx - b * W / 2 * sx, oy + a * L / 2 * sx + b * W / 2 * cx))
    px = np.stack([c[0] for c in cols], -1)
    py = np.stack([c[1] for c in cols], -1)
    row = np.floor(PX_PER_M * px).astype(np.int64)
    col = np.floor(PX_PER_M * py).astype(np.int64) + BEV_W // 2
    inside = (row >= 0) & (row < BEV_H) & (col >= 0) & (col < BEV_W)
    flat = np.arange(N)[:, None, None, None] * BEV_H * BEV_W + np.clip(row, 0, BEV_H - 1) * BEV_W + np.clip(col, 0, BEV_W - 1)
    cell = bev.reshape(-1)[flat]
    ok = np.isin(cell, ROAD)
    if box_pass:
        ok |= np.isin(cell, BOX)
    return (~inside | ok).all(axis=(2, 3))


def load_sim(split, tokens):
    """token -> (K+1, 41, 3) scorer-simulated ego poses (last row = human), aligned to `tokens`."""
    m = {}
    for f in sorted(glob.glob(str(SIM_DIR[split] / "*.npz"))):
        d = np.load(f)
        for t, x in zip(d["tokens"], d["sim"]):
            m[str(t)] = x
    return np.stack([m[str(t)] for t in tokens])


def run_gate(d, bev, v):
    """(candidate pass (N,K), human pass (N,)) for variant v on BEV `bev`."""
    geom, bp = VARIANTS[v]
    if v == "V3":
        sim_ = d["sim"]
        K = sim_.shape[1] - 1
        return gate(sim_[:, :K, 1:], bev, geom, bp), gate(sim_[:, K:, 1:], bev, geom, bp)[:, 0]
    return gate(d["traj"], bev, geom, bp), gate(d["gt_future"][:, None], bev, geom, bp)[:, 0]


def cache_dac(pdm):
    return pdm[..., DAC_COL] >= 0.5


def gate_stats(pass_, human_pass, dac):
    fail = ~dac
    rej = ~pass_
    return dict(
        human_pass=float(human_pass.mean()),
        cand_pass=float(pass_.mean()),
        agree_dac=float((pass_ == dac).mean()),
        dac_fail_recall=float((rej & fail).sum() / max(fail.sum(), 1)),
        dac_fail_precision=float((rej & fail).sum() / max(rej.sum(), 1)),
    )


def iou_by_band(pred, gt):
    """Road IoU (classes ROAD, boxes excluded: IoU A; boxes counted as road: IoU B), per scene, per range band. Arrays (N,H,W)."""
    res = {}
    box_p = np.isin(pred, BOX)
    box_g = np.isin(gt, BOX)
    road_p = np.isin(pred, ROAD)
    road_g = np.isin(gt, ROAD)
    valid = ~(box_p | box_g)
    road_pB = road_p | box_p
    road_gB = road_g | box_g
    for name, (rp, rg, msk) in {"A": (road_p, road_g, valid), "B": (road_pB, road_gB, np.ones_like(valid))}.items():
        for lo, hi in BANDS:
            band = (DIST >= lo) & (DIST < hi)
            m = msk & band[None]
            inter = (rp & rg & m).sum((1, 2))
            union = ((rp | rg) & m).sum((1, 2))
            per_scene = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
            pooled = inter.sum() / max(union.sum(), 1)
            res[f"iou{name}_{lo}_{int(hi)}_mean"] = float(np.nanmean(per_scene))
            res[f"iou{name}_{lo}_{int(hi)}_pooled"] = float(pooled)
            res[f"recall{name}_{lo}_{int(hi)}_pooled"] = float((rp & rg & m).sum() / max((rg & m).sum(), 1))
    return res


def paired_ci(diff, log_ids, n_boot=20000, seed=0):
    """Mean diff with scene bootstrap and cluster-by-log bootstrap, 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    est = diff.mean()
    scene = np.empty(n_boot)
    for i in range(0, n_boot, 500):
        idx = rng.integers(0, n, size=(min(500, n_boot - i), n))
        scene[i:i + 500] = diff[idx].mean(1)
    uniq, inv = np.unique(log_ids, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    cnts = np.bincount(inv).astype(float)
    cl = np.empty(n_boot)
    for i in range(0, n_boot, 500):
        idx = rng.integers(0, len(uniq), size=(min(500, n_boot - i), len(uniq)))
        cl[i:i + 500] = sums[idx].sum(1) / cnts[idx].sum(1)
    return dict(est=float(est), scene_lo=float(np.percentile(scene, 2.5)), scene_hi=float(np.percentile(scene, 97.5)),
                cluster_lo=float(np.percentile(cl, 2.5)), cluster_hi=float(np.percentile(cl, 97.5)))


def gate_variant_table(d, gt=None):
    """Per variant: gate on predicted BEV (+ optional GT BEV) statistics."""
    dac = cache_dac(d["pdm"])
    out = {}
    for v in VARIANTS:
        cand, human = run_gate(d, d["pred_bev"], v)
        out[v] = dict(pred=gate_stats(cand, human, dac))
    return out


def main():
    navtest = load_pool("navtest")
    nt_gt = load_gt("gtbev_navtest", navtest["tokens"])
    keep = np.isin(navtest["tokens"], list(nt_gt))
    navtest = {k: v[keep] for k, v in navtest.items()}
    navtest["gt_bev"] = np.stack([nt_gt[str(t)] for t in navtest["tokens"]])
    navtest["sim"] = load_sim("navtest", navtest["tokens"])
    tok20 = [t.strip() for t in open(DIAG / "navtrain20k_tokens.txt") if t.strip()]
    navtr = load_pool("navtrain", tokens=tok20)
    tr_gt = load_gt("gtbev_navtrain20k", navtr["tokens"])
    keep = np.isin(navtr["tokens"], list(tr_gt))
    navtr = {k: v[keep] for k, v in navtr.items()}
    navtr["gt_bev"] = np.stack([tr_gt[str(t)] for t in navtr["tokens"]])
    navtr["sim"] = load_sim("navtrain", navtr["tokens"])
    print(f"navtest n={len(navtest['tokens'])} navtrain20k n={len(navtr['tokens'])}", flush=True)

    res = {"navtest": {}, "navtrain20k": {}}
    for name, d in (("navtest", navtest), ("navtrain20k", navtr)):
        res[name]["gate_pred"] = gate_variant_table(d)
        dac = cache_dac(d["pdm"])
        res[name]["gate_gt"] = {}
        for v in VARIANTS:
            cand, human = run_gate(d, d["gt_bev"], v)
            res[name]["gate_gt"][v] = gate_stats(cand, human, dac)
        res[name]["iou"] = iou_by_band(d["pred_bev"], d["gt_bev"])
        print(name, "done", flush=True)

    # choose variant on navtrain-20k: highest human-pass + DAC agreement (predicted BEV). Navtest read once after.
    tr = res["navtrain20k"]["gate_pred"]
    chosen = max(VARIANTS, key=lambda v: tr[v]["pred"]["human_pass"] + tr[v]["pred"]["agree_dac"])
    res["chosen_variant"] = chosen
    res["chosen_rule"] = "argmax over V0-V3 of (human_pass + agree_dac) on navtrain-20k predicted BEV"

    # step 4: filter-then-rank on navtest with chosen variant
    p, _ = run_gate(navtest, navtest["pred_bev"], chosen)
    has = p.any(1)
    score = np.where(has[:, None], np.where(p, navtest["logit"], -np.inf), navtest["logit"])
    pick = score.argmax(1)
    pdms = navtest["pdm"][..., PDMS_COL]
    ar = np.arange(len(pdms))
    ftr_pdms = pdms[ar, pick]
    ddcls_pdms = pdms[ar, navtest["logit"].argmax(1)]
    diff = ftr_pdms - ddcls_pdms
    ci = paired_ci(diff, navtest["log_name"])
    res["ftr"] = dict(
        chosen=chosen, pdms_ftr=float(ftr_pdms.mean()), pdms_ddcls=float(ddcls_pdms.mean()),
        pdms_oracle=float(pdms.max(1).mean()), scenes_no_pass=float((~has).mean()),
        pass_rate=float(p.mean()), diff_vs_ddcls=ci,
        share_of_map_gain=float((ftr_pdms.mean() - ddcls_pdms.mean()) / (0.9063 - ddcls_pdms.mean())),
    )
    print(json.dumps(res, indent=1, default=float), flush=True)
    write_run(res, navtest, chosen)
    return res


def decision(res):
    """Gate (i)/(ii)/(iii) from Selector_Experiment_Plan.md §5i, applied to the chosen variant."""
    c = res["chosen_variant"]
    v2_hp = res["navtest"]["gate_pred"]["V2"]["pred"]["human_pass"]
    v2_gt_agree = res["navtest"]["gate_gt"]["V2"]["agree_dac"]
    iou_a = res["navtest"]["iou"]
    band_iou = iou_a["iouA_10_20_mean"], iou_a["iouA_20_32_mean"]
    if v2_gt_agree < 0.95:
        return "(iii) GT-BEV agreement < 0.95: our rasterisation/geometry still differs; fix before anything else"
    if res["navtest"]["gate_pred"]["V2"]["pred"]["human_pass"] >= 0.95 and v2_gt_agree >= 0.95:
        return "(i) F13 gap was class mapping; predicted BEV carries drivable info; read step 4"
    if v2_hp < 0.90 and min(band_iou) < 0.8:
        return "(ii) perception is the limit (human-pass < 0.90, road IoU < 0.8 at 10-32 m): go to §5j"
    return "mixed: not (i), not (ii) clean; report both tables"


def write_run(res, navtest, chosen):
    run = RUNS / f"sel_dd_k20_public_v2_5i_bevmap_{chosen}_none_none_s0"
    run.mkdir(parents=True, exist_ok=True)
    (run / "result.json").write_text(json.dumps(res, indent=1, default=float))
    git = os.popen("git -C /mnt/jimmys/BridgeDrive rev-parse --short HEAD").read().strip()
    params = dict(pool_id="dd_k20_public_v2", split="navtest", n_scenes=len(navtest["tokens"]), selector_family="bev_vs_map_5i",
                  inputs="traj+pred_bev+gt_bev(logic check only)", labels="map_dac_none", loss="none", seed=0,
                  chosen_variant=chosen, git_commit=git, decision=decision(res))
    metrics = dict(pdms_select=res["ftr"]["pdms_ftr"], pdms_ddcls=res["ftr"]["pdms_ddcls"], pdms_oracle=res["ftr"]["pdms_oracle"],
                   ftr_diff_vs_ddcls=res["ftr"]["diff_vs_ddcls"]["est"], share_of_map_gain=res["ftr"]["share_of_map_gain"],
                   v2_human_pass_navtest=res["navtest"]["gate_pred"]["V2"]["pred"]["human_pass"],
                   v2_gt_agree_dac_navtest=res["navtest"]["gate_gt"]["V2"]["agree_dac"])
    (run / "params.json").write_text(json.dumps(params, indent=1))
    (run / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (run / "tags.json").write_text(json.dumps(dict(diag="bev_vs_map", public=1, deployable=1, privileged=0, post_hoc=0, smoke=0), indent=1))
    print("run dir", run, flush=True)


if __name__ == "__main__":
    main()
