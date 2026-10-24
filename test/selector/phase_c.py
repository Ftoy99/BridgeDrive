"""Phase C selector zoo on the P1 pool `dd_k20_public_v2` (Selector_Experiment_Plan v3 §5).

Learned selectors train on navtrain 20k (90% fit, 10% calibration by scene, seed-0 split) and are evaluated once on navtest
at the epoch with the best calibration pick PDMS. Baselines: DD cls (deployed argmax) and filter-then-rank (analytic
drivable-area gate on the predicted BEV, then DD cls; no training).

  python test/selector/phase_c.py --selector dd_cls
  python test/selector/phase_c.py --selector ftr
  python test/selector/phase_c.py --selector {a0,a1,r3} --loss {bce,pairwise} --seed S
  python test/selector/phase_c.py --selector {gt_fut,gt_ag,gt_fa} --loss pairwise --seed S

  a0 = trajectory only, a1 = + predicted BEV and agents, r3 = a1 + GT ego future (privileged, teacher only).
  gt_fut / gt_ag / gt_fa = trajectory + GT ego future / GT agents at t0 / both. No sensor or DD input (privileged-only ladder, Plan §5b).
Run dirs: /mnt/ds/navsim-main/exp/selector/runs/<run_name>/ ; log with test/selector/log_selector_run.py.
SMOKE=1: 7 shards per split, 1 epoch, tagged smoke=1.
"""
import argparse, glob, json, os, subprocess, sys, time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/mnt/jimmys/BridgeDrive/test")
from compare_pdm_runs import paired_bootstrap

POOL_ID = "dd_k20_public_v2"
POOL_DIR = Path("/mnt/ds/navsim-main/exp/selector") / POOL_ID
SMOKE = os.environ.get("SMOKE", "0") == "1"
RUNS = Path("/mnt/ds/navsim-main/exp/selector/runs_smoke" if SMOKE else "/mnt/ds/navsim-main/exp/selector/runs")
EPOCHS = int(os.environ.get("EPOCHS", 1 if SMOKE else 12))  # sensitivity runs set EPOCHS; name gets _ep<N>
BATCH = 64
LR, WD = 3e-4, 1e-4
CAL_FRAC = 0.10
# bev_semantic_classes (transfuser_config.py): 1 = lane + intersection polygons, 3 = lane centrelines (thin lines on lanes).
# The predicted map puts the centreline class on ~half of GT poses, so drivable = {1, 3}.
ROAD_CLASSES = (1, 3)
PX_PER_M = 4.0  # bev_pixel_size 0.25 m
BEV_H, BEV_W = 128, 256
EGO_L, EGO_W, EGO_CENTER_FWD = 4.9, 2.0, 1.45  # approximate ego box; box centre = pose + offset along heading
PDMS_COL, NC_COL, DAC_COL = 6, 0, 1
METRIC_COLS = {"NC": 0, "DAC": 1, "EP": 2, "TTC": 3, "Comf": 4, "DDC": 5}
ARCH = "Scene: 4xConv(7->128,s2) + FC256 | agents FC128 | cand MLP 32->128->128 | head 256->256->1"


def load_split(split):
    shards = sorted(glob.glob(str(POOL_DIR / split / "shard_*.npz")))
    if SMOKE:
        shards = shards[:7]
    parts = [np.load(s) for s in shards]
    keys = ["tokens", "traj", "logit", "pdm", "ok", "pred_bev", "pred_agents", "pred_agent_prob", "gt_future", "gt_agents", "n_gt_agents"]
    d = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    keep = d["ok"].astype(bool) & np.isfinite(d["pdm"]).all(axis=(1, 2))
    return {k: v[keep] for k, v in d.items()}


def traj_feats(t):
    """(..., P, 3) poses (x, y, heading) -> (..., P*4) features."""
    x, y, h = t[..., 0] / 32.0, t[..., 1] / 32.0, t[..., 2]
    return torch.stack([x, y, torch.cos(h), torch.sin(h)], -1).flatten(-2)


class Scene(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(7, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 96, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(96, 128, 3, 2, 1), nn.ReLU(),
        )  # 128x256 -> 8x16
        self.bev = nn.Linear(128 * 8 * 16, 256)
        self.ag = nn.Sequential(nn.Linear(30 * 6, 128), nn.ReLU())

    def forward(self, bev, ag):
        onehot = F.one_hot(bev.long(), 7).permute(0, 3, 1, 2).float()
        return torch.cat([F.relu(self.bev(self.cnn(onehot).flatten(1))), self.ag(ag.flatten(1))], 1)


N_GT_AG = 30  # nearest GT agents kept (privileged-only ladder)


def gt_agent_feats(gt_agents, n_gt):
    """(N,128,7) GT agents (x, y, h, l, w, vx, vy; ego frame, first n valid) -> (N,30,9) nearest 30 with a present flag."""
    valid = np.arange(gt_agents.shape[1])[None] < n_gt[:, None]
    dist = np.where(valid, np.hypot(gt_agents[..., 0], gt_agents[..., 1]), np.inf)
    idx = np.argsort(dist, 1)[:, :N_GT_AG]
    a = np.take_along_axis(gt_agents, idx[..., None], 1)
    present = np.take_along_axis(valid, idx, 1)[..., None].astype(np.float32)
    x, y, h = a[..., 0] / 32.0, a[..., 1] / 32.0, a[..., 2]
    f = np.stack([x, y, np.cos(h), np.sin(h), a[..., 3], a[..., 4], a[..., 5] / 10.0, a[..., 6] / 10.0], -1)
    return np.concatenate([f * present, present], -1).astype(np.float32)


class Selector(nn.Module):
    def __init__(self, scene, fut, gta=False, logit=False, traj=True):
        super().__init__()
        self.use_scene, self.use_fut, self.use_gta, self.use_logit, self.use_traj = scene, fut, gta, logit, traj
        self.cand = nn.Sequential(nn.Linear(32 * traj + logit, 128), nn.ReLU(), nn.Linear(128, 128))
        width = 128
        if scene:
            self.scene = Scene()
            width += 384
        if fut:
            self.fut = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 128))
            width += 128
        if gta:
            self.gta = nn.Sequential(nn.Linear(N_GT_AG * 9, 128), nn.ReLU())
            width += 128
        self.head = nn.Sequential(nn.Linear(width, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, traj, bev=None, ag=None, fut=None, gta=None, lg=None):
        cand_in = ([traj_feats(traj)] if self.use_traj else []) + ([lg[..., None]] if self.use_logit else [])
        c = self.cand(torch.cat(cand_in, -1))  # (B, K, 128)
        ctx = []
        if self.use_scene:
            ctx.append(self.scene(bev, ag))
        if self.use_fut:
            ctx.append(self.fut(traj_feats(fut)))
        if self.use_gta:
            ctx.append(self.gta(gta.flatten(1)))
        if ctx:
            ctx = torch.cat(ctx, 1)[:, None].expand(-1, c.shape[1], -1)
            c = torch.cat([c, ctx], 2)
        return self.head(c).squeeze(-1)  # (B, K) scores


def loss_fn(s, y, kind):
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(s, y.clamp(0, 1))
    d = y[:, :, None] - y[:, None, :]  # pairwise logistic ranking on PDMS differences, ties dropped
    S = s[:, :, None] - s[:, None, :]
    m = d.abs() > 1e-3
    return F.softplus(-torch.sign(d) * S)[m].mean()


def make_inputs(D, idx, dev, scene, fut, gta=False, logit=False, traj=True):
    i = torch.as_tensor(idx)
    inp = {"traj": D["traj"][i].to(dev)} if traj else {"traj": None}
    if logit:
        inp["lg"] = D["lg"][i].to(dev)
    if scene:
        inp.update(bev=D["bev"][i].to(dev), ag=D["ag"][i].to(dev))
    if fut:
        inp["fut"] = D["fut"][i].to(dev)
    if gta:
        inp["gta"] = D["gta"][i].to(dev)
    return inp


@torch.no_grad()
def score_all(model, D, idx, dev, scene, fut, gta=False, logit=False, traj=True):
    model.eval()
    out = [model(**make_inputs(D, idx[i:i + 256], dev, scene, fut, gta, logit, traj)).float().cpu().numpy() for i in range(0, len(idx), 256)]
    return np.concatenate(out)


def gate_pass(traj, bev):
    """Analytic drivable-area proxy on the predicted BEV. traj (N,K,P,3) -> (N,K) bool.
    Pixel map follows transfuser_features._compute_map_*: row = 4x (forward), col = 4y + 128 (ego at col 128).
    Ego box corners at every pose must sit on road (class 1). Points outside the 32 m BEV count as pass (not observable)."""
    N, K = traj.shape[:2]
    x, y, h = traj[..., 0], traj[..., 1], traj[..., 2]
    cx, sx_ = np.cos(h), np.sin(h)
    ox, oy = x + EGO_CENTER_FWD * cx, y + EGO_CENTER_FWD * sx_
    corners = []
    for a in (1, -1):
        for b in (1, -1):
            corners.append((ox + a * EGO_L / 2 * cx - b * EGO_W / 2 * sx_, oy + a * EGO_L / 2 * sx_ + b * EGO_W / 2 * cx))
    px = np.stack([c[0] for c in corners], -1)  # (N,K,P,4)
    py = np.stack([c[1] for c in corners], -1)
    row = np.floor(PX_PER_M * px).astype(np.int64)
    col = np.floor(PX_PER_M * py).astype(np.int64) + BEV_W // 2
    inside = (row >= 0) & (row < BEV_H) & (col >= 0) & (col < BEV_W)
    flat = np.arange(N)[:, None, None, None] * BEV_H * BEV_W + np.clip(row, 0, BEV_H - 1) * BEV_W + np.clip(col, 0, BEV_W - 1)
    on_road = np.isin(bev.reshape(-1)[flat], ROAD_CLASSES)
    return (~inside | on_road).all(axis=(2, 3))


def metrics_of(pdm, pick, score=None):
    """pdm (N,K,7) official sub-scores, pick (N,). Names follow audit_headroom.py."""
    N = len(pick)
    ar = np.arange(N)
    pdms = pdm[:, :, PDMS_COL]
    sel = pdms[ar, pick]
    orc = pdms.max(1)
    out = dict(
        n_scenes=N, pdms_select=sel.mean(), pdms_oracle=orc.mean(), pool_mean=pdms.mean(), headroom=(orc - sel).mean(),
        regret_top1=(orc - sel).mean(), hit_best_pct=100 * (sel >= orc - 1e-6).mean(),
        gate_fail_rate=((pdm[ar, pick, NC_COL] < 0.5) | (pdm[ar, pick, DAC_COL] < 0.5)).mean(),
    )
    for name, c in METRIC_COLS.items():
        out[name] = pdm[ar, pick, c].mean()
    if score is not None:
        order = np.argsort(-score, axis=1)
        for k in (1, 2, 4, 8):
            out[f"best_of_top_{k}"] = np.take_along_axis(pdms, order[:, :k], 1).max(1).mean()
    return out


def dd_cls_pick(logit):
    return logit.argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", required=True, choices=["dd_cls", "ftr", "a0", "a1", "r3", "gt_fut", "gt_ag", "gt_fa", "a1l", "r0l"])
    ap.add_argument("--loss", default="pairwise", choices=["bce", "pairwise"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    te = load_split("navtest")
    pdm_te, logit_te = te["pdm"], te["logit"]
    n_te = len(pdm_te)
    git = subprocess.run(["git", "-C", "/mnt/jimmys/BridgeDrive", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    cache_sha = json.loads((POOL_DIR / "navtest" / "pool_manifest.json").read_text())["cache_sha"]
    train_sha = json.loads((POOL_DIR / "navtrain" / "pool_manifest.json").read_text())["cache_sha"]

    base = dict(pool_id=POOL_ID, split="navtest", n_scenes=n_te, K=20, cache_sha=cache_sha, train_cache_sha=train_sha,
                git_commit=git, seed=a.seed, public=1, internal=0, smoke=int(SMOKE))
    tags = {"public": "1", "internal": "0", "smoke": str(int(SMOKE))}

    if a.selector in ("dd_cls", "ftr"):
        if a.selector == "dd_cls":
            pick = dd_cls_pick(logit_te)
            score = logit_te
            run_name = f"sel_{POOL_ID}_C_ddcls_nearest_anchor_own_s0"
            params = dict(base, selector_family="dd_cls", inputs="traj", labels="nearest_anchor", loss="own",
                          arch="DiffusionDrive cls head (public ckpt, deployed argmax)", privileged=0, deployable=1)
        else:
            gate = gate_pass(te["traj"], te["pred_bev"])
            masked = np.where(gate, logit_te, -np.inf)
            has = gate.any(1)
            score = np.where(has[:, None], masked, logit_te)
            pick = score.argmax(1)
            human_gate = gate_pass(te["gt_future"][:, None], te["pred_bev"])[:, 0]
            dac_fail = pdm_te[:, :, DAC_COL] < 0.5
            agree = (gate != dac_fail).mean()
            recall = (~gate[dac_fail]).mean() if dac_fail.any() else float("nan")
            run_name = f"sel_{POOL_ID}_C_ftr_none_none_s0"
            params = dict(base, selector_family="filter_then_rank", inputs="traj+pred_bev(road mask)+dd_cls logit",
                          labels="none", loss="none", arch="analytic gate (ego box corners on BEV road) then DD cls argmax",
                          privileged=0, deployable=1, ftr_ego_L=EGO_L, ftr_ego_W=EGO_W, ftr_center_fwd=EGO_CENTER_FWD,
                          ftr_no_pass_fallback="dd_cls")
            extra = dict(ftr_gate_pass_rate=gate.mean(), ftr_gate_agree_dac=agree, ftr_gate_dac_fail_recall=recall,
                         ftr_human_pass_rate=human_gate.mean(), ftr_scenes_no_pass=(~has).mean())
        m = metrics_of(pdm_te, pick, score)
        if a.selector == "ftr":
            m.update(extra)
        pdms_pick = pdm_te[np.arange(n_te), pick, PDMS_COL]
        out_dir = RUNS / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "picks.npz", tokens=te["tokens"], pick_idx=pick, score=score, pdms_pick=pdms_pick)
        m["wall_s"] = time.time() - t0
        (out_dir / "params.json").write_text(json.dumps(params, indent=2, default=float))
        (out_dir / "metrics.json").write_text(json.dumps(m, indent=2, default=float))
        (out_dir / "tags.json").write_text(json.dumps(tags | {"privileged": "0", "deployable": "1"}, indent=2))
        print(json.dumps(m, indent=2, default=float))
        print("RUN_DIR", out_dir)
        return

    # learned selectors
    scene = a.selector in ("a1", "r3", "a1l")
    fut = a.selector in ("r3", "gt_fut", "gt_fa")
    gta = a.selector in ("gt_ag", "gt_fa")
    logit = a.selector in ("a1l", "r0l")  # post-hoc: DD cls logit as input (Selector_Results.md, DD logit ladder)
    traj = a.selector != "r0l"
    privileged = a.selector in ("r3", "gt_fut", "gt_ag", "gt_fa")
    tr = load_split("navtrain")
    n_tr = len(tr["pdm"])
    perm = np.random.RandomState(0).permutation(n_tr)
    n_cal = int(round(CAL_FRAC * n_tr))
    cal_idx, fit_idx = np.sort(perm[:n_cal]), np.sort(perm[n_cal:])

    ag_all = np.concatenate([tr["pred_agents"].astype(np.float32), tr["pred_agent_prob"].astype(np.float32)[..., None]], -1)
    mu = ag_all[fit_idx, :, :5].reshape(-1, 5).mean(0)
    sd = ag_all[fit_idx, :, :5].reshape(-1, 5).std(0) + 1e-6

    def pack(split_dict):
        ag = np.concatenate([split_dict["pred_agents"].astype(np.float32), split_dict["pred_agent_prob"].astype(np.float32)[..., None]], -1)
        ag[..., :5] = (ag[..., :5] - mu) / sd
        lg = split_dict["logit"].astype(np.float32)
        lg = (lg - lg.mean(1, keepdims=True)) / (lg.std(1, keepdims=True) + 1e-6)  # z-score over K per scene
        return dict(traj=torch.from_numpy(split_dict["traj"]), lg=torch.from_numpy(lg), pdms=torch.from_numpy(split_dict["pdm"][:, :, PDMS_COL].copy()),
                    bev=torch.from_numpy(split_dict["pred_bev"]), ag=torch.from_numpy(ag),
                    fut=torch.from_numpy(split_dict["gt_future"].astype(np.float32)),
                    gta=torch.from_numpy(gt_agent_feats(split_dict["gt_agents"], split_dict["n_gt_agents"])))

    D_tr, D_te = pack(tr), pack(te)
    model = Selector(scene, fut, gta, logit, traj).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    g = torch.Generator().manual_seed(a.seed)
    best = dict(cal=-1.0, state=None, epoch=-1)
    log = []
    for ep in range(EPOCHS):
        model.train()
        order = fit_idx[torch.randperm(len(fit_idx), generator=g).numpy()]
        tot, nb = 0.0, 0
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            s = model(**make_inputs(D_tr, idx, dev, scene, fut, gta, logit, traj))
            y = D_tr["pdms"][idx].to(dev)
            loss = loss_fn(s, y, a.loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        cal_s = score_all(model, D_tr, cal_idx, dev, scene, fut, gta, logit, traj)
        cal_pick = D_tr["pdms"][torch.as_tensor(cal_idx)].numpy()[np.arange(len(cal_idx)), cal_s.argmax(1)].mean()
        log.append(dict(epoch=ep, train_loss=tot / nb, cal_pdms_select=float(cal_pick)))
        print(log[-1], flush=True)
        if cal_pick > best["cal"]:
            best = dict(cal=float(cal_pick), state=deepcopy(model.state_dict()), epoch=ep)
    model.load_state_dict(best["state"])

    # navtest touched once, at the calibration-selected epoch
    score = score_all(model, D_te, np.arange(n_te), dev, scene, fut, gta, logit, traj)
    pick = score.argmax(1)
    m = metrics_of(pdm_te, pick, score)
    pdms_pick = pdm_te[np.arange(n_te), pick, PDMS_COL]

    dd_dir = RUNS / f"sel_{POOL_ID}_C_ddcls_nearest_anchor_own_s0"
    dd_pdms = np.load(dd_dir / "picks.npz")["pdms_pick"]
    mean, lo, hi, p_le0 = paired_bootstrap(pdms_pick - dd_pdms, n_boot=20000, seed=0)
    m.update(delta_vs_dd_cls=mean, delta_ci_lo=lo, delta_ci_hi=hi, delta_p_le0=p_le0,
             best_epoch=best["epoch"], cal_pdms_select=best["cal"], n_train=len(fit_idx), n_cal=len(cal_idx),
             wall_s=time.time() - t0)

    labels = "real_pdms" if a.selector != "dd_cls" else "nearest_anchor"
    inputs = {"a0": "traj", "a1": "traj+pred_bev+pred_agents", "r3": "traj+pred_bev+pred_agents+gt_future(ego)",
              "gt_fut": "traj+gt_future(ego)", "gt_ag": "traj+gt_agents_t0(30 nearest)",
              "gt_fa": "traj+gt_future(ego)+gt_agents_t0(30 nearest)",
              "a1l": "traj+pred_bev+pred_agents+dd_cls_logit(z over K)", "r0l": "dd_cls_logit(z over K) only"}[a.selector]
    run_name = f"sel_{POOL_ID}_C_{a.selector}_{labels}_{a.loss}_s{a.seed}" + (f"_ep{EPOCHS}" if EPOCHS != 12 else "")
    params = dict(base, selector_family=a.selector, inputs=inputs, labels=labels, loss=a.loss, arch=ARCH,
                  privileged=int(privileged), deployable=int(not privileged), epochs=EPOCHS, batch=BATCH, lr=LR, wd=WD,
                  calib_split_frac=CAL_FRAC, calib_seed=0, train_split="navtrain", train_cache_sha=train_sha,
                  best_epoch=best["epoch"], device=dev, post_hoc=int(logit), logit_input=int(logit))
    out_dir = RUNS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "picks.npz", tokens=te["tokens"], pick_idx=pick, score=score, pdms_pick=pdms_pick)
    (out_dir / "params.json").write_text(json.dumps(params, indent=2, default=float))
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2, default=float))
    (out_dir / "tags.json").write_text(json.dumps(tags | {"privileged": str(int(privileged)), "deployable": str(int(not privileged))}, indent=2))
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    (out_dir / "config.yaml").write_text(json.dumps(params, indent=2, default=float) + "\n")  # JSON is valid YAML
    # paired bootstrap vs DD cls as its own run (selector_compare): mean(B - A), 95% CI, one-sided p for B > A
    cmp_dir = RUNS / f"cmp_{run_name}_vs_ddcls"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    (cmp_dir / "params.json").write_text(json.dumps(dict(a="dd_cls", b=run_name, split="navtest", n_scenes=n_te, pool_id=POOL_ID,
                                                        cache_sha=cache_sha, metric="pdms_v1_pick", n_boot=20000, seed=0), indent=2))
    (cmp_dir / "metrics.json").write_text(json.dumps(dict(delta_vs_dd_cls=mean, delta_ci_lo=lo, delta_ci_hi=hi, delta_p_le0=p_le0,
                                                         n_scenes=n_te), indent=2, default=float))
    (cmp_dir / "tags.json").write_text(json.dumps(tags, indent=2))
    print(json.dumps(m, indent=2, default=float))
    print("RUN_DIR", out_dir)


if __name__ == "__main__":
    main()
