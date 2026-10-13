"""
BridgeDrive denoising visualization + timing + pool PDMS (select / best / worst).

For N navtest scenes:
  * runs the real agent forward (batch 1) and captures the trajectory-head inputs via a pre-hook,
  * replays DDBMTrajectoryHead.forward_test step by step (same math) with CUDA-synced timing per
    denoising step, records all K candidates at every step (step 0 = anchors, step 20 = final),
  * checks the replica reproduces the real forward output (same seed),
  * scores every final candidate with the official pdm_score() (select / best / worst),
  * saves per scene: PNG (anchors before | final after coloured by PDMS | intermediate steps |
    per-step timing), NPZ with all trajectories + per-candidate sub-metrics.

Run via test/run_viz_bridgedrive.sh (sets env vars). Config is composed from the same hydra
config run_pdm_score_fast.py uses.
"""
import json
import lzma
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.agents.diffusiondrive.model_diffusion_head_ddbm_v5 import gen_sineembed_for_position

DEVKIT = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
CKPT = os.environ["CKPT"]
BKB = os.environ["BKB_PATH"]
ANCHOR = os.environ["ANCHOR_PATH"]
N_SCENES = int(os.environ.get("N_SCENES", 4))
OUT = Path(os.environ.get("VIZ_OUT", str(Path(__file__).parent / "viz_out")))
METRICS = [
    "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
    "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score",
]
SHOW_STEPS = [0, 1, 5, 10, 15, 20]  # states shown in the grid (0 = anchors, 20 = final)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def replay_forward_test(head, ego_query, agents_query, bev_feature, bev_spatial_shape, global_img):
    """Same math as TrajectoryHead.forward_test, plus per-step states and timings (ms)."""
    device = ego_query.device
    bs = ego_query.shape[0]
    step_num = 20
    ts = torch.arange(0, 1001, 1000 // step_num).to(device)
    plan_anchor = head.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
    timings = {}

    sync(); t0 = time.perf_counter()
    K = plan_anchor.shape[1]
    emb = gen_sineembed_for_position(plan_anchor, hidden_dim=64).flatten(-2)
    feat = head.plan_anchor_encoder(emb).view(bs, K, -1)
    cls_list = head.diff_decoder_cls(feat, plan_anchor, bev_feature, bev_spatial_shape, agents_query, ego_query, global_img)
    mode_idx = cls_list[-1].argmax(dim=-1)
    sync(); timings["classifier"] = (time.perf_counter() - t0) * 1e3

    states = [plan_anchor[0].detach().cpu().numpy()]  # (K,P,2)
    x0_preds = []
    step_ms = []
    xt_prev = plan_anchor
    for i in range(step_num, 0, -1):
        sync(); t0 = time.perf_counter()
        xt = xt_prev
        t = ts[i] * torch.ones([bs], device=device)
        t_prev = ts[i - 1] * torch.ones([bs], device=device)
        emb = gen_sineembed_for_position(xt, hidden_dim=64).flatten(-2)
        feat = head.plan_anchor_encoder(emb).view(bs, xt.shape[1], -1)
        time_embed = head.time_mlp(t).view(bs, 1, -1)
        reg_list, _ = head.diff_decoder(feat, xt, plan_anchor, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, global_img)
        x0_mean = reg_list[-1][..., :2]
        head0_mean = reg_list[-1][..., [-1]]
        xt_prev_norm = head.diffusion_scheduler.sample_step(
            t=t, t_prev=t_prev, xt=head.norm_odo(xt.clone()), x0=head.norm_odo(x0_mean), xT=head.norm_odo(plan_anchor.clone()),
        )
        xt_prev = head.denorm_odo(xt_prev_norm)
        sync(); step_ms.append((time.perf_counter() - t0) * 1e3)
        states.append(xt_prev[0].detach().cpu().numpy())
        x0_preds.append(x0_mean[0].detach().cpu().numpy())
    timings["steps_ms"] = step_ms
    final = torch.cat([xt_prev, head0_mean], dim=-1)  # (bs,K,P,3)
    traj = torch.gather(final, 1, mode_idx[..., None, None, None].repeat(1, 1, final.shape[2], 3)).squeeze(1)
    return {
        "traj": traj, "final": final[0].detach().cpu().numpy(), "logits": cls_list[-1][0].detach().cpu().numpy(),
        "states": np.stack(states), "x0_preds": np.stack(x0_preds), "timings": timings,
        "mode_idx": int(mode_idx[0]),
    }


def draw(ax, trajs, colors=None, lw=0.8, alpha=0.8, **kw):
    # ego frame: x forward, y left -> plot (-y, x) so forward is up, left is left
    for k, tr in enumerate(trajs):
        ax.plot(-tr[:, 1], tr[:, 0], color=None if colors is None else colors[k], lw=lw, alpha=alpha, **kw)


def style(ax, title, xlim, ylim):
    ax.set_title(title, fontsize=9)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal"); ax.grid(alpha=0.2)
    ax.plot(0, 0, "k^", ms=6)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_dir = str(DEVKIT / "navsim" / "planning" / "script" / "config" / "pdm_scoring")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(
            config_name="default_run_pdm_score",
            overrides=[
                "train_test_split=navtest", "agent=bridgedrive_agent_k80_beta10",
                f"agent.checkpoint_path={CKPT}", f"+agent.config.bkb_path={BKB}",
                f"agent.config.plan_anchor_path={ANCHOR}", "experiment_name=viz",
                "train_test_split.scene_filter.max_scenes=600",
            ],
        )
    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    agent = instantiate(cfg.agent)
    agent.initialize()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent.to(device).eval()
    head = agent._transfuser_model._trajectory_head
    head.eval()

    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path), data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter), sensor_config=agent.get_sensor_config(),
    )
    cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    tokens = sorted(set(scene_loader.tokens) & set(cache_loader.tokens))
    pick = np.linspace(0, len(tokens) - 1, N_SCENES).astype(int)
    tokens = [tokens[i] for i in pick]
    print("scenes:", tokens, flush=True)

    captured = {}
    head.register_forward_pre_hook(lambda m, a, kw: captured.update(args=a, kwargs=kw), with_kwargs=True)

    summary = []
    warmed = False
    for si, token in enumerate(tokens):
        agent_input = scene_loader.get_agent_input_from_token(token)
        scene = scene_loader.get_scene_from_token(token)
        gt = scene.get_future_trajectory(num_trajectory_frames=8).poses
        feats = {}
        for b in agent.get_feature_builders():
            feats.update(b.compute_features(agent_input))
        feats = {k: v.unsqueeze(0).to(device) for k, v in feats.items()}

        with torch.no_grad():
            if not warmed:  # cuda warmup so timings aren't dominated by first-call overhead
                agent.forward(feats); agent.forward(feats); warmed = True
            torch.manual_seed(0)
            sync(); t0 = time.perf_counter()
            real = agent.forward(feats)["trajectory"][0].float().cpu().numpy()
            sync(); full_ms = (time.perf_counter() - t0) * 1e3
            a = captured["args"]
            torch.manual_seed(0)
            rep = replay_forward_test(head, a[0], a[1], a[2], a[3], captured["kwargs"].get("global_img"))
        mismatch = float(np.abs(rep["traj"][0].float().cpu().numpy() - real).max())
        timings = rep["timings"]
        print(f"[{si}] {token} full-forward {full_ms:.1f} ms | classifier {timings['classifier']:.1f} ms | "
              f"20 steps {sum(timings['steps_ms']):.1f} ms (mean {np.mean(timings['steps_ms']):.1f}) | replica-vs-real max diff {mismatch:.2e}", flush=True)

        with lzma.open(cache_loader.metric_cache_paths[token], "rb") as f:
            metric_cache = pickle.load(f)
        final = rep["final"]  # (K,P,3)
        sub = np.zeros((len(final), len(METRICS)))
        for k in range(len(final)):
            r = pdm_score(metric_cache=metric_cache, model_trajectory=Trajectory(final[k].astype(np.float32)),
                          future_sampling=simulator.proposal_sampling, simulator=simulator, scorer=scorer)
            sub[k] = [getattr(r, m) for m in METRICS]
        sel, best, worst = rep["mode_idx"], int(sub[:, -1].argmax()), int(sub[:, -1].argmin())
        print(f"    PDMS select {sub[sel, -1]:.3f} (mode {sel}) | best {sub[best, -1]:.3f} (mode {best}) | "
              f"worst {sub[worst, -1]:.3f} (mode {worst}) | pool mean {sub[:, -1].mean():.3f}", flush=True)

        np.savez(OUT / f"{si:02d}_{token}.npz", states=rep["states"], final=final, logits=rep["logits"], sub=sub,
                 metrics=np.array(METRICS), gt=gt, step_ms=np.array(timings["steps_ms"]))

        # ---------------- figure ----------------
        states = rep["states"]
        allx = states[..., 0]; ally = states[..., 1]
        xlim = (-max(8, np.abs(ally).max() * 1.05), max(8, np.abs(ally).max() * 1.05))
        ylim = (min(-2, allx.min() - 1), allx.max() + 2)
        cmap = plt.get_cmap("RdYlGn")
        cols = [cmap(float(np.clip(s, 0, 1))) for s in sub[:, -1]]
        fig = plt.figure(figsize=(22, 13))
        gs = fig.add_gridspec(3, 6, height_ratios=[1.6, 1.0, 0.8])

        ax = fig.add_subplot(gs[0, 0:2])
        draw(ax, states[0], colors=["tab:gray"] * len(final), lw=0.7, alpha=0.6)
        draw(ax, states[0][[sel]], colors=["tab:blue"], lw=2.5, alpha=1)
        ax.plot(-gt[:, 1], gt[:, 0], "k--", lw=1.5, label="GT")
        style(ax, f"BEFORE denoising: all {len(final)} anchors (blue = classifier pick)", xlim, ylim); ax.legend(fontsize=7)

        ax = fig.add_subplot(gs[0, 2:4])
        order = np.argsort(sub[:, -1])  # draw bad first, good on top
        draw(ax, final[order], colors=[cols[k] for k in order], lw=0.9, alpha=0.8)
        ax.plot(-gt[:, 1], gt[:, 0], "k--", lw=1.5, label="GT")
        for k, c, lab in [(worst, "red", f"worst {sub[worst, -1]:.2f}"), (best, "limegreen", f"best {sub[best, -1]:.2f}"),
                          (sel, "blue", f"select {sub[sel, -1]:.2f}")]:
            ax.plot(-final[k][:, 1], final[k][:, 0], color=c, lw=2.8, label=lab)
        style(ax, "AFTER denoising: all candidates coloured by PDMS", xlim, ylim); ax.legend(fontsize=7)

        ax = fig.add_subplot(gs[0, 4:6])
        for k, c, lab in [(worst, "red", "worst"), (best, "limegreen", "best"), (sel, "blue", "select")]:
            ax.plot(-final[k][:, 1], final[k][:, 0], color=c, lw=2.5, label=f"{lab} (mode {k})")
            ax.plot(-states[0][k][:, 1], states[0][k][:, 0], color=c, lw=1.2, ls=":", alpha=0.9)
        ax.plot(-gt[:, 1], gt[:, 0], "k--", lw=1.5)
        style(ax, "select / best / worst: anchor (dotted) -> denoised (solid)", xlim, ylim); ax.legend(fontsize=7)

        for j, s in enumerate(SHOW_STEPS):
            ax = fig.add_subplot(gs[1, j])
            draw(ax, states[s], colors=cols, lw=0.7, alpha=0.7)
            ax.plot(-gt[:, 1], gt[:, 0], "k--", lw=1)
            style(ax, "step 0 (anchors)" if s == 0 else f"after step {s}/20", xlim, ylim)

        ax = fig.add_subplot(gs[2, 0:3])
        ax.bar(range(1, 21), timings["steps_ms"], color="tab:blue")
        ax.set_title(f"per-step time (ms) | classifier {timings['classifier']:.1f} ms | 20 steps {sum(timings['steps_ms']):.0f} ms | "
                     f"full forward (backbone+head) {full_ms:.0f} ms", fontsize=9)
        ax.set_xlabel("denoising step")

        ax = fig.add_subplot(gs[2, 3:6])
        ax.axis("off")
        lines = [f"{'':8s}" + "".join(f"{m[:9]:>11s}" for m in METRICS)]
        for tag, k in [("select", sel), ("best", best), ("worst", worst)]:
            lines.append(f"{tag:8s}" + "".join(f"{v:11.3f}" for v in sub[k]))
        lines.append(f"pool mean PDMS {sub[:, -1].mean():.3f} | token {token[:12]}")
        ax.text(0, 0.8, "\n".join(lines), family="monospace", fontsize=8, va="top")

        fig.suptitle(f"scene {si} | token {token}", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / f"{si:02d}_{token}.png", dpi=110)
        plt.close(fig)

        summary.append({
            "token": token, "full_forward_ms": full_ms, "classifier_ms": timings["classifier"],
            "steps_ms": timings["steps_ms"], "replica_max_diff": mismatch,
            "select": sub[sel].tolist(), "best": sub[best].tolist(), "worst": sub[worst].tolist(),
            "pool_mean_pdms": float(sub[:, -1].mean()),
        })

    with open(OUT / "summary.json", "w") as f:
        json.dump({"metrics": METRICS, "scenes": summary}, f, indent=1)
    m = lambda key: np.mean([s[key] for s in summary], axis=0)
    print("\n=== averages over", len(summary), "scenes ===")
    print("full forward ms %.1f | classifier %.1f | per denoise step %.1f (x20 = %.1f)" % (
        m("full_forward_ms"), m("classifier_ms"), np.mean([np.mean(s["steps_ms"]) for s in summary]),
        np.mean([sum(s["steps_ms"]) for s in summary])))
    for tag in ["select", "best", "worst"]:
        print(f"{tag:7s} PDMS {m(tag)[-1]:.4f} | " + " ".join(f"{n[:6]}={v:.3f}" for n, v in zip(METRICS[:-1], m(tag)[:-1])))
    print("saved to", OUT)


if __name__ == "__main__":
    main()
