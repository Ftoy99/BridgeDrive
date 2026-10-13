"""Overlay the saved BridgeDrive candidates (test/viz_out/*.npz from viz_bridgedrive_denoise.py) on the
NAVSIM BEV (nuPlan map + annotated agents + ego box). Per scene: anchors | all denoised candidates coloured
by PDMS | select/best/worst."""
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.visualization.bev import add_configured_bev_on_ax
from navsim.visualization.plots import configure_ax

DEVKIT = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
OUT = Path(os.environ.get("VIZ_OUT", str(Path(__file__).parent / "viz_out")))
XLIM, YLIM = (-12, 58), (-26, 26)  # x forward (up), y left (plot axis inverted)


def bev_ax(ax, scene, frame_idx):
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    ax.set_aspect("equal")
    ax.set_xlim(*YLIM); ax.set_ylim(*XLIM)
    ax.invert_xaxis()  # left is y positive
    configure_ax(ax)


def lines(ax, trajs, colors, lw, alpha, z):
    for tr, c in zip(trajs, colors):
        pts = np.concatenate([[[0.0, 0.0]], tr[:, :2]])  # start at ego rear axle like navsim's own plot
        ax.plot(pts[:, 1], pts[:, 0], color=c, lw=lw, alpha=alpha, zorder=z)


def main():
    cfg_dir = str(DEVKIT / "navsim" / "planning" / "script" / "config" / "pdm_scoring")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="default_run_pdm_score", overrides=[
            "train_test_split=navtest", "agent=bridgedrive_agent_k80_beta10", "experiment_name=viz",
            "train_test_split.scene_filter.max_scenes=600"])
    loader = SceneLoader(sensor_blobs_path=None, data_path=Path(cfg.navsim_log_path),
                         scene_filter=instantiate(cfg.train_test_split.scene_filter),
                         sensor_config=SensorConfig.build_no_sensors())
    cmap = plt.get_cmap("RdYlGn")
    for npz in sorted(OUT.glob("[0-9][0-9]_*.npz")):
        token = npz.stem.split("_", 1)[1]
        d = np.load(npz)
        states, final, sub, gt = d["states"], d["final"], d["sub"], d["gt"]
        sc = sub[:, -1]
        sel, best, worst = int(d["logits"].argmax()), int(sc.argmax()), int(sc.argmin())
        scene = loader.get_scene_from_token(token)
        fi = scene.scene_metadata.num_history_frames - 1
        order = np.argsort(sc)

        fig, axs = plt.subplots(1, 3, figsize=(21, 10))
        ax = axs[0]; bev_ax(ax, scene, fi)
        lines(ax, states[0], ["tab:gray"] * len(final), 0.9, 0.7, 4)
        lines(ax, states[0][[sel]], ["tab:blue"], 2.8, 1, 6)
        lines(ax, [gt], ["k"], 2.5, 1, 7)
        ax.set_title(f"BEFORE denoising: {len(final)} anchors (blue = classifier pick, black = GT)")

        ax = axs[1]; bev_ax(ax, scene, fi)
        lines(ax, final[order], [cmap(float(np.clip(sc[k], 0, 1))) for k in order], 1.2, 0.9, 4)
        lines(ax, [gt], ["k"], 2.5, 1, 7)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)); sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, label="candidate PDMS")
        ax.set_title("AFTER denoising: all candidates coloured by PDMS")

        ax = axs[2]; bev_ax(ax, scene, fi)
        for k, c, lab in [(worst, "red", "worst"), (best, "limegreen", "best"), (sel, "blue", "select")]:
            lines(ax, [final[k]], [c], 3.2, 1, 6)
            ax.plot([], [], color=c, lw=3, label=f"{lab} PDMS {sc[k]:.2f} (mode {k})")
        lines(ax, [gt], ["k"], 2.5, 1, 7); ax.plot([], [], "k", lw=2.5, label="GT")
        ax.legend(loc="lower left", fontsize=9)
        ax.set_title("select / best / worst")

        fig.suptitle(f"token {token}", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT / f"{npz.stem}_bev.png", dpi=100)
        plt.close(fig)
        print("saved", OUT / f"{npz.stem}_bev.png", flush=True)

        # ---- noise-level figure: anchors | noise injected (after step 1) | INPUT to last step | clean output ----
        n = len(states) - 1  # 20
        clean = final[..., :2]
        dev = lambda s: np.linalg.norm(s - clean, axis=-1).mean()  # mean per-waypoint distance to the clean output, m
        panels = [
            (states[0], f"t=1000 anchors (start)\nmean dist to clean output {dev(states[0]):.2f} m"),
            (states[1], f"after step 1: noise injected\nmean dist to clean output {dev(states[1]):.2f} m"),
            (states[n - 1], f"input to LAST step (t=50, still noisy)\nmean dist to clean output {dev(states[n - 1]):.2f} m"),
            (clean, "after LAST step: clean output (t=0)\nmean dist to clean output 0.00 m"),
        ]
        fig, axs = plt.subplots(1, 4, figsize=(28, 9))
        for ax, (st, title) in zip(axs, panels):
            bev_ax(ax, scene, fi)
            lines(ax, st[order], [cmap(float(np.clip(sc[k], 0, 1))) for k in order], 1.2, 0.9, 4)
            lines(ax, [gt], ["k"], 2.5, 1, 7)
            ax.set_title(title, fontsize=11)
        fig.colorbar(sm, ax=axs, fraction=0.012, pad=0.01, label="candidate PDMS (of the final clean trajectory)")
        fig.suptitle(f"token {token} | black = GT | all 80 candidates, coloured by final PDMS | per-step mean dist to clean: "
                     + " ".join(f"{dev(states[i]):.1f}" for i in range(n + 1)) + " m", fontsize=11)
        fig.savefig(OUT / f"{npz.stem}_bev_noise.png", dpi=90, bbox_inches="tight")
        plt.close(fig)
        print("saved", OUT / f"{npz.stem}_bev_noise.png", flush=True)

        # ---- BEV step grid: anchors, then after step 1 / 5 / 10 / 15 / 20 ----
        fig, axs = plt.subplots(1, 6, figsize=(36, 9))
        for ax, st in zip(axs, [0, 1, 5, 10, 15, 20]):
            bev_ax(ax, scene, fi)
            lines(ax, (states[st] if st < 20 else clean)[order], [cmap(float(np.clip(sc[k], 0, 1))) for k in order], 1.2, 0.9, 4)
            lines(ax, [gt], ["k"], 2.5, 1, 7)
            ax.set_title("step 0 (anchors)" if st == 0 else f"after step {st}/20" + (" (clean)" if st == 20 else ""), fontsize=12)
        fig.suptitle(f"token {token} | all 80 candidates coloured by final PDMS | black = GT", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / f"{npz.stem}_bev_steps.png", dpi=70)
        plt.close(fig)
        print("saved", OUT / f"{npz.stem}_bev_steps.png", flush=True)

        # ---- last-step before/after: same trajectories, input (t=50) vs output (t=0) ----
        before, after = states[n - 1], clean  # (K,P,2)
        step_move = np.linalg.norm(after - before, axis=-1)  # (K,P) metres moved by the last step
        fig, axs = plt.subplots(1, 3, figsize=(26, 10))

        ax = axs[0]; bev_ax(ax, scene, fi)  # all 80: before faint dashed, after solid
        for k in order:
            c = cmap(float(np.clip(sc[k], 0, 1)))
            for tr, ls, a, w in [(before[k], "--", 0.45, 0.9), (after[k], "-", 0.95, 1.3)]:
                pts = np.concatenate([[[0.0, 0.0]], tr]); ax.plot(pts[:, 1], pts[:, 0], color=c, ls=ls, alpha=a, lw=w, zorder=4)
        lines(ax, [gt], ["k"], 2.5, 1, 7)
        ax.set_title(f"all 80: BEFORE last step (dashed) -> AFTER (solid)\nmean move {step_move.mean():.2f} m, max {step_move.max():.2f} m")

        def zoomed(ax, k_list, title):
            ks = np.array(k_list)
            pts = np.concatenate([before[ks].reshape(-1, 2), after[ks].reshape(-1, 2), [[0, 0]]])
            xmin, xmax = pts[:, 0].min() - 3, pts[:, 0].max() + 3
            ymid, yhalf = (pts[:, 1].min() + pts[:, 1].max()) / 2, max((pts[:, 1].max() - pts[:, 1].min()) / 2 + 3, (xmax - xmin) / 2 * 0.6)
            add_configured_bev_on_ax(ax, scene.map_api, scene.frames[fi])
            ax.set_aspect("equal"); ax.set_xlim(ymid - yhalf, ymid + yhalf); ax.set_ylim(xmin, xmax); ax.invert_xaxis(); configure_ax(ax)
            for k, c, lab in [(worst, "red", "worst"), (best, "limegreen", "best"), (sel, "blue", "select")]:
                if k not in ks: continue
                b = np.concatenate([[[0.0, 0.0]], before[k]]); a = np.concatenate([[[0.0, 0.0]], after[k]])
                for i in range(1, len(b)):  # connector per waypoint = what the last step moved
                    ax.plot([b[i, 1], a[i, 1]], [b[i, 0], a[i, 0]], color=c, lw=1, alpha=0.8, zorder=5)
                ax.plot(b[:, 1], b[:, 0], color=c, ls="--", lw=1.8, marker="o", ms=4, mfc="white", zorder=6, label=f"{lab} before (t=50)")
                ax.plot(a[:, 1], a[:, 0], color=c, ls="-", lw=2.6, marker="o", ms=5, zorder=7, label=f"{lab} after (clean), moved {step_move[k].mean():.2f} m avg")
            lines(ax, [gt], ["k"], 2.0, 1, 8); ax.plot([], [], "k", lw=2, label="GT")
            ax.legend(fontsize=8, loc="lower left"); ax.set_title(title)

        zoomed(axs[1], [sel, best, worst], "select / best / worst, zoomed\ndashed+hollow = before, solid = after")
        zoomed(axs[2], [sel], f"select only, zoomed (mode {sel})\nper-waypoint move (m): "
               + " ".join(f"{v:.2f}" for v in step_move[sel]))
        fig.suptitle(f"token {token} | last denoising step (t=50 -> 0)", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT / f"{npz.stem}_bev_laststep.png", dpi=90)
        plt.close(fig)
        print("saved", OUT / f"{npz.stem}_bev_laststep.png", flush=True)


if __name__ == "__main__":
    main()
