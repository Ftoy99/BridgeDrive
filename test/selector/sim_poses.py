"""§5i V3 input: scorer-simulated ego poses (LQR tracker, 0.1 s steps) for every cached candidate and the human trajectory.

Same path as navsim.evaluate.pdm_score and map_gate.py (transform_trajectory -> get_trajectory_as_array -> PDMSimulator), but
no map is loaded: the simulator needs only the initial ego state from the metric cache. Output converted back to the ego frame
at t0 so the BEV gate can use it. Row K of `sim` is the human trajectory (gt_future), rows 0..K-1 the candidates.
Output per shard: tokens, sim (N, K+1, 41, 3) float32 (x, y, heading; index 0 = ego at t0 = (0, 0, 0)).

  /home/harris/miniconda3/envs/hidden/bin/python test/selector/sim_poses.py --split navtest --out <dir> [--workers 6]
"""
import argparse, glob, lzma, os, pickle, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

DEVKIT = "/mnt/jimmys/public_repos/DiffusionDrive"  # same devkit as the DD pool dump
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NAVSIM_EXP_ROOT", "/mnt/ds/navsim-main/exp")
os.environ.setdefault("OPENSCENE_DATA_ROOT", "/mnt/ds/navsim-main/dataset")
os.environ.setdefault("NUPLAN_MAPS_ROOT", os.environ["OPENSCENE_DATA_ROOT"] + "/maps")
sys.path.insert(0, DEVKIT)

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)  # default_scoring_parameters.yaml
sim = PDMSimulator(proposal_sampling=SAMPLING)
EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
METRIC = {"navtest": "metric_cache", "navtrain": "metric_cache_navtrain"}
_paths = None


def _init(split):
    global _paths
    _paths = MetricCacheLoader(EXP / METRIC[split]).metric_cache_paths


def _to_ego(states, ego_pose):
    """global (K,T,3) -> ego frame at t0 (K,T,3)."""
    x0, y0, h0 = ego_pose
    dx, dy = states[..., 0] - x0, states[..., 1] - y0
    c, s = np.cos(h0), np.sin(h0)
    out = np.empty_like(states)
    out[..., 0] = c * dx + s * dy
    out[..., 1] = -s * dx + c * dy
    out[..., 2] = np.arctan2(np.sin(states[..., 2] - h0), np.cos(states[..., 2] - h0))
    return out


def sim_one(args):
    token, cands = args
    with lzma.open(_paths[token], "rb") as f:
        mc = pickle.load(f)
    ego = mc.ego_state
    pose = (ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading)
    states = np.stack([
        get_trajectory_as_array(transform_trajectory(Trajectory(c.astype(np.float32)), ego), SAMPLING, ego.time_point)
        for c in cands])
    sim_states = sim.simulate_proposals(states, ego)[:, :, :3]
    return token, _to_ego(sim_states.astype(np.float64), pose).astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(METRIC))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-shards", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(glob.glob(f"/mnt/ds/navsim-main/exp/selector/dd_k20_public_v2/{a.split}/shard_*.npz"))
    if a.max_shards:
        shards = shards[:a.max_shards]
    with Pool(a.workers, initializer=_init, initargs=(a.split,)) as pool:
        for sp in shards:
            dst = out / f"sim_{Path(sp).stem}.npz"
            if dst.exists():
                continue
            d = np.load(sp)
            ok = d["ok"].astype(bool)
            toks = d["tokens"][ok]
            cands = np.concatenate([d["traj"][ok], d["gt_future"][ok][:, None]], axis=1)  # (N, K+1, 8, 3), human last
            res = pool.map(sim_one, [(t, c) for t, c in zip(toks, cands)], chunksize=4)
            np.savez(dst, tokens=np.array([r[0] for r in res]), sim=np.stack([r[1] for r in res]))
            print(f"{Path(sp).name} scenes={len(res)}", flush=True)
    print("done", out, flush=True)
