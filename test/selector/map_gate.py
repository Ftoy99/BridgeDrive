"""Map gate `mapgate` (Selector_Experiment_Plan.md §5c): recompute drivable-area compliance (DAC) per cached DD candidate
from the on-disk NAVSIM map, then argmax DD logit among candidates that pass. Training-free, CPU.

DAC uses only (candidate, ego state at t0, HD map). Labels (cache `pdm`) are read only AFTER the gate is fixed, to score it.
Map: nuPlan maps in NUPLAN_MAPS_ROOT via get_maps_api; PDMDrivableMap.from_simulation at radius 100 m (metric_cache_processor).
Candidate states: same path as navsim.evaluate.pdm_score (transform_trajectory -> get_trajectory_as_array -> PDMSimulator),
DAC from PDMScorer's own drivable-area code (_calculate_ego_area / _calculate_drivable_area_compliance).
Initial ego state: metric cache `ego_state` (built from the navsim log at t0); map name: navsim_logs `map_location`.
Resumable per shard: shards with an existing dac_*.npz are skipped.
  /home/harris/miniconda3/envs/hidden/bin/python test/selector/map_gate.py --pool <pool/split> --out <dir> [--max-shards N]
"""
import argparse, json, os, pickle, lzma, subprocess, sys, time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True, help="dir of shard_*.npz (keys tokens, traj, logit, pdm, ok)")
ap.add_argument("--out", required=True)
ap.add_argument("--max-shards", type=int, default=0, help="smoke: stop after N shards (0 = all)")
ap.add_argument("--metric-cache", default="metric_cache", help="under NAVSIM_EXP_ROOT (navtrain: metric_cache_navtrain)")
a = ap.parse_args()

DEVKIT = os.environ.get("NAVSIM_DEVKIT_ROOT", "/mnt/jimmys/BridgeDrive/BridgeDrive_adaptation_DiffusionDrive")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NAVSIM_EXP_ROOT", "/mnt/ds/navsim-main/exp")
os.environ.setdefault("OPENSCENE_DATA_ROOT", "/mnt/ds/navsim-main/dataset")
os.environ.setdefault("NUPLAN_MAPS_ROOT", os.environ["OPENSCENE_DATA_ROOT"] + "/maps")
sys.path.insert(0, DEVKIT)

from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, MultiMetricIndex
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap

MAP_RADIUS = 100  # metric_cache_processor.py
NC, DAC, EP, TTC, COMF, DDC, PDMS = range(7)  # cache pdm column order
SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)  # default_scoring_parameters.yaml
sim = PDMSimulator(proposal_sampling=SAMPLING)
scorer = PDMScorer(proposal_sampling=SAMPLING)
exp = Path(os.environ["NAVSIM_EXP_ROOT"])
MAPS = {}  # map_location -> nuPlan map api


def map_locations():
    """frame token -> map_location, from navsim logs (scene token = first frame token)."""
    out = {}
    for f in (Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs").glob("*/*.pkl"):
        for fr in pickle.load(open(f, "rb")):
            out[fr["token"]] = fr["map_location"]
    return out


def mc_path_map(name):
    """scene token -> metric cache pickle path (same loader as run_bridge_dump)."""
    from navsim.common.dataloader import MetricCacheLoader
    return MetricCacheLoader(exp / name).metric_cache_paths


def dac_for_scene(mc_path, loc, cands):
    """cands (K,8,3) ego-frame. Returns (K,) bool DAC pass."""
    with lzma.open(mc_path, "rb") as f:
        mc = pickle.load(f)
    if loc not in MAPS:
        MAPS[loc] = get_maps_api(os.environ["NUPLAN_MAPS_ROOT"], "nuplan-maps-v1.0", loc)
    ego = mc.ego_state
    pmap: PDMDrivableMap = PDMDrivableMap.from_simulation(MAPS[loc], ego, MAP_RADIUS)
    states = np.stack([
        get_trajectory_as_array(transform_trajectory(Trajectory(c.astype(np.float32)), ego), SAMPLING, ego.time_point)
        for c in cands])
    sim_states = sim.simulate_proposals(states, ego)
    scorer._reset(sim_states, None, None, [], pmap)
    scorer._calculate_ego_area()
    scorer._calculate_drivable_area_compliance()
    return scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA] >= 1


if __name__ == "__main__":
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    locs = map_locations()
    mcp = mc_path_map(a.metric_cache)
    shards = sorted(Path(a.pool).glob("shard_*.npz"))
    if a.max_shards:
        shards = shards[:a.max_shards]
    n_err = 0
    for sp in shards:
        dst = out / f"dac_{sp.stem}.npz"
        if dst.exists():
            continue
        d = np.load(sp)
        toks, traj, ok = d["tokens"], d["traj"], d["ok"].astype(bool)
        K = traj.shape[1]
        dac = np.zeros((len(toks), K), dtype=bool)
        err = np.zeros(len(toks), dtype=bool)
        for i in np.where(ok)[0]:
            try:
                dac[i] = dac_for_scene(mcp[toks[i]], locs[toks[i]], traj[i])
            except Exception as e:  # counted, never silently passed
                err[i] = True
                print(f"[err] {toks[i]}: {type(e).__name__}: {e}", flush=True)
        np.savez(dst, tokens=toks, dac_map=dac, err=err, ok=ok)
        n_err += int(err.sum())
        print(f"{sp.name} done, scenes={int(ok.sum())} err={int(err.sum())} elapsed={time.time() - t0:.0f}s", flush=True)
    print(f"total err={n_err}")
