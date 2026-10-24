"""Label-free observable terms for the soft tilt (Selector_Experiment_Plan.md §5e).
Per candidate: DAC (map), EP_proxy (scorer progress normalisation on [PDM-closed reference, candidate] pairs, DAC in place
of the full product, NC excluded), Comf (scorer comfort on simulated states). No cache labels are read.
Reuses simulator, scorer and map loaders from map_gate.py (same sanity path as the DAC gate).
  /home/harris/miniconda3/envs/hidden/bin/python test/selector/observable_terms.py --pool <pool/split> --out <dir> --metric-cache metric_cache
Resumable per shard.
"""
import argparse, lzma, pickle, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import map_gate as mg  # sim, scorer, SAMPLING, MAPS, map_locations, mc_path_map, PDMDrivableMap, get_trajectory_as_array, transform_trajectory, Trajectory
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import MultiMetricIndex, WeightedMetricIndex

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--metric-cache", default="metric_cache")
ap.add_argument("--max-shards", type=int, default=0)
a = ap.parse_args()

THRESH = mg.scorer._config.progress_distance_threshold  # 5 m, as scorer


def terms_for_scene(mc_path, loc, cands):
    """cands (K,8,3) ego frame. Returns dict of (K,) arrays: dac, ep, comf."""
    with lzma.open(mc_path, "rb") as f:
        mc = pickle.load(f)
    if loc not in mg.MAPS:
        mg.MAPS[loc] = mg.get_maps_api(mg.os.environ["NUPLAN_MAPS_ROOT"], "nuplan-maps-v1.0", loc)
    ego = mc.ego_state
    pmap = mg.PDMDrivableMap.from_simulation(mg.MAPS[loc], ego, mg.MAP_RADIUS)
    ref = mg.get_trajectory_as_array(mc.trajectory, mg.SAMPLING, ego.time_point)  # PDM-closed reference, already global
    cand = np.stack([mg.get_trajectory_as_array(mg.transform_trajectory(mg.Trajectory(c.astype(np.float32)), ego),
                                                mg.SAMPLING, ego.time_point) for c in cands])
    sim_states = mg.sim.simulate_proposals(np.concatenate([ref[None], cand], axis=0), ego)
    K = len(cands)
    dac = np.zeros(K); ep = np.zeros(K); comf = np.zeros(K)
    for k in range(K):
        pair = sim_states[[0, k + 1]]  # [reference, candidate], as pdm_score pairs them
        s = mg.scorer
        s._reset(pair, None, mc.centerline, [], pmap)
        s._calculate_ego_area()
        s._calculate_drivable_area_compliance()
        s._calculate_progress()
        s._calculate_is_comfortable()
        mult = s._multi_metrics[MultiMetricIndex.DRIVABLE_AREA]          # DAC only (NC excluded, see §5e)
        raw = s._progress_raw * mult
        mx = raw.max()
        if mx > THRESH:
            e = raw / mx
        else:
            e = np.ones(2); e[mult == 0] = 0.0
        dac[k] = mult[1] >= 1; ep[k] = e[1]
        comf[k] = s._weighted_metrics[WeightedMetricIndex.COMFORTABLE][1]
    return {"dac": dac, "ep": ep, "comf": comf}


if __name__ == "__main__":
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    locs = mg.map_locations()
    mcp = mg.mc_path_map(a.metric_cache)
    shards = sorted(Path(a.pool).glob("shard_*.npz"))
    if a.max_shards:
        shards = shards[:a.max_shards]
    n_err = 0
    for sp in shards:
        dst = out / f"terms_{sp.stem}.npz"
        if dst.exists():
            continue
        d = np.load(sp)
        toks, traj, ok = d["tokens"], d["traj"], d["ok"].astype(bool)
        K = traj.shape[1]
        res = {k: np.zeros((len(toks), K)) for k in ("dac", "ep", "comf")}
        err = np.zeros(len(toks), dtype=bool)
        for i in np.where(ok)[0]:
            try:
                r = terms_for_scene(mcp[toks[i]], locs[toks[i]], traj[i])
                for k in res:
                    res[k][i] = r[k]
            except Exception as e:  # counted, never silently passed
                err[i] = True
                print(f"[err] {toks[i]}: {type(e).__name__}: {e}", flush=True)
        np.savez(dst, tokens=toks, dac=res["dac"], ep=res["ep"], comf=res["comf"], err=err, ok=ok)
        n_err += int(err.sum())
        print(f"{sp.name} done, scenes={int(ok.sum())} err={int(err.sum())} elapsed={time.time() - t0:.0f}s", flush=True)
    print(f"total err={n_err}")
