"""GT BEV semantic raster per scene, from the devkit target builder (§5i measurement 3 and the map reference for measurement 2).

Same builder and config as the DD pool (`TransfuserTargetBuilder`, `TransfuserConfig`): class 1 road, 2 walkway, 3 centreline,
4 static, 5 vehicle, 6 pedestrian at the scene's t0 frame. Output (N,128,256) uint8, same grid as the cache `pred_bev`.
  /home/harris/miniconda3/envs/hidden/bin/python test/selector/gt_bev.py --split navtest --tokens <file|all> --out <dir> [--workers 20]
Resumable: one shard file per chunk of 256 tokens; existing shards are skipped.
"""
import argparse, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

DEVKIT = "/mnt/jimmys/public_repos/DiffusionDrive"  # the devkit the DD pool was dumped with (dump_dd_pool.sh); not NAVSIM_DEVKIT_ROOT
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NAVSIM_EXP_ROOT", "/mnt/ds/navsim-main/exp")
os.environ.setdefault("OPENSCENE_DATA_ROOT", "/mnt/ds/navsim-main/dataset")
os.environ.setdefault("NUPLAN_MAPS_ROOT", os.environ["OPENSCENE_DATA_ROOT"] + "/maps")
sys.path.insert(0, DEVKIT)

from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features import TransfuserTargetBuilder

LOGS = {"navtest": "test", "navtrain": "trainval"}
CHUNK = 256
_loader = None
_builder = None


def _init(split, logs=None):
    global _loader, _builder
    root = Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs" / LOGS[split]
    # log_names keeps each worker from loading all 1310 trainval logs (20 workers x full load exhausted 62 GB RAM)
    _loader = SceneLoader(sensor_blobs_path=None, data_path=root, scene_filter=SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True, log_names=logs),
                          sensor_config=SensorConfig.build_no_sensors())
    _builder = TransfuserTargetBuilder(TransfuserConfig())


def _one(token):
    scene = _loader.get_scene_from_token(token)
    return _builder.compute_targets(scene)["bev_semantic_map"].numpy().astype(np.uint8)


def run_grouped(split, tokens, logs, out, group=50):
    """Single process, one SceneLoader per group of logs (a pool of loaders exhausted 62 GB RAM on navtrain)."""
    import gc
    global _loader
    want = set(tokens)
    for gi in range(0, len(logs), group):
        dst = out / f"gtbev_grp_{gi // group:04d}.npz"
        if dst.exists():
            continue
        _init(split, logs[gi:gi + group])
        toks = sorted(want & set(_loader.tokens))
        bev = np.stack([_one(t) for t in toks]) if toks else np.zeros((0, 128, 256), np.uint8)
        np.savez(dst, tokens=np.array(toks), gt_bev=bev)
        print(f"group {gi // group + 1}/{(len(logs) + group - 1) // group} scenes={len(toks)}", flush=True)
        _loader = None
        gc.collect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(LOGS))
    ap.add_argument("--tokens", required=True, help="file with one token per line, or 'all'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--logs", default=None, help="file with one log name per line (restricts the loader)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.tokens == "all":
        _init(a.split)
        tokens = sorted(_loader.tokens)
    else:
        tokens = [t.strip() for t in open(a.tokens) if t.strip()]
    chunks = [tokens[i:i + CHUNK] for i in range(0, len(tokens), CHUNK)]
    if a.logs:  # navtrain: grouped, single process (see run_grouped)
        logs = [l.strip() for l in open(a.logs) if l.strip()]
        run_grouped(a.split, tokens, logs, out)
        print(f"total scenes={len(tokens)} out={out}")
        sys.exit(0)
    with Pool(a.workers, initializer=_init, initargs=(a.split, logs)) as pool:
        for ci, chunk in enumerate(chunks):
            dst = out / f"gtbev_{ci:04d}.npz"
            if dst.exists():
                continue
            bev = np.stack(pool.map(_one, chunk, chunksize=8))
            np.savez(dst, tokens=np.array(chunk), gt_bev=bev)
            print(f"chunk {ci + 1}/{len(chunks)} done", flush=True)
    print(f"total scenes={len(tokens)} out={out}")
