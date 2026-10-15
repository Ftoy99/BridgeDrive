"""Register a dump as a frozen selector pool: symlink shards into exp/selector/<pool_id>/<split>/, write pool_manifest.json
(ckpt, sha256 of every shard, cache_sha), summary metrics, and a params/metrics pair for log_selector_run.py.
  python test/selector/make_manifest.py SRC_DIR POOL_ID --split navtest --model BridgeDrive --ckpt /path.ckpt --notes "..."
Then: uv run ... python test/selector/log_selector_run.py exp/selector/<pool_id>/<split> --experiment selector_cache
"""
import argparse, glob, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_bridge_selectors import load

ROOT = Path(os.environ.get("NAVSIM_EXP_ROOT", "/mnt/ds/navsim-main/exp")) / "selector"

def sha(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("pool_id")
ap.add_argument("--split", default="navtest"); ap.add_argument("--model", required=True)
ap.add_argument("--ckpt", default=""); ap.add_argument("--sampler", default="")
ap.add_argument("--dump-script", default="test/run_bridge_dump.py"); ap.add_argument("--notes", default="")
a = ap.parse_args()

src = Path(a.src).resolve()
dst = ROOT / a.pool_id / a.split
dst.mkdir(parents=True, exist_ok=True)
files = sorted(glob.glob(str(src / "shard_*.npz"))) + [p for p in (src / "anchors.npy", src / "recon_ts.npy", src / "columns.json") if p.exists()]
hashes = {}
for f in files:
    f = Path(f); link = dst / f.name
    if not link.exists():
        link.symlink_to(f)
    hashes[f.name] = sha(f)
cache_sha = hashlib.sha256("".join(f"{k}:{v}" for k, v in sorted(hashes.items())).encode()).hexdigest()

d = load(str(dst))
cols = json.loads((src / "columns.json").read_text())["columns"] if (src / "columns.json").exists() else None
sc_idx = cols.index("pdm_score") if cols else -1  # vocabulary label pools (GTRS) carry named columns and no model logits
score = d["pdm"][..., sc_idx].astype(np.float32)
N, K = score.shape
sel = score[np.arange(N), d["logit1"].argmax(1)].mean() if "logit1" in d else float("nan")
metrics = {"n_scenes": N, "K": K, "pdms_select": float(sel), "pdms_oracle": float(score.max(1).mean()),
           "pool_mean": float(score.mean()), "headroom": float(score.max(1).mean() - sel)}
metrics = {k: v for k, v in metrics.items() if v == v}
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parents[2]).decode().strip()
except Exception:
    commit = "unknown"
manifest = {"pool_id": a.pool_id, "model": a.model, "ckpt_path": a.ckpt, "ckpt_sha256": sha(a.ckpt) if a.ckpt and Path(a.ckpt).exists() else "",
            "sampler": a.sampler, "dump_script": a.dump_script, "git_commit": commit, "split": a.split, "src_dir": str(src),
            "n_scenes": N, "K": K, "keys": list(d.keys()), "shard_sha256": hashes, "cache_sha": cache_sha,
            "date": time.strftime("%Y-%m-%d %H:%M"), "notes": a.notes}
(dst / "pool_manifest.json").write_text(json.dumps(manifest, indent=1))
(dst / "params.json").write_text(json.dumps({k: v for k, v in manifest.items() if k not in ("shard_sha256", "keys")} | {"keys": ",".join(d.keys())}))
(dst / "metrics.json").write_text(json.dumps(metrics))
(dst / "tags.json").write_text(json.dumps({"kind": "cache", "deployable": "1"}))
os.remove(dst / "params.json") if False else None
print(json.dumps({**metrics, "cache_sha": cache_sha[:12], "dir": str(dst)}, indent=1))
