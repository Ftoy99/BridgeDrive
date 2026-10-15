"""Log one run_pdm_score_fast.py result CSV to MLflow (SQLite store, readable through the mlflow MCP).

Usage (needs mlflow, run it through uv so no env changes are needed):
  uv run --with 'mlflow>=3.5' python test/log_pdm_to_mlflow.py <csv | exp_dir> --run-name NAME --param hb_profile=levy ...
An exp_dir resolves to the newest *.csv below it. Tracking URI: $MLFLOW_TRACKING_URI or sqlite:////mnt/ds/mlflow/mlflow.db.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # newer mlflow refuses the file store otherwise; the MCP server reads the same store
import mlflow
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--experiment", default="navsim_pdm")
ap.add_argument("--run-name", default=None)
ap.add_argument("--param", action="append", default=[], help="key=value, repeatable")
ap.add_argument("--tag", action="append", default=[], help="key=value, repeatable")
a = ap.parse_args()

p = Path(a.path)
csv = max(p.rglob("*.csv"), key=lambda f: f.stat().st_mtime) if p.is_dir() else p
df = pd.read_csv(csv, index_col=0)
avg = df[df["token"] == "average"].iloc[0]
body = df[df["token"] != "average"]

metrics = {"n_scenes": float(len(body)), "n_valid": float(body["valid"].astype(bool).sum())}
for c in df.columns:
    if c in ("token", "valid"):
        continue
    try:
        metrics[c] = float(avg[c])
    except (TypeError, ValueError):
        pass
for k in (20, 80):  # forced-anchor catastrophes: scenes with PDMS < 0.5 at classifier rank k (paper Table 5/12)
    if f"rank_score_{k}" in body.columns:
        metrics[f"catas_k{k}"] = float((body[f"rank_score_{k}"] < 0.5).sum())
metrics["pdms"] = metrics.get("select_score", metrics.get("score", float("nan")))  # deployed score, one name for dashboards

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////mnt/ds/mlflow/mlflow.db"))
mlflow.set_experiment(a.experiment)
kv = lambda xs: dict(x.split("=", 1) for x in xs)
with mlflow.start_run(run_name=a.run_name or csv.parent.name) as run:
    mlflow.log_params({"csv": str(csv), **kv(a.param)})
    mlflow.set_tags(kv(a.tag))
    mlflow.log_metrics({k: v for k, v in metrics.items() if v == v})  # drop NaN
    mlflow.log_artifact(str(csv))
    print(f"logged {csv} -> run {run.info.run_id} (pdms={metrics['pdms']:.4f})")
