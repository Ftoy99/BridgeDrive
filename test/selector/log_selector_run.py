"""Log one selector run dir to MLflow (SQLite store, readable through the mlflow MCP). Spec: second_brain/PhD/Selector_Runbook.md §3.

A run dir holds params.json (flat), metrics.json (flat, numeric), optional tags.json; every file in it is attached as artifact.
  uv run --no-project --python 3.11 --with 'mlflow>=3.5' python test/selector/log_selector_run.py RUN_DIR --experiment selector_audit
"""
import argparse, json, os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--experiment", required=True)
ap.add_argument("--run-name", default=None)
a = ap.parse_args()
d = Path(a.run_dir)
params = json.loads((d / "params.json").read_text())
metrics = json.loads((d / "metrics.json").read_text())
tags = json.loads((d / "tags.json").read_text()) if (d / "tags.json").exists() else {}

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////mnt/ds/mlflow/mlflow.db"))
mlflow.set_experiment(a.experiment)
with mlflow.start_run(run_name=a.run_name or d.name) as run:
    mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
    mlflow.set_tags({k: str(v) for k, v in tags.items()})
    mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v == v})
    for f in sorted(d.iterdir()):
        if f.is_file() and f.stat().st_size < 200e6:
            mlflow.log_artifact(str(f))
    (d / "mlflow_run_id.txt").write_text(run.info.run_id + "\n")
    print(f"logged {d} -> {a.experiment} run {run.info.run_id}")
