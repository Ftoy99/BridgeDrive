"""Log one Bench2Drive closed-loop result JSON to MLflow experiment b2d_closed_loop.

Usage:
  uv run --with 'mlflow>=3.5' python test/log_b2d_to_mlflow.py <result.json> --run-name NAME --param k=v ...
<result.json> is a merged.json or checkpoint_endpoint.json written by leaderboard_evaluator_v2.
Tracking URI: $MLFLOW_TRACKING_URI or sqlite:////mnt/ds/mlflow/mlflow.db.
"""
import argparse
import json
import os
from collections import Counter

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--experiment", default="b2d_closed_loop")
ap.add_argument("--run-name", default=None)
ap.add_argument("--param", action="append", default=[], help="key=value, repeatable")
ap.add_argument("--tag", action="append", default=[], help="key=value, repeatable")
a = ap.parse_args()

d = json.load(open(a.path))
c = d.get("_checkpoint", d)
records = c.get("records", [])

metrics = {}
for k, name in (("driving score", "driving_score"), ("success rate", "success_rate")):
    if k in d:
        metrics[name] = float(d[k])
if "driving score" not in d and records:  # checkpoint_endpoint.json (partial or per-run): compute from records
    scores = [r["scores"]["score_composed"] for r in records if r.get("scores")]
    if scores:
        metrics["driving_score"] = float(sum(scores) / len(scores))
    metrics["success_rate"] = float(sum(r.get("status", "").startswith("Completed") for r in records) / len(records))
metrics["n_routes"] = float(len(records))
metrics["n_completed"] = float(sum(r.get("status", "").startswith("Completed") for r in records))
infr = Counter()
for r in records:
    for kind, items in (r.get("infractions") or {}).items():
        infr[kind] += len(items)
metrics["infractions_total"] = float(sum(infr.values()))
for kind, n in infr.items():
    metrics[f"infr_{kind}"] = float(n)
status = Counter(r.get("status", "unknown") for r in records)

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////mnt/ds/mlflow/mlflow.db"))
mlflow.set_experiment(a.experiment)
kv = lambda xs: dict(x.split("=", 1) for x in xs)
with mlflow.start_run(run_name=a.run_name or os.path.basename(os.path.dirname(os.path.abspath(a.path)))) as run:
    mlflow.log_params({"result_json": os.path.abspath(a.path), **kv(a.param)})
    mlflow.set_tags(kv(a.tag))
    mlflow.log_metrics(metrics)
    mlflow.log_dict({k: v for k, v in status.items()}, "status_counts.json")
    mlflow.log_artifact(a.path)
    print(f"logged {a.path} -> run {run.info.run_id} DS={metrics.get('driving_score')} SR={metrics.get('success_rate')}")
