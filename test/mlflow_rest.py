"""Dependency-free MLflow logging over the tracking server's REST API (urllib only).

Lets the training env (no `mlflow` package) log live to the same store as the UI/MCP: the server started with
`mlflow ui --backend-store-uri sqlite:////mnt/ds/mlflow/mlflow.db --port 5000` is a full tracking server.
Every call swallows errors (logging must never kill a training run) and warns once.
"""
import json
import os
import time
import urllib.request

import pytorch_lightning as pl

SERVER = os.environ.get("MLFLOW_SERVER", "http://127.0.0.1:5000")
_warned = False


def _post(path, body, timeout=10):
    global _warned
    req = urllib.request.Request(SERVER + "/api/2.0/mlflow/" + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except Exception as e:  # noqa: BLE001
        if not _warned:
            print(f"[mlflow_rest] WARNING logging to {SERVER} failed ({e}); training continues without it")
            _warned = True
        return None


def experiment_id(name, kind="custom_model_development"):
    try:  # get-by-name is a GET endpoint
        with urllib.request.urlopen(f"{SERVER}/api/2.0/mlflow/experiments/get-by-name?experiment_name={name}", timeout=10) as f:
            return json.loads(f.read())["experiment"]["experiment_id"]
    except Exception:  # noqa: BLE001  (404 = does not exist yet)
        r = _post("experiments/create", {"name": name, "tags": [{"key": "mlflow.experimentKind", "value": kind}]})
        return r["experiment_id"] if r else None


class MLflowRestCallback(pl.Callback):
    """Creates a run at fit start, logs step losses every `step_every` optimizer steps and all callback metrics each epoch."""

    def __init__(self, experiment, run_name, params, tags=None, step_every=100):
        self.experiment, self.run_name, self.params, self.tags = experiment, run_name, params, tags or {}
        self.step_every, self.run_id = step_every, None

    def _log(self, metrics, step):
        if not self.run_id:
            return
        ts = int(time.time() * 1000)
        ms = [{"key": k, "value": float(v), "timestamp": ts, "step": int(step)} for k, v in metrics.items() if v == v]
        for i in range(0, len(ms), 900):
            _post("runs/log-batch", {"run_id": self.run_id, "metrics": ms[i:i + 900]})

    @staticmethod
    def _scalars(d):
        out = {}
        for k, v in d.items():
            try:
                out[k.replace("/", ".")] = float(v)  # mlflow keys: alnum _ - . space /  (kept dotted for safety)
            except (TypeError, ValueError):
                pass
        return out

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        exp = experiment_id(self.experiment)
        if not exp:
            return
        r = _post("runs/create", {"experiment_id": exp, "run_name": self.run_name, "start_time": int(time.time() * 1000),
                                  "tags": [{"key": k, "value": str(v)} for k, v in self.tags.items()]})
        if not r:
            return
        self.run_id = r["run"]["info"]["run_id"]
        params = [{"key": k, "value": str(v)[:500]} for k, v in self.params.items()]
        _post("runs/log-batch", {"run_id": self.run_id, "params": params})
        print(f"[mlflow_rest] run {self.run_id} in experiment '{self.experiment}' ({SERVER})")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.run_id and trainer.global_rank == 0 and trainer.global_step % self.step_every == 0:
            self._log({k: v for k, v in self._scalars(trainer.callback_metrics).items() if "train" in k and "epoch" not in k},
                      trainer.global_step)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.run_id and trainer.global_rank == 0 and not trainer.sanity_checking:
            self._log(self._scalars(trainer.callback_metrics), trainer.current_epoch)

    def _end(self, status):
        if self.run_id:
            _post("runs/update", {"run_id": self.run_id, "status": status, "end_time": int(time.time() * 1000)})
            self.run_id = None

    def on_fit_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            self._end("FINISHED")

    def on_exception(self, trainer, pl_module, exception):
        if trainer.global_rank == 0:
            self._end("FAILED")
