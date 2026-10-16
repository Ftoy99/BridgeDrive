#!/bin/bash
# Bench2Drive closed-loop eval of BridgeDrive (arm A) + MLflow log (experiment b2d_closed_loop).
# Needs a CARLA 0.9.15 server already running on CARLA_PORT (see README-style notes in PhD/Bench2Drive_Plan.md).
# Usage: GPU=0 CARLA_PORT=8848 RUN_NAME=armA_k60_s0 SMOKE=0 bash test/run_b2d_bridgedrive.sh
set -u
LEAD_DIR=${LEAD_DIR:-/mnt/jimmys/LEAD_bridgedrive}
BD_DIR=${BD_DIR:-/mnt/jimmys/BridgeDrive}
GPU=${GPU:-0}
CARLA_PORT=${CARLA_PORT:-8848}
RUN_NAME=${RUN_NAME:-armA_k60_$(date +%m%d_%H%M)}
# routes and output dir are fixed in scripts/eval_bench2drive_bridgedrive.sh (bench2drive_111)
ROUTE_FILE=data/benchmark_routes/bench2drive_split/bench2drive_111.xml
EVAL_OUT=$LEAD_DIR/outputs/local_evaluation/bench2drive_111

cd "$LEAD_DIR" || exit 1
echo "launch: RESUME=${RESUME:-0} GPU=$GPU CARLA_PORT=$CARLA_PORT ROUTE_FILE=$ROUTE_FILE RUN_NAME=$RUN_NAME" | tee -a "$BD_DIR/test/b2d_launches.log"
env GPU=$GPU CARLA_PORT=$CARLA_PORT bash scripts/eval_bench2drive_bridgedrive.sh 2>&1 | tee "outputs/eval_${RUN_NAME}.log"
rc=${PIPESTATUS[0]}
echo "eval exit code: $rc"

RESULT=$(find "$EVAL_OUT" -maxdepth 2 \( -name merged.json -o -name checkpoint_endpoint.json \) 2>/dev/null | head -1)
if [ -z "$RESULT" ]; then
  echo "no result json under $EVAL_OUT; not logged" >&2
  exit 1
fi
cd "$BD_DIR" && MLFLOW_DISABLE_AGENT_HINT=1 uv run --no-project --python 3.11 --with 'mlflow>=3.5' python test/log_b2d_to_mlflow.py "$RESULT" \
  --run-name "$RUN_NAME" --param ckpt=model_BridgeDrive_m2_k60_0030 --param variant=bridgedrive --param routes=$ROUTE_FILE \
  --param carla_port=$CARLA_PORT --param gpu=$GPU --param eval_exit_code=$rc --tag smoke=${SMOKE:-0}
