#!/bin/bash
# Bench2Drive closed-loop eval of the HeavyBridge head (arm B levy / arm C ones) + MLflow log (experiment b2d_closed_loop).
# Resumes automatically if a partial result exists for this RUN_NAME; FRESH=1 forces a clean start.
# Needs a CARLA 0.9.15 server already running on CARLA_PORT (eval starts its own CARLA too, see Bench2Drive_Plan.md).
# Usage: GPU=0 CARLA_PORT=8848 RUN_NAME=B_levy_s0 CHECKPOINT_DIR=outputs/local_training/<ckpt_dir> HB_PROFILE=levy \
#        [FRESH=1] [SMOKE=1] [ROUTES=...] bash test/run_b2d_heavybridge.sh
set -u
LEAD_DIR=${LEAD_DIR:-/mnt/jimmys/LEAD_bridgedrive}
BD_DIR=${BD_DIR:-/mnt/jimmys/BridgeDrive}
GPU=${GPU:-0}
CARLA_PORT=${CARLA_PORT:-8848}
HB_PROFILE=${HB_PROFILE:?set HB_PROFILE=levy|ones}
HB_STEPS=${HB_STEPS:-2}
CHECKPOINT_DIR=${CHECKPOINT_DIR:?set CHECKPOINT_DIR (dir with model*.pth + config.json, relative to LEAD_DIR)}
RUN_NAME=${RUN_NAME:-hb_${HB_PROFILE}_$(date +%m%d_%H%M)}
FRESH=${FRESH:-0}
ROUTES=${ROUTES:-3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml}
EVAL_OUT=outputs/local_evaluation/$RUN_NAME
if [ "$FRESH" != 1 ] && [ -f "$LEAD_DIR/$EVAL_OUT/checkpoint_endpoint.json" ]; then RESUME=1; else RESUME=0; fi

cd "$LEAD_DIR" || exit 1
echo "launch: RUN_NAME=$RUN_NAME variant=heavybridge profile=$HB_PROFILE steps=$HB_STEPS ckpt=$CHECKPOINT_DIR RESUME=$RESUME GPU=$GPU CARLA_PORT=$CARLA_PORT ROUTES=$ROUTES" | tee -a "$BD_DIR/test/b2d_launches.log"
RESUME=$RESUME EVAL_OUT_DIR=$EVAL_OUT/ RUN_NAME=$RUN_NAME GPU=$GPU CARLA_PORT=$CARLA_PORT CHECKPOINT_DIR=$CHECKPOINT_DIR \
  HB_PROFILE=$HB_PROFILE HB_STEPS=$HB_STEPS ROUTES=$ROUTES bash scripts/eval_bench2drive_heavybridge.sh 2>&1 | tee -a "outputs/eval_${RUN_NAME}.log"
rc=${PIPESTATUS[0]}
echo "eval exit code: $rc"

RESULT=$(find "$EVAL_OUT" -maxdepth 2 \( -name merged.json -o -name checkpoint_endpoint.json \) 2>/dev/null | head -1)
if [ -z "$RESULT" ]; then
  echo "no result json under $EVAL_OUT; not logged" >&2
  exit 1
fi
cd "$BD_DIR" && MLFLOW_DISABLE_AGENT_HINT=1 uv run --no-project --python 3.11 --with 'mlflow>=3.5' python test/log_b2d_to_mlflow.py "$LEAD_DIR/$RESULT" \
  --run-name "$RUN_NAME" --param ckpt=$CHECKPOINT_DIR --param variant=heavybridge --param profile=$HB_PROFILE --param steps=$HB_STEPS \
  --param routes=$ROUTES --param carla_port=$CARLA_PORT --param gpu=$GPU --param resumed=$RESUME --param eval_exit_code=$rc \
  --tag smoke=${SMOKE:-0} --tag eval_status=$([ "$rc" -eq 0 ] && echo ok || echo crashed)
