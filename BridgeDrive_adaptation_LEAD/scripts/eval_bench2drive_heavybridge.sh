#!/bin/bash

# Switch the planner head without touching code: BRIDGE_VARIANT=bridgedrive (original head) | heavybridge (tempered Levy bridge)
# Usage: CHECKPOINT_DIR=outputs/local_training/heavybridge_k60 BRIDGE_VARIANT=heavybridge . scripts/eval_bench2drive_heavybridge.sh
export BRIDGE_VARIANT=${BRIDGE_VARIANT:-heavybridge}
export HB_ALPHA=${HB_ALPHA:-1.8}
export HB_VARPI=${HB_VARPI:-1.25}
export HB_ETA=${HB_ETA:-0.5}
export HB_PROFILE=${HB_PROFILE:-levy}  # levy | ones (lambda=1: Gaussian bridge on the HeavyBridge code path)
export HB_STEPS=${HB_STEPS:-2}  # denoising strides over the N=20 grid; must divide 20. BridgeDrive default is 20.

# Checkpoints
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-outputs/local_training/heavybridge_k60}
export ROUTES=data/benchmark_routes/bench2drive_split/bench2drive_111.xml

# Set environment variables
export LEAD_PROJECT_ROOT=$(pwd)
export LD_LIBRARY_PATH=""
source $(pwd)/scripts/main.sh
export BENCHMARK_ROUTE_ID=$(basename $ROUTES .xml) # Last part of the route file name, e.g., 0 for 0.xml
export EVALUATION_OUTPUT_DIR=outputs/local_evaluation/${BRIDGE_VARIANT}_${HB_PROFILE}_$BENCHMARK_ROUTE_ID/
export PYTHONPATH=3rd_party/Bench2Drive/leaderboard:$PYTHONPATH
export PYTHONPATH=3rd_party/Bench2Drive/scenario_runner:$PYTHONPATH
export SCENARIO_RUNNER_ROOT=3rd_party/Bench2Drive/scenario_runner
export LEADERBOARD_ROOT=3rd_party/Bench2Drive/leaderboard
export IS_BENCH2DRIVE=1
export PLANNER_TYPE=only_traj
export SAVE_PATH=$EVALUATION_OUTPUT_DIR/
export PYTHONUNBUFFERED=1
export LEAD_CLOSED_LOOP_CONFIG="steer_modality=route throttle_modality=target_speed brake_modality=target_speed step_num=20 diffusion_speed=False"
if [ "$BRIDGE_VARIANT" = "heavybridge" ]; then
  export LEAD_TRAINING_CONFIG="diffusion_speed=False bridge_variant=heavybridge step_num=$HB_STEPS hb_alpha=$HB_ALPHA hb_varpi=$HB_VARPI hb_eta=$HB_ETA hb_profile=$HB_PROFILE"
else
  export LEAD_TRAINING_CONFIG="diffusion_speed=False bridge_variant=bridgedrive"
fi

set -x
set +e

# Recreate output folders
rm -rf $EVALUATION_OUTPUT_DIR/
mkdir -p $EVALUATION_OUTPUT_DIR

CUDA_VISIBLE_DEVICES=4 python3 3rd_party/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator_v2.py \
    --routes=$ROUTES \
    --track=SENSORS \
    --checkpoint=$EVALUATION_OUTPUT_DIR/checkpoint_endpoint.json \
    --agent=lead/inference/sensor_agent_bridgedrive.py \
    --agent-config=$CHECKPOINT_DIR \
    --debug=0 \
    --record=None \
    --resume=False \
    --port=8848 \
    --traffic-manager-port=8000 \
    --timeout=60 \
    --debug-checkpoint=$EVALUATION_OUTPUT_DIR/debug_checkpoint/debug_checkpoint_endpoint.txt \
    --traffic-manager-seed=0 \
    --repetitions=1 