#!/usr/bin/env bash
# Fast PDMS eval for HeavyBridge (copy of run_pdm_fast_bridgedrive.sh with agent=heavybridge_agent_k80): select (classifier pick) / best (oracle) / worst candidate.
# Usage:
#   CKPT=/path/to.ckpt bash test/run_pdm_fast_bridgedrive.sh
#   MAX_SCENES=50 CAND_TOPK=10 CKPT=... bash test/run_pdm_fast_bridgedrive.sh   # quick check
#   POOL_EVAL=0 ...   # only score the deployed pick (fast, == run_pdm_score.py)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVKIT="${REPO_ROOT}/BridgeDrive_adaptation_DiffusionDrive"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/ds/navsim-main/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/ds/navsim-main/exp}"
export NAVSIM_DEVKIT_ROOT="${DEVKIT}"
export PYTHONPATH="${DEVKIT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export POOL_EVAL="${POOL_EVAL:-1}"

CKPT="${CKPT:?set CKPT to a HeavyBridge ckpt (from test/run_train_heavybridge.sh)}"
BKB_PATH="${BKB_PATH:-${REPO_ROOT}/weights/resnet34.a1_in1k/pytorch_model.bin}"
ANCHOR_PATH="${ANCHOR_PATH:-${DEVKIT}/kmeans_navsim/navsim_anchor_k80.npy}"

EXTRA=()
[[ -n "${MAX_SCENES:-}" ]] && EXTRA+=("train_test_split.scene_filter.max_scenes=${MAX_SCENES}")

python "${DEVKIT}/navsim/planning/script/run_pdm_score_fast.py" \
  train_test_split=navtest \
  agent=heavybridge_agent_k80 \
  agent.checkpoint_path="${CKPT}" \
  +agent.config.bkb_path="${BKB_PATH}" \
  agent.config.plan_anchor_path="${ANCHOR_PATH}" \
  agent.config.beta_d="${BETA_D:-2.0}" \
  agent.config.hb_profile="${HB_PROFILE:-levy}" agent.config.hb_alpha="${HB_ALPHA:-1.8}" agent.config.hb_varpi="${HB_VARPI:-1.25}" agent.config.hb_eta="${HB_ETA:-0.5}" agent.config.hb_step_num="${HB_STEPS:-2}" \
  experiment_name="${EXP_NAME:-heavybridge_pdm_fast_${HB_PROFILE:-levy}}" \
  "${EXTRA[@]}"

# Log the result to MLflow (SQLite store /mnt/ds/mlflow/mlflow.db, readable via the mlflow MCP). Never fails the eval. MLFLOW_LOG=0 skips.
if [[ "${MLFLOW_LOG:-1}" == "1" ]]; then
  uv run --no-project --python 3.11 --with 'mlflow>=3.5' python "${REPO_ROOT}/test/log_pdm_to_mlflow.py" \
    "${NAVSIM_EXP_ROOT}/${EXP_NAME:-heavybridge_pdm_fast_${HB_PROFILE:-levy}}" \
    --run-name "${EXP_NAME:-heavybridge_pdm_fast_${HB_PROFILE:-levy}}" \
    --param ckpt="${CKPT}" --param variant=heavybridge --param beta_d="${BETA_D:-2.0}" \
    --param hb_profile="${HB_PROFILE:-levy}" --param hb_alpha="${HB_ALPHA:-1.8}" --param hb_varpi="${HB_VARPI:-1.25}" \
    --param hb_eta="${HB_ETA:-0.5}" --param hb_step_num="${HB_STEPS:-2}" --param max_scenes="${MAX_SCENES:-all}" \
    ${MLFLOW_TAGS:+--tag "${MLFLOW_TAGS}"} 2>&1 | grep -v "mlflow.agent.hint" || echo "WARN: mlflow logging failed (eval result is still in the exp dir)"
fi
