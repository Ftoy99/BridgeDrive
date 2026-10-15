#!/usr/bin/env bash
# Fast PDMS eval for BridgeDrive: select (classifier pick) / best (oracle) / worst candidate.
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

CKPT="${CKPT:-/mnt/ds/navsim-main/exp/training_bridgedrive_k80_beta10/2026.10.09.04.50.09/checkpoints/best-epoch045-valtraj3.797.ckpt}"
BKB_PATH="${BKB_PATH:-${REPO_ROOT}/weights/resnet34.a1_in1k/pytorch_model.bin}"
ANCHOR_PATH="${ANCHOR_PATH:-${DEVKIT}/kmeans_navsim/navsim_anchor_k80.npy}"

EXTRA=()
[[ -n "${MAX_SCENES:-}" ]] && EXTRA+=("train_test_split.scene_filter.max_scenes=${MAX_SCENES}")

python "${DEVKIT}/navsim/planning/script/run_pdm_score_fast.py" \
  train_test_split=navtest \
  agent=bridgedrive_agent_k80_beta10 \
  agent.checkpoint_path="${CKPT}" \
  +agent.config.bkb_path="${BKB_PATH}" \
  agent.config.plan_anchor_path="${ANCHOR_PATH}" \
  agent.config.beta_d="${BETA_D:-2.0}" \
  experiment_name="${EXP_NAME:-bridgedrive_pdm_fast_ep45}" \
  "${EXTRA[@]}"

# Log the result to MLflow (file store /mnt/ds/mlflow, readable via the mlflow MCP). Never fails the eval. MLFLOW_LOG=0 skips.
if [[ "${MLFLOW_LOG:-1}" == "1" ]]; then
  uv run --no-project --python 3.11 --with 'mlflow>=3.5' python "${REPO_ROOT}/test/log_pdm_to_mlflow.py" \
    "${NAVSIM_EXP_ROOT}/${EXP_NAME:-bridgedrive_pdm_fast_ep45}" \
    --run-name "${EXP_NAME:-bridgedrive_pdm_fast_ep45}" \
    --param ckpt="${CKPT}" --param variant=bridgedrive --param beta_d="${BETA_D:-2.0}" \
    --param step_num=20 --param max_scenes="${MAX_SCENES:-all}" \
    ${MLFLOW_TAGS:+--tag "${MLFLOW_TAGS}"} 2>&1 | grep -v "mlflow.agent.hint" || echo "WARN: mlflow logging failed (eval result is still in the exp dir)"
fi
