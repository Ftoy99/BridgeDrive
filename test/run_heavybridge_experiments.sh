#!/usr/bin/env bash
# HeavyBridge experiment chain on the one GPU, strictly sequential:
#   fine-tune levy (10 ep) -> full-navtest PDM eval -> fine-tune ones (control, same recipe) -> full-navtest PDM eval.
# Everything logs to MLflow (train: experiment navsim_train, eval: navsim_pdm). Baseline to beat: ep150 through the same
# head with lambda=1 (no fine-tune): PDMS 0.8714 (run heavybridge_pdm_ones_ep150_2step).
# Usage: nohup bash test/run_heavybridge_experiments.sh > chain.log 2>&1 &     PROFILES="levy ones" EPOCHS=10 override via env.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/ds/navsim-main/exp}"
RUN="conda run --no-capture-output -n hidden bash"
for P in ${PROFILES:-levy ones}; do
  TRAIN_EXP="training_heavybridge_k80_${P}"
  echo "=== $(date) train ${P} -> ${TRAIN_EXP}"
  HB_PROFILE="$P" EXP_NAME="$TRAIN_EXP" $RUN "${REPO_ROOT}/test/run_train_heavybridge.sh"
  RUN_DIR="$(ls -1d "${NAVSIM_EXP_ROOT}/${TRAIN_EXP}"/*/ | sort | tail -1)"
  CKPT="${RUN_DIR}checkpoints/last.ckpt"          # fixed recipe: final epoch, not best-val
  [[ -e "$CKPT" ]] || { echo "missing $CKPT" >&2; exit 1; }
  echo "=== $(date) eval ${P} ckpt ${CKPT}"
  HB_PROFILE="$P" CKPT="$CKPT" EXP_NAME="heavybridge_pdm_${P}_ft10" MLFLOW_TAGS="train_exp=${TRAIN_EXP}" \
    $RUN "${REPO_ROOT}/test/run_pdm_fast_heavybridge.sh"
done
echo "=== $(date) chain done"
