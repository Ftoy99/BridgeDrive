#!/usr/bin/env bash
# HeavyBridge experiment chain on the one GPU, strictly sequential:
#   for each (LR, profile): fine-tune 10 ep -> full-navtest PDM eval. 'ones' (lambda=1) is the control, same recipe.
#   Default order: gentle LR 3e-5 first (restarting a converged ckpt at the paper's peak 3e-4 for only 10 epochs likely
#   degrades it), then the paper LR 3e-4.
# Everything logs to MLflow (train: experiment navsim_train, eval: navsim_pdm). Baseline to beat: ep150 through the same
# head with lambda=1 (no fine-tune): PDMS 0.8714 (run heavybridge_pdm_ones_ep150_2step).
# Usage: nohup bash test/run_heavybridge_experiments.sh > chain.log 2>&1 &     CONFIGS="3e-5:levy 3e-5:ones 3e-4:levy 3e-4:ones" override via env.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/ds/navsim-main/exp}"
RUN="conda run --no-capture-output -n hidden bash"
for CFG in ${CONFIGS:-3e-5:levy 3e-5:ones 3e-4:levy 3e-4:ones}; do
  LR_="${CFG%%:*}"; P="${CFG##*:}"
  TRAIN_EXP="training_heavybridge_k80_${P}_lr${LR_}"
  echo "=== $(date) train ${P} lr ${LR_} -> ${TRAIN_EXP}"
  LR="$LR_" HB_PROFILE="$P" EXP_NAME="$TRAIN_EXP" $RUN "${REPO_ROOT}/test/run_train_heavybridge.sh"
  RUN_DIR="$(ls -1d "${NAVSIM_EXP_ROOT}/${TRAIN_EXP}"/*/ | sort | tail -1)"
  CKPT="${RUN_DIR}checkpoints/last.ckpt"          # fixed recipe: final epoch, not best-val
  [[ -e "$CKPT" ]] || { echo "missing $CKPT" >&2; exit 1; }
  echo "=== $(date) eval ${P} lr ${LR_} ckpt ${CKPT}"
  HB_PROFILE="$P" CKPT="$CKPT" EXP_NAME="heavybridge_pdm_${P}_lr${LR_}_ft10" MLFLOW_TAGS="train_exp=${TRAIN_EXP}" \
    $RUN "${REPO_ROOT}/test/run_pdm_fast_heavybridge.sh"
done
echo "=== $(date) chain done"
