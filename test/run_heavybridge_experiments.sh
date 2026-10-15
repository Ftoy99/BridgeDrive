#!/usr/bin/env bash
# HeavyBridge paper-faithful screening on NAVSIM, strictly sequential on the one GPU.
# For each profile:seed -> fine-tune from the ep150 BridgeDrive ckpt with the paper's screening recipe (HB_RECIPE=screening in
# run_train_heavybridge.sh: 15 epochs, lr 1e-4, grad clip 1, bf16, cosine hot-stopped via T0=100) -> full-navtest PDM eval.
#   ones = Gaussian bridge (lambda=1), paper R0       levy = per-step tempered Levy (alpha 1.8, varpi 5/4), paper R3g
# Order puts both profiles of a seed together so a partial chain still gives a paired comparison.
# Everything logs to MLflow (train: navsim_train, eval: navsim_pdm; forced-anchor rank_score_k in the eval).
# Baselines (no training): B0 original sampler 0.8712, B1 paper sampler lambda=1 (run heavybridge_pdm_B1_ep150_paper_sampler).
# Usage: nohup bash test/run_heavybridge_experiments.sh > chain.log 2>&1 &     CONFIGS="ones:0 levy:0" to shorten.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/ds/navsim-main/exp}"
RUN="conda run --no-capture-output -n hidden bash"
for CFG in ${CONFIGS:-ones:0 levy:0 ones:1 levy:1}; do
  P="${CFG%%:*}"; S="${CFG##*:}"
  TRAIN_EXP="training_heavybridge_${P}_s${S}"
  echo "=== $(date) train ${P} seed ${S} -> ${TRAIN_EXP}"
  SEED="$S" HB_PROFILE="$P" EXP_NAME="$TRAIN_EXP" $RUN "${REPO_ROOT}/test/run_train_heavybridge.sh"
  RUN_DIR="$(ls -1d "${NAVSIM_EXP_ROOT}/${TRAIN_EXP}"/*/ | sort | tail -1)"
  CKPT="${RUN_DIR}checkpoints/last.ckpt"          # fixed recipe: final epoch, not best-val
  [[ -e "$CKPT" ]] || { echo "missing $CKPT" >&2; exit 1; }
  echo "=== $(date) eval ${P} seed ${S} ckpt ${CKPT}"
  SEED="$S" HB_PROFILE="$P" CKPT="$CKPT" EXP_NAME="heavybridge_pdm_${P}_s${S}" MLFLOW_TAGS="train_exp=${TRAIN_EXP}" \
    $RUN "${REPO_ROOT}/test/run_pdm_fast_heavybridge.sh"
done
echo "=== $(date) chain done"
