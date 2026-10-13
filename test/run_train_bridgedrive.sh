#!/usr/bin/env bash
# Train BridgeDrive (k80, beta10 = paper config) on a NAVSIM cache made by
# test/run_cache_bridgedrive.sh. Outputs go to a NEW experiment dir under
# $NAVSIM_EXP_ROOT/<EXP_NAME>/<timestamp>; nothing existing is overwritten.
#
# Usage:
#   bash test/run_train_bridgedrive.sh                       # full training
#   SMOKE=1 bash test/run_train_bridgedrive.sh               # 2 train batches on the smoke cache
#   CLIP=0 (default, no clipping like the repo; CLIP=1.0 opts in) PRECISION=bf16-mixed (default; PRECISION=16-mixed = repo/authors, 32-true = fp32)
#   BATCH_SIZE=48 GPU=0 EPOCHS=100 bash test/run_train_bridgedrive.sh
#   TOPK=3 PATIENCE=0   # keep 3 best val-traj-loss ckpts (+last.ckpt); PATIENCE>0 enables early stopping
#   RESUME=/path/to/last.ckpt bash test/run_train_bridgedrive.sh
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
# Paper note: single-GPU training gives better performance.
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

BKB_PATH="${BKB_PATH:-${REPO_ROOT}/weights/resnet34.a1_in1k/pytorch_model.bin}"
ANCHOR_PATH="${ANCHOR_PATH:-${DEVKIT}/kmeans_navsim/navsim_anchor_k80.npy}"

CACHE_NAME="${CACHE_NAME:-training_cache_bridgedrive_k80}"
# PRESET=paper (default): training hyper-parameters as stated in the BridgeDrive paper (Appendix C.3): AdamW, lr 3e-4,
#   cosine annealing with warm restarts (T0=10 epochs, T_mult=2), 10 epochs, beta_d=2.0. The LOSS is the repo's own,
#   unchanged. Not stated in the paper for NAVSIM (kept from the repo): K=80 anchors, weight decay 1e-4, batch 64.
# PRESET=authors: the authors' NAVSIM script settings (100 epochs, eff. batch 256, lr 6e-4, beta_d 1.0, WarmupCosLR).
PRESET="${PRESET:-paper}"
# CYCLES=N (paper preset): run N full cosine warm-restart cycles in ONE go, restarts happen on their own.
#   cycle lengths 10, 20, 40, 80 epochs -> total epochs 10, 30, 70, 150 for N=1,2,3,4.  Default N=1 (= the paper's 10 epochs).
CYCLES="${CYCLES:-1}"
if [[ "${PRESET}" == "paper" ]]; then
  : "${EPOCHS:=$((10 * ((1 << CYCLES) - 1)))}"; : "${ACCUM:=1}"; : "${LR:=3e-4}"; : "${BETA_D:=2.0}"
  EXP_NAME="${EXP_NAME:-training_bridgedrive_k80_paper}"
  CYCLE_END_EPOCHS=""; _e=0; _len=10
  while (( _e + _len <= EPOCHS )); do _e=$((_e + _len)); CYCLE_END_EPOCHS="${CYCLE_END_EPOCHS:+${CYCLE_END_EPOCHS},}$((_e - 1))"; _len=$((_len * 2)); done
  export CYCLE_END_EPOCHS
  PAPER_OVERRIDES=(+agent.config.scheduler_type=CosineAnnealingWarmRestarts +agent.config.sched_T0=10 +agent.config.sched_T_mult=2 agent.config.beta_d="${BETA_D}")
else
  EXP_NAME="${EXP_NAME:-training_bridgedrive_k80_beta10}"
  PAPER_OVERRIDES=()
fi
EPOCHS="${EPOCHS:-100}"
# Author script uses 256, which will not fit on a 24 GB card. Keep LR as in the agent yaml (6e-4)
# unless LR is set; changing batch size changes the effective recipe vs. the paper.
BATCH_SIZE="${BATCH_SIZE:-64}"
# Authors' NAVSIM recipe is batch 256 / lr 6e-4 on one GPU. 64 x ACCUM 4 = effective 256 on a 24 GB card.
ACCUM="${ACCUM:-4}"
# Gradient clipping: OFF by default, as in the authors' repo / paper (default_training.yaml gradient_clip_val: 0.0).
# CLIP=1.0 is available as an opt-in safety net if NaN reappears.
CLIP="${CLIP:-0}"
# Authors use fp16-mixed (yaml default); bf16 avoids fp16 overflow/loss-scaler issues behind the NaN at epoch ~46 (issue #3 too).
PRECISION="${PRECISION:-bf16-mixed}"
NUM_WORKERS="${NUM_WORKERS:-8}"

EXTRA=()
if [[ -n "${SMOKE:-}" ]]; then
  CACHE_NAME="${CACHE_NAME}_smoke${SMOKE}"
  EXP_NAME="${EXP_NAME}_smoke"
  EPOCHS=1
  # smoke cache holds only a few logs that may all fall in the val split: train on whatever is cached
  SMOKE_LOGS="$(ls "${NAVSIM_EXP_ROOT}/${CACHE_NAME}" | paste -sd, -)"
  EXTRA+=("train_logs=[${SMOKE_LOGS}]" "val_logs=[${SMOKE_LOGS}]")
  EXTRA+=(trainer.params.limit_train_batches=2 trainer.params.limit_val_batches=2 trainer.params.num_sanity_val_steps=0)
fi
EXTRA+=("${PAPER_OVERRIDES[@]}")
# Pinned host memory OFF: a run crashed with "CUDA error: invalid argument" inside the dataloader pin-memory thread.
# PIN_MEMORY=true restores the repo default (default_training.yaml: pin_memory: true).
EXTRA+=(dataloader.params.pin_memory="${PIN_MEMORY:-false}")
EXTRA+=(trainer.params.accumulate_grad_batches="${ACCUM}" trainer.params.gradient_clip_val="${CLIP}" trainer.params.precision="${PRECISION}")
[[ -n "${LR:-}" ]] && EXTRA+=(agent.lr="${LR}")
# extra hydra overrides, e.g. OVERRIDES="trainer.params.gradient_clip_val=1.0 trainer.params.precision=bf16-mixed"
[[ -n "${OVERRIDES:-}" ]] && read -r -a _OV <<< "${OVERRIDES}" && EXTRA+=("${_OV[@]}")
[[ -n "${RESUME:-}" ]] && EXTRA+=(agent.checkpoint_path="${RESUME}")

CACHE_PATH="${NAVSIM_EXP_ROOT}/${CACHE_NAME}"
for p in "${BKB_PATH}" "${ANCHOR_PATH}" "${CACHE_PATH}"; do
  [[ -e "$p" ]] || { echo "Missing: $p (run test/run_cache_bridgedrive.sh first?)" >&2; exit 1; }
done
# cache must be in BridgeDrive's Transfuser format, not another project's
if ! find "${CACHE_PATH}" -maxdepth 3 -name transfuser_feature.gz -print -quit | grep -q .; then
  echo "No transfuser_feature.gz in ${CACHE_PATH}: wrong cache format." >&2; exit 1
fi

echo "cache  : ${CACHE_PATH}"
echo "exp    : ${NAVSIM_EXP_ROOT}/${EXP_NAME}"
echo "cycle-end checkpoints at epochs: ${CYCLE_END_EPOCHS:-none}"
echo "preset=${PRESET} lr=${LR:-yaml} gpu=${CUDA_VISIBLE_DEVICES} epochs=${EPOCHS} batch=${BATCH_SIZE} x accum=${ACCUM} (eff $((BATCH_SIZE*ACCUM))) clip=${CLIP} precision=${PRECISION} workers=${NUM_WORKERS}"

cd "${DEVKIT}"
python "${REPO_ROOT}/test/run_training_bridgedrive.py" \
  agent=bridgedrive_agent_k80_beta10 \
  +agent.config.bkb_path="${BKB_PATH}" \
  agent.config.plan_anchor_path="${ANCHOR_PATH}" \
  experiment_name="${EXP_NAME}" \
  train_test_split=navtrain \
  split=trainval \
  trainer.params.max_epochs="${EPOCHS}" \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${NUM_WORKERS}" \
  cache_path="${CACHE_PATH}" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  "${EXTRA[@]}"
