#!/usr/bin/env bash
# §5j (Amendment A1): footprint-sampled DD BEV probabilities. GPU. Writes only exp/selector/footprint_feats/<split>/.
# Usage: SPLIT=navtrain20k bash test/selector/footprint_feats.sh   (then SPLIT=navtest)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEVKIT="/mnt/jimmys/public_repos/DiffusionDrive"
PY="${PY:-/home/harris/miniconda3/envs/hidden/bin/python}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/ds/navsim-main/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/ds/navsim-main/exp}"
export NAVSIM_DEVKIT_ROOT="${DEVKIT}"
export PYTHONPATH="${DEVKIT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
CKPT="${CKPT:-/mnt/ds/public_ckpts/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS}"
BKB_PATH="${BKB_PATH:-${REPO_ROOT}/weights/resnet34.a1_in1k/pytorch_model.bin}"
ANCHOR_PATH="${ANCHOR_PATH:-/home/harris/Desktop/Jimmys/DiffusionDriveV2/kmeans_navsim_traj_20.npy}"
export SPLIT="${SPLIT:?set SPLIT=navtrain20k or navtest}"
export POOL_OUT="${POOL_OUT:-${NAVSIM_EXP_ROOT}/selector/footprint_feats/${SPLIT}}"
cd "${REPO_ROOT}"
"${PY}" test/selector/footprint_feats.py \
  train_test_split=navtest agent=diffusiondrive_agent \
  agent.checkpoint_path="${CKPT}" +agent.config.bkb_path="${BKB_PATH}" \
  +agent.config.plan_anchor_path="${ANCHOR_PATH}" experiment_name=footprint_feats
