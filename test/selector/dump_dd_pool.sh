#!/usr/bin/env bash
# DiffusionDrive public-ckpt P1 pool (K=20). Official PDM, batched GPU + CPU pool.
#   SMOKE=1 MAX_SCENES=50 bash test/selector/dump_dd_pool.sh          # smoke, tagged smoke=1
#   bash test/selector/dump_dd_pool.sh                                 # full navtest (12,146 scenes)
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
POOL_ID="${POOL_ID:-dd_k20_public}"
CKPT="${CKPT:-/mnt/ds/public_ckpts/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS}"
BKB_PATH="${BKB_PATH:-/mnt/jimmys/BridgeDrive/weights/resnet34.a1_in1k/pytorch_model.bin}"
ANCHOR_PATH="${ANCHOR_PATH:-/home/harris/Desktop/Jimmys/DiffusionDriveV2/kmeans_navsim_traj_20.npy}"  # md5 0378349a, not the 1 KB placeholder
SPLIT="${SPLIT:-navtest}"
TAG="${SMOKE:+smoke_}${POOL_ID}_${SPLIT}"
export POOL_ID
export DUMP_DIR="${DUMP_DIR:-${NAVSIM_EXP_ROOT}/selector/${POOL_ID}/${SPLIT}${SMOKE:+_smoke}}"
export RUN_DIR="${NAVSIM_EXP_ROOT}/selector/runs/${TAG}_dump"
export SMOKE="${SMOKE:-0}"
EXTRA=()
[[ -n "${MAX_SCENES:-}" ]] && EXTRA+=("train_test_split.scene_filter.max_scenes=${MAX_SCENES}")
[[ "${SPLIT}" == "navtrain" ]] && EXTRA+=("metric_cache_path=${NAVSIM_EXP_ROOT}/metric_cache_navtrain")
cd "${REPO_ROOT}"
"${PY}" test/selector/dump_dd_pool.py \
  train_test_split="${SPLIT}" agent=diffusiondrive_agent \
  agent.checkpoint_path="${CKPT}" +agent.config.bkb_path="${BKB_PATH}" \
  +agent.config.plan_anchor_path="${ANCHOR_PATH}" experiment_name="dd_pool_${TAG}" "${EXTRA[@]}"
