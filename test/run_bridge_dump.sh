#!/usr/bin/env bash
# Dump all-candidate PDMS + bridge signals on navtest for offline selector research.
#   MAX_SCENES=64 DUMP_DIR=/tmp/bd bash test/run_bridge_dump.sh     # quick test
#   bash test/run_bridge_dump.sh                                    # full navtest (~12k scenes)
#   SPLIT=navtrain SUBSAMPLE=10000 SAMPLES=0 RECON_TS=500 DUMP_DIR=... bash test/run_bridge_dump.sh   # ranker training data
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
CKPT="${CKPT:-/mnt/ds/navsim-main/exp/training_bridgedrive_k80_beta10/2026.10.09.04.50.09/checkpoints/best-epoch045-valtraj3.797.ckpt}"
BKB_PATH="${BKB_PATH:-${REPO_ROOT}/weights/resnet34.a1_in1k/pytorch_model.bin}"
ANCHOR_PATH="${ANCHOR_PATH:-${DEVKIT}/kmeans_navsim/navsim_anchor_k80.npy}"
SPLIT="${SPLIT:-navtest}"
EXTRA=()
if [[ "${SPLIT}" == "navtrain" ]]; then
  EXTRA+=("metric_cache_path=${NAVSIM_EXP_ROOT}/metric_cache_navtrain")
fi
[[ -n "${MAX_SCENES:-}" ]] && EXTRA+=("train_test_split.scene_filter.max_scenes=${MAX_SCENES}")
python "${REPO_ROOT}/test/run_bridge_dump.py" \
  train_test_split="${SPLIT}" agent=bridgedrive_agent_k80_beta10 \
  agent.checkpoint_path="${CKPT}" +agent.config.bkb_path="${BKB_PATH}" \
  agent.config.plan_anchor_path="${ANCHOR_PATH}" experiment_name=bridge_dump \
  "${EXTRA[@]}"
