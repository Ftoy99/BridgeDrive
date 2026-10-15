#!/usr/bin/env bash
# Selector pool dump (dump_pool.py) for BridgeDrive K80. Defaults = paper ep150 ckpt, navtest.
#   POOL_ID=bd_k80_ep150_s20 bash test/selector/dump_pool.sh                        # navtest, ~30 min
#   SPLIT=navtrain SUBSAMPLE=20000 POOL_ID=bd_k80_ep150_s20 bash test/selector/dump_pool.sh   # navtrain 20k, ~50 min
#   MAX_SCENES=50 POOL_ID=smoke bash test/selector/dump_pool.sh                     # smoke
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEVKIT="${REPO_ROOT}/BridgeDrive_adaptation_DiffusionDrive"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/ds/navsim-main/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/ds/navsim-main/exp}"
export NAVSIM_DEVKIT_ROOT="${DEVKIT}"
export PYTHONPATH="${DEVKIT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
CKPT="${CKPT:-$(ls -d ${NAVSIM_EXP_ROOT}/training_bridgedrive_k80_paper/*/checkpoints/last.ckpt | tail -1)}"
BKB_PATH="${BKB_PATH:-${REPO_ROOT}/weights/resnet34.a1_in1k/pytorch_model.bin}"
ANCHOR_PATH="${ANCHOR_PATH:-${DEVKIT}/kmeans_navsim/navsim_anchor_k80.npy}"
SPLIT="${SPLIT:-navtest}"
POOL_ID="${POOL_ID:?set POOL_ID}"
export DUMP_DIR="${DUMP_DIR:-${NAVSIM_EXP_ROOT}/selector/${POOL_ID}/${SPLIT}}"
EXTRA=()
[[ "${SPLIT}" == "navtrain" ]] && EXTRA+=("metric_cache_path=${NAVSIM_EXP_ROOT}/metric_cache_navtrain")
[[ -n "${MAX_SCENES:-}" ]] && EXTRA+=("train_test_split.scene_filter.max_scenes=${MAX_SCENES}")
python "${REPO_ROOT}/test/selector/dump_pool.py" \
  train_test_split="${SPLIT}" agent="${AGENT:-bridgedrive_agent_k80_beta10}" \
  agent.checkpoint_path="${CKPT}" +agent.config.bkb_path="${BKB_PATH}" \
  agent.config.plan_anchor_path="${ANCHOR_PATH}" experiment_name="selector_dump_${POOL_ID}" "${EXTRA[@]}"
python "${REPO_ROOT}/test/selector/make_manifest.py" "${DUMP_DIR}" "${POOL_ID}" --split "${SPLIT}" --model "BridgeDrive K80" \
  --ckpt "${CKPT}" --sampler "ddbm 20 steps, seeded per batch" --dump-script test/selector/dump_pool.py --notes "${NOTES:-}"
