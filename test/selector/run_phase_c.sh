#!/usr/bin/env bash
# Phase C selector zoo on dd_k20_public_v2 (navtrain 20k fit -> navtest once). Plan: PhD/Selector_Experiment_Plan.md §5.
# Launch: cd /mnt/jimmys/BridgeDrive && nohup bash test/selector/run_phase_c.sh > /mnt/ds/navsim-main/exp/selector/phase_c_driver.log 2>&1 &
# Each finished run dir is logged to MLflow: selector_train (selectors), selector_compare (paired bootstrap vs DD cls).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/harris/miniconda3/envs/hidden/bin/python
RUNS=/mnt/ds/navsim-main/exp/selector/runs
LOGGER="uv run --no-project --python 3.11 --with mlflow>=3.5 python test/selector/log_selector_run.py"

run() {  # run <phase_c args...>: train/eval, then log the run dir and its compare dir
  out=$($PY test/selector/phase_c.py "$@") || { echo "FAILED: $*"; return; }
  echo "$out"
  name=$(echo "$out" | awk '/^RUN_DIR/{print $2}')
  $LOGGER "$name" --experiment selector_train
  cmp=$RUNS/cmp_$(basename "$name")_vs_ddcls
  [ -d "$cmp" ] && $LOGGER "$cmp" --experiment selector_compare
}

run --selector dd_cls
run --selector ftr
for seed in 0 1 2; do
  run --selector a0 --loss bce --seed $seed
  run --selector a1 --loss bce --seed $seed
  run --selector a1 --loss pairwise --seed $seed
  run --selector r3 --loss pairwise --seed $seed
done
echo PHASE_C_DONE
