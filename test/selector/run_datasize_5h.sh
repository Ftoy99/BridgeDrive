#!/usr/bin/env bash
# §5h data-size test (Selector_Experiment_Plan §5h): sensor rungs fit on navtrain-20k, scored once on navtest.
# Launch: cd /mnt/jimmys/BridgeDrive && nohup bash test/selector/run_datasize_5h.sh > /mnt/ds/navsim-main/exp/selector/datasize_5h_driver.log 2>&1 &
# Each rung run dir -> MLflow selector_datasize (logged after the whole script finishes).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/harris/miniconda3/envs/hidden/bin/python
S=/mnt/ds/navsim-main/exp/selector
OUT=$S/datasize_5h
LOGGER="uv run --no-project --python 3.11 --with mlflow>=3.5 python test/selector/log_selector_run.py"

$PY -u test/selector/sensor_rungs_train.py --train-pool $S/sensor_pool_navtrain20k/sensor_pool.npz \
  --test-pool $S/sensor_pool/sensor_pool.npz --out $OUT
for d in $OUT/rung_*; do
  $LOGGER "$d" --experiment selector_datasize
done
echo DATASIZE_5H_LOGGED
