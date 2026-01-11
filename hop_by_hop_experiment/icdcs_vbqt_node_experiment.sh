#!/bin/bash

# Configuration
START_NODE=7
END_NODE=10
DEPOLAR_RATE=8641
DISTANCE=1.0
TARGET_FID=0.0

echo "Starting node sweep experiments at distance ${DISTANCE}km"
echo "Node range: ${START_NODE} to ${END_NODE}"
echo "Depolar rate: ${DEPOLAR_RATE}"
echo "Target fidelity: ${TARGET_FID}"
echo "With verification, no purification"
echo "----------------------------------------"

# Loop from START_NODE to END_NODE
for NODE_COUNT in $(seq $START_NODE $END_NODE); do

    echo ">> Starting run for Node Count: ${NODE_COUNT}"

    python -u sim_vbqt_transportation_concurrent.py \
          --distance $DISTANCE \
          --target-fid $TARGET_FID \
          --depolar-rate $DEPOLAR_RATE \
          --with-verification \
          --node-count $NODE_COUNT \
          --batch-size 4 \
          --total-runs 1000 \
          --preload

    # Check exit status of the python command
    if [ $? -eq 0 ]; then
        echo "Successfully completed experiment for Node Count: ${NODE_COUNT} at ${DISTANCE}km"
    else
        echo "Error occurred for Node Count: ${NODE_COUNT}"
    fi

    echo "----------------------------------------"
done

echo "All experiments in range [${START_NODE}-${END_NODE}] completed."