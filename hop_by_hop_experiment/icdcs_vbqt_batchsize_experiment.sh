#!/bin/bash

# Configuration
NODE_COUNT=3
DEPOLAR_RATE=8641
DISTANCE=1
TARGET_FID=0.0
BATCH_SIZES=(6 8 10) # List of batch sizes to run

echo "Starting batch size experiments"
echo "Node count: ${NODE_COUNT}"
echo "Distance: ${DISTANCE}km"
echo "Depolar rate: ${DEPOLAR_RATE}"
echo "Target fidelity: ${TARGET_FID}"
echo "Batch sizes to run: ${BATCH_SIZES[*]}"
echo "With verification, no purification"
echo "----------------------------------------"

# Loop through batch sizes
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    echo "Running experiment for Batch Size: ${BATCH_SIZE}"

    python -u sim_vbqt_transportation_concurrent.py \
        --distance $DISTANCE \
        --target-fid $TARGET_FID \
        --depolar-rate $DEPOLAR_RATE \
        --with-verification \
        --node-count $NODE_COUNT \
        --batch-size $BATCH_SIZE \
        --total-runs 1000 \
        --save_path "./vbqt_batchsize_experiments/" \
        --preload

    if [ $? -eq 0 ]; then
        echo "Successfully completed experiment for Batch Size: ${BATCH_SIZE}"
    else
        echo "Error occurred for Batch Size: ${BATCH_SIZE}"
    fi

    echo "----------------------------------------"
done

echo "All experiments completed"