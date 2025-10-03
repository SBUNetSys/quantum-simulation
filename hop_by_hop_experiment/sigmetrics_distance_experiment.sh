#!/bin/bash

# Configuration
NODE_COUNT=3
DEPOLAR_RATE=8641
START_DISTANCE=1.0
END_DISTANCE=5.0
STEP=0.5

# Target fidelities for different distances
declare -A TARGET_FIDS
TARGET_FIDS[1.0]=0.99
TARGET_FIDS[1.5]=0.98
TARGET_FIDS[2.0]=0.96
TARGET_FIDS[2.5]=0.94
TARGET_FIDS[3.0]=0.92
TARGET_FIDS[3.5]=0.91
TARGET_FIDS[4.0]=0.90
TARGET_FIDS[4.5]=0.88
TARGET_FIDS[5.0]=0.87

echo "Starting distance experiments from ${START_DISTANCE}km to ${END_DISTANCE}km"
echo "Node count: ${NODE_COUNT}"
echo "Depolar rate: ${DEPOLAR_RATE}"
echo "With verification, no purification"
echo "----------------------------------------"

# Loop through distances
distance=$START_DISTANCE
while (( $(echo "$distance <= $END_DISTANCE" | bc -l) )); do
    target_fid=${TARGET_FIDS[$distance]}

    echo "Running experiment for distance: ${distance}km with target fidelity: ${target_fid}"

    python -u sim_multihop_transportation_concurrent.py \
        --distance $distance \
        --target-fid $target_fid \
        --depolar-rate $DEPOLAR_RATE \
        --node-count $NODE_COUNT \
        --batch-size 4 \
        --with-verification \

    if [ $? -eq 0 ]; then
        echo "Successfully completed experiment for ${distance}km"
    else
        echo "Error occurred for ${distance}km"
    fi

    echo "----------------------------------------"

    # Increment distance
    distance=$(echo "$distance + $STEP" | bc)
done
echo "Starting distance experiments from ${START_DISTANCE}km to ${END_DISTANCE}km"
echo "Node count: ${NODE_COUNT}"
echo "Depolar rate: ${DEPOLAR_RATE}"
echo "no verification, only purification"
echo "----------------------------------------"
# Loop through distances
distance=$START_DISTANCE
while (( $(echo "$distance <= $END_DISTANCE" | bc -l) )); do
    target_fid=${TARGET_FIDS[$distance]}

    echo "Running experiment for distance: ${distance}km with target fidelity: ${target_fid}"

    python -u sim_multihop_transportation_concurrent.py \
        --distance $distance \
        --target-fid $target_fid \
        --depolar-rate $DEPOLAR_RATE \
        --node-count $NODE_COUNT \
        --with-purify \

    if [ $? -eq 0 ]; then
        echo "Successfully completed experiment for ${distance}km"
    else
        echo "Error occurred for ${distance}km"
    fi

    echo "----------------------------------------"

    # Increment distance
    distance=$(echo "$distance + $STEP" | bc)
done
echo "All experiments completed"