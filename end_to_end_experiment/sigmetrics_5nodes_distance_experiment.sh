#!/bin/bash

# Configuration
NODE_COUNT=5
DEPOLAR_RATE=24483
START_DISTANCE=0.5
END_DISTANCE=2.5
STEP=0.25

# Target fidelities for different distances
declare -A TARGET_FIDS
TARGET_FIDS[1.0]=0.98
TARGET_FIDS[1.5]=0.97
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
echo "no verification, no purification"
echo "----------------------------------------"

# Loop through distances
distance=$START_DISTANCE
while (( $(echo "$distance <= $END_DISTANCE" | bc -l) )); do
    target_fid=${TARGET_FIDS[$distance]}

    echo "Running experiment for distance: ${distance}km with target fidelity: ${target_fid}"

    python -u sim_end_to_end_transport.py \
        --node_count $NODE_COUNT \
        --depolar_rate $DEPOLAR_RATE \
        --node_distance=$distance \
        --preload

    if [ $? -eq 0 ]; then
        echo "Successfully completed experiment for ${distance}km"
    else
        echo "Error occurred for ${distance}km"
    fi

    echo "----------------------------------------"

    # Increment distance
    distance=$(echo "$distance + $STEP" | bc)
done