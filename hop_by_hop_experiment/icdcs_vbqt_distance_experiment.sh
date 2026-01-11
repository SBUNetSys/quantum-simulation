# Configuration
NODE_COUNT=3
DEPOLAR_RATE=8641
START_DISTANCE=1.0
END_DISTANCE=5.0
STEP=0.5
TARGET_FID=0.0

echo "Starting distance experiments from ${START_DISTANCE}km to ${END_DISTANCE}km"
echo "Node count: ${NODE_COUNT}"
echo "Depolar rate: ${DEPOLAR_RATE}"
echo "Target fidelity: ${TARGET_FID}"
echo "With verification, no purification"
echo "----------------------------------------"

# Loop through distances
distance=$START_DISTANCE
while (( $(echo "$distance <= $END_DISTANCE" | bc -l) )); do
    echo "Running experiment for distance: ${distance}km with target fidelity: ${TARGET_FID}"

    python -u sim_vbqt_transportation_concurrent.py \
        --distance $distance \
        --target-fid $TARGET_FID \
        --depolar-rate $DEPOLAR_RATE \
	      --with-verification \
        --node-count $NODE_COUNT \
        --batch-size 4 \
        --total-runs 1000 \
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

DEPOLAR_RATE=24483
echo "Starting distance experiments from ${START_DISTANCE}km to ${END_DISTANCE}km"
echo "Node count: ${NODE_COUNT}"
echo "Depolar rate: ${DEPOLAR_RATE}"
echo "Target fidelity: ${TARGET_FID}"
echo "With verification, no purification"
echo "----------------------------------------"

# Loop through distances
distance=$START_DISTANCE
while (( $(echo "$distance <= $END_DISTANCE" | bc -l) )); do
    echo "Running experiment for distance: ${distance}km with target fidelity: ${TARGET_FID}"

    python -u sim_vbqt_transportation_concurrent.py \
        --distance $distance \
        --target-fid $TARGET_FID \
        --depolar-rate $DEPOLAR_RATE \
	      --with-verification \
        --node-count $NODE_COUNT \
        --batch-size 4 \
        --total-runs 1000 \
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

echo "All experiments completed"
