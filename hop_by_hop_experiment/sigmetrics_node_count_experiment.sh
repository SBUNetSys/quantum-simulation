#!/bin/bash

# Configuration
START_NODES=4
END_NODES=10
DEPOLAR_RATE=8641
DISTANCE=1.0
TARGET_FID=0.98
WITH_VERIFICATION="--with-verification"

# Create results directory if it doesn't exist
mkdir -p ./transportation_results

echo "========================================"
echo "Multi-Node Count Experiment"
echo "========================================"
echo "Node count range: ${START_NODES} to ${END_NODES}"
echo "Distance: ${DISTANCE} km"
echo "Depolar rate: ${DEPOLAR_RATE}"
echo "Target fidelity: ${TARGET_FID}"
echo "With verification, no purification"
echo "----------------------------------------"

# Loop through node counts
for ((node_count=$START_NODES; node_count<=$END_NODES; node_count++)); do
    echo ""
    echo "Running experiment with ${node_count} nodes..."
    echo "Started at: $(date)"

    # Run the simulation
    python -u sim_multihop_transportation_concurrent.py \
        --distance ${DISTANCE} \
        --target-fid ${TARGET_FID} \
        --depolar-rate ${DEPOLAR_RATE} \
        --node-count ${node_count} \
        --with-verification \
        --preload
    # Check if the run was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed ${node_count} nodes experiment"
    else
        echo "✗ Failed to complete ${node_count} nodes experiment"
    fi

    echo "Finished at: $(date)"
    echo "----------------------------------------"
done
echo "========================================"
echo "Multi-Node Count Experiment"
echo "========================================"
echo "Node count range: ${START_NODES} to ${END_NODES}"
echo "Distance: ${DISTANCE} km"
echo "Depolar rate: ${DEPOLAR_RATE}"
echo "Target fidelity: ${TARGET_FID}"
echo "no verification, with purification"
echo "----------------------------------------"

# Loop through node counts
for ((node_count=$START_NODES; node_count<=$END_NODES; node_count++)); do
    echo ""
    echo "Running experiment with ${node_count} nodes..."
    echo "Started at: $(date)"

    # Run the simulation
    python -u sim_multihop_transportation_concurrent.py \
        --distance ${DISTANCE} \
        --target-fid ${TARGET_FID} \
        --depolar-rate ${DEPOLAR_RATE} \
        --node-count ${node_count} \
        --with-purify \

    # Check if the run was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed ${node_count} nodes experiment"
    else
        echo "✗ Failed to complete ${node_count} nodes experiment"
    fi

    echo "Finished at: $(date)"
    echo "----------------------------------------"
done

echo ""
echo "All experiments completed at: $(date)"