#!/bin/bash

# Configuration
START_NODES=4
END_NODES=10
DEPOLAR_RATE=8641
DISTANCE=1.0
TARGET_FID=0.99

# Create results directory if it doesn't exist
mkdir -p ./transportation_results

echo "========================================"
echo "Multi-Node Count Experiment"
echo "========================================"
echo "Node count range: ${START_NODES} to ${END_NODES}"
echo "Distance: ${DISTANCE} km"
echo "Depolar rate: ${DEPOLAR_RATE}"
echo "Target fidelity: ${TARGET_FID}"
echo "no verification, no purification"
echo "----------------------------------------"

for ((node_count=$START_NODES; node_count<=$END_NODES; node_count++)); do
    echo ""
    echo "Running experiment with ${node_count} nodes..."
    echo "Started at: $(date)"

    python -u sim_end_to_end_transport.py \
            --node_count $node_count \
            --depolar_rate $DEPOLAR_RATE \
            --node_distance=$DISTANCE \

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

for ((node_count=$START_NODES; node_count<=$END_NODES; node_count++)); do
    echo ""
    echo "Running experiment with ${node_count} nodes..."
    echo "Started at: $(date)"

    python -u sim_end_to_end_transport.py \
            --node_count $node_count \
            --depolar_rate $DEPOLAR_RATE \
            --node_distance=$DISTANCE \
            --with_purification \
            --target_fidelity=$TARGET_FID
    # Check if the run was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed ${node_count} nodes experiment"
    else
        echo "✗ Failed to complete ${node_count} nodes experiment"
    fi

    echo "Finished at: $(date)"
    echo "----------------------------------------"
done
echo "All experiments completed"