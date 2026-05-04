#!/bin/bash
# Fixed total path length experiment.
# Holds total distance at TOTAL_KM while increasing node count 5→30 (step 5).
# Per-segment distance = TOTAL_KM / (nodes - 1), computed with bc.
# Run with an optional depolar rate argument:
#   bash run_fixed_total_distance_experiment.sh 8641
#   bash run_fixed_total_distance_experiment.sh 24483
# If no argument is given, both rates are run sequentially.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TOTAL_KM=10.0
NODES=(5 10 15 20 25 30)
TOTAL_RUNS=1000
SAVE_PATH="./gate_noise_fixed_total_results/"

GATE_DEPOLAR_RATE=20000
GATE_DURATION_NS=50
CNOT_DEPOLAR_RATE=33000
CNOT_DURATION_NS=300

if [ -n "$1" ]; then
    DEPOLAR_RATES=("$1")
else
    DEPOLAR_RATES=(8641 24483)
fi

mkdir -p "$SAVE_PATH"

echo "=========================================="
echo "  Fixed Total Distance Experiment"
echo "  Total path: ${TOTAL_KM} km"
echo "  Nodes:      ${NODES[*]}"
echo "  Depolar:    ${DEPOLAR_RATES[*]} Hz"
echo "  Gate noise: gd=${GATE_DEPOLAR_RATE} Hz / ${GATE_DURATION_NS} ns"
echo "              CNOT=${CNOT_DEPOLAR_RATE} Hz / ${CNOT_DURATION_NS} ns"
echo "  Runs:       ${TOTAL_RUNS}"
echo "  Save:       ${SAVE_PATH}"
echo "  $(date)"
echo "=========================================="

for DEPOLAR_RATE in "${DEPOLAR_RATES[@]}"; do
    echo ""
    echo "--- Depolar rate: ${DEPOLAR_RATE} Hz ---"
    for N in "${NODES[@]}"; do
        SEGMENTS=$((N - 1))
        DIST=$(echo "scale=6; $TOTAL_KM / $SEGMENTS" | bc)
        echo ""
        echo ">> nodes=${N}  segment_dist=${DIST}km  total=${TOTAL_KM}km"

        echo "   [1/2] multihop (end-of-path correction)"
        python -u sim_multihop_gate_noise.py \
            --distance "$DIST" \
            --node-count "$N" \
            --depolar-rate "$DEPOLAR_RATE" \
            --total-runs "$TOTAL_RUNS" \
            --gate-depolar-rate "$GATE_DEPOLAR_RATE" \
            --gate-duration-ns "$GATE_DURATION_NS" \
            --cnot-depolar-rate "$CNOT_DEPOLAR_RATE" \
            --cnot-duration-ns "$CNOT_DURATION_NS" \
            --save-path "$SAVE_PATH" \
            --preload
        if [ $? -ne 0 ]; then
            echo "   ERROR: multihop failed for nodes=${N}"
        fi

        echo "   [2/2] transport (per-hop correction)"
        python -u sim_transport_gate_noise.py \
            --distance "$DIST" \
            --node-count "$N" \
            --depolar-rate "$DEPOLAR_RATE" \
            --total-runs "$TOTAL_RUNS" \
            --gate-depolar-rate "$GATE_DEPOLAR_RATE" \
            --gate-duration-ns "$GATE_DURATION_NS" \
            --cnot-depolar-rate "$CNOT_DEPOLAR_RATE" \
            --cnot-duration-ns "$CNOT_DURATION_NS" \
            --save-path "$SAVE_PATH" \
            --preload
        if [ $? -ne 0 ]; then
            echo "   ERROR: transport failed for nodes=${N}"
        fi

        echo "----------------------------------------"
    done
done

echo ""
echo "=========================================="
echo "  Fixed total distance experiments completed."
echo "  $(date)"
echo "=========================================="
