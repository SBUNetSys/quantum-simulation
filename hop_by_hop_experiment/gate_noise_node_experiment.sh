#!/bin/bash

# Node count sweep for gate noise transport experiments.
# Runs both end-of-path (multihop) and per-hop correction protocols
# at two depolar rates per node count, no purification, no verification.

# Fixed parameters
DISTANCE=1.0
START_NODE=3
END_NODE=10
TOTAL_RUNS=1000
SAVE_PATH="./gate_noise_results/"

# Gate noise (superconducting qubit ~2024 typical values)
GATE_DEPOLAR_RATE=20000   # ~0.1% error at 50 ns
GATE_DURATION_NS=50
CNOT_DEPOLAR_RATE=33000   # ~1% error at 300 ns
CNOT_DURATION_NS=300

# If DEPOLAR_RATE is set in the environment, run only that rate; otherwise run both.
if [ -n "$DEPOLAR_RATE" ]; then
    DEPOLAR_RATES=("$DEPOLAR_RATE")
else
    DEPOLAR_RATES=(8641 24483)
fi

echo "Gate noise node count experiment"
echo "  Distance:  ${DISTANCE} km"
echo "  Nodes:     ${START_NODE} to ${END_NODE}"
echo "  Depolar:   ${DEPOLAR_RATES[*]} Hz"
echo "  Gate noise: gd=${GATE_DEPOLAR_RATE} Hz / ${GATE_DURATION_NS} ns, CNOT=${CNOT_DEPOLAR_RATE} Hz / ${CNOT_DURATION_NS} ns"
echo "  Purification: False | Verification: False"
echo "  Runs:      ${TOTAL_RUNS}"
echo "----------------------------------------"

for NODE_COUNT in $(seq "$START_NODE" "$END_NODE"); do
    for DEPOLAR_RATE in "${DEPOLAR_RATES[@]}"; do
        echo ">> nodes=${NODE_COUNT}  depolar=${DEPOLAR_RATE}Hz"

        echo "   [1/2] multihop (end-of-path correction)"
        python -u sim_multihop_gate_noise.py \
            --distance "$DISTANCE" \
            --depolar-rate "$DEPOLAR_RATE" \
            --node-count "$NODE_COUNT" \
            --total-runs "$TOTAL_RUNS" \
            --gate-depolar-rate "$GATE_DEPOLAR_RATE" \
            --gate-duration-ns "$GATE_DURATION_NS" \
            --cnot-depolar-rate "$CNOT_DEPOLAR_RATE" \
            --cnot-duration-ns "$CNOT_DURATION_NS" \
            --save-path "$SAVE_PATH" \
            --preload

        if [ $? -ne 0 ]; then
            echo "   ERROR: multihop failed for nodes=${NODE_COUNT} depolar=${DEPOLAR_RATE}Hz"
        fi

        echo "   [2/2] transport (per-hop correction)"
        python -u sim_transport_gate_noise.py \
            --distance "$DISTANCE" \
            --depolar-rate "$DEPOLAR_RATE" \
            --node-count "$NODE_COUNT" \
            --total-runs "$TOTAL_RUNS" \
            --gate-depolar-rate "$GATE_DEPOLAR_RATE" \
            --gate-duration-ns "$GATE_DURATION_NS" \
            --cnot-depolar-rate "$CNOT_DEPOLAR_RATE" \
            --cnot-duration-ns "$CNOT_DURATION_NS" \
            --save-path "$SAVE_PATH" \
            --preload

        if [ $? -ne 0 ]; then
            echo "   ERROR: transport failed for nodes=${NODE_COUNT} depolar=${DEPOLAR_RATE}Hz"
        fi

        echo "----------------------------------------"
    done
done

echo "All node count experiments completed."
