#!/bin/bash
# Fixed total path length experiment — all jobs run in parallel.
# Holds total distance at TOTAL_KM while increasing node count 5→30 (step 5).
# Per-segment distance = TOTAL_KM / (nodes - 1), computed with bc.
# Logs go to ./gate_noise_logs/fixed_total/<rate>/.
#
# Usage:
#   bash run_fixed_total_distance_experiment.sh 8641    # single rate
#   bash run_fixed_total_distance_experiment.sh 24483
#   bash run_fixed_total_distance_experiment.sh         # both rates in parallel

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
echo "  Fixed Total Distance Experiment (parallel)"
echo "  Total path: ${TOTAL_KM} km"
echo "  Nodes:      ${NODES[*]}"
echo "  Depolar:    ${DEPOLAR_RATES[*]} Hz"
echo "  Gate noise: gd=${GATE_DEPOLAR_RATE} Hz / ${GATE_DURATION_NS} ns"
echo "              CNOT=${CNOT_DEPOLAR_RATE} Hz / ${CNOT_DURATION_NS} ns"
echo "  Runs:       ${TOTAL_RUNS}"
echo "  Save:       ${SAVE_PATH}"
echo "  $(date)"
echo "=========================================="

pids=()

for DEPOLAR_RATE in "${DEPOLAR_RATES[@]}"; do
    LOG_DIR="./gate_noise_logs/fixed_total/${DEPOLAR_RATE}"
    mkdir -p "$LOG_DIR"

    COMMON_ARGS="--depolar-rate $DEPOLAR_RATE --total-runs $TOTAL_RUNS \
        --gate-depolar-rate $GATE_DEPOLAR_RATE --gate-duration-ns $GATE_DURATION_NS \
        --cnot-depolar-rate $CNOT_DEPOLAR_RATE --cnot-duration-ns $CNOT_DURATION_NS \
        --save-path $SAVE_PATH --preload"

    for N in "${NODES[@]}"; do
        SEGMENTS=$((N - 1))
        DIST=$(echo "scale=6; $TOTAL_KM / $SEGMENTS" | bc)
        log_prefix="${LOG_DIR}/nodes_${N}"

        python -u sim_multihop_gate_noise.py \
            --distance "$DIST" --node-count "$N" \
            $COMMON_ARGS \
            > "${log_prefix}_multihop.log" 2>&1 &
        pids+=($!)
        echo "  [PID $!] multihop  nodes=${N}  dist=${DIST}km  @${DEPOLAR_RATE}Hz"

        python -u sim_transport_gate_noise.py \
            --distance "$DIST" --node-count "$N" \
            $COMMON_ARGS \
            > "${log_prefix}_transport.log" 2>&1 &
        pids+=($!)
        echo "  [PID $!] transport nodes=${N}  dist=${DIST}km  @${DEPOLAR_RATE}Hz"
    done
done

echo ""
echo "Launched ${#pids[@]} processes. Waiting for all to finish..."
echo "Monitor: tail -f ./gate_noise_logs/fixed_total/*/*.log"
echo ""

wait "${pids[@]}"

echo "=========================================="
echo "  All fixed total distance experiments completed."
echo "  $(date)"
echo "=========================================="
