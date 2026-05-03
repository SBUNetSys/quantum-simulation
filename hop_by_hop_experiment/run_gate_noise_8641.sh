#!/bin/bash

# Runs all gate noise experiments at depolar rate 8641 Hz in parallel.
# Each (distance, protocol) and (node_count, protocol) pair is launched
# as a background process. Logs go to ./gate_noise_logs/8641/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEPOLAR_RATE=8641
TOTAL_RUNS=1000
NODE_COUNT_FIXED=3
DISTANCE_FIXED=1.0
SAVE_PATH="./gate_noise_results/"
LOG_DIR="./gate_noise_logs/8641"
mkdir -p "$LOG_DIR" "$SAVE_PATH"

GATE_DEPOLAR_RATE=20000
GATE_DURATION_NS=50
CNOT_DEPOLAR_RATE=33000
CNOT_DURATION_NS=300

DISTANCES=(1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0)
NODES=(3 4 5 6 7 8 9 10)

COMMON_ARGS="--depolar-rate $DEPOLAR_RATE --total-runs $TOTAL_RUNS \
    --gate-depolar-rate $GATE_DEPOLAR_RATE --gate-duration-ns $GATE_DURATION_NS \
    --cnot-depolar-rate $CNOT_DEPOLAR_RATE --cnot-duration-ns $CNOT_DURATION_NS \
    --save-path $SAVE_PATH --preload"

echo "=========================================="
echo "  Gate Noise Experiments @ ${DEPOLAR_RATE} Hz (parallel)"
echo "  $(date)"
echo "  Logs: ${LOG_DIR}/"
echo "=========================================="

pids=()

# --- Distance sweep ---
for D in "${DISTANCES[@]}"; do
    log_prefix="${LOG_DIR}/dist_${D}km"

    python -u sim_multihop_gate_noise.py \
        --distance "$D" --node-count "$NODE_COUNT_FIXED" \
        $COMMON_ARGS \
        > "${log_prefix}_multihop.log" 2>&1 &
    pids+=($!)
    echo "  [PID $!] multihop  distance=${D}km"

    python -u sim_transport_gate_noise.py \
        --distance "$D" --node-count "$NODE_COUNT_FIXED" \
        $COMMON_ARGS \
        > "${log_prefix}_transport.log" 2>&1 &
    pids+=($!)
    echo "  [PID $!] transport distance=${D}km"
done

# --- Node count sweep ---
for N in "${NODES[@]}"; do
    log_prefix="${LOG_DIR}/nodes_${N}"

    python -u sim_multihop_gate_noise.py \
        --distance "$DISTANCE_FIXED" --node-count "$N" \
        $COMMON_ARGS \
        > "${log_prefix}_multihop.log" 2>&1 &
    pids+=($!)
    echo "  [PID $!] multihop  nodes=${N}"

    python -u sim_transport_gate_noise.py \
        --distance "$DISTANCE_FIXED" --node-count "$N" \
        $COMMON_ARGS \
        > "${log_prefix}_transport.log" 2>&1 &
    pids+=($!)
    echo "  [PID $!] transport nodes=${N}"
done

echo ""
echo "Launched ${#pids[@]} processes. Waiting for all to finish..."
echo "Monitor progress: tail -f ${LOG_DIR}/*.log"
echo ""

wait "${pids[@]}"

echo "=========================================="
echo "  All 8641 Hz experiments completed."
echo "  $(date)"
echo "=========================================="
