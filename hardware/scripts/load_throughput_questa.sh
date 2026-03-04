#!/bin/bash

# Copyright 2021 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

# Load-throughput sweep using QuestaSim (simc).
# Compiles RTL + DPI once, then sweeps (req_prob, seq_prob) via env vars.
# Use PARALLEL=N to run N seq_prob sweeps in parallel (default: sequential).

MEMPOOL_DIR=$(git rev-parse --show-toplevel 2>/dev/null || echo $MEMPOOL_DIR)
cd $MEMPOOL_DIR/hardware

tg_ncycles=${TG_NCYCLES:-10000}
max_parallel=${PARALLEL:-1}

# QuestaSim version and command (must match Makefile)
questa_version=${QUESTA_VERSION:-2022.3-bt}
questa_cmd="questa-${questa_version}"

# Timestamp
timestamp=$(date +%Y%m%d_%H%M%S)
result_dir=load_thru_questa_$timestamp
mkdir -p $result_dir

echo "=========================================="
echo " Load-Throughput Sweep (QuestaSim)"
echo " Results: $result_dir"
echo " Cycles per run: $tg_ncycles"
echo " Parallel jobs:  $max_parallel"
echo "=========================================="

# Step 1: Compile RTL and DPI once (with TRAFFIC_GEN enabled)
echo ""
echo "[1/2] Compiling RTL + DPI (once)..."
make clean
tg=1 tg_ncycles=$tg_ncycles make compile
if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi
echo "[1/2] Compilation done."
echo ""

# ── Helper: run inner req_prob sweep for one seq_prob ──
run_seq_prob_sweep() {
    local seq_prob=$1
    local result_dir=$2
    local tg_ncycles=$3
    local questa_cmd=$4

    for req_prob in $(seq 0.02 0.02 0.6); do
        local transcript=$result_dir/transcript_seq${seq_prob}_req${req_prob}

        # Run vsim directly (shared compiled work library, unique transcript)
        TG_REQ_PROB=${req_prob} TG_SEQ_PROB=${seq_prob} TG_NCYCLES=${tg_ncycles} \
          $questa_cmd vsim -c \
            "+DRAMSYS_RES=$MEMPOOL_DIR/hardware/deps/dram_rtl_sim/dramsys_lib/DRAMSys/configs" \
            -sv_lib deps/dram_rtl_sim/dramsys_lib/DRAMSys/build/lib/libsystemc \
            -sv_lib deps/dram_rtl_sim/dramsys_lib/DRAMSys/build/lib/libDRAMSys_Simulator \
            -sv_lib build/work-dpi/mempool_dpi \
            -work build/work \
            -suppress vsim-12070 \
            "+tg_ncycles=${tg_ncycles}" \
            work.mempool_tb \
            -l "$transcript" \
            -do "run -a" &> /dev/null

        # Parse results
        local avg_lat=$(grep "Average latency" "$transcript" | cut -d: -f2 | tr -d ' ')
        local throughput=$(grep "Throughput" "$transcript" | cut -d: -f2 | tr -d ' ')

        echo "$req_prob $avg_lat $throughput" >> $result_dir/results_seqprob${seq_prob}
        echo "  [seq=$seq_prob] req_prob=$req_prob | Avg Latency: $avg_lat cycles | Throughput: $throughput req/core/cycle"
    done
}

export -f run_seq_prob_sweep

# Step 2: Sweep over seq_prob values
echo "[2/2] Starting sweep..."

if [ "$max_parallel" -gt 1 ]; then
    # ── Parallel mode: launch one background job per seq_prob ──
    pids=()
    running=0

    for seq_prob in $(seq 0 0.2 1); do
        echo "Launching sweep for seq_prob=${seq_prob} ..."
        run_seq_prob_sweep "$seq_prob" "$result_dir" "$tg_ncycles" "$questa_cmd" &
        pids+=($!)
        running=$((running + 1))

        # Throttle: wait if we hit the parallel limit
        if [ "$running" -ge "$max_parallel" ]; then
            wait "${pids[0]}"
            pids=("${pids[@]:1}")
            running=$((running - 1))
        fi
    done

    # Wait for remaining jobs
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
else
    # ── Sequential mode (default) ──
    for seq_prob in $(seq 0 0.2 1); do
        echo ""
        echo "--- seq_prob = ${seq_prob} ---"
        run_seq_prob_sweep "$seq_prob" "$result_dir" "$tg_ncycles" "$questa_cmd"
    done
fi

echo ""
echo "=========================================="
echo " Sweep complete. Results in: $result_dir"
echo "=========================================="
