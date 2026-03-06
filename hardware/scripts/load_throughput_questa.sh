#!/bin/bash

# Copyright 2021 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

# Load-throughput sweep using QuestaSim (simc).
# Compiles RTL + DPI once, then sweeps (req_prob, seq_prob) via env vars.
# Use PARALLEL=N to run N simulations concurrently (default: 1 = sequential).

# Ensure Ctrl-C kills the whole process group
trap 'echo ""; echo "Interrupted! Killing child jobs..."; kill -- -$$; exit 130' INT TERM

MEMPOOL_DIR=$(git rev-parse --show-toplevel 2>/dev/null || echo $MEMPOOL_DIR)
cd $MEMPOOL_DIR/hardware

tg_ncycles=${TG_NCYCLES:-10000}
tg_tile_range=${TG_TILE_RANGE:-0}
max_parallel=${PARALLEL:-1}

# QuestaSim version and command (must match Makefile)
questa_version=${QUESTA_VERSION:-2022.3-bt}
questa_cmd="questa-${questa_version}"

# Timestamp
timestamp=$(date +%Y%m%d_%H%M%S)
result_dir=load_thru_questa_$timestamp
mkdir -p $result_dir/tmp

echo "=========================================="
echo " Load-Throughput Sweep (QuestaSim)"
echo " Results: $result_dir"
echo " Cycles per run: $tg_ncycles"
echo " Tiles per partition: $tg_tile_range"
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

# ── Single simulation job ──
run_one() {
    local seq_prob=$1
    local req_prob=$2
    local result_dir=$3
    local tg_ncycles=$4
    local questa_cmd=$5
    local tg_tile_range=$6

    local transcript=$MEMPOOL_DIR/hardware/$result_dir/transcript_seq${seq_prob}_req${req_prob}

    # Run vsim from inside build/ (matching Makefile simc target)
    pushd $MEMPOOL_DIR/hardware/build > /dev/null
    TG_REQ_PROB=${req_prob} TG_SEQ_PROB=${seq_prob} TG_NCYCLES=${tg_ncycles} \
      TG_TILE_RANGE=${tg_tile_range} \
      $questa_cmd vsim -c \
        "+DRAMSYS_RES=$MEMPOOL_DIR/hardware/deps/dram_rtl_sim/dramsys_lib/DRAMSys/configs" \
        -sv_lib ../deps/dram_rtl_sim/dramsys_lib/DRAMSys/build/lib/libsystemc \
        -sv_lib ../deps/dram_rtl_sim/dramsys_lib/DRAMSys/build/lib/libDRAMSys_Simulator \
        -sv_lib work-dpi/mempool_dpi \
        -work work \
        -suppress vsim-12070 \
        "+tg_ncycles=${tg_ncycles}" \
        work.mempool_tb \
        -l "$transcript" \
        -do "run -a" > /dev/null 2>&1
    popd > /dev/null

    # Parse results → write to per-job temp file (avoids race conditions)
    local avg_lat=$(grep "Average latency" "$transcript" | cut -d: -f2 | tr -d ' ')
    local throughput=$(grep "Throughput" "$transcript" | cut -d: -f2 | tr -d ' ')
    echo "$req_prob $avg_lat $throughput" > $result_dir/tmp/seq${seq_prob}_req${req_prob}.dat

    echo "  [seq=$seq_prob req=$req_prob] Lat: ${avg_lat:-?} | Thru: ${throughput:-?}"
}

# Step 2: Launch all jobs with throttling
echo "[2/2] Starting sweep ($(echo '6 * 30' | bc) data points, $max_parallel concurrent)..."

pids=()

for seq_prob in $(seq 0 0.2 1); do
    for req_prob in $(seq 0.02 0.02 0.6); do
        # Throttle: wait for a slot if at max
        while [ ${#pids[@]} -ge $max_parallel ]; do
            # Wait for any one child to finish
            wait -n 2>/dev/null || true
            # Reap finished PIDs
            new_pids=()
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    new_pids+=("$pid")
                fi
            done
            pids=("${new_pids[@]}")
        done

        run_one "$seq_prob" "$req_prob" "$result_dir" "$tg_ncycles" "$questa_cmd" "$tg_tile_range" &
        pids+=($!)
    done
done

# Wait for all remaining jobs
wait

# Step 3: Merge per-job results into final sorted files
echo ""
echo "Merging results..."
for seq_prob in $(seq 0 0.2 1); do
    cat $result_dir/tmp/seq${seq_prob}_req*.dat 2>/dev/null | sort -g > $result_dir/results_seqprob${seq_prob}
done
rm -rf $result_dir/tmp

echo ""
echo "=========================================="
echo " Sweep complete. Results in: $result_dir"
echo "=========================================="
