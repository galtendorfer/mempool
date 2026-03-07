#!/bin/bash

# Copyright 2021 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

# Load-throughput sweep using QuestaSim (simc).
# Compiles RTL + DPI once, then sweeps (req_prob, seq_prob) via env vars.
# Set USE_WARMUP=1 to try a one-time optimized-design warm-up before the sweep.
# Use PARALLEL=N to run N simulations concurrently (default: 1 = sequential).
# Build artifacts default to scratch if available; set BUILD_ROOT or BUILD_DIR
# to override this when running multiple sweeps from the same checkout.

# Ensure Ctrl-C kills the whole process group
trap 'echo ""; echo "Interrupted! Killing child jobs..."; kill -- -$$; exit 130' INT TERM

MEMPOOL_DIR=$(git rev-parse --show-toplevel 2>/dev/null || echo $MEMPOOL_DIR)
cd "$MEMPOOL_DIR/hardware"

tg_ncycles=${TG_NCYCLES:-10000}
tg_tile_range=${TG_TILE_RANGE:-0}
max_parallel=${PARALLEL:-1}
use_warmup=${USE_WARMUP:-0}
build_dir=${BUILD_DIR:-build_tilerange${tg_tile_range}}
host_name=${HOSTNAME:-$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown-host)}

resolve_build_root() {
    if [ -n "$BUILD_ROOT" ]; then
        echo "$BUILD_ROOT"
        return 0
    fi

    for candidate in "${SCRATCH:-}" /scratch /scratch1 /scratch2; do
        if [ -n "$candidate" ] && [ -d "$candidate" ] && [ -w "$candidate" ]; then
            echo "$candidate/$USER/mempool_questa_builds/$host_name"
            return 0
        fi
    done

    echo "$MEMPOOL_DIR/hardware"
}

build_root=$(resolve_build_root)
if [[ "$build_dir" = /* ]]; then
    build_dir_path="$build_dir"
else
    build_dir_path="$build_root/$build_dir"
fi

dramsys_root="$MEMPOOL_DIR/hardware/deps/dram_rtl_sim/dramsys_lib/DRAMSys"
dramsys_res_path="$dramsys_root/configs"
dramsys_lib_path="$dramsys_root/build/lib"
dpi_lib_path="$build_dir_path/work-dpi/mempool_dpi"

# QuestaSim version and command (must match Makefile)
questa_version=${QUESTA_VERSION:-2022.3-bt}
questa_cmd="questa-${questa_version}"

# Timestamp
timestamp=$(date +%Y%m%d_%H%M%S)
result_dir=load_thru_questa_tilerange${tg_tile_range}_$timestamp
mkdir -p "$result_dir/tmp"
total_points=$((6 * 30))

echo "=========================================="
echo " Load-Throughput Sweep (QuestaSim)"
echo " Results: $result_dir"
echo " Build directory: $build_dir_path"
echo " Cycles per run: $tg_ncycles"
echo " Tiles per partition: $tg_tile_range"
echo " Parallel jobs:  $max_parallel"
echo " Warm-up snapshot: $use_warmup"
echo "=========================================="

# Step 1: Compile RTL and DPI once (with TRAFFIC_GEN enabled)
echo ""
echo "[1/3] Compiling RTL + DPI (once)..."
make clean buildpath="$build_dir_path"
if ! tg=1 tg_ncycles="$tg_ncycles" make compile buildpath="$build_dir_path"; then
    echo "ERROR: Compilation failed!"
    exit 1
fi
echo "[1/3] Compilation done."
echo ""

run_vsim() {
    local transcript=$1
    local do_cmd=$2
    local tg_req_prob=$3
    local tg_seq_prob=$4

    TG_REQ_PROB=${tg_req_prob} TG_SEQ_PROB=${tg_seq_prob} TG_NCYCLES=${tg_ncycles} \
      TG_TILE_RANGE=${tg_tile_range} \
      $questa_cmd vsim -c \
                "+DRAMSYS_RES=$dramsys_res_path" \
                -sv_lib "$dramsys_lib_path/libsystemc" \
                -sv_lib "$dramsys_lib_path/libDRAMSys_Simulator" \
                -sv_lib "$dpi_lib_path" \
        -work work \
        -suppress vsim-12070 \
        "+tg_ncycles=${tg_ncycles}" \
        work.mempool_tb \
        -l "$transcript" \
        -do "$do_cmd" > /dev/null 2>&1
}

if [ "$use_warmup" = "1" ]; then
    # Optional warm-up: in some Questa configurations this can reduce later
    # optimization work, but it is not always safe at very high concurrency.
    echo "[2/3] Creating optimized design snapshot..."
    warmup_transcript=$MEMPOOL_DIR/hardware/$result_dir/transcript_warmup
    pushd "$build_dir_path" > /dev/null
    run_vsim "$warmup_transcript" "quit -f" "0.02" "0"
    warmup_status=$?
    popd > /dev/null
    if [ $warmup_status -ne 0 ]; then
        echo "ERROR: Warm-up optimization failed!"
        echo "Check transcript: $warmup_transcript"
        exit 1
    fi
    echo "[2/3] Optimized design ready."
    echo ""
fi

# ── Single simulation job ──
run_one() {
    local seq_prob=$1
    local req_prob=$2
    local result_dir=$3
    local transcript=$MEMPOOL_DIR/hardware/$result_dir/transcript_seq${seq_prob}_req${req_prob}
    local vsim_status

    # Run vsim from inside build/ (matching Makefile simc target)
    pushd "$build_dir_path" > /dev/null
    run_vsim "$transcript" "run -a" "$req_prob" "$seq_prob"
    vsim_status=$?
    popd > /dev/null

    if [ $vsim_status -ne 0 ]; then
        echo "  [seq=$seq_prob req=$req_prob] ERROR: QuestaSim failed, see $transcript"
        return $vsim_status
    fi

    # Parse results → write to per-job temp file (avoids race conditions)
    local avg_lat=$(grep "Average latency" "$transcript" | cut -d: -f2 | tr -d ' ')
    local throughput=$(grep "Throughput" "$transcript" | cut -d: -f2 | tr -d ' ')
    echo "$req_prob $avg_lat $throughput" > $result_dir/tmp/seq${seq_prob}_req${req_prob}.dat

    echo "  [seq=$seq_prob req=$req_prob] Lat: ${avg_lat:-?} | Thru: ${throughput:-?}"
}

# Step 2/3: Launch all jobs with throttling
if [ "$use_warmup" = "1" ]; then
    echo "[3/3] Starting sweep ($total_points data points, $max_parallel concurrent)..."
else
    echo "[2/2] Starting sweep ($total_points data points, $max_parallel concurrent)..."
fi

pids=()
overall_status=0

for seq_prob in $(seq 0 0.2 1); do
    for req_prob in $(seq 0.02 0.02 0.6); do
        # Throttle: wait for a slot if at max
        while [ ${#pids[@]} -ge $max_parallel ]; do
            # Wait for any one child to finish
            if ! wait -n 2>/dev/null; then
                overall_status=1
            fi
            # Reap finished PIDs
            new_pids=()
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    new_pids+=("$pid")
                fi
            done
            pids=("${new_pids[@]}")
        done

        run_one "$seq_prob" "$req_prob" "$result_dir" &
        pids+=($!)
    done
done

# Wait for all remaining jobs
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        overall_status=1
    fi
done

# Final: Merge per-job results into final sorted files
echo ""
echo "Merging results..."
for seq_prob in $(seq 0 0.2 1); do
    cat "$result_dir"/tmp/seq${seq_prob}_req*.dat 2>/dev/null | sort -g > "$result_dir/results_seqprob${seq_prob}"
done
rm -rf "$result_dir/tmp"

if [ $overall_status -ne 0 ]; then
    echo ""
    echo "WARNING: One or more simulations failed. Partial results in: $result_dir"
    exit 1
fi

echo ""
echo "=========================================="
echo " Sweep complete. Results in: $result_dir"
echo "=========================================="
