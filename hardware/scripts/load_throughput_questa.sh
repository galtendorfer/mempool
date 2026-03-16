#!/bin/bash

# Copyright 2021 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

# Load-throughput sweep using QuestaSim (simc).
# Compiles RTL + DPI once, then sweeps (req_prob, partition_prob) via env vars.
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
summary_csv="$result_dir/run_summary.csv"
summary_txt="$result_dir/run_summary.txt"
summary_transcript="$result_dir/transcript_summary.txt"

echo "=========================================="

printf '%s\n' 'partition_prob,req_prob,status,avg_latency,throughput,transcript,tile_util_csv,group_util_csv,subgroup_util_csv' > "$summary_csv"
{
    echo "Load-throughput sweep transcript summary"
    echo "result_dir=$result_dir"
    echo "tg_tile_range=$tg_tile_range"
    echo "tg_ncycles=$tg_ncycles"
    echo "parallel_jobs=$max_parallel"
    echo
} > "$summary_transcript"
echo " Load-Throughput Sweep (QuestaSim)"
echo " Results: $result_dir"
echo " Build directory: $build_dir_path"
echo " Cycles per run: $tg_ncycles"
echo " Tiles per partition: $tg_tile_range"
echo " In-partition probability knob: TG_SEQ_PROB"
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
    local tg_partition_prob=$4
    local tile_util_file=$5
    local group_util_file=$6
    local subgroup_util_file=$7

    TG_REQ_PROB=${tg_req_prob} TG_SEQ_PROB=${tg_partition_prob} TG_NCYCLES=${tg_ncycles} \
      TG_TILE_RANGE=${tg_tile_range} \
      $questa_cmd vsim -c \
        "+DRAMSYS_RES=$dramsys_res_path" \
        "+tg_tile_port_util_file=$tile_util_file" \
        "+tg_group_port_util_file=$group_util_file" \
        "+tg_subgroup_port_util_file=$subgroup_util_file" \
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
    echo "[2/3] Creating optimized design snapshot..."
    warmup_transcript=$MEMPOOL_DIR/hardware/$result_dir/transcript_warmup
    pushd "$build_dir_path" > /dev/null
    run_vsim "$warmup_transcript" "quit -f" "0.02" "0" "" "" ""
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

run_one() {
    local partition_prob=$1
    local req_prob=$2
    local result_dir=$3
    local transcript=$MEMPOOL_DIR/hardware/$result_dir/transcript_partition${partition_prob}_req${req_prob}
    local tile_util_file=$MEMPOOL_DIR/hardware/$result_dir/tile_port_util_partition${partition_prob}_req${req_prob}.csv
    local group_util_file=$MEMPOOL_DIR/hardware/$result_dir/group_port_util_partition${partition_prob}_req${req_prob}.csv
    local subgroup_util_file=$MEMPOOL_DIR/hardware/$result_dir/subgroup_port_util_partition${partition_prob}_req${req_prob}.csv
    local summary_tmp=$result_dir/tmp/summary_partition${partition_prob}_req${req_prob}.csv
    local transcript_summary_tmp=$result_dir/tmp/transcript_summary_partition${partition_prob}_req${req_prob}.txt
    local vsim_status

    printf '%s\n' 'tile,port,cycles,accepts,stalls,util_pct' > "$tile_util_file"
    printf '%s\n' 'group,remote_group,subgroup,tile,cycles,accepts,stalls,util_pct' > "$group_util_file"
    printf '%s\n' 'group,subgroup,remote_subgroup,tile,cycles,accepts,stalls,util_pct' > "$subgroup_util_file"

    pushd "$build_dir_path" > /dev/null
    run_vsim "$transcript" "run -a" "$req_prob" "$partition_prob" \
        "$tile_util_file" "$group_util_file" "$subgroup_util_file"
    vsim_status=$?
    popd > /dev/null

    if [ $vsim_status -ne 0 ]; then
        echo "$partition_prob,$req_prob,error,,,${transcript#${MEMPOOL_DIR}/hardware/},${tile_util_file#${MEMPOOL_DIR}/hardware/},${group_util_file#${MEMPOOL_DIR}/hardware/},${subgroup_util_file#${MEMPOOL_DIR}/hardware/}" > "$summary_tmp"
        {
            echo "=== partition_prob=$partition_prob req_prob=$req_prob status=error ==="
            echo "transcript=${transcript#${MEMPOOL_DIR}/hardware/}"
            echo "tile_util_csv=${tile_util_file#${MEMPOOL_DIR}/hardware/}"
            echo "group_util_csv=${group_util_file#${MEMPOOL_DIR}/hardware/}"
            echo "subgroup_util_csv=${subgroup_util_file#${MEMPOOL_DIR}/hardware/}"
            echo
        } > "$transcript_summary_tmp"
        echo "  [partition=$partition_prob req=$req_prob] ERROR: QuestaSim failed, see $transcript"
        return $vsim_status
    fi

    local avg_lat=$(grep "Average latency" "$transcript" | cut -d: -f2 | tr -d ' ')
    local throughput=$(grep "Throughput" "$transcript" | cut -d: -f2 | tr -d ' ')
    echo "$req_prob $avg_lat $throughput" > "$result_dir/tmp/partition${partition_prob}_req${req_prob}.dat"
    echo "$partition_prob,$req_prob,ok,$avg_lat,$throughput,${transcript#${MEMPOOL_DIR}/hardware/},${tile_util_file#${MEMPOOL_DIR}/hardware/},${group_util_file#${MEMPOOL_DIR}/hardware/},${subgroup_util_file#${MEMPOOL_DIR}/hardware/}" > "$summary_tmp"
    {
        echo "=== partition_prob=$partition_prob req_prob=$req_prob status=ok ==="
        echo "Average latency: ${avg_lat:-?}"
        echo "Throughput: ${throughput:-?}"
        echo "tile_util_csv=${tile_util_file#${MEMPOOL_DIR}/hardware/}"
        echo "group_util_csv=${group_util_file#${MEMPOOL_DIR}/hardware/}"
        echo "subgroup_util_csv=${subgroup_util_file#${MEMPOOL_DIR}/hardware/}"
        echo
    } > "$transcript_summary_tmp"
    rm -f "$transcript"

    echo "  [partition=$partition_prob req=$req_prob] Lat: ${avg_lat:-?} | Thru: ${throughput:-?}"
}

if [ "$use_warmup" = "1" ]; then
    echo "[3/3] Starting sweep ($total_points data points, $max_parallel concurrent)..."
else
    echo "[2/2] Starting sweep ($total_points data points, $max_parallel concurrent)..."
fi

pids=()
overall_status=0

for partition_prob in $(seq 0 0.2 1); do
    for req_prob in $(seq 0.02 0.02 0.6); do
        while [ ${#pids[@]} -ge $max_parallel ]; do
            if ! wait -n 2>/dev/null; then
                overall_status=1
            fi
            new_pids=()
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    new_pids+=("$pid")
                fi
            done
            pids=("${new_pids[@]}")
        done

        run_one "$partition_prob" "$req_prob" "$result_dir" &
        pids+=($!)
    done
done

for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        overall_status=1
    fi
done

echo ""
echo "Merging results..."
for partition_prob in $(seq 0 0.2 1); do
    cat "$result_dir"/tmp/partition${partition_prob}_req*.dat 2>/dev/null | sort -g > "$result_dir/results_partitionprob${partition_prob}"
done

cat "$result_dir"/tmp/summary_partition*.csv 2>/dev/null | sort -t, -k1,1g -k2,2g >> "$summary_csv"
cat "$result_dir"/tmp/transcript_summary_partition*.txt 2>/dev/null | sort >> /dev/null
for transcript_part in $(printf '%s\n' "$result_dir"/tmp/transcript_summary_partition*.txt | sort -V); do
    [ -f "$transcript_part" ] || continue
    cat "$transcript_part" >> "$summary_transcript"
done

success_count=$(awk -F, 'NR > 1 && $3 == "ok" {count++} END {print count + 0}' "$summary_csv")
error_count=$(awk -F, 'NR > 1 && $3 == "error" {count++} END {print count + 0}' "$summary_csv")

{
    echo "Load-throughput sweep summary"
    echo "result_dir=$result_dir"
    echo "tg_tile_range=$tg_tile_range"
    echo "tg_ncycles=$tg_ncycles"
    echo "parallel_jobs=$max_parallel"
    echo "total_points=$total_points"
    echo "successful_runs=$success_count"
    echo "failed_runs=$error_count"
    echo "summary_csv=$(basename "$summary_csv")"
    echo "summary_transcript=$(basename "$summary_transcript")"
} > "$summary_txt"

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
