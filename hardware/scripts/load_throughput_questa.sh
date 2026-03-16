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
overall_status=0
build_session_root=
cleanup_build_session=0

cleanup_build_artifacts() {
    if [ "$cleanup_build_session" != "1" ]; then
        return
    fi

    if [ "$overall_status" -ne 0 ]; then
        return
    fi

    if [ -n "$build_session_root" ] && [ -d "$build_session_root" ]; then
        rm -rf "$build_session_root"
    fi
}

trap 'overall_status=130; echo ""; echo "Interrupted! Killing child jobs..."; kill -- -$$; exit 130' INT TERM
trap cleanup_build_artifacts EXIT

MEMPOOL_DIR=$(git rev-parse --show-toplevel 2>/dev/null || echo $MEMPOOL_DIR)
cd "$MEMPOOL_DIR/hardware"
script_start_epoch=$(date +%s)

resolve_active_config() {
    if [ -n "$config" ]; then
        echo "$config"
        return 0
    fi

    if [ -n "$MEMPOOL_CONFIGURATION" ]; then
        echo "$MEMPOOL_CONFIGURATION"
        return 0
    fi

    awk '
        /^[[:space:]]*config[[:space:]]*:=[[:space:]]*/ {
            if ($3 !~ /^\$\(/) {
                print $3
                exit
            }
        }
    ' "$MEMPOOL_DIR/config/config.mk"
}

make_unique_run_dir() {
    local parent_dir=$1
    local base_name=$2
    local candidate="$parent_dir/$base_name"
    local suffix=2

    while [ -e "$candidate" ]; do
        candidate="$parent_dir/${base_name}_$suffix"
        suffix=$((suffix + 1))
    done

    echo "$candidate"
}

replace_dir_copy() {
    local source_dir=$1
    local target_dir=$2

    rm -rf "$target_dir"
    mkdir -p "$(dirname "$target_dir")"
    cp -a "$source_dir" "$target_dir"
}

sync_latest_view() {
    local latest_dir=$1

    mkdir -p "$latest_dir"
    replace_dir_copy "$data_dir" "$latest_dir/data"
    replace_dir_copy "$plots_dir" "$latest_dir/plots"
    replace_dir_copy "$config_dir" "$latest_dir/config"
    replace_dir_copy "$timing_dir" "$latest_dir/timing"
    cp -a "$metadata_json" "$latest_dir/metadata.json"

    if [ -d "$audit_dir" ]; then
        replace_dir_copy "$audit_dir" "$latest_dir/audit"
    fi
}

clone_build_dir() {
    local source_dir=$1
    local target_dir=$2

    mkdir -p "$target_dir"
    cp -a --reflink=auto "$source_dir/." "$target_dir/"
}

format_duration() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))

    printf '%02d:%02d:%02d' "$hours" "$minutes" "$seconds"
}

write_metadata_json() {
    local duration_seconds=$1
    local duration_hms=$2
    local duration_seconds_json=null
    local duration_hms_json=null

    if [ -n "$duration_seconds" ]; then
        duration_seconds_json=$duration_seconds
    fi

    if [ -n "$duration_hms" ]; then
        duration_hms_json="\"$duration_hms\""
    fi

    cat > "$metadata_json" <<EOF
{
    "config": "$active_config",
    "tile_range": $tg_tile_range,
    "ncycles": $tg_ncycles,
    "parallel_jobs": $max_parallel,
    "build_shards": $build_shard_count,
    "build_shard_size": $build_shard_size,
    "use_warmup": $use_warmup,
    "questa_version": "$questa_version",
    "host": "$host_name",
    "build_dir": "$build_session_root",
    "result_dir": "$result_dir_rel",
    "git_commit": "$git_commit",
    "timestamp": "$timestamp",
    "duration_seconds": $duration_seconds_json,
    "duration_hms": $duration_hms_json
}
EOF
}

tg_ncycles=${TG_NCYCLES:-10000}
tg_tile_range=${TG_TILE_RANGE:-0}
max_parallel=${PARALLEL:-1}
build_shard_size=${BUILD_SHARD_SIZE:-20}
use_warmup=${USE_WARMUP:-0}
keep_build=${KEEP_BUILD:-0}
build_dir=${BUILD_DIR:-build_tilerange${tg_tile_range}}
host_name=${HOSTNAME:-$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown-host)}
active_config=$(resolve_active_config)

if ! [[ "$max_parallel" =~ ^[0-9]+$ ]] || [ "$max_parallel" -lt 1 ]; then
    echo "ERROR: PARALLEL must be a positive integer"
    exit 1
fi

if ! [[ "$build_shard_size" =~ ^[0-9]+$ ]] || [ "$build_shard_size" -lt 1 ]; then
    echo "ERROR: BUILD_SHARD_SIZE must be a positive integer"
    exit 1
fi

build_shard_count=$(((max_parallel + build_shard_size - 1) / build_shard_size))

# The first four partition settings are consistently the heavier half of the
# sweep. With 120 jobs and 120 slots, launching these 4 * 30 points first keeps
# the machine full of longer jobs while lighter jobs refill freed slots later.
heavy_partition_prob_values=(0.0 0.2 0.4 0.6)
light_partition_prob_values=(0.8 1.0)

# Within each partition, the high req_prob region dominates the slow tail.
req_prob_values=($(seq 0.60 -0.02 0.02))

# Timestamp shared by build and result artifacts for this sweep
timestamp=$(date +%Y-%m-%d_%H-%M)

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
    build_dir_base="$build_dir"
    auto_build_dir=0
else
    auto_build_dir=1
    build_dir_base="$build_root/$build_dir"
fi

if [ "$auto_build_dir" = "1" ]; then
    build_session_root=$(make_unique_run_dir "$build_root" "${build_dir}_${timestamp}")
    if [ "$keep_build" != "1" ]; then
        cleanup_build_session=1
    fi
else
    build_session_root="$build_dir_base"
fi

declare -a build_shard_paths
declare -a shard_parallel_limits

# Each shard gets a private mutable work/ tree so concurrent vsim jobs do not
# all fight over the same Questa optimization scratch state.
for ((shard_index = 1; shard_index <= build_shard_count; shard_index++)); do
    build_shard_paths[$shard_index]="$build_session_root/shard${shard_index}"
done

remaining_parallel=$max_parallel
for ((shard_index = 1; shard_index <= build_shard_count; shard_index++)); do
    if [ "$remaining_parallel" -gt "$build_shard_size" ]; then
        shard_parallel_limits[$shard_index]=$build_shard_size
        remaining_parallel=$((remaining_parallel - build_shard_size))
    else
        shard_parallel_limits[$shard_index]=$remaining_parallel
        remaining_parallel=0
    fi
done

dramsys_root="$MEMPOOL_DIR/hardware/deps/dram_rtl_sim/dramsys_lib/DRAMSys"
dramsys_res_path="$dramsys_root/configs"
dramsys_lib_path="$dramsys_root/build/lib"

# QuestaSim version and command (must match Makefile)
questa_version=${QUESTA_VERSION:-2022.3-bt}
questa_cmd="questa-${questa_version}"

# Organized result layout
topology_results_root="$MEMPOOL_DIR/hardware/results/$active_config"
runs_root="$topology_results_root/runs/tilerange${tg_tile_range}"
latest_dir="$topology_results_root/latest/tilerange${tg_tile_range}"
mkdir -p "$runs_root"
result_dir=$(make_unique_run_dir "$runs_root" "$timestamp")
result_dir_rel=${result_dir#${MEMPOOL_DIR}/hardware/}
raw_dir="$result_dir/raw"
raw_tile_dir="$raw_dir/tile"
raw_group_dir="$raw_dir/group"
raw_subgroup_dir="$raw_dir/subgroup"
raw_transcript_dir="$raw_dir/transcripts"
data_dir="$result_dir/data"
timing_dir="$result_dir/timing"
plots_dir="$result_dir/plots"
audit_dir="$result_dir/audit"
config_dir="$result_dir/config"
timing_tmp_dir="$timing_dir/tmp"
data_tmp_dir="$data_dir/tmp"
mkdir -p "$raw_tile_dir" "$raw_group_dir" "$raw_subgroup_dir" "$raw_transcript_dir" "$data_tmp_dir" "$timing_tmp_dir" "$plots_dir" "$config_dir"
total_points=$(((${#heavy_partition_prob_values[@]} + ${#light_partition_prob_values[@]}) * ${#req_prob_values[@]}))
summary_csv="$data_dir/run_summary.csv"
summary_txt="$data_dir/run_summary.txt"
summary_transcript="$data_dir/transcript_summary.txt"
# Per-job timing stays separate from throughput summaries so we can inspect
# long-tail behavior without mixing analysis metadata into the main CSVs.
job_timing_csv="$timing_dir/job_timing.csv"
job_timing_summary="$timing_dir/job_timing_summary.txt"
metadata_json="$result_dir/metadata.json"
git_commit=$(git -C "$MEMPOOL_DIR" rev-parse HEAD 2>/dev/null || echo unknown)

cp "$MEMPOOL_DIR/config/config.mk" "$config_dir/config.mk"
if [ -f "$MEMPOOL_DIR/config/${active_config}.mk" ]; then
    cp "$MEMPOOL_DIR/config/${active_config}.mk" "$config_dir/${active_config}.mk"
fi

write_metadata_json "" ""

echo "=========================================="

printf '%s\n' 'partition_prob,req_prob,status,avg_latency,throughput,transcript,tile_util_csv,group_util_csv,subgroup_util_csv' > "$summary_csv"
printf '%s\n' 'partition_prob,req_prob,shard,status,start_epoch,end_epoch,duration_seconds,duration_hms' > "$job_timing_csv"
{
    echo "Load-throughput sweep transcript summary"
    echo "result_dir=$result_dir_rel"
    echo "tg_tile_range=$tg_tile_range"
    echo "tg_ncycles=$tg_ncycles"
    echo "parallel_jobs=$max_parallel"
    echo "build_shards=$build_shard_count"
    echo "build_shard_size=$build_shard_size"
    echo "build_session=$build_session_root"
    echo "config=$active_config"
    echo "questa_version=$questa_version"
    echo
} > "$summary_transcript"
echo " Load-Throughput Sweep (QuestaSim)"
echo " Results: $result_dir_rel"
echo " Build session: $build_session_root"
echo " Config: $active_config"
echo " Cycles per run: $tg_ncycles"
echo " Tiles per partition: $tg_tile_range"
echo " In-partition probability knob: TG_SEQ_PROB"
echo " Parallel jobs:  $max_parallel"
echo " Build shards:   $build_shard_count"
echo " Shard size:     $build_shard_size"
echo " Warm-up snapshot: $use_warmup"
echo "=========================================="

# Step 1: Compile RTL and DPI once (with TRAFFIC_GEN enabled)
echo ""
echo "[1/3] Compiling RTL + DPI (once)..."
mkdir -p "$build_session_root"
seed_build_dir=${build_shard_paths[1]}
make clean buildpath="$seed_build_dir"
if ! tg=1 tg_ncycles="$tg_ncycles" make compile buildpath="$seed_build_dir"; then
    overall_status=1
    echo "ERROR: Compilation failed!"
    exit 1
fi

if [ "$build_shard_count" -gt 1 ]; then
    echo "[1/3] Cloning clean build into $build_shard_count shards..."
    for ((shard_index = 2; shard_index <= build_shard_count; shard_index++)); do
        clone_build_dir "$seed_build_dir" "${build_shard_paths[$shard_index]}"
    done
fi

echo "[1/3] Compilation done."
echo ""

run_vsim() {
    local build_dir_path=$1
    local transcript=$2
    local do_cmd=$3
    local tg_req_prob=$4
    local tg_partition_prob=$5
    local tile_util_file=$6
    local group_util_file=$7
    local subgroup_util_file=$8
    local dpi_lib_path="$build_dir_path/work-dpi/mempool_dpi"

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
    # Optional warm-up: in some Questa configurations this can reduce later
    # optimization work, but it is not always safe at very high concurrency.
    echo "[2/3] Creating optimized design snapshot..."
    for ((shard_index = 1; shard_index <= build_shard_count; shard_index++)); do
        warmup_transcript="$raw_transcript_dir/transcript_warmup_shard${shard_index}"
        pushd "${build_shard_paths[$shard_index]}" > /dev/null
        run_vsim "${build_shard_paths[$shard_index]}" "$warmup_transcript" "quit -f" "0.02" "0" "" "" ""
        warmup_status=$?
        popd > /dev/null
        if [ $warmup_status -ne 0 ]; then
            overall_status=1
            echo "ERROR: Warm-up optimization failed on shard $shard_index!"
            echo "Check transcript: $warmup_transcript"
            exit 1
        fi
    done
    echo "[2/3] Optimized design ready."
    echo ""
fi

reap_finished_jobs() {
    local new_pids=()
    local new_pid_shards=()
    local pid
    local shard_index
    local idx

    for ((shard_index = 1; shard_index <= build_shard_count; shard_index++)); do
        shard_active_counts[$shard_index]=0
    done

    for idx in "${!pids[@]}"; do
        pid=${pids[$idx]}
        shard_index=${pid_shards[$idx]}
        if kill -0 "$pid" 2>/dev/null; then
            new_pids+=("$pid")
            new_pid_shards+=("$shard_index")
            shard_active_counts[$shard_index]=$((shard_active_counts[$shard_index] + 1))
        fi
    done

    pids=("${new_pids[@]}")
    pid_shards=("${new_pid_shards[@]}")
}

wait_for_available_shard() {
    local shard_index

    while true; do
        for ((shard_index = 1; shard_index <= build_shard_count; shard_index++)); do
            if [ ${shard_active_counts[$shard_index]:-0} -lt ${shard_parallel_limits[$shard_index]} ]; then
                available_shard_index=$shard_index
                return 0
            fi
        done

        if ! wait -n 2>/dev/null; then
            overall_status=1
        fi
        reap_finished_jobs
    done
}

# ── Single simulation job ──
run_one() {
    local partition_prob=$1
    local req_prob=$2
    local result_dir=$3
    local build_dir_path=$4
    local shard_index=$5
    local transcript="$raw_transcript_dir/transcript_partition${partition_prob}_req${req_prob}"
    local tile_util_file="$raw_tile_dir/tile_port_util_partition${partition_prob}_req${req_prob}.csv"
    local group_util_file="$raw_group_dir/group_port_util_partition${partition_prob}_req${req_prob}.csv"
    local subgroup_util_file="$raw_subgroup_dir/subgroup_port_util_partition${partition_prob}_req${req_prob}.csv"
    local summary_tmp="$data_tmp_dir/summary_partition${partition_prob}_req${req_prob}.csv"
    local transcript_summary_tmp="$data_tmp_dir/transcript_summary_partition${partition_prob}_req${req_prob}.txt"
    local timing_tmp="$timing_tmp_dir/timing_partition${partition_prob}_req${req_prob}.csv"
    local vsim_status
    local job_start_epoch=$(date +%s)
    local job_end_epoch
    local job_duration_seconds
    local job_duration_hms

    printf '%s\n' 'tile,port,direction,cycles,accepts,stalls,util_pct' > "$tile_util_file"
    printf '%s\n' 'group,remote_group,subgroup,tile,direction,cycles,accepts,stalls,util_pct' > "$group_util_file"
    printf '%s\n' 'group,subgroup,remote_subgroup,tile,direction,cycles,accepts,stalls,util_pct' > "$subgroup_util_file"

    # Run vsim from inside build/ (matching Makefile simc target)
    pushd "$build_dir_path" > /dev/null
        run_vsim "$build_dir_path" "$transcript" "run -a" "$req_prob" "$partition_prob" \
            "$tile_util_file" "$group_util_file" "$subgroup_util_file"
    vsim_status=$?
    popd > /dev/null
    job_end_epoch=$(date +%s)
    job_duration_seconds=$((job_end_epoch - job_start_epoch))
    job_duration_hms=$(format_duration "$job_duration_seconds")

    if [ $vsim_status -ne 0 ]; then
        echo "$partition_prob,$req_prob,error,,,${transcript#${result_dir}/},${tile_util_file#${result_dir}/},${group_util_file#${result_dir}/},${subgroup_util_file#${result_dir}/}" > "$summary_tmp"
        echo "$partition_prob,$req_prob,$shard_index,error,$job_start_epoch,$job_end_epoch,$job_duration_seconds,$job_duration_hms" > "$timing_tmp"
        {
            echo "=== partition_prob=$partition_prob req_prob=$req_prob status=error ==="
            echo "transcript=${transcript#${result_dir}/}"
            echo "tile_util_csv=${tile_util_file#${result_dir}/}"
            echo "group_util_csv=${group_util_file#${result_dir}/}"
            echo "subgroup_util_csv=${subgroup_util_file#${result_dir}/}"
            echo "duration_seconds=$job_duration_seconds"
            echo "duration_hms=$job_duration_hms"
            echo
        } > "$transcript_summary_tmp"
        echo "  [partition=$partition_prob req=$req_prob] ERROR: QuestaSim failed, see $transcript"
        return $vsim_status
    fi

    # Parse results → write to per-job temp file (avoids race conditions)
    local avg_lat=$(grep "Average latency" "$transcript" | cut -d: -f2 | tr -d ' ')
    local throughput=$(grep "Throughput" "$transcript" | cut -d: -f2 | tr -d ' ')
    echo "$req_prob $avg_lat $throughput" > "$data_tmp_dir/partition${partition_prob}_req${req_prob}.dat"
    echo "$partition_prob,$req_prob,ok,$avg_lat,$throughput,${transcript#${result_dir}/},${tile_util_file#${result_dir}/},${group_util_file#${result_dir}/},${subgroup_util_file#${result_dir}/}" > "$summary_tmp"
    echo "$partition_prob,$req_prob,$shard_index,ok,$job_start_epoch,$job_end_epoch,$job_duration_seconds,$job_duration_hms" > "$timing_tmp"
    {
        echo "=== partition_prob=$partition_prob req_prob=$req_prob status=ok ==="
        echo "Average latency: ${avg_lat:-?}"
        echo "Throughput: ${throughput:-?}"
        echo "tile_util_csv=${tile_util_file#${result_dir}/}"
        echo "group_util_csv=${group_util_file#${result_dir}/}"
        echo "subgroup_util_csv=${subgroup_util_file#${result_dir}/}"
        echo "duration_seconds=$job_duration_seconds"
        echo "duration_hms=$job_duration_hms"
        echo
    } > "$transcript_summary_tmp"
    rm -f "$transcript"

    echo "  [partition=$partition_prob req=$req_prob shard=$shard_index] Lat: ${avg_lat:-?} | Thru: ${throughput:-?} | Time: $job_duration_hms"
}

# Step 2/3: Launch all jobs with throttling
if [ "$use_warmup" = "1" ]; then
    echo "[3/3] Starting sweep ($total_points data points, $max_parallel concurrent)..."
else
    echo "[2/2] Starting sweep ($total_points data points, $max_parallel concurrent)..."
fi

declare -a pids
declare -a pid_shards
declare -a shard_active_counts

for ((shard_index = 1; shard_index <= build_shard_count; shard_index++)); do
    shard_active_counts[$shard_index]=0
done

for partition_prob in "${heavy_partition_prob_values[@]}"; do
    for req_prob in "${req_prob_values[@]}"; do
        wait_for_available_shard
        shard_index=$available_shard_index
        run_one "$partition_prob" "$req_prob" "$result_dir" "${build_shard_paths[$shard_index]}" "$shard_index" &
        pids+=($!)
        pid_shards+=("$shard_index")
        shard_active_counts[$shard_index]=$((shard_active_counts[$shard_index] + 1))
    done
done

for partition_prob in "${light_partition_prob_values[@]}"; do
    for req_prob in "${req_prob_values[@]}"; do
        wait_for_available_shard
        shard_index=$available_shard_index
        run_one "$partition_prob" "$req_prob" "$result_dir" "${build_shard_paths[$shard_index]}" "$shard_index" &
        pids+=($!)
        pid_shards+=("$shard_index")
        shard_active_counts[$shard_index]=$((shard_active_counts[$shard_index] + 1))
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
for partition_prob in "${heavy_partition_prob_values[@]}" "${light_partition_prob_values[@]}"; do
    cat "$data_tmp_dir"/partition${partition_prob}_req*.dat 2>/dev/null | sort -g > "$data_dir/results_partitionprob${partition_prob}"
done

cat "$data_tmp_dir"/summary_partition*.csv 2>/dev/null | sort -t, -k1,1g -k2,2g >> "$summary_csv"
cat "$timing_tmp_dir"/timing_partition*.csv 2>/dev/null | sort -t, -k1,1g -k2,2g >> "$job_timing_csv"
cat "$data_tmp_dir"/transcript_summary_partition*.txt 2>/dev/null | sort >> /dev/null
for transcript_part in $(printf '%s\n' "$data_tmp_dir"/transcript_summary_partition*.txt | sort -V); do
    [ -f "$transcript_part" ] || continue
    cat "$transcript_part" >> "$summary_transcript"
done

success_count=$(awk -F, 'NR > 1 && $3 == "ok" {count++} END {print count + 0}' "$summary_csv")
error_count=$(awk -F, 'NR > 1 && $3 == "error" {count++} END {print count + 0}' "$summary_csv")
script_end_epoch=$(date +%s)
duration_seconds=$((script_end_epoch - script_start_epoch))
duration_hms=$(format_duration "$duration_seconds")
job_timing_count=$(awk -F, 'NR > 1 {count++} END {print count + 0}' "$job_timing_csv")
job_timing_min=$(awk -F, 'NR > 1 {if (min == "" || $7 < min) min = $7} END {print min + 0}' "$job_timing_csv")
job_timing_max=$(awk -F, 'NR > 1 {if ($7 > max) max = $7} END {print max + 0}' "$job_timing_csv")
job_timing_avg=$(awk -F, 'NR > 1 {sum += $7; count++} END {if (count > 0) printf "%.2f", sum / count; else print "0.00"}' "$job_timing_csv")

write_metadata_json "$duration_seconds" "$duration_hms"

{
    echo "Load-throughput sweep summary"
    echo "result_dir=$result_dir_rel"
    echo "config=$active_config"
    echo "tg_tile_range=$tg_tile_range"
    echo "tg_ncycles=$tg_ncycles"
    echo "parallel_jobs=$max_parallel"
    echo "build_shards=$build_shard_count"
    echo "build_shard_size=$build_shard_size"
    echo "total_points=$total_points"
    echo "successful_runs=$success_count"
    echo "failed_runs=$error_count"
    echo "duration_seconds=$duration_seconds"
    echo "duration_hms=$duration_hms"
    echo "summary_csv=${summary_csv#${result_dir}/}"
    echo "summary_transcript=${summary_transcript#${result_dir}/}"
    echo "job_timing_csv=${job_timing_csv#${result_dir}/}"
    echo "job_timing_summary=${job_timing_summary#${result_dir}/}"
    echo "metadata_json=$(basename "$metadata_json")"
} > "$summary_txt"

{
    echo "Per-job timing summary"
    echo "result_dir=$result_dir_rel"
    echo "job_count=$job_timing_count"
    echo "min_duration_seconds=$job_timing_min"
    echo "min_duration_hms=$(format_duration "$job_timing_min")"
    echo "max_duration_seconds=$job_timing_max"
    echo "max_duration_hms=$(format_duration "$job_timing_max")"
    echo "avg_duration_seconds=$job_timing_avg"
    echo "job_timing_csv=$(basename "$job_timing_csv")"
} > "$job_timing_summary"

rm -rf "$data_tmp_dir"
rm -rf "$timing_tmp_dir"

if [ $overall_status -ne 0 ]; then
    cleanup_build_session=0
    echo ""
    echo "WARNING: One or more simulations failed. Partial results in: $result_dir_rel"
    exit 1
fi

sync_latest_view "$latest_dir"

echo ""
echo "=========================================="
echo " Sweep complete. Results in: $result_dir_rel"
echo " Duration: $duration_hms"
echo "=========================================="
