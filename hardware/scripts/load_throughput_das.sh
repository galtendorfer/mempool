#!/bin/bash

# DAS-aware load-throughput sweep using QuestaSim.
# Compiles RTL + DPI once, then sweeps (group_tiles, req_prob) via env vars.
# Produces one results directory per group_tiles value.
# Use PARALLEL=N to run N simulations concurrently (default: 8).

set -uo pipefail

# Ensure Ctrl-C kills the whole process group
trap 'echo ""; echo "Interrupted! Killing child jobs..."; kill -- -$$; exit 130' INT TERM

MEMPOOL_DIR=$(git rev-parse --show-toplevel 2>/dev/null || echo $MEMPOOL_DIR)
cd "$MEMPOOL_DIR/hardware"

# ── Configuration ──
tg_ncycles=${TG_NCYCLES:-10000}
max_parallel=${PARALLEL:-8}
tg_seed=${TG_SEED:-42}

# Sweep dimensions
GROUP_TILES_LIST="${GROUP_TILES_LIST:-1 2 4}"          # tiles per DAS group
LOCAL_TILE_PROB="${LOCAL_TILE_PROB:-0.0}"               # probability of local tile
INGROUP_PROB="${INGROUP_PROB:-1.0}"                     # probability of in-group (non-local) tile
REQ_PROBS="${REQ_PROBS:-$(seq 0.02 0.02 0.6)}"         # injection rate sweep points
SEQ_PROB=0.0                                            # always 0 for DAS sweeps

# QuestaSim
questa_version=${QUESTA_VERSION:-2022.3-bt}
questa_cmd="questa-${questa_version}"

# Timestamp
timestamp=$(date +%Y%m%d_%H%M%S)
base_dir="das_sweep_$timestamp"
mkdir -p "$base_dir"

echo "============================================"
echo " DAS Load-Throughput Sweep (QuestaSim)"
echo " Base dir:        $base_dir"
echo " Cycles per run:  $tg_ncycles"
echo " Seed:            $tg_seed"
echo " Group tiles:     $GROUP_TILES_LIST"
echo " Local tile prob: $LOCAL_TILE_PROB"
echo " In-group prob:   $INGROUP_PROB"
echo " Parallel jobs:   $max_parallel"
echo "============================================"

# Step 1: Compile RTL + DPI once (with TRAFFIC_GEN enabled)
echo ""
echo "[1/3] Compiling RTL + DPI (once)..."
make clean > /dev/null 2>&1
tg=1 tg_ncycles=$tg_ncycles tg_seed=$tg_seed make compile
if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi
echo "[1/3] Compilation done."
echo ""

# ── Single simulation job ──
run_one() {
    local group_tiles=$1
    local req_prob=$2
    local out_dir=$3
    local tg_ncycles=$4
    local questa_cmd=$5
    local local_prob=$6
    local ingroup_prob=$7
    local seed=$8

    local label="g${group_tiles}_req${req_prob}"
    local transcript="$MEMPOOL_DIR/hardware/$out_dir/transcript_${label}"

    # Run vsim from inside build/ (matching Makefile simc target)
    pushd "$MEMPOOL_DIR/hardware/build" > /dev/null
    TG_REQ_PROB="${req_prob}" \
    TG_SEQ_PROB=0.0 \
    TG_NCYCLES="${tg_ncycles}" \
    TG_GROUP_TILES="${group_tiles}" \
    TG_LOCAL_TILE_PROB="${local_prob}" \
    TG_INGROUP_PROB="${ingroup_prob}" \
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

    # Parse results
    local avg_lat=$(grep "Average latency" "$transcript" 2>/dev/null | cut -d: -f2 | tr -d ' ')
    local throughput=$(grep "Throughput" "$transcript" 2>/dev/null | cut -d: -f2 | tr -d ' ')
    local tg_mix=$(grep "TG_MIX" "$transcript" 2>/dev/null || echo "")

    # Write per-job temp file
    echo "$req_prob $avg_lat $throughput" > "$out_dir/tmp/${label}.dat"

    echo "  [$label] Lat: ${avg_lat:-?} | Thru: ${throughput:-?}  ${tg_mix:+($tg_mix)}"
}

# Step 2: Launch jobs for each grouping size
echo "[2/3] Starting sweep..."

total_jobs=0
for gt in $GROUP_TILES_LIST; do
    for rp in $REQ_PROBS; do
        total_jobs=$((total_jobs + 1))
    done
done
echo "  Total data points: $total_jobs"
echo ""

pids=()

for gt in $GROUP_TILES_LIST; do
    out_dir="$base_dir/group_tiles_${gt}"
    mkdir -p "$out_dir/tmp"
    echo "--- group_tiles=$gt (local_prob=$LOCAL_TILE_PROB, ingroup_prob=$INGROUP_PROB) ---"

    for rp in $REQ_PROBS; do
        # Throttle: wait for a slot if at max
        while [ ${#pids[@]} -ge $max_parallel ]; do
            wait -n 2>/dev/null || true
            new_pids=()
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    new_pids+=("$pid")
                fi
            done
            pids=("${new_pids[@]}")
        done

        run_one "$gt" "$rp" "$out_dir" "$tg_ncycles" "$questa_cmd" \
                "$LOCAL_TILE_PROB" "$INGROUP_PROB" "$tg_seed" &
        pids+=($!)
    done
done

# Wait for all remaining jobs
wait

# Step 3: Merge results into sorted files + CSV
echo ""
echo "[3/3] Merging results..."

# Create combined CSV
csv="$base_dir/results.csv"
echo "group_tiles,req_prob,avg_latency,throughput" > "$csv"

for gt in $GROUP_TILES_LIST; do
    out_dir="$base_dir/group_tiles_${gt}"

    # Merge per-job .dat files into one sorted results file
    cat "$out_dir/tmp"/g${gt}_req*.dat 2>/dev/null | sort -g > "$out_dir/results_group${gt}"
    rm -rf "$out_dir/tmp"

    # Append to CSV
    while read -r rp lat thr; do
        echo "$gt,$rp,$lat,$thr" >> "$csv"
    done < "$out_dir/results_group${gt}"

    echo "  group_tiles=$gt: $(wc -l < "$out_dir/results_group${gt}") data points"
done

# Save per-port bottleneck data
echo ""
echo "Extracting per-port counters..."
port_csv="$base_dir/port_counters.csv"
echo "group_tiles,req_prob,tile,port,accepts,backpressure" > "$port_csv"

for gt in $GROUP_TILES_LIST; do
    out_dir="$base_dir/group_tiles_${gt}"
    for t in "$out_dir"/transcript_g${gt}_req*; do
        rp=$(echo "$t" | sed 's/.*_req\([0-9.]*\)$/\1/')
        grep '\[REMOTE_PORT\]' "$t" 2>/dev/null | while read -r line; do
            tile=$(echo "$line" | sed 's/.*tile=\([0-9]*\).*/\1/')
            port=$(echo "$line" | sed 's/.*port=\([0-9]*\).*/\1/')
            accepts=$(echo "$line" | sed 's/.*accepts=\([0-9]*\).*/\1/')
            bp=$(echo "$line" | sed 's/.*backpressure=\([0-9]*\).*/\1/')
            echo "$gt,$rp,$tile,$port,$accepts,$bp" >> "$port_csv"
        done
    done
done

echo "  Port counters: $(wc -l < "$port_csv") rows"

echo ""
echo "============================================"
echo " Sweep complete!"
echo " Results:        $base_dir/results.csv"
echo " Port counters:  $base_dir/port_counters.csv"
echo " Per-grouping:   $base_dir/group_tiles_*/results_group*"
echo "============================================"
