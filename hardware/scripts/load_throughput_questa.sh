#!/bin/bash

# Copyright 2021 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

# Load-throughput sweep using QuestaSim (simc).
# Compiles RTL + DPI once, then sweeps (req_prob, seq_prob) via env vars.

MEMPOOL_DIR=$(git rev-parse --show-toplevel 2>/dev/null || echo $MEMPOOL_DIR)
cd $MEMPOOL_DIR/hardware

tg_ncycles=${TG_NCYCLES:-10000}

# Timestamp
timestamp=$(date +%Y%m%d_%H%M%S)
result_dir=load_thru_questa_$timestamp
mkdir -p $result_dir

echo "=========================================="
echo " Load-Throughput Sweep (QuestaSim)"
echo " Results: $result_dir"
echo " Cycles per run: $tg_ncycles"
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

# Step 2: Sweep over (seq_prob, req_prob) — simulation only
echo "[2/2] Starting sweep..."

for seq_prob in $(seq 0 0.2 1); do
    echo ""
    echo "--- seq_prob = ${seq_prob} ---"

    for req_prob in $(seq 0.02 0.02 0.6); do
        # Set runtime parameters via environment variables
        export TG_REQ_PROB=${req_prob}
        export TG_SEQ_PROB=${seq_prob}
        export TG_NCYCLES=${tg_ncycles}

        # Run simulation (skip recompilation — compile target is up to date)
        tg=1 tg_ncycles=$tg_ncycles make simc &> /dev/null

        # Parse results from transcript
        avg_lat=$(grep "Average latency" build/transcript | cut -d: -f2 | tr -d ' ')
        throughput=$(grep "Throughput" build/transcript | cut -d: -f2 | tr -d ' ')

        # Append to results file
        echo "$req_prob $avg_lat $throughput" >> $result_dir/results_seqprob${seq_prob}

        echo "  req_prob=$req_prob | Avg Latency: $avg_lat cycles | Throughput: $throughput req/core/cycle"
    done
done

echo ""
echo "=========================================="
echo " Sweep complete. Results in: $result_dir"
echo "=========================================="
