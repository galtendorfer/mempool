# Benchmark Pipeline

## Overview

```
make benchmark ──► traces ──► CSVs ──► make plots ──► plots
  (Stage 1)                         (Stage 2)
```

Three user-facing commands do everything. All other scripts are internal.

If you are starting fresh, begin with `make benchmark` from `hardware/`.
Once that finishes, run `make plots` from `hardware/` for Stage 2.
Use `make rerun_stall_timeseries` only when you want to rebuild the stall CSV
from an existing `result_dir` without re-running simulation.

---

## Pipeline Tree

```
YOU RUN                          CALLED INTERNALLY                 OUTPUT
═══════                          ═════════════════                 ══════

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Simulation + Trace Generation                                      │
│                                                                             │
│  make benchmark                                                             │
│  ├── app=matmul_i32  config=mempool  variant=baseline  [force=1]           │
│  │                                                                          │
│  ├─► log                                                                    │
│  │     └── copies binary, env, config, topology.env, git-info → result_dir/│
│  │                                                                          │
│  ├─► simcvcs                                                                │
│  │     └── VCS simulation → build/*.dasm  (raw trace per hart)             │
│  │                                                                          │
│  ├─► trace                                                                  │
│  │     ├─► pre_trace       (cleans build/traces/)                          │
│  │     ├─► *.dasm→*.trace  (per hart, in parallel by Make)                 │
│  │     │     ├── spike-dasm            .dasm → build/traces/spike output   │
│  │     │     ├── gen_trace.py          spike output → .trace + results.csv │
│  │     │     └── outdated_gen_timeseries_windowed.py                       │
│  │     │                            (optional, if timeline_window set)     │
│  │     └─► post_trace                                                      │
│  │           ├── cp *.trace         → result_dir/traces/                   │
│  │           ├── cp results.csv     → result_dir/data/                     │
│  │           └── gen_avg.py         → result_dir/avg.txt                   │
│  │                                                                          │
│  └─► stall_timeseries                                                       │
│        └── _gen_stall_timeseries_batch.py                                   │
│              └── _gen_stall_timeseries.py  (×256 or ×1024, one per trace)  │
│              → result_dir/data/stall_timeseries_benchmark.csv              │
│                                                                             │
│  Output:                                                                    │
│    result_dir/                                                              │
│      traces/    trace_hart_*.trace                                          │
│      data/      results.csv, stall_timeseries_benchmark.csv                │
│      avg.txt, transcript, config, env, topology.env, git-info.diff         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Plot Generation                                                    │
│                                                                             │
│  make plots                                                                 │
│  ├── app=... config=... variant=...   or   result_dir=...                  │
│  ├── [plot_section=1] [plot_tiles="..."] [plot_overview=1]                │
│  ├── [plot_window=N] [plot_topology=...] [force=1]                         │
│  │                                                                          │
│  ├─► (reads data/stall_timeseries_benchmark.csv, discovers tile IDs)       │
│  ├─► (loads topology from result_dir/topology.env when available)          │
│  ├─► (falls back to path detection only if metadata is absent)             │
│  │                                                                          │
│  ├─► plot_all_tiles.py                                                      │
│  ├─► [--overview]  _plot_specific_tile.py --overview                       │
│  │     └── _stall_plot_common.py                                           │
│  │     → result_dir/plots/overview/                                        │
│  │                                                                          │
│  └─► per tile:    _plot_specific_tile.py <csv> <tile_id>                   │
│        └── _stall_plot_common.py  (shared helpers)                         │
│              └── locate_trace_file()  (finds trace in traces/)             │
│        (detail page only; no overview unless --overview is passed)         │
│        → result_dir/plots/group{N}/[subgroup{N}/]                          │
│                                                                             │
│  Output:                                                                    │
│    result_dir/plots/                                                        │
│      overview/             cluster overview PNGs                            │
│      group0/               tile PNGs for group 0                           │
│      group1/ ...           (mempool: tiles directly in group)              │
│      group0/subgroup0/     (terapool: tiles in subgroups)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2b: Single-core drill-down (ad-hoc, when you spot something)          │
│                                                                             │
│  plot_specific_core.py <csv> <core_id> [<core_id>...]                      │
│  ├── --section 1  [--traces-dir ...]  [--output-dir ...]                   │
│  └── _stall_plot_common.py                                                  │
│        → per-core 5-subplot detail report                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## User-Facing Commands

### 1. `make benchmark` — Simulate & Generate Data

Run from `hardware/`.

```bash
cd hardware
app=matmul_i32 config=mempool variant=baseline make benchmark
```

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `app` | **yes** | — | Application name (e.g. `matmul_i32`) |
| `variant` | **yes** | — | System variant: `baseline`, `das`, or `redmule` |
| `config` | no | `mempool` | Hardware topology: `mempool` (256 cores) or `terapool` (1024 cores) |
| `result_dir` | no | `results/<app>_<config>/<variant>` | Override output path (bypasses variant check) |
| `force` | no | — | Set `force=1` to allow overwriting existing results |

**Produces:**

```
result_dir/
  traces/       trace_hart_*.trace  (1 per core: 256 for mempool, 1024 for terapool)
  data/         results.csv, stall_timeseries_benchmark.csv
  avg.txt       average performance stats per section
  transcript    simulation log
  config        snapshot of config.mk at build time
  env           environment variables at run time
  topology.env  exact topology used for post-processing
  git-info.diff source state at run time
```

### 2. `make plots` — Public Plotting Entry Point

Run from `hardware/`.

```bash
cd hardware
app=matmul_i32 config=mempool variant=baseline make plots
```

This wraps `plot_all_tiles.py` so users do not need to remember the script
path, result directory layout, or standard benchmark flags.

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `result_dir` | no | `results/<app>_<config>/<variant>` | Existing result directory to plot |
| `plot_section` | no | `1` | Section to plot |
| `plot_overview` | no | `1` | Generate the cluster overview too |
| `plot_window` | no | `64` | Sliding-window width (cycles) |
| `plot_tiles` | no | all | Optional tile list, e.g. `plot_tiles="0 1 2"` |
| `plot_topology` | no | auto | Override topology detection if needed |
| `force` | no | off | Overwrite existing PNGs |

### 3. `plot_all_tiles.py` — Batch Plot Generation

Run from `hardware/scripts/stall_analysis/`.

```bash
cd hardware/scripts/stall_analysis
python plot_all_tiles.py ../../../results/matmul_i32_mempool/baseline --section 1 --overview
```

| Flag | Required | Default | Purpose |
|------|----------|---------|---------|
| `result_dir` (positional) | **yes** | — | Path to variant directory |
| `--section N` | recommended | all | Filter by section (repeatable). Use `--section 1` for benchmark. |
| `--topology` | no | auto-detected | Force `mempool` or `terapool` (normally loaded from `topology.env`) |
| `--tiles T...` | no | all from CSV | Only plot specific tile IDs |
| `--overview` | no | off | Also generate cluster overview page |
| `--window N` | no | 64 | Sliding-window width (cycles) for timeseries aggregation |
| `--force` | no | off | Overwrite existing PNGs (default: skip existing) |
| `--dry-run` | no | off | Print actions without executing |

### 4. `make rerun_stall_timeseries` — Public Reprocessing Entry Point

Run from `hardware/`.

```bash
cd hardware
result_dir=results/matmul_i32_mempool/baseline make rerun_stall_timeseries force=1
```

This is the public reprocessing command when you want to rebuild only the
stall CSV from an existing `result_dir` without re-running simulation.

### 5. `rerun_stall_timeseries.py` — Direct Wrapper

Run from `hardware/scripts/stall_analysis/`.

```bash
python rerun_stall_timeseries.py ../../../results/matmul_i32_mempool/baseline --force
```

This wraps `_gen_stall_timeseries_batch.py` and derives `traces/` and the
default output CSV from `result_dir` automatically.

### 6. `plot_specific_core.py` — Single-Core Drill-Down (Ad-Hoc)

Run from `hardware/scripts/stall_analysis/`. For investigating a specific core spotted in a tile plot.

```bash
python plot_specific_core.py \
    ../../../results/matmul_i32_mempool/baseline/data/stall_timeseries_benchmark.csv \
    42 \
    --traces-dir ../../../results/matmul_i32_mempool/baseline/traces \
    --section 1
```

| Flag | Required | Default | Purpose |
|------|----------|---------|---------|
| `csv` (positional) | **yes** | — | Path to stall_timeseries_benchmark.csv |
| `core` (positional) | **yes** | — | Core ID(s) to plot |
| `--traces-dir` | recommended | searches near CSV | Directory with trace_hart_*.trace files |
| `--section N` | recommended | all | Filter by section |
| `--output-dir` | no | `<csv-dir>/plots` | Where to save PNGs |

---

## Internal Scripts (Not User-Facing)

All internal scripts live in `stall_analysis/` with an underscore prefix.

| Script | Called by | What it does |
|--------|-----------|-------------|
| `rerun_stall_timeseries.py` | Makefile `rerun_stall_timeseries`, users | Public wrapper for safe stall CSV regeneration from `result_dir` |
| `_gen_stall_timeseries_batch.py` | Makefile `stall_timeseries`, `rerun_stall_timeseries.py` | Loops all traces, auto-loads topology metadata, calls `_gen_stall_timeseries.py` for each |
| `_gen_stall_timeseries.py` | `_gen_stall_timeseries_batch.py` | Single-trace → cycle-by-cycle stall rows in CSV |
| `_plot_specific_tile.py` | `plot_all_tiles.py` | Single-tile 6-subplot detail page |
| `_stall_plot_common.py` | `_plot_specific_tile.py`, `plot_specific_core.py` | Shared helpers: data loading, trace lookup, plot formatting |

Upstream scripts called by the Makefile (not in this folder):

| Script | Called by | What it does |
|--------|-----------|-------------|
| `gen_trace.py` | Makefile `%.trace` rule | Parses spike-dasm output → annotated `.trace` + `results.csv` |
| `outdated_gen_timeseries_windowed.py` | Makefile `%.trace` rule | Optional legacy window-based timeline CSV |
| `gen_avg.py` | Makefile `post_trace` | Averages results.csv across all cores → `avg.txt` |

---

## Topology Awareness

| Script | How it knows mempool vs terapool |
|--------|----------------------------------|
| Makefile | `config=mempool` or `config=terapool` → includes `config/<name>.mk` |
| `_gen_stall_timeseries.py` | **Environment variables**: `NUM_CORES`, `NUM_GROUPS`, `NUM_CORES_PER_TILE` (normally injected by `_gen_stall_timeseries_batch.py`) |
| `plot_all_tiles.py` | Reads `result_dir/topology.env`, then falls back to result-dir metadata/path |
| `_plot_specific_tile.py` | Topology-agnostic (works with whatever CSV says) |
| `plot_specific_core.py` | Topology-agnostic |

**Important:** New benchmark runs write `result_dir/topology.env`, and
`_gen_stall_timeseries_batch.py` uses it automatically when you point the
script at `result_dir/traces` and `result_dir/data/...csv`.

If you run `_gen_stall_timeseries_batch.py` outside a benchmark result
directory, you must either pass `--topology <config>` or set the topology
environment variables yourself.

For mempool:

```bash
  python _gen_stall_timeseries_batch.py \
    --folder .../traces --csv .../data/stall_timeseries_benchmark.csv \
    --benchmark-only -p --force --topology mempool
```

For terapool:

```bash
  python _gen_stall_timeseries_batch.py \
    --folder .../traces --csv .../data/stall_timeseries_benchmark.csv \
    --benchmark-only -p --force --topology terapool
```

---

## Safety Guards

| Step | Protection | Override |
|------|-----------|---------|
| `make benchmark` | Refuses if `variant` not set | Set `variant=...` |
| `make benchmark` | Refuses if `result_dir/traces/` has files | `force=1` |
| `_gen_stall_timeseries_batch.py` | Refuses if output CSV exists; rejects missing or conflicting topology metadata | `--force`, `--topology`, or explicit env |
| `plot_all_tiles.py` | Skips tile if PNG already exists | `--force` |

---

## Quick Reference

```bash
# ── SIMULATE ──────────────────────────────────────────────────
cd hardware
app=matmul_i32 config=mempool variant=baseline make benchmark

# ── PLOT ALL TILES ────────────────────────────────────────────
cd hardware
app=matmul_i32 config=mempool variant=baseline make plots

# ── REBUILD ONLY THE STALL CSV ────────────────────────────────
result_dir=results/matmul_i32_mempool/baseline make rerun_stall_timeseries force=1

# ── DRILL INTO CORE ──────────────────────────────────────────
cd scripts/stall_analysis
python plot_specific_core.py \
    ../../../results/matmul_i32_mempool/baseline/data/stall_timeseries_benchmark.csv \
    42 --traces-dir ../../../results/matmul_i32_mempool/baseline/traces \
    --section 1
```

---

## Folder Structure

```
hardware/scripts/
  stall_analysis/                        ← this folder
    PIPELINE.md                          pipeline documentation (this file)
    plot_all_tiles.py                    user-facing: batch tile plotter
    plot_specific_core.py                user-facing: single-core drill-down
    _plot_specific_tile.py               internal: per-tile 6-subplot detail
    _stall_plot_common.py                internal: shared plotting library
    _gen_stall_timeseries_batch.py       internal: batch trace→CSV wrapper
    _gen_stall_timeseries.py             internal: single-trace→CSV parser
  plotting/                              traffic-gen / port analysis (separate)
    _plotting_common.py                  internal: shared helpers
    plot_port_utilization.py             port utilization heatmaps
    plot_load_throughput.py              latency/throughput curves
    plot_reconstructed_load_throughput.py reconstructed latency/throughput
    compare_tilerange_plots.py           side-by-side comparison
```
