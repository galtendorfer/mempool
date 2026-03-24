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

For speed, the safe parallelism boundary is one `result_dir` per job.
In practice: run different kernels in parallel, but do not try to parallelize
the communication extraction inside a single result directory.

If you do not want all kernels every time, set the kernel subset explicitly
with `parallel_kernels="..."` when using `make plots_parallel`.

---

## Fresh-Start Workflow

From `hardware/`:

```bash
make \
  app=matmul_i32 config=mempool kernel=4x4 variant=baseline benchmark

make \
  app=matmul_i32 config=mempool kernel=4x4 variant=baseline plots
```

By default, the Makefile uses `mempoolvenv/bin/python` for both the software
generation scripts and the communication-analysis scripts. You only need to
override `python` or `python_venv` if you deliberately want a different
environment.

What this gives you:

- `make benchmark`: traces plus `data/results.csv` and `data/stall_timeseries_benchmark.csv`
- `make plots`: tile/overview plots, `data/comm_events_benchmark.csv`, `data/comm_summary/`, `data/comm_timeseries/`, and `plots/communication/`

---

## Parallel Stage 2

The recommended minimal parallel workflow is to run Stage 2 for different
kernels in parallel, because each kernel writes to a separate result directory.

Good:

- `4x4` and `4x4_conflict_opt` in parallel
- `4x4_asm` and `4x4_conflict_opt_asm` in parallel

Avoid:

- `make -j` inside a single communication extraction run
- parallelizing multiple writers into the same `result_dir`

Minimal knobs:

- `parallel_kernels="..."`: which kernels to run
- `parallel_jobs=N`: optional cap on how many kernels to run at once

Default behavior:

- if you omit `parallel_jobs`, `make plots_parallel` runs one job per kernel
- if you want less concurrency, set `parallel_jobs` explicitly

Examples for the baseline 4x4 family:

Run only two kernels:

```bash
cd /home/bsc26f10/thesis/mempool/hardware

make \
  app=matmul_i32 \
  config=mempool \
  variant=baseline \
  parallel_kernels="4x4 4x4_asm" \
  plots_parallel
```

Run all four kernels:

```bash
cd /home/bsc26f10/thesis/mempool/hardware

make \
  app=matmul_i32 \
  config=mempool \
  variant=baseline \
  parallel_kernels="4x4 4x4_conflict_opt 4x4_asm 4x4_conflict_opt_asm" \
  plots_parallel
```

If you want to limit concurrency, for example to two jobs:

```bash
make \
  app=matmul_i32 \
  config=mempool \
  variant=baseline \
  parallel_kernels="4x4 4x4_conflict_opt 4x4_asm 4x4_conflict_opt_asm" \
  parallel_jobs=2 \
  plots_parallel
```

This is safe because each job writes only to its own directory:

- `results/matmul_i32_mempool/4x4/baseline/`
- `results/matmul_i32_mempool/4x4_conflict_opt/baseline/`
- `results/matmul_i32_mempool/4x4_asm/baseline/`
- `results/matmul_i32_mempool/4x4_conflict_opt_asm/baseline/`

---

## Pipeline Tree

```
YOU RUN                          CALLED INTERNALLY                 OUTPUT
═══════                          ═════════════════                 ══════

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Simulation + Trace Generation                                      │
│                                                                             │
│  make benchmark                                                             │
│  ├── app=matmul_i32  config=mempool  kernel=2x2_xpulpv2                    │
│  │   variant=baseline  [force=1]                                            │
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
│  ├── app=... config=... kernel=... variant=...   or   result_dir=...       │
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
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2c: Thesis-quality communication figures                              │
│                                                                             │
│  make plots          (default: includes communication analysis)             │
│      or                                                                     │
│  make plots_comm                                                            │
│  ├── app=... kernel=... variant=...   or   result_dir=...                  │
│  ├── [plot_section=1]                                                      │
│  │                                                                          │
│  └─► plot_comm_thesis.py <result_dir> --section N [--figures ...]          │
│        reads: data/comm_summary/, data/comm_timeseries/,                   │
│               data/comm_events_benchmark.csv,                               │
│               data/stall_timeseries_benchmark.csv                          │
│        → result_dir/plots/communication/      (PNG)                        │
│        → result_dir/plots/communication/pdf/  (PDF)                        │
│                                                                             │
│  Output:                                                                    │
│    result_dir/plots/communication/                                          │
│      traffic_matrix[_sectionN]_<kernel>_<variant>.png                       │
│      traffic_matrix_groups[_sectionN]_<kernel>_<variant>.png                │
│      locality_overview[_sectionN]_<kernel>_<variant>.png                    │
│      comm_vs_stall[_sectionN]_<kernel>_<variant>.png                        │
│      temporal_overview[_sectionN]_<kernel>_<variant>.png                    │
│      latency_timeseries[_sectionN]_<kernel>_<variant>.png                   │
│      latency_tile_g{0,1}[_sectionN]_<kernel>_<variant>.png                 │
│      latency_matrix[_sectionN]_<kernel>_<variant>.png                       │
│      latency_contention[_sectionN]_<kernel>_<variant>.png                   │
│      latency_excess_matrix[_sectionN]_<kernel>_<variant>.png                │
│    result_dir/plots/communication/pdf/                                      │
│      (same stems as above, .pdf)                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## User-Facing Commands

### 1. `make benchmark` — Simulate & Generate Data

Run from `hardware/`.

```bash
cd hardware
app=matmul_i32 config=mempool kernel=2x2_xpulpv2 variant=baseline make benchmark
```

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `app` | **yes** | — | Application name (e.g. `matmul_i32`) |
| `kernel` | **yes** | — | Kernel result label (e.g. `2x2_xpulpv2`, `4x4_asm`) |
| `variant` | **yes** | — | System variant: `baseline`, `das`, or `redmule` |
| `config` | no | `mempool` | Hardware topology: `mempool` (256 cores) or `terapool` (1024 cores) |
| `result_dir` | no | `results/<app>_<config>/<kernel>/<variant>` | Override output path (bypasses kernel/variant checks) |
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
app=matmul_i32 config=mempool kernel=2x2_xpulpv2 variant=baseline make plots
```

This wraps `plot_all_tiles.py` so users do not need to remember the script
path, result directory layout, or standard benchmark flags.

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `result_dir` | no | `results/<app>_<config>/<kernel>/<variant>` | Existing result directory to plot |
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
python plot_all_tiles.py ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline --section 1 --overview
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
result_dir=results/matmul_i32_mempool/2x2_xpulpv2/baseline make rerun_stall_timeseries force=1
```

This is the public reprocessing command when you want to rebuild only the
stall CSV from an existing `result_dir` without re-running simulation.

### 5. `rerun_stall_timeseries.py` — Direct Wrapper

Run from `hardware/scripts/stall_analysis/`.

```bash
python rerun_stall_timeseries.py ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline --force
```

This wraps `_gen_stall_timeseries_batch.py` and derives `traces/` and the
default output CSV from `result_dir` automatically.

### 6. `plot_specific_core.py` — Single-Core Drill-Down (Ad-Hoc)

Run from `hardware/scripts/stall_analysis/`. For investigating a specific core spotted in a tile plot.

```bash
python plot_specific_core.py \
  ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline/data/stall_timeseries_benchmark.csv \
    42 \
  --traces-dir ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline/traces \
    --section 1
```

| Flag | Required | Default | Purpose |
|------|----------|---------|---------|
| `csv` (positional) | **yes** | — | Path to stall_timeseries_benchmark.csv |
| `core` (positional) | **yes** | — | Core ID(s) to plot |
| `--traces-dir` | recommended | searches near CSV | Directory with trace_hart_*.trace files |
| `--section N` | recommended | all | Filter by section |
| `--output-dir` | no | `<csv-dir>/plots` | Where to save PNGs |

### 7. `extract_comm_events.py` — Communication Event Extraction

Run from `hardware/scripts/stall_analysis/` when you want a source/destination
event CSV for later communication analysis.

```bash
python extract_comm_events.py ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline --force
```

This derives:
  - traces folder: `<result_dir>/traces`
  - default output: `<result_dir>/data/comm_events_benchmark.csv`

### 8. `extract_comm_events_batch.py` — Direct Batch Extraction

Run from `hardware/scripts/stall_analysis/` when you want to point directly at
`traces/` instead of a full `result_dir`.

```bash
python extract_comm_events_batch.py \
  --folder ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline/traces \
  --csv ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline/data/comm_events_benchmark.csv \
    --benchmark-only --force
```

### 9. `summarize_comm_events.py` — First Summary Layer

Run from `hardware/scripts/stall_analysis/` after communication extraction when
you want compact CSVs that are easier to inspect than the raw event log.

```bash
python summarize_comm_events.py ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline --benchmark-only
```

This derives the default input/output paths automatically:
  - input CSV: `<result_dir>/data/comm_events_benchmark.csv`
  - output dir: `<result_dir>/data/comm_summary/`

It writes:
  - `source_dest_counts.csv`: source tile -> destination tile event counts
  - `source_tile_locality.csv`: local vs remote communication per source tile
  - `dest_tile_load_latency.csv`: load-return latency summary per destination tile

### 10. `plot_comm_thesis.py` — Thesis-Quality Communication Figures

Run from `hardware/scripts/stall_analysis/` after `summarize_comm_events.py`
and `summarize_comm_timeseries.py`. Generates up to 8 publication-ready
communication analysis figures.

```bash
# All figures for 4×4 kernel:
python plot_comm_thesis.py ../../../results/matmul_i32_mempool/4x4/baseline --section 1

# All figures for 2×2 kernel:
python plot_comm_thesis.py ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline --section 1

# Specific figures only:
python plot_comm_thesis.py ../../../results/matmul_i32_mempool/4x4/baseline --section 1 \
  --figures matrix zoom contention latency_over_minimum
```

This derives the default input/output paths automatically:
  - input summary dir: `<result_dir>/data/comm_summary/`
  - input timeseries dir: `<result_dir>/data/comm_timeseries/`
  - input events CSV: `<result_dir>/data/comm_events_benchmark.csv`
  - input stalls CSV: `<result_dir>/data/stall_timeseries_benchmark.csv`
  - output PNG dir: `<result_dir>/plots/communication/`
  - output PDF dir: `<result_dir>/plots/communication/pdf/`

| Flag | Required | Default | Purpose |
|------|----------|---------|----------|
| `input_path` (positional) | **yes** | — | Result directory (contains `data/` and `plots/`) |
| `--section N` | recommended | all | Filter by section |
| `--n-groups N` | no | 4 | Number of tile groups |
| `--figures F...` | no | all | Subset: `matrix zoom locality correlation temporal latency tile_latency contention latency_over_minimum` |

It writes PNG files to `plots/communication/` and matching PDF files to `plots/communication/pdf/`:
  - `traffic_matrix`: zoomed active-groups rectangular heatmap
  - `traffic_matrix_groups`: group-level aggregate heatmap
  - `locality_overview`: remote fraction strip + per-group stacked bars + latency by distance
  - `comm_vs_stall`: scatter (non-local traffic vs LSU stalls) + temporal overlay
  - `temporal_overview`: stacked area + incoming heatmap + overall/local/same-group/remote latency over time
  - `latency_timeseries`: system-wide + per-group average latency
  - `latency_tile_g{N}`: per-tile latency within a group (G0 and G1)
  - `latency_matrix`: full tile-pair latency heatmap (green→yellow→red)
  - `latency_contention`: traffic volume vs latency scatter
  - `latency_excess_matrix`: latency heatmap normalized by ideal hierarchy minimum (local=1, same-group=3, remote=5)

File names include `_sectionN` when `--section` is used, and a `_<kernel>_<variant>` suffix derived from the result directory path.

### 11. `summarize_comm_timeseries.py` — Windowed Communication Timeseries

Run from `hardware/scripts/stall_analysis/` when you want communication data
that still keeps the time axis, so it can later be aligned with tile/core stall
plots.

```bash
python summarize_comm_timeseries.py ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline --benchmark-only --window 64
```

This derives the default input/output paths automatically:
  - input CSV: `<result_dir>/data/comm_events_benchmark.csv`
  - output dir: `<result_dir>/data/comm_timeseries/`

It writes:
  - `comm_timeseries_tiles.csv`: one row per section/window/tile
  - `comm_timeseries_edges.csv`: one row per section/window/source_tile/dest_tile
  - `comm_timeseries_metadata.json`: time-window and schema metadata for later plot alignment/splicing

The metadata file is the contract for later combined plots. It records:
  - cycle range
  - chosen window size
  - number of windows
  - tile list and section list
  - row granularity and field groups for both CSVs
  - the x-axis field to use when aligning with stall plots (`window_center_cycle`)

---

## Internal Scripts (Not User-Facing)

All internal scripts live in `stall_analysis/` with an underscore prefix.

| Script | Called by | What it does |
|--------|-----------|-------------|
| `rerun_stall_timeseries.py` | Makefile `rerun_stall_timeseries`, users | Public wrapper for safe stall CSV regeneration from `result_dir` |
| `extract_comm_events.py` | users | Public wrapper for building `comm_events_benchmark.csv` from `result_dir` |
| `extract_comm_events_batch.py` | users | Direct folder-based communication-event extraction |
| `summarize_comm_events.py` | users | First summary layer on top of `comm_events_benchmark.csv` |
| `plot_comm_thesis.py` | users, Makefile `plots_comm` | Thesis-quality communication figures (8 plot types, PNG+PDF) |
| `summarize_comm_timeseries.py` | users | Generate windowed communication timeseries CSVs + metadata for later combined plots |
| `_gen_stall_timeseries_batch.py` | Makefile `stall_timeseries`, `rerun_stall_timeseries.py` | Loops all traces, auto-loads topology metadata, calls `_gen_stall_timeseries.py` for each |
| `_gen_stall_timeseries.py` | `_gen_stall_timeseries_batch.py` | Single-trace → cycle-by-cycle stall rows in CSV |
| `_extract_comm_events.py` | `extract_comm_events_batch.py` | Single-trace → communication event rows in CSV |
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
| `extract_comm_events.py` / `extract_comm_events_batch.py` | Refuse if output CSV exists; reject missing or conflicting topology metadata | `--force`, `--topology`, or explicit env |
| `plot_all_tiles.py` | Skips tile if PNG already exists | `--force` |

---

## Quick Reference

```bash
# ── SIMULATE ──────────────────────────────────────────────────
cd hardware
app=matmul_i32 config=mempool kernel=2x2_xpulpv2 variant=baseline make benchmark

# ── PLOT ALL TILES ────────────────────────────────────────────
cd hardware
app=matmul_i32 config=mempool kernel=2x2_xpulpv2 variant=baseline make plots

# ── REBUILD ONLY THE STALL CSV ────────────────────────────────
result_dir=results/matmul_i32_mempool/2x2_xpulpv2/baseline make rerun_stall_timeseries force=1

# ── DRILL INTO CORE ──────────────────────────────────────────
cd scripts/stall_analysis
python plot_specific_core.py \
    ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline/data/stall_timeseries_benchmark.csv \
    42 --traces-dir ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline/traces \
    --section 1

# ── EXTRACT COMMUNICATION EVENTS ────────────────────────────
python extract_comm_events.py \
  ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline --force

# ── SUMMARIZE COMMUNICATION EVENTS ──────────────────────────
python summarize_comm_events.py \
  ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline --benchmark-only

# ── PLOT THESIS COMMUNICATION FIGURES ────────────────────────
# From hardware/scripts/stall_analysis/:
python plot_comm_thesis.py \
  ../../../results/matmul_i32_mempool/4x4/baseline --section 1
python plot_comm_thesis.py \
  ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline --section 1
# Or via Makefile (from hardware/):
# `make plots` now runs the communication extraction/summaries and then
# invokes `make plots_comm` unless you disable it with `plot_comm=0`.
app=matmul_i32 kernel=4x4 variant=baseline make plots
app=matmul_i32 kernel=4x4 variant=baseline make plots plot_comm=0
app=matmul_i32 kernel=4x4 variant=baseline make plots_comm
app=matmul_i32 kernel=2x2_xpulpv2 variant=baseline make plots_comm

# ── BUILD COMMUNICATION TIMESERIES FOR COMBINED PLOTS ───────
python summarize_comm_timeseries.py \
  ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline --benchmark-only --window 64
```

---

## Folder Structure

```
hardware/scripts/
  stall_analysis/                        ← this folder
    PIPELINE.md                          pipeline documentation (this file)
    plot_all_tiles.py                    user-facing: batch tile plotter
    plot_specific_core.py                user-facing: single-core drill-down
    extract_comm_events.py               user-facing: result_dir → comm_events CSV
    extract_comm_events_batch.py         user-facing: traces folder → comm_events CSV
    summarize_comm_events.py             user-facing: comm_events CSV → compact summaries
    plot_comm_thesis.py                  user-facing: thesis-quality communication figures (PNG+PDF)
    summarize_comm_timeseries.py         user-facing: comm_events CSV → windowed timeseries + metadata
    _plot_specific_tile.py               internal: per-tile 6-subplot detail
    _stall_plot_common.py                internal: shared plotting library
    _gen_stall_timeseries_batch.py       internal: batch trace→CSV wrapper
    _gen_stall_timeseries.py             internal: single-trace→CSV parser
    _extract_comm_events.py              internal: single-trace comm-event parser
  plotting/                              traffic-gen / port analysis (separate)
    _plotting_common.py                  internal: shared helpers
    plot_port_utilization.py             port utilization heatmaps
    plot_load_throughput.py              latency/throughput curves
    plot_reconstructed_load_throughput.py reconstructed latency/throughput
    compare_tilerange_plots.py           side-by-side comparison
```
