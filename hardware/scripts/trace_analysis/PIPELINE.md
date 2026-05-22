# Benchmark Pipeline

> **Note (2026-04-21):** Directory restructured from `stall_analysis/` (flat) to
> `trace_analysis/` with `extract/` and `plot/` subdirectories. Makefile paths
> updated. Standard benchmark output is now CSV-first: trace intermediates live
> under `build/` during Stage 1 and are cleaned after successful extraction
> unless you preserve them manually. The current thesis workflow defaults to
> `app=matmul_i32` and `variant=das`; the canonical kernel is
> `4x4_das_thesis_asm`.

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
make kernel=4x4_das_thesis_asm benchmark

make kernel=4x4_das_thesis_asm plots
```

For these workflow targets, the Makefile defaults to `app=matmul_i32`,
`variant=das`, and `config=mempool`. Set `config=terapool` for TeraPool runs,
or set `variant=baseline` only for an explicit baseline comparison.

By default, the Makefile uses `mempoolvenv/bin/python` for both the software
generation scripts and the communication-analysis scripts. You only need to
override `python` or `python_venv` if you deliberately want a different
environment.

What this gives you:

- `make benchmark`: raw `.dasm` provenance and temporary trace intermediates in `build/` during Stage 1, plus final `data/results.csv`, `data/comm_events_benchmark.csv`, `data/stall_timeseries_benchmark.csv`, the copied benchmark ELF, and reproducibility metadata in `result_dir/`
- `make plots`: tile/overview plots plus thesis communication figures in `plots/communication/`

Trace layers used in this document:

- raw trace: `build/*.dasm` from simulation
- canonical analysis trace: `build/traces/trace_hart_*` after `spike-dasm` during Stage 1
- reconstructed trace: `build/*.trace` produced later by `gen_trace.py` during Stage 1
- optional preserved traces: `result_dir/traces_raw/` and `result_dir/traces_dasm/` only when you explicitly keep them; older pre-rename runs may still use `real_traces/` and `traces/`

---

## Parallel Stage 2

The recommended minimal parallel workflow is to run Stage 2 for different
kernels in parallel, because each kernel writes to a separate result directory.
For the current single-kernel thesis flow, plain `make plots` is usually enough;
use `plots_parallel` only when you intentionally have multiple kernel result
directories to process.

Good:

- different kernel result directories in one `plots_parallel` invocation
- separate shell invocations for different configs or result families, because
  `plots_parallel` varies `kernel`, not `config`

Avoid:

- `make -j` inside a single communication extraction run
- parallelizing multiple writers into the same `result_dir`

Minimal knobs:

- `parallel_kernels="..."`: which kernels to run
- `parallel_jobs=N`: optional cap on how many kernels to run at once

Default behavior:

- if you omit `parallel_jobs`, `make plots_parallel` runs one job per kernel
- if you want less concurrency, set `parallel_jobs` explicitly

Example shape for the current thesis kernel:

Run only one kernel through the wrapper:

```bash
cd /home/bsc26f10/thesis/mempool/hardware

make \
  config=mempool \
  parallel_kernels="4x4_das_thesis_asm" \
  plots_parallel
```

For TeraPool, use a separate invocation with `config=terapool`:

```bash
cd /home/bsc26f10/thesis/mempool/hardware

make \
  config=terapool \
  parallel_kernels="4x4_das_thesis_asm" \
  plots_parallel
```

These invocations are safe because each one writes only to its own directory:

- `results/matmul_i32_mempool/4x4_das_thesis_asm/das/`
- `results/matmul_i32_terapool/4x4_das_thesis_asm/das/`

---

## Pipeline Tree

```
YOU RUN                          CALLED INTERNALLY                 OUTPUT
═══════                          ═════════════════                 ══════

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Simulation + Trace Generation                                      │
│                                                                             │
│  make benchmark                                                             │
│  ├── app=matmul_i32  config=mempool  kernel=4x4_das_thesis_asm          │
│  │   variant=das  [force=1]                                                 │
│  │                                                                          │
│  ├─► log                                                                    │
│  │     └── copies binary, env, config, topology.env, git-info → result_dir/│
│  │                                                                          │
│  ├─► simcvcs                                                                │
│  │     └── VCS simulation → build/*.dasm  (raw trace per hart)             │
│  │                                                                          │
│  ├─► trace                                                                  │
│  │     ├─► pre_trace       (cleans build/traces/)                          │
│  │     ├─► *.dasm→trace artifacts  (per hart, in parallel by Make)         │
│  │     │     ├── spike-dasm            .dasm → build/traces/trace_hart_*   │
│  │     │     │                         (canonical analysis traces)          │
│  │     │     ├── gen_trace.py          spike-dasm output → .trace +        │
│  │     │     │                         results.csv                          │
│  │     │     └── outdated_gen_timeseries_windowed.py                       │
│  │     │                            (optional, if timeline_window set)     │
│  │     └─► post_trace                                                      │
│  │           ├── merge_trace_results_csv.py → result_dir/data/results.csv  │
│  │           └── gen_avg.py                → result_dir/avg.txt            │
│  │                                                                          │
│  ├─► comm_events_real                                                       │
│  │     └── extract_comm_events_batch.py                                     │
│  │           build/traces/trace_hart_* → result_dir/data/comm_events_benchmark.csv │
│  │                                                                          │
│  └─► stall_timeseries_build                                                 │
│        └── _gen_stall_timeseries_batch.py                                   │
│              └── _gen_stall_timeseries.py  (×256 or ×1024, one per trace)  │
│              build/*.trace → result_dir/data/stall_timeseries_benchmark.csv│
│                                                                             │
│  Output:                                                                    │
│    result_dir/                                                              │
│      <app>, <app>.dump                                                     │
│      data/      results.csv, comm_events_benchmark.csv,                    │
│                 stall_timeseries_benchmark.csv                             │
│      avg.txt, transcript, config, env, topology.env, git-info.diff         │
│      (temporary trace intermediates remain in build/ and are cleaned)      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Plot Generation                                                    │
│                                                                             │
│  make plots                                                                 │
│  ├── app=... config=... kernel=... variant=...   or   result_dir=...       │
│  ├── [plot_section=1] [plot_tiles="..."] [plot_overview=1]                │
│  ├── [plot_group_details=0] [plot_tile_details=1]                           │
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
│  ├─► [--group-details]  group/subgroup detail pages                         │
│  │     └── _plot_specific_tile.py  (shared tile-detail visual style)        │
│  │     → result_dir/plots/group{N}/[subgroup{N}/]                          │
│  └─► per tile, unless plot_tile_details=0:                                 │
│        _plot_specific_tile.py <csv> <tile_id>                              │
│        └── _stall_plot_common.py  (shared helpers)                         │
│              └── locate_trace_file()  (finds trace in traces_dasm/)        │
│        (detail page only; no overview unless --overview is passed)         │
│        → result_dir/plots/group{N}/[subgroup{N}/]                          │
│                                                                             │
│  Output:                                                                    │
│    result_dir/plots/                                                        │
│      overview/             cluster overview and per-group breakdown PNGs     │
│      group0/               group-detail and tile PNGs for group 0          │
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
│        → per-core 3-subplot detail report                                   │
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
│        reads: data/comm_events_benchmark.csv                               │
│        computes summary & timeseries data inline from raw events           │
│        → result_dir/plots/communication/      (PNG)                        │
│        → result_dir/plots/communication/pdf/  (PDF)                        │
│                                                                             │
│  Output:                                                                    │
│    result_dir/plots/communication/                                          │
│      traffic_matrix_<kernel-tag>.png                                        │
│      traffic_matrix_groups_<kernel-tag>.png                                 │
│      request_pressure_by_tile_<kernel-tag>.png                              │
│      latency_timeseries_<kernel-tag>.png                                    │
│      latency_tile_g{N}_<kernel-tag>.png                                     │
│      latency_matrix_<kernel-tag>.png                                        │
│      latency_matrix_refined_<kernel-tag>.png                                │
│      latency_excess_matrix_<kernel-tag>.png                                 │
│      latency_excess_matrix_refined_<kernel-tag>.png                         │
│    result_dir/plots/communication/pdf/                                      │
│      (same stems as above, .pdf)                                            │
│    result_dir/plots/overview/                                               │
│      overview_temporal_<kernel-tag>.png  (from the temporal figure)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## User-Facing Commands

### 1. `make benchmark` — Simulate & Generate Data

Run from `hardware/`.

```bash
cd hardware
make kernel=4x4_das_thesis_asm benchmark
```

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `app` | no | `matmul_i32` | Application name for benchmark workflow targets |
| `kernel` | **yes** | — | Kernel result label; current thesis default is `4x4_das_thesis_asm` |
| `variant` | no | `das` | System variant: `baseline`, `das`, or `redmule` |
| `config` | no | `mempool` | Hardware topology: `mempool` (256 cores) or `terapool` (1024 cores) |
| `result_app` | no | derived from `app` | Logical result-family name used for the default result directory. In the standard workflow it normally matches `app`. |
| `result_dir` | no | `results/<result_app>_<config>/<kernel>/<variant>` | Override output path (bypasses kernel/variant checks) |
| `force` | no | — | Set `force=1` to allow overwriting existing results |

**Produces:**

```
result_dir/
  <app>         copied benchmark ELF
  <app>.dump    copied objdump for the benchmark ELF
  data/         results.csv, comm_events_benchmark.csv, stall_timeseries_benchmark.csv
  avg.txt       average performance stats per section
  transcript    simulation log
  config        snapshot of config.mk at build time
  env           environment variables at run time
  topology.env  exact topology used for post-processing
  git-info.diff source state at run time
```

`comm_events_benchmark.csv` and `stall_timeseries_benchmark.csv` may be
scratch-backed symlinks depending on `result_data_storage`.

### 2. `make plots` — Public Plotting Entry Point

Run from `hardware/`.

```bash
cd hardware
make kernel=4x4_das_thesis_asm plots
```

This wraps `plot_all_tiles.py` so users do not need to remember the script
path, result directory layout, or standard benchmark flags.

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `result_dir` | no | `results/<result_app>_<config>/<kernel>/<variant>` | Existing result directory to plot |
| `plot_section` | no | `1` | Section to plot |
| `plot_overview` | no | `1` | Generate the cluster overview too |
| `plot_group_details` | no | `0` | Generate group/subgroup detail pages using the tile-detail visual style, with one heatmap row per tile |
| `plot_tile_details` | no | `1` | Generate per-tile detail pages; set `0` to skip them |
| `plot_window` | no | `64` | Sliding-window width (cycles) |
| `plot_tiles` | no | all | Optional tile list, e.g. `plot_tiles="0 1 2"` |
| `plot_topology` | no | auto | Override topology detection if needed |
| `force` | no | off | Overwrite existing PNGs |

Important default behavior:

- `config=terapool kernel=4x4_das_thesis_asm make benchmark` writes by default
  to `results/matmul_i32_terapool/4x4_das_thesis_asm/das/`
- baseline and DAS matmul runs use the same `matmul_i32` app entry point;
  `variant` and `das` control the memory-placement mode
- direct `das=0` RTL-only commands should use `variant=baseline`; the default
  `variant=das` requires `das=1`

Important behavior:

- `make benchmark` creates both `data/comm_events_benchmark.csv` and `data/stall_timeseries_benchmark.csv` during Stage 1.
- `make plots` requires those CSVs to already exist.
- `force=1` affects plot outputs only; it does not rebuild communication CSVs.
- If `comm_events_benchmark.csv` is missing, `make plots` fails. The normal recovery path is to rerun `make benchmark`.
- `extract_comm_events.py` is only useful when you have a manually preserved or legacy `result_dir` that still contains `traces_raw/`.

### 3. `plot_all_tiles.py` — Batch Plot Generation

Run from `hardware/scripts/trace_analysis/plot/`.

```bash
cd hardware/scripts/trace_analysis/plot
python plot_all_tiles.py ../../../../results/matmul_i32_mempool/4x4_das_thesis_asm/das --section 1 --overview
```

| Flag | Required | Default | Purpose |
|------|----------|---------|---------|
| `result_dir` (positional) | **yes** | — | Path to variant directory |
| `--section N` | recommended | all | Filter by section (repeatable). Use `--section 1` for benchmark. |
| `--topology` | no | auto-detected | Force `mempool` or `terapool` (normally loaded from `topology.env`) |
| `--tiles T...` | no | all from CSV | Only plot specific tile IDs |
| `--overview` | no | off | Also generate cluster overview page |
| `--group-details` | no | off | Generate group/subgroup detail pages with one tile row per page |
| `--skip-tile-details` | no | off | Skip per-tile detail pages |
| `--window N` | no | 64 | Sliding-window width (cycles) for timeseries aggregation |
| `--force` | no | off | Overwrite existing PNGs (default: skip existing) |
| `--dry-run` | no | off | Print actions without executing |

### 4. `make rerun_stall_timeseries` — Public Reprocessing Entry Point

Run from `hardware/`.

```bash
cd hardware
result_dir=results/matmul_i32_mempool/4x4_das_thesis_asm/das make rerun_stall_timeseries force=1
```

This is the public reprocessing command when you want to rebuild only the
stall CSV from an existing `result_dir` without re-running simulation.
It requires a result directory that still contains preserved `traces_dasm/`.

### 5. `rerun_stall_timeseries.py` — Direct Wrapper

Run from `hardware/scripts/trace_analysis/extract/`.

```bash
python rerun_stall_timeseries.py ../../../../results/matmul_i32_mempool/4x4_das_thesis_asm/das --force
```

This wraps `_gen_stall_timeseries_batch.py` and derives `traces_dasm/` and the
default output CSV from `result_dir` automatically.

### 6. `plot_specific_core.py` — Single-Core Drill-Down (Ad-Hoc)

Run from `hardware/scripts/trace_analysis/plot/`. For investigating a specific core spotted in a tile plot.

```bash
python plot_specific_core.py \
  ../../../../results/matmul_i32_mempool/4x4_das_thesis_asm/das/data/stall_timeseries_benchmark.csv \
    42 \
    --section 1
```

| Flag | Required | Default | Purpose |
|------|----------|---------|---------|
| `csv` (positional) | **yes** | — | Path to stall_timeseries_benchmark.csv |
| `core` (positional) | **yes** | — | Core ID(s) to plot |
| `--traces-dir` | no | ignored | Deprecated compatibility flag from the old trace-backed workflow |
| `--section N` | recommended | all | Filter by section |
| `--output-dir` | no | `<csv-dir>/plots` | Where to save PNGs |

### 7. `extract_comm_events.py` — Communication Event Extraction

Run from `hardware/scripts/trace_analysis/extract/` when you want a source/destination
event CSV for later communication analysis.

```bash
python extract_comm_events.py ../../../../results/matmul_i32_mempool/4x4_das_thesis_asm/das --force
```

This derives:
  - canonical analysis trace folder: `<result_dir>/traces_raw`
  - default output: `<result_dir>/data/comm_events_benchmark.csv`

Important:
  - standard `make benchmark` result directories do **not** include `traces_raw/` by default
  - this wrapper is therefore mainly for legacy or manually preserved result directories
  - if your standard result dir only has the CSVs, rerun `make benchmark` rather than expecting `extract_comm_events.py` to recover them

Useful flags:
  - `--force`: overwrite an existing communication CSV

### 8. `extract_comm_events_batch.py` — Direct Batch Extraction

Run from `hardware/scripts/trace_analysis/extract/` when you want to point directly at
trace folders instead of a full `result_dir`.

```bash
python extract_comm_events_batch.py \
  --folder ../../../../results/matmul_i32_mempool/4x4_das_thesis_asm/das/traces_raw \
  --csv ../../../../results/matmul_i32_mempool/4x4_das_thesis_asm/das/data/comm_events_benchmark.csv \
    --benchmark-only --force
```

The normal `--folder` input is a canonical-analysis-trace folder containing
`trace_hart_*` files from `spike-dasm`, usually either `build/traces/` during
Stage 1 or a manually preserved `result_dir/traces_raw/`. Legacy reconstructed
`trace_hart_*.trace` files may still parse for manual recovery, but they are
decommissioned from the standard pipeline.

### 9. `plot_comm_thesis.py` — Thesis-Quality Communication Figures

Run from `hardware/scripts/trace_analysis/plot/` after communication extraction.
Generates 7 communication plot families.
On standard 4-group MemPool/TeraPool runs, that yields 13 PNG outputs plus 13
matching PDF files because `tile_latency` emits one figure per group.
All summary and timeseries data is computed inline from the raw events CSV.

```bash
# All figures for the current thesis kernel:
python plot_comm_thesis.py ../../../../results/matmul_i32_mempool/4x4_das_thesis_asm/das --section 1

# Specific figures only:
python plot_comm_thesis.py ../../../../results/matmul_i32_mempool/4x4_das_thesis_asm/das --section 1 \
  --figures matrix latency_matrix latency_over_minimum
```

This derives the default input/output paths automatically:
  - input events CSV: `<result_dir>/data/comm_events_benchmark.csv`
  - output PNG dir: `<result_dir>/plots/communication/`
  - output PDF dir: `<result_dir>/plots/communication/pdf/`

| Flag | Required | Default | Purpose |
|------|----------|---------|----------|
| `input_path` (positional) | **yes** | — | Result directory (contains `data/` and `plots/`) |
| `--section N` | recommended | all | Filter by section |
| `--n-groups N` | no | 4 | Number of tile groups |
| `--figures F...` | no | all | Subset: `matrix pressure temporal latency tile_latency latency_matrix latency_over_minimum` |

It writes PNG files to `plots/communication/` and matching PDF files to `plots/communication/pdf/`:
  - `traffic_matrix`: zoomed active-groups rectangular heatmap
  - `traffic_matrix_groups`: group-level aggregate heatmap
  - `request_pressure_by_tile`: row/column reductions of the tile request matrix, showing outgoing source pressure and incoming destination pressure per tile
  - `overview_temporal`: stacked area + incoming heatmap + overall/local/same-group/remote latency over time, written under `plots/overview/`
  - `latency_timeseries`: system-wide + per-group average latency
  - `latency_tile_g{N}`: per-tile latency within a group (typically G0-G3)
  - `latency_matrix`: full tile-pair latency heatmap (green→yellow→red)
  - `latency_matrix_refined`: latency heatmap with outlier-robust colour scale
  - `latency_excess_matrix`: latency heatmap normalized by topology-aware ideal hierarchy minimum (MemPool: local=1, same-subgroup=3, same-group=3, remote=5; TeraPool: local=1, same-subgroup=3, same-group=5, remote=7)
  - `latency_excess_matrix_refined`: excess-latency heatmap with outlier-robust colour scale

File names include a kernel-tag suffix derived from the result directory path.
`--section` filters the data but is not encoded in the current output filename.

---

## Internal Scripts (Not User-Facing)

All internal scripts live in `trace_analysis/extract/` and `trace_analysis/plot/` with an underscore prefix.

| Script | Called by | What it does |
|--------|-----------|-------------|
| `rerun_stall_timeseries.py` | Makefile `rerun_stall_timeseries`, users | Public wrapper for safe stall CSV regeneration from `result_dir` |
| `extract_comm_events.py` | users | Public wrapper for building `comm_events_benchmark.csv` from `result_dir` |
| `extract_comm_events_batch.py` | users | Direct folder-based communication-event extraction |
| `plot_comm_thesis.py` | users, Makefile `plots_comm` | Thesis-quality communication figures (7 plot families; typically 13 PNG+PDF outputs on 4-group runs). Computes summaries and timeseries inline from raw events. |
| `_gen_stall_timeseries_batch.py` | Makefile `stall_timeseries`, `rerun_stall_timeseries.py` | Loops all traces, auto-loads topology metadata, calls `_gen_stall_timeseries.py` for each |
| `_gen_stall_timeseries.py` | `_gen_stall_timeseries_batch.py` | Single-trace → cycle-by-cycle stall rows in CSV |
| `_extract_comm_events.py` | `extract_comm_events_batch.py` | Single-trace → communication event rows in CSV |
| `_plot_specific_tile.py` | `plot_all_tiles.py` | Cluster overview, group/subgroup detail pages, and single-tile detail pages |
| `_stall_plot_common.py` | `_plot_specific_tile.py`, `plot_specific_core.py` | Shared helpers: data loading, trace lookup, plot formatting |

Upstream scripts called by the Makefile (not in this folder):

| Script | Called by | What it does |
|--------|-----------|-------------|
| `gen_trace.py` | Makefile `%.trace` rule | Parses spike-dasm output → reconstructed `.trace` + `results.csv` |
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

**Important:** New benchmark runs write `result_dir/topology.env`, but the
standard benchmark flow does not archive `traces_dasm/` by default.
`_gen_stall_timeseries_batch.py` uses the metadata automatically when you point
it at a preserved `result_dir/traces_dasm` and `result_dir/data/...csv`.

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
| `make benchmark` | Defaults to `variant=das`; rejects inconsistent explicit `das` values | Set `variant=baseline` for `das=0`, or keep `variant=das` for `das=1` |
| `make benchmark` | Refuses if `result_dir/data/` already contains benchmark CSVs | `force=1` |
| `_gen_stall_timeseries_batch.py` | Refuses if output CSV exists; rejects missing or conflicting topology metadata | `--force`, `--topology`, or explicit env |
| `extract_comm_events.py` / `extract_comm_events_batch.py` | Refuse if output CSV exists; reject missing or conflicting topology metadata | `--force`, `--topology`, or explicit env |
| `make plots` | Requires existing stall and communication CSVs; `force=1` only overwrites plot outputs | Rerun `make benchmark`, or run `extract_comm_events.py` only on preserved `traces_raw/` |
| `extract_comm_events.py` | Requires preserved canonical analysis traces in `traces_raw/`; reconstructed traces are decommissioned | Use a legacy/manual-preserved result dir, or rerun `make benchmark` |
| `plot_all_tiles.py` | Skips tile if PNG already exists | `--force` |

---

## Quick Reference

```bash
# ── SIMULATE ──────────────────────────────────────────────────
cd hardware
make kernel=4x4_das_thesis_asm benchmark

# ── PLOT ALL TILES ────────────────────────────────────────────
cd hardware
make kernel=4x4_das_thesis_asm plots

# ── REBUILD ONLY THE STALL CSV ────────────────────────────────
result_dir=results/matmul_i32_mempool/4x4_das_thesis_asm/das make rerun_stall_timeseries force=1

# ── DRILL INTO CORE ──────────────────────────────────────────
cd scripts/trace_analysis/plot
python plot_specific_core.py \
  ../../../../results/matmul_i32_mempool/4x4_das_thesis_asm/das/data/stall_timeseries_benchmark.csv \
  42 --section 1

# ── EXTRACT COMMUNICATION EVENTS ────────────────────────────
python extract_comm_events.py \
  ../../../../results/matmul_i32_mempool/4x4_das_thesis_asm/das --force

# ── REBUILD COMMUNICATION DATA EXPLICITLY ───────────────────
python extract_comm_events.py \
  ../../../../results/matmul_i32_mempool/4x4_das_thesis_asm/das --force

# ── PLOT THESIS COMMUNICATION FIGURES ────────────────────────
# From hardware/scripts/trace_analysis/plot/:
python plot_comm_thesis.py \
  ../../../../results/matmul_i32_mempool/4x4_das_thesis_asm/das --section 1
# Or via Makefile (from hardware/):
# `make plots` uses the existing communication CSV and then invokes
# `make plots_comm` unless you disable it with `plot_comm=0`.
make kernel=4x4_das_thesis_asm plots
make kernel=4x4_das_thesis_asm plots plot_comm=0
make kernel=4x4_das_thesis_asm plots_comm
```

---

## Folder Structure

```
hardware/scripts/
  trace_analysis/                        ← this folder
    _workflow_metadata.py                shared: topology resolution & path helpers
    PIPELINE.md                          pipeline documentation (this file)
    extract/                             Stage 1: traces → CSV
      extract_comm_events.py             user-facing: result_dir → comm_events CSV
      extract_comm_events_batch.py       user-facing: traces folder → comm_events CSV
      rerun_stall_timeseries.py          user-facing: rebuild stall CSV from result_dir
      _extract_comm_events.py            internal: single-trace comm-event parser
      _gen_stall_timeseries_batch.py     internal: batch trace→CSV wrapper
      _gen_stall_timeseries.py           internal: single-trace→CSV parser
    plot/                                Stage 2: CSV → PNG/PDF
      plot_all_tiles.py                  user-facing: batch tile plotter
      plot_specific_core.py              user-facing: single-core drill-down
      plot_comm_thesis.py                user-facing: thesis-quality communication figures (PNG+PDF)
      _plot_specific_tile.py             internal: per-tile 4-subplot detail
      _stall_plot_common.py              internal: shared plotting library
  plotting/                              traffic-gen / port analysis (separate)
    _plotting_common.py                  internal: shared helpers
    plot_port_utilization.py             port utilization heatmaps
    plot_load_throughput.py              latency/throughput curves
    plot_reconstructed_load_throughput.py reconstructed latency/throughput
    compare_tilerange_plots.py           side-by-side comparison
```
