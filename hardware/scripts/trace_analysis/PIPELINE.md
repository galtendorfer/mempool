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

## Monitor Evidence Workflow

This is the optional evidence flow used when the normal benchmark/plot outputs
show an IPC or stall change, but you need to explain the hardware-side request
path. It is CSV-first: run a monitor-enabled benchmark, normalize the monitor
CSVs once, then generate focused plots from the normalized `analysis/path_graph`
dataset.

Monitor dependency map:

| Dependency | Scripts | Monitor required? |
|------------|---------|-------------------|
| Standard benchmark CSVs only | `plot_all_tiles.py`, `plot_specific_core.py`, `plot_comm_thesis.py`, `rerun_stall_timeseries.py`, `extract_comm_events.py`, `extract_comm_events_batch.py` | No |
| Compact source-port monitor CSVs in `<result_dir>/monitor/` | `path_graph_dataset.py --point tcdm_remote --no-html` | Yes |
| Compact normalized `analysis/path_graph` dataset | `classify_source_targets.py`, `plot_port_pressure.py`, `plot_port_fanin_heatmap.py`, `plot_fanin_flow_overlay.py` | Yes, via `path_graph_dataset.py` |
| Full path monitor dataset | `path_route_checkpoints.py` | Yes, full tile/group path probes required |
| Source-target classifier CSVs | `plot_source_target_classification.py`, `scripts/port_pressure/plot_source_target_fanin_comparison.py` | Yes, via `classify_source_targets.py` |

In short: the normal benchmark and thesis communication plots do not need the
monitor. The port-pressure, fan-in, and source-target tools only need compact
source-port monitor data. Route-checkpoint drill-downs still need the full path
monitor.

Current thesis evidence usually uses all-tile, load-only, source-port monitor
capture over a bounded window:

```bash
cd /home/bsc26f10/thesis/mempool/hardware

make \
  app=matmul_i32 \
  config=mempool \
  kernel=4x4_das_thesis_asm \
  variant=das \
  result_dir=results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000 \
  tile_path_monitor=1 \
  path_util_start=10000 \
  path_util_end=16000 \
  tile_path_active_only=1 \
  tile_path_load_only=1 \
  tile_path_source_ports_only=1 \
  benchmark
```

For a Back2Local comparison, add `back2local=1` and write to a separate
`result_dir`, for example `.../das_back2local_sourceport_loadmon_alltiles_c10000_16000`.

The compact source-port monitor records both the transformed route address
(`addr`) and original source/core address (`source_addr`). Operand labels such
as A/B/C must be derived from `source_addr` plus operand-region metadata when
you need exact labels; target tile/group decoding still uses the route `addr`.
Back2Local runs are decoded with the per-row `back2local` bit, so same-group
rerouted requests are not misclassified as remote-group targets.

The benchmark binary is not modified to print these regions. For thesis-grade
operand plots, provide out-of-band metadata in
`<result_dir>/analysis/operand_regions.json` or pass explicit CLI ranges. The
sidecar format is:

```json
{
  "address_field": "source_addr",
  "regions": [
    {"name": "A", "start": "0x80000", "end": "0x87fff"},
    {"name": "B", "start": "0x88000", "end": "0x8ffff"}
  ]
}
```

Monitor benchmark flags:

| Flag | Meaning |
|------|---------|
| `tile_path_monitor=1` | Enable tile-level request/response monitor CSV capture. |
| `path_util_start`, `path_util_end` | Inclusive cycle window; tile monitor uses these as the default start/end if tile-specific values are not provided. |
| `tile_path_active_only=1` | Emit only active tile monitor rows instead of every idle probe row. |
| `tile_path_load_only=1` | Keep tile-level monitor output focused on valid load traffic. |
| `tile_path_source_ports_only=1` | Emit only source-side `tcdm_remote` route-port rows; this is the compact mode for port-pressure/fan-in/source-target plots. |
| `tile_path_tile=<T>` | Optional single-tile capture filter; omit for all-tile evidence. |
| `path_util_monitor=1` | Optional full-path debug mode: also emit group-level path-util lane CSVs for route-checkpoint drill-downs. |
| `path_util_load_only=1` | In full-path debug mode, keep group-level monitor output focused on valid load traffic. |
| `result_dir=<path>` | Explicit result directory, important for keeping compared runs separate. |
| `back2local=1` | Enable Back2Local routing for the comparison run. |

Normalize monitor CSVs into graph-time tables:

```bash
python scripts/trace_analysis/path_graph_dataset.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000 \
  --point tcdm_remote \
  --cycle-start 10000 \
  --cycle-end 16000 \
  --no-html
```

Generate the all-port pressure guardrail, then the port-0 fan-in views from the
normalized dataset:

```bash
python scripts/trace_analysis/plot/plot_port_pressure.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph \
  --metric-source node-state \
  --node-point tcdm_remote \
  --cycle-start 10000 \
  --cycle-end 16000 \
  --window 40 \
  --prefix all_ports_pressure \
  --force

python scripts/trace_analysis/plot/plot_port_fanin_heatmap.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph \
  --cycle-start 10000 \
  --cycle-end 16000 \
  --window 40 \
  --port 0 \
  --prefix port0_tile_fanin \
  --require-exact-operands \
  --force

python scripts/trace_analysis/plot/plot_fanin_flow_overlay.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph \
  --cycle-start 10000 \
  --cycle-end 16000 \
  --window 40 \
  --focus-tile auto \
  --threshold 3 \
  --prefix all_ports_fanin_flow \
  --require-exact-operands \
  --force
```

Classify source/target traffic and plot the mechanism:

```bash
python scripts/trace_analysis/classify_source_targets.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph \
  --cycle-start 10000 \
  --cycle-end 16000 \
  --port 0 \
  --node-point tcdm_remote \
  --prefix port0_source_target \
  --require-exact-operands \
  --force

python scripts/trace_analysis/plot/plot_source_target_classification.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph/port0_source_target_classification/port0_source_target_source_target_matrix.csv \
  --prefix port0_source_target \
  --force

python scripts/port_pressure/plot_source_target_fanin_comparison.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph/port0_source_target_classification/port0_source_target_tile_cycles.csv \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_back2local_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph/port0_source_target_classification/port0_source_target_tile_cycles.csv \
  --label DAS \
  --label Back2Local \
  --output-dir results/matmul_i32_mempool/4x4_das_thesis_asm/das_back2local_sourceport_loadmon_alltiles_c10000_16000/plots/port_pressure \
  --prefix port0_fanin_mechanism \
  --force
```

Default monitor-evidence plot outputs go to `<result_dir>/plots/port_pressure/`
when the input is a normal `<result_dir>/analysis/path_graph` dataset.
PNG figures stay in the selected plot directory, matching PDFs are written to a
`pdf/` subdirectory, and derived plot CSV/TXT support files are written to a
`data/` subdirectory.

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
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2d: Monitor graph-time dataset (ad-hoc evidence runs)                 │
│                                                                             │
│  path_graph_dataset.py <result_dir-with-monitor/>                          │
│  ├── --cycle-start N --cycle-end M                                          │
│  ├── [--tile T] [--group G] [--window-size N]                              │
│  └── reads monitor/tile_path_tile*.csv and monitor/path_util_group*.csv    │
│                                                                             │
│  Output:                                                                    │
│    result_dir/analysis/path_graph/                                          │
│      nodes.csv, lanes.csv, edges.csv                                        │
│      cycle_node_state.csv, cycle_lane_state.csv                             │
│      cycle_summary.csv, subject_summary.csv, window_summary.csv             │
│      path_timeline.html, schema.md, manifest.txt                            │
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

### 10. `path_graph_dataset.py` — Monitor Graph-Time Dataset

Run from `hardware/` on a result directory that contains `monitor/` CSVs.
For the current thesis port-pressure flow, normalize only source-side route-port
rows and skip the optional HTML timeline:

```bash
python scripts/trace_analysis/path_graph_dataset.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000 \
  --point tcdm_remote \
  --cycle-start 10000 \
  --cycle-end 16000 \
  --no-html
```

Use the full path monitor only when you need node/path state per cycle, idle
lanes, or route-checkpoint evidence.

Useful full-path scoped version while iterating:

```bash
python scripts/trace_analysis/path_graph_dataset.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_tilemon_t0_c8306_8600 \
  --cycle-start 8320 \
  --cycle-end 8383 \
  --tile 3 \
  --group 0
```

Single-group outgoing request view:

```bash
python scripts/trace_analysis/path_graph_dataset.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_tilemon_t0_c8306_8600 \
  --cycle-start 8320 \
  --cycle-end 8383 \
  --tile-group 0 \
  --group 0 \
  --outgoing-only
```

| Output | Purpose |
|--------|---------|
| `nodes.csv` | Static tile-level monitor subjects, keyed by tile/point/core/port/bank |
| `lanes.csv` | Static group path-util lane subjects, keyed by group/tile/port/channel/stage |
| `edges.csv` | Inferred adjacency for route and checkpoint tooling |
| `cycle_node_state.csv` | One row per observed tile node per cycle, with `state` from valid/ready/fire; source-port rows include route `addr` and, for new runs, original `source_addr` |
| `cycle_lane_state.csv` | One row per observed group lane stage per cycle |
| `cycle_summary.csv` | Per-cycle aggregate pressure for all subjects, tile nodes, and group lanes |
| `subject_summary.csv` | Whole-window aggregate per node/lane, sorted by stall pressure |
| `window_summary.csv` | Per-window aggregate per node/lane for zooming into hot regions |
| `path_timeline.html` | Optional compact heatmap of the highest-stall subjects; skipped with `--no-html` |

`subject_id` values are stable machine keys for joining CSVs. The `label`
column and optional timeline use the human-readable form
`G#/T## abs### | node/lane ... | qualifiers`.

Useful filters:

- `--tile T`: include only specific tile monitor files
- `--tile-group G`: include tile monitor nodes from one group
- `--point P`: include selected tile monitor points such as `tile_master_req_out`
- `--group G`: include path-util lanes from one group
- `--lane-channel C`: include `req` or `resp` lanes
- `--lane-stage S`: include lane stages such as `out`, `in0`, or `post0`
- `--outgoing-only`: shortcut for `tile_master_req_out` plus `req/out` group lanes

Key flags:

| Flag | Meaning |
|------|---------|
| `input_path` | Result directory containing `monitor/`, a monitor directory, or a previously scoped monitor input. |
| `--output-dir DIR` | Override the default `<result_dir>/analysis/path_graph` output directory. |
| `--cycle-start N`, `--cycle-end M` | Inclusive cycle filter for all generated graph-time tables. |
| `--window-size N` | Aggregation window for `window_summary.csv`; default is 64 cycles. |
| `--tile`, `--tile-group`, `--point` | Scope tile-monitor node rows by absolute tile, tile group, or monitor point. |
| `--group`, `--lane-channel`, `--lane-stage` | Scope group path-util lane rows. |
| `--outgoing-only` | Shortcut for an outgoing-request view: `tile_master_req_out` plus `req/out` lanes. |
| `--no-html` | Skip `path_timeline.html` when only CSV outputs are needed. |

State meanings:

- `flow`: `valid=1`, `ready=1`, `fire=1`
- `blocked`: `valid=1`, `ready=0`, `fire=0`; direct backpressure
- `idle_ready`: `valid=0`, `ready=1`, `fire=0`; available but unused
- `inactive`: `valid=0`, `ready=0`, `fire=0`
- `valid_no_fire`: valid and ready were asserted but no fire was observed
- `mixed_blocked_flow`: aggregate row containing both blocked and firing observations

This tool normalizes monitor observations only. It does not prove an exact
request identity through every merge/split unless the probe metadata is enough
to correlate it; use route `addr`, original `source_addr`, `meta_id`, and
`payload_core` in the detailed cycle tables for those manual drill-downs.

### 11. `path_route_checkpoints.py` — Route Checkpoint CSVs

Run from `hardware/` on a normalized `path_graph_dataset.py` output directory.
This full-path drill-down tool writes detailed checkpoint rows plus a compact
route-bottleneck summary that can be plotted with `plot_port_pressure.py`. It
requires full path-monitor data; compact `tcdm_remote`-only datasets do not
contain enough downstream checkpoints.

```bash
python scripts/trace_analysis/path_route_checkpoints.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_fullpath_loadmon_alltiles_c10000_16000/analysis/path_graph \
  --tile 46 \
  --port 0 \
  --cycle-start 10000 \
  --cycle-end 10080 \
  --output results/matmul_i32_mempool/4x4_das_thesis_asm/das_fullpath_loadmon_alltiles_c10000_16000/analysis/path_graph/plots/support_examples_2026_06_02/route_checkpoints_tile46_port0_c10000_10080.csv \
  --summary-output results/matmul_i32_mempool/4x4_das_thesis_asm/das_fullpath_loadmon_alltiles_c10000_16000/analysis/path_graph/plots/support_examples_2026_06_02/route_bottlenecks_tile46_port0_c10000_10080.csv
```

Useful follow-up plot:

```bash
python scripts/trace_analysis/plot/plot_port_pressure.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_fullpath_loadmon_alltiles_c10000_16000/analysis/path_graph/plots/support_examples_2026_06_02/route_bottlenecks_tile46_port0_c10000_10080.csv \
  --metric-source route-summary \
  --cycle-start 10000 \
  --cycle-end 10080 \
  --window 10 \
  --port 0 \
  --output-dir results/matmul_i32_mempool/4x4_das_thesis_asm/das_fullpath_loadmon_alltiles_c10000_16000/analysis/path_graph/plots/support_examples_2026_06_02 \
  --prefix route_checkpoint_tile46_port0_c10000_10080 \
  --force
```

When `plot_port_pressure.py` is run on a normal result directory or its
`analysis/path_graph` dataset without `--output-dir`, figures are written to
`<result_dir>/plots/port_pressure/`.

Default output names, when explicit paths are not supplied:

```
<path_graph_dataset_dir>/route_checkpoints_all_tiles.csv
<path_graph_dataset_dir>/route_bottlenecks_all_tiles.csv
```

Key flags:

| Flag | Meaning |
|------|---------|
| `graph_dir` | Normalized `analysis/path_graph` directory from `path_graph_dataset.py`. |
| `--tile T` | Restrict to one or more source tiles; comma-separated and repeatable. |
| `--core C` / `--source C` | Restrict to one local source core. |
| `--port P` | Restrict to route port(s); comma-separated and repeatable. |
| `--cycle-start N`, `--cycle-end M` | Inclusive cycle filter. |
| `--output PATH` | Detailed checkpoint CSV output path. |
| `--summary-output PATH` | Compact route-bottleneck summary CSV output path. |
| `--include-idle-lanes` | Include idle same-port group lanes in the detailed CSV. |
| `--target-window N` | Look-ahead window, in cycles, for decoded target-side evidence. |

### 12. `classify_source_targets.py` + source-target plotters — Source-Target Classification

Run from `hardware/` on a normalized `path_graph_dataset.py` output directory.
The classifier materializes source tile, source core, operand class, route port,
and decoded target tile. The companion plotter renders that classifier matrix as
request/stall heatmaps plus traffic-class summaries. The fan-in comparison
plotter compares classifier tile-cycle CSVs across runs to show whether source
pressure comes from single-source traffic or multi-source fan-in cycles.

Operand labels are exact when the classifier has `source_addr` in
`cycle_node_state.csv` and source-address operand ranges from
`analysis/operand_regions.json`, `--operand-regions-json`, `--a-range`,
`--b-range`, `--c-range`, or repeated `--operand-region NAME:START:END`
arguments. If explicit metadata is unavailable, operands remain `other` by
default. The old route-address A/B fallback is available only with
`--allow-legacy-route-operands` and should be treated as legacy/debug evidence,
not thesis evidence. Transcript parsing remains supported for debug runs that
already contain `MATMUL_I32_REGION` lines, but the benchmark-preserving workflow
does not print those lines from the application.

```bash
python scripts/trace_analysis/classify_source_targets.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph \
  --cycle-start 10000 \
  --cycle-end 10080 \
  --port 0 \
  --node-point tcdm_remote \
  --require-exact-operands \
  --output-dir results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph/plots/support_examples_2026_06_02/source_target_port0_c10000_10080 \
  --prefix support_port0_source_target \
  --force

python scripts/trace_analysis/plot/plot_source_target_classification.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph/plots/support_examples_2026_06_02/source_target_port0_c10000_10080/support_port0_source_target_source_target_matrix.csv \
  --output-dir results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph/plots/support_examples_2026_06_02/source_target_port0_c10000_10080/plots \
  --prefix support_port0_source_target_c10000_10080 \
  --force

python scripts/port_pressure/plot_source_target_fanin_comparison.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph/port0_source_target_classification/port0_source_target_tile_cycles.csv \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_back2local_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph/port0_source_target_classification/port0_source_target_tile_cycles.csv \
  --label DAS \
  --label Back2Local \
  --output-dir results/matmul_i32_mempool/4x4_das_thesis_asm/das_back2local_sourceport_loadmon_alltiles_c10000_16000/plots/port_pressure \
  --force

# Single benchmark operand-offset view. Outputs go to the benchmark's
# `plots/port_pressure/` directory by default.
python scripts/trace_analysis/plot/plot_source_target_operand_offsets.py \
  results/test_2026_06_02/b2l_4x4 \
  --label b2l_4x4 \
  --prefix port0_operand_offsets \
  --force

# Multi-benchmark comparison view. Outputs go to the common sweep plot
# directory by default.
python scripts/trace_analysis/plot/plot_source_target_operand_offsets.py \
  results/test_2026_06_02/b2l_4x4 \
  results/test_2026_06_02/b2l_thesis \
  results/test_2026_06_02/das_thesis \
  results/test_2026_06_02/das_4x4 \
  --label b2l_4x4 \
  --label b2l_thesis \
  --label das_thesis \
  --label das_4x4 \
  --prefix port0_operand_offset_sweep \
  --force
```

Classifier outputs:

- `<prefix>_details.csv`
- `<prefix>_tile_cycles.csv`
- `<prefix>_source_target_matrix.csv`
- `<prefix>_target_tile_in_group.csv`
- `<prefix>_summary.csv`
- `<prefix>_operand_region_audit.csv`

Plotter outputs:

- `<prefix>_source_target_requests.png` and `pdf/<prefix>_source_target_requests.pdf`
- `<prefix>_source_target_stalls.png` and `pdf/<prefix>_source_target_stalls.pdf`
- `<prefix>_traffic_class_by_signed_offset.png` and `pdf/<prefix>_traffic_class_by_signed_offset.pdf` as a multi-row raw signed-offset traffic-class view, with valid, accepted, blocked, blocked-share, and high-fanin rows when high-fanin events exist. Pass `--offset-mode wrapped` for the older shortest-offset view, or `--combine` on a path-graph directory to combine multiple port classifier matrices.
- `<prefix>_request_class_totals.png` and `pdf/<prefix>_request_class_totals.pdf`
- `plot_source_target_operand_offsets.py`: `<prefix>.png` and `pdf/<prefix>.pdf` for the combined operand-offset view: accepted requests, blocked request-cycles, and normalized per-operand offset shape. It also emits `<prefix>_a_neighbor_source_tiles.png` plus matching PDF/CSV to show operand-A `+1`/`-1` neighbor traffic by source-target tile pair. `data/<prefix>.csv` contains the underlying offset shares/counts, including accepted requests per observed source tile. Accepted requests are `fire` handshakes, while blocked traffic is counted as stalled request-cycles.
- `plot_source_target_fanin_comparison.py`: `<prefix>_comparison.png`, `pdf/<prefix>_comparison.pdf`, and `data/<prefix>_comparison.csv` for fan-in bucket share and stall-rate comparison across runs

For matrices under a normal `analysis/path_graph` dataset, the plotter defaults
to `<result_dir>/plots/port_pressure/` unless `--output-dir` is supplied.

Classifier flags:

| Flag | Meaning |
|------|---------|
| `input_path` | `analysis/path_graph`, `cycle_node_state.csv`, or a result directory containing `analysis/path_graph`. |
| `--cycle-start N`, `--cycle-end M` | Inclusive cycle filter. |
| `--port P` | Route port to classify; port 0 is the current source-pressure focus. |
| `--node-point NAME` | Monitor point to classify, usually `tcdm_remote`. |
| `--high-fanin-threshold N` | Tile-cycle fan-in count treated as high fan-in; default is 2. |
| `--cores-per-tile N` | Source cores per tile; default is 4 for MemPool. |
| `--operand-regions-json PATH` | Explicit sidecar path; otherwise the tools look for `<result_dir>/analysis/operand_regions.json`. |
| `--operand-address-field source_addr\|source_addr_or_addr\|addr` | Address field used for CLI/transcript ranges; thesis operand plots should use `source_addr`. |
| `--a-range START:END`, `--b-range START:END`, `--c-range START:END` | Override transcript-derived operand source-address ranges. Use `0x` prefixes for hexadecimal values. |
| `--operand-region NAME:START:END` | Add or override an operand source-address range; may be repeated. |
| `--require-exact-operands` | Fail unless operand labels come from `source_addr` plus sidecar/transcript/CLI ranges. |
| `--allow-legacy-route-operands` | Allow old route-address A/B fallback for legacy/debug plots. |
| `--no-legacy-operand-fallback` | Deprecated compatibility flag; legacy fallback is disabled by default. |
| `--output-dir DIR` | CSV output directory; default is `<graph_dir>/<prefix>_classification`. |
| `--prefix NAME` | Output filename prefix, e.g. `port0_source_target`. |
| `--force` | Overwrite existing CSV outputs. |

`plot_source_target_classification.py` flags:

| Flag | Meaning |
|------|---------|
| `input_path` | Classification directory or explicit `*_source_target_matrix.csv`. |
| `--tiles-per-group N` | Tile slots per group for local source/target axes; default is 16. |
| `--offset-mode raw\|wrapped` | Plot direct target-source offsets by default, or wrapped shortest offsets for the older compact view. |
| `--combine` | Combine all matching port classifier matrix CSVs found under the input directory. |
| `--allow-legacy-route-operands` | Allow plotting classifier CSVs produced with legacy route-address operand labels. Without this flag, operand-labeled classifier plots require exact provenance in `<prefix>_summary.csv`. |
| `--output-dir DIR` | Figure output directory; defaults to `<result_dir>/plots/port_pressure/` for normal datasets. |
| `--prefix NAME` | Figure filename prefix; default derives from the matrix CSV stem. |
| `--formats png pdf` | Figure formats to write; default is both PNG and PDF. |
| `--force` | Overwrite existing figures. |

`plot_source_target_operand_offsets.py` flags:

| Flag | Meaning |
|------|---------|
| `input_path...` | One or more result directories, path-graph directories, classification directories, or explicit `*_source_target_matrix.csv` files. |
| `--label NAME` | Label for each input path; repeat once per input in the same order. |
| `--port P` | Port classifier directory to resolve when passing directories; default is 0. |
| `--offset-mode raw\|wrapped` | Plot direct target-source offsets by default, or wrapped shortest offsets. |
| `--operand A`, `--operand B` | Operand columns to include; default is A and B. |
| `--allow-legacy-route-operands` | Allow plotting classifier CSVs produced with legacy route-address operand labels. Without this flag, operand-offset plots require exact provenance in `<prefix>_summary.csv`. |
| `--output-dir DIR` | Figure/data output directory; defaults to the benchmark result directory for one input, or the common sweep parent for multiple result directories. |
| `--prefix NAME` | Output filename prefix; default is `port0_operand_offset_sweep`. |
| `--formats png pdf` | Figure formats to write; default is both PNG and PDF. |
| `--force` | Overwrite existing outputs. |

`scripts/port_pressure/plot_source_target_fanin_comparison.py` flags:

| Flag | Meaning |
|------|---------|
| `input_path...` | One or more classification directories, result directories, or explicit `*_tile_cycles.csv` files. |
| `--label NAME` | Label for each input path; repeat once per input in the same order. |
| `--max-bucket N` | Largest explicit fan-in bucket before the overflow bucket `>N`; default is 4. |
| `--output-dir DIR` | Output directory for the comparison PNG/PDF/CSV. |
| `--prefix NAME` | Output filename prefix; default is `port0_fanin_mechanism`. |
| `--formats png pdf` | Figure formats to write; default is both PNG and PDF. |
| `--force` | Overwrite existing outputs. |

### 13. Monitor Port-Pressure Plotters

These plotters operate directly on `analysis/path_graph` datasets and are the
fastest way to move from raw monitor observations to pressure/fan-in figures.
They do not rerun simulation.

All-port route pressure from node-state observations:

```bash
python scripts/trace_analysis/plot/plot_port_pressure.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph \
  --metric-source node-state \
  --node-point tcdm_remote \
  --cycle-start 10000 \
  --cycle-end 16000 \
  --window 40 \
  --prefix all_ports_pressure \
  --force
```

If the visual plot bounds intentionally include margin cycles around the
benchmark, keep `--cycle-start/--cycle-end` as the visual bounds and add
`--average-section 1`. The aggregate bars, generated summary CSV, and
utilization-style denominators then use section 1 from
`data/stall_timeseries_benchmark.csv`, while the time-series x-axis still shows
the wider context.

Per-tile fan-in heatmap:

```bash
python scripts/trace_analysis/plot/plot_port_fanin_heatmap.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph \
  --cycle-start 10000 \
  --cycle-end 16000 \
  --window 40 \
  --port 0 \
  --prefix port0_tile_fanin \
  --require-exact-operands \
  --force
```

Fan-in versus flow/stall overlay:

```bash
python scripts/trace_analysis/plot/plot_fanin_flow_overlay.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph \
  --cycle-start 10000 \
  --cycle-end 16000 \
  --window 40 \
  --focus-tile auto \
  --threshold 3 \
  --prefix all_ports_fanin_flow \
  --require-exact-operands \
  --force
```

Common flags:

| Flag | Meaning |
|------|---------|
| `input_path` | `analysis/path_graph`, `cycle_node_state.csv`, or a result directory containing `analysis/path_graph`. |
| `--cycle-start N`, `--cycle-end M` | Inclusive cycle filter. Required by `plot_fanin_flow_overlay.py`; optional for the others. |
| `--window N` | Cycle aggregation window; use 1 for exact per-cycle data. For fan-in heatmaps, windowed cells default to averaged fan-in rendered with discrete 0.25-step colours. |
| `--port P` | Route port to analyze or plot. For `plot_port_pressure.py`, omit this for the thesis pressure aggregate; port-filtered runs are diagnostic only and omit aggregate/summary outputs. For `plot_fanin_flow_overlay.py`, omit this for the all-port by-request-count mechanism plot; port-filtered runs omit by-request-count outputs. |
| `--node-point NAME` | Monitor node point to count, usually `tcdm_remote` for source-side route pressure. |
| `--group G`, `--tile T` | Scope plots to one group or selected source tiles. |
| `--operand-regions-json PATH` | Explicit sidecar path; otherwise the tools look for `<result_dir>/analysis/operand_regions.json`. |
| `--operand-address-field source_addr\|source_addr_or_addr\|addr` | Address field used for CLI/transcript ranges; thesis operand plots should use `source_addr`. |
| `--a-range START:END`, `--b-range START:END`, `--c-range START:END` | Override transcript-derived operand source-address ranges for operand-split fan-in plots. |
| `--operand-region NAME:START:END` | Add or override an operand source-address range; may be repeated. |
| `--require-exact-operands` | Fail unless operand labels come from `source_addr` plus sidecar/transcript/CLI ranges. |
| `--allow-legacy-route-operands` | Allow old route-address A/B fallback for legacy/debug plots. |
| `--no-legacy-operand-fallback` | Deprecated compatibility flag; legacy fallback is disabled by default. |
| `--output-dir DIR` | Override default output under `<result_dir>/plots/port_pressure/`. |
| `--prefix NAME` | Output filename prefix. |
| `--formats png pdf` | Figure formats to write. |
| `--force` | Overwrite existing outputs. |

Additional `plot_port_pressure.py` flags:

| Flag | Meaning |
|------|---------|
| `--metric-source route-summary` | Count route-checkpoint summary rows from `route_bottlenecks_all_tiles.csv`. |
| `--metric-source node-state` | Count valid/stall/fire observations directly from `cycle_node_state.csv`. |
| `--per-tile-average` | Divide counts by the number of source tiles in the scope. |
| `--average-section N` | Use section `N` from `data/stall_timeseries_benchmark.csv` as the metric/denominator window while keeping `--cycle-start/--cycle-end` as visual bounds. |
| `--average-cycle-start N`, `--average-cycle-end M` | Override the metric/denominator window directly. |
| `--all-groups` | Emit one scoped plot set per source group. |
| `--tile-local T` | Select tile index `T` inside `--group`. |

Additional `plot_fanin_flow_overlay.py` flags:

| Flag | Meaning |
|------|---------|
| `--window-stat mean|max` | Window reduction for heatmap cells; default `mean` averages fan-in and maps colours to the nearest 0.25, while `max` shows peak exact 0..4 fan-in within each window. Also supported by `plot_port_fanin_heatmap.py`. |
| `--focus-tile T\|auto` | Tile used for worst-cycle drilldown tables. The default `auto` picks the tile with the strongest threshold-matching fan-in evidence, or falls back to the best available peak fan-in when no tile reaches the requested threshold. |
| `--threshold N` | Preferred minimum focus-tile fan-in for drilldown rows. In automatic fallback mode, the generated focus-selection CSV records both this requested threshold and the lower effective threshold used for non-empty drilldown rows. |
| `--max-cores N` | Maximum source cores per tile; default is 4. |

Main outputs:

- `plot_port_pressure.py`: unfiltered runs write `<prefix>_timeseries`, `<prefix>_aggregate`, and `<prefix>_utilization` PNG figures, matching PDFs in `pdf/`, plus `data/<prefix>_summary.csv` and `data/<prefix>_caption.txt`; port-filtered runs write only the filtered time-series/utilization diagnostics
- `plot_port_fanin_heatmap.py`: `<prefix>_total_heatmap` and `<prefix>_operand_heatmap` PNG figures, matching PDFs in `pdf/`, and `<prefix>_heatmap.csv` plus `<prefix>_summary.txt` in `data/`
- `plot_fanin_flow_overlay.py`: unfiltered runs write `<prefix>_overlay_heatmap`, `<prefix>_by_request_count`, and `<prefix>_by_tile_hotspots` PNG figures, matching PDFs in `pdf/`, plus exact/overlay/by-request/by-tile/summary/focus-selection CSVs in `data/`; port-filtered runs omit by-request-count and by-tile hotspot outputs

---

## Script Reference

Public wrappers, permanent monitor helpers, and internal extraction/plotting
pieces are listed here. Internal-only scripts use an underscore prefix.

| Script | Called by | Monitor needed? | What it does |
|--------|-----------|-----------------|-------------|
| `plot_all_tiles.py` | users, Makefile `plots` | No | Batch tile, group, and overview plotting from `stall_timeseries_benchmark.csv`. |
| `plot_specific_core.py` | users | No | Single-core drill-down from `stall_timeseries_benchmark.csv`. |
| `plot_comm_thesis.py` | users, Makefile `plots_comm` | No | Thesis-quality communication figures from `comm_events_benchmark.csv`. |
| `rerun_stall_timeseries.py` | Makefile `rerun_stall_timeseries`, users | No | Public wrapper for safe stall CSV regeneration from preserved traces in `result_dir`. |
| `extract_comm_events.py` | users | No | Public wrapper for building `comm_events_benchmark.csv` from preserved trace inputs in `result_dir`. |
| `extract_comm_events_batch.py` | users | No | Direct folder-based communication-event extraction. |
| `path_graph_dataset.py` | users | Yes, raw monitor CSVs | Normalizes monitor valid/ready/fire CSVs into graph-time node/lane state, summaries, and an optional compact HTML timeline. |
| `path_route_checkpoints.py` | users | Yes, full `analysis/path_graph` | Builds detailed route-checkpoint CSVs and compact bottleneck summaries from a normalized full path graph dataset. |
| `classify_source_targets.py` | users | Yes, `analysis/path_graph` | Classifies source-side route requests by operand, source tile/core, route port, fan-in, and decoded target tile. |
| `plot_port_pressure.py` | users | Yes, `analysis/path_graph` or route summary | Plots route-port request, blocked, fire, and utilization pressure from monitor-derived data. |
| `plot_port_fanin_heatmap.py` | users | Yes, `analysis/path_graph` | Plots per-tile source-core fan-in heatmaps from monitor node-state data. |
| `plot_fanin_flow_overlay.py` | users | Yes, `analysis/path_graph` | Plots fan-in versus flow/stall outcomes and exact tile-cycle drill-down CSVs. |
| `plot_source_target_classification.py` | users | Yes, classifier CSVs | Plots source-target request/stall heatmaps and traffic-class summaries from `classify_source_targets.py` matrix CSVs. |
| `port_pressure/plot_source_target_fanin_comparison.py` | users | Yes, classifier tile-cycle CSVs | Compares source-target tile-cycle CSVs across runs to show source-side fan-in bucket share and stall rates. |
| `_gen_stall_timeseries_batch.py` | Makefile `stall_timeseries`, `rerun_stall_timeseries.py` | No | Loops all traces, auto-loads topology metadata, calls `_gen_stall_timeseries.py` for each. |
| `_gen_stall_timeseries.py` | `_gen_stall_timeseries_batch.py` | No | Single-trace → cycle-by-cycle stall rows in CSV. |
| `_extract_comm_events.py` | `extract_comm_events_batch.py` | No | Single-trace → communication event rows in CSV. |
| `_plot_specific_tile.py` | `plot_all_tiles.py` | No | Cluster overview, group/subgroup detail pages, and single-tile detail pages. |
| `_stall_plot_common.py` | `_plot_specific_tile.py`, `plot_specific_core.py` | No | Shared helpers: data loading, trace lookup, plot formatting. |

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

# ── MONITOR NORMALIZATION ───────────────────────────────────
python scripts/trace_analysis/path_graph_dataset.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000 \
  --point tcdm_remote --cycle-start 10000 --cycle-end 16000 --no-html

# ── ALL-PORT PRESSURE GUARDRAIL ─────────────────────────────
python scripts/trace_analysis/plot/plot_port_pressure.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph \
  --metric-source node-state --node-point tcdm_remote \
  --cycle-start 10000 --cycle-end 16000 --window 40 \
  --prefix all_ports_pressure --force

# ── PORT-0 SOURCE-TARGET CLASSIFICATION ─────────────────────
python scripts/trace_analysis/classify_source_targets.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph \
  --cycle-start 10000 --cycle-end 16000 \
  --port 0 --node-point tcdm_remote \
  --prefix port0_source_target --require-exact-operands --force

python scripts/trace_analysis/plot/plot_source_target_classification.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph/port0_source_target_classification/port0_source_target_source_target_matrix.csv \
  --prefix port0_source_target --force

# ── DAS VS BACK2LOCAL FAN-IN MECHANISM ──────────────────────
python scripts/port_pressure/plot_source_target_fanin_comparison.py \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph/port0_source_target_classification/port0_source_target_tile_cycles.csv \
  results/matmul_i32_mempool/4x4_das_thesis_asm/das_back2local_sourceport_loadmon_alltiles_c10000_16000/analysis/path_graph/port0_source_target_classification/port0_source_target_tile_cycles.csv \
  --label DAS --label Back2Local \
  --output-dir results/matmul_i32_mempool/4x4_das_thesis_asm/das_back2local_sourceport_loadmon_alltiles_c10000_16000/plots/port_pressure \
  --prefix port0_fanin_mechanism --force
```

---

## Folder Structure

```
hardware/scripts/
  trace_analysis/                        ← this folder
    _workflow_metadata.py                shared: topology resolution & path helpers
    PIPELINE.md                          pipeline documentation (this file)
    path_graph_common.py                 shared helpers for path-graph scripts
    path_graph_dataset.py                monitor CSVs → graph-time normalized data
    path_route_checkpoints.py            graph-time data → route checkpoint and bottleneck CSVs
    classify_source_targets.py           graph-time data → source-target classification CSVs
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
      plot_port_pressure.py              monitor route-port request/block/fire plots
      plot_port_fanin_heatmap.py         monitor source-core fan-in heatmaps
      plot_fanin_flow_overlay.py         monitor fan-in versus flow/stall plots
      plot_source_target_classification.py source-target classifier heatmaps and summaries
      _plot_specific_tile.py             internal: per-tile 4-subplot detail
      _stall_plot_common.py              internal: shared plotting library
  port_pressure/                         port-pressure comparison tools
    plot_source_target_fanin_comparison.py compare source-target fan-in mechanisms across runs
  plotting/                              traffic-gen / port analysis (separate)
    _plotting_common.py                  internal: shared helpers
    plot_port_utilization.py             port utilization heatmaps
    plot_load_throughput.py              latency/throughput curves
    plot_reconstructed_load_throughput.py reconstructed latency/throughput
    compare_tilerange_plots.py           side-by-side comparison
```
