# Copyright 2021 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

# Author: Matheus Cavalcante, ETH Zurich
#         Samuel Riedel, ETH Zurich

# ── Benchmark Workflow Overview ───────────────────────────────────────────
#
# `make benchmark` drives both the software build and the hardware Stage-1
# trace/CSV pipeline. For thesis DAS work, the primary software path today is
# `app=matmul_i32` with `kernel=4x4_das_thesis_asm`.
#
# Call tree for the matmul_i32 benchmark family:
#
#   make benchmark
#   |-- build_app
#   |   `-- software/apps/baremetal/Makefile
#   |       `-- matmul_i32
#   |           |-- data_%.h
#   |           |   `-- software/data/gendata_header.py
#   |           |       `-- generate_imatmul()
#   |           |           `-- emits data_matmul_i32.h
#   |           `-- matmul_i32/main.c
#   |               |-- #include "data_matmul_i32.h"
#   |               |-- #include "baremetal/mempool_matmul_i32p.h"
#   |               `-- kernel dispatch
#   |                   `-- kernel=4x4_das_thesis_asm
#   |                       `-- mat_mul_unrolled_4x4_das_thesis_parallel_asm()
#   `-- Stage 1 hardware pipeline
#       |-- log
#       |-- simcvcs
#       |-- trace
#       |   |-- spike-dasm
#       |   |-- scripts/gen_trace.py
#       |   |-- scripts/merge_trace_results_csv.py
#       |   `-- scripts/gen_avg.py
#       |-- comm_events_real
#       |   `-- scripts/trace_analysis/extract/extract_comm_events_batch.py
#       `-- stall_timeseries_build
#           `-- scripts/trace_analysis/extract/_gen_stall_timeseries_batch.py
#
# Exact target order for `make benchmark`:
#   build_app -> log -> simcvcs -> trace -> comm_events_real -> stall_timeseries_build
#
# Public workflow targets:
#   make benchmark               Stage 1: software build, simulation, traces, CSVs
#   make plots                   Stage 2: tile plots plus communication figures from an existing result_dir
#   make plots_parallel          Stage 2: run `make plots` across selected kernels in parallel
#   make plots_comm              Stage 2b: thesis communication figures from an existing result_dir
#   make archive_result          Save a selected finished result_dir for thesis/reference
#   make freeze                  Alias for `make archive_result`
#   make rerun_stall_timeseries  Rebuild only stall_timeseries_benchmark.csv
#
# Start here as a new user:
#   1. Run `make benchmark` to produce a complete result_dir.
#   2. Run `make plots` on that result_dir to generate the overview/tile plots
#      and, by default, the communication-analysis artifacts and thesis figures.
#      If you want multiple kernels, use `make plots_parallel` with
#      `parallel_kernels="..."` and `parallel_jobs=N`.
#   3. Only use `make rerun_stall_timeseries` when you want to rebuild the stall CSV
#      from saved traces without re-running simulation.
#
# Required variables:
#   kernel=<name>     Kernel folder name, e.g. kernel=2x2_xpulpv2.
#                     Select this to separate benchmark outputs by kernel.
#
# Defaults for thesis workflow targets:
#   app=<name>        Defaults to app=matmul_i32 for build_app/benchmark/plot
#                     workflow targets. Resolved via software/bin/ search.
#   variant=<name>    System variant: baseline, das, or redmule.
#                     Defaults to variant=das and controls the result subdirectory.
#
# Optional variables:
#   config=<name>     Hardware topology (default: mempool).
#                     Set config=terapool for the 1024-core system.
#                     Selects config/<name>.mk for core/group/tile counts.
#   scratch_storage=<mode>
#                     Storage location policy for generated build data.
#                     `auto` (default) uses /scratch/$$USER/mempool/hardware when
#                     /scratch/$$USER exists and is writable, otherwise falls back
#                     to local `build/`. Set `scratch_storage=0` to force local
#                     workspace paths, or `scratch_storage=1` to force
#                     /scratch-backed build paths.
#   scratch_storage_root=<path>
#                     Override the /scratch root used when scratch storage is
#                     enabled. Default: /scratch/$$USER/mempool/hardware
#   result_data_storage=<mode>
#                     Storage location policy for the heavy benchmark CSVs.
#                     `auto` (default) keeps result directories local under
#                     `results/`, but places `comm_events_benchmark.csv` and
#                     `stall_timeseries_benchmark.csv` on scratch when available,
#                     exposing them in the local result directory via symlinks.
#                     Set `result_data_storage=0` to keep those CSVs local too,
#                     or `result_data_storage=1` to force scratch-backed CSVs.
#   result_data_root=<path>
#                     Override the scratch root used for heavy result CSVs.
#                     Default: $(scratch_storage_root)/results_data
#   result_app=<name> Optional logical result-family name used for the
#                     auto-generated result directory.
#                     Default: $(app)
#   result_dir=<path> Override the auto-generated result directory.
#                     Default: $(resultpath)/<result_app>_<config>/<kernel>/<variant>
#                     Setting this bypasses the kernel/variant requirements.
#   force=1           Allow overwriting an existing result directory.
#                     Without this, `make benchmark` refuses to run if
#                     result_dir/data/ already contains benchmark CSV outputs.
#   extract_jobs=<N>  For benchmark-time and rerun CSV extraction: number of
#                     workers for communication/stall extraction. Defaults to 16.
#   trace_jobs=<N>    For `make benchmark`: number of parallel per-hart trace
#                     decode/extract workers used in the Stage-1 `trace`
#                     submake. Defaults to extract_jobs.
#   benchmark_comm=0  For `make benchmark`: skip communication extraction.
#   benchmark_stall=0 For `make benchmark`: skip stall-timeseries extraction.
#   plot_overview=1   For `make plots`: also generate the cluster overview.
#                     Default: enabled. Set plot_overview=0 to skip it.
#   plot_group_details=1
#                     For `make plots`: generate group/subgroup detail pages
#                     with one tile-row heatmap per page.
#   plot_tile_details=0
#                     For `make plots`: skip the slow per-tile detail pages.
#   plot_window=<N>   For `make plots`: sliding aggregation window.
#                     Default: 64 cycles.
#   plot_comm=0       For `make plots`: skip communication figures and the
#                     requirement to have `comm_events_benchmark.csv`.
#   plot_port=1       For `make plots`: also generate monitor-derived
#                     route-port pressure/fan-in/source-target plots.
#                     Requires a monitor-enabled result directory.
#   plot_port_cycle_start=<cycle>
#                     First monitor cycle for port plots. Default: 7500.
#   plot_port_cycle_end=<cycle|auto>
#                     Last monitor cycle for port plots. Default: auto, which
#                     uses the end of plot_section rounded up to 500 cycles.
#   plot_port_id=<P>  Route port for focused fan-in/source-target plots.
#                     Default: 0.
#   plot_port_threshold=<N>
#                     Same-tile source-core fan-in threshold for contention
#                     drilldown CSVs. Default: 2.
#   parallel_kernels="..."
#                     For `make plots_parallel`: kernel subset to run.
#                     Example: "4x4_das_thesis_asm"
#   parallel_jobs=<N> For `make plots_parallel`: number of concurrent kernels.
#                     Default: number of kernels in `parallel_kernels`.
#   plot_tiles="..."  For `make plots`: optional tile list, e.g. "0 1 2".
#   plot_topology=<name>
#                     For `make plots`: override topology auto-detection.
#                     Normally not needed because result metadata is used.
#   path_util_monitor=1
#                     Compile and enable group-level path monitor CSV capture.
#   tile_path_monitor=1
#                     Enable tile-level path monitor CSV capture.
#   path_util_start=<cycle> / path_util_end=<cycle>
#                     Optional cycle window for both monitor writers.
#   path_util_load_only=1
#                     For group monitor rows, emit only valid load requests.
#   tile_path_tile=<tile>
#                     Optional single-tile filter for tile path capture.
#   tile_path_active_only=1
#                     For tile monitor rows, emit only valid requests.
#   tile_path_load_only=1
#                     For tile monitor rows, emit only valid load requests.
#   tile_path_source_ports_only=1
#                     For tile monitor rows, emit only source-side `tcdm_remote`
#                     route-port rows. This is the compact monitor mode used by
#                     port-pressure, fan-in, and source-target plots.
#   back2local=0/1    Enable the Back2Local routing path for same-group loads.
#                     Default is in config/config.mk; command-line override is
#                     still accepted for one-off runs.
#   archive_dir=<path>
#                     For `make archive_result` / `make freeze`: destination directory for a
#                     frozen copy of a finished benchmark result.
#   archive_include_traces=1
#                     For `make archive_result` / `make freeze`: also keep archived traces if
#                     they exist in the source result dir. Default: off.
#   archive_note="..."
#                     For `make archive_result` / `make freeze`: optional note stored in
#                     archive.meta alongside the frozen result.
#
# The pipeline produces:
#   result_dir/
#     <app>            copied benchmark ELF
#     <app>.dump       copied objdump for the benchmark ELF
#     data/            results.csv, comm_events_benchmark.csv,
#                      stall_timeseries_benchmark.csv
#     avg.txt          Average performance stats
#     transcript       Simulation log
#     config, env      Build-time snapshots
#     topology.env     Exact topology used for post-processing
#     git-info.diff    Source state at run time
#
# Temporary Stage-1 trace intermediates still exist internally:
#   $(buildpath)/*.dasm   raw simulator output
#   $(buildpath)/traces/* spike-dasm canonical analysis traces
#   $(buildpath)/*.trace  gen_trace.py temporary parsed trace format
# These are cleaned after benchmark extraction completes successfully.
#
# Examples:
#   kernel=4x4_das_thesis_asm make benchmark
#   kernel=4x4_das_thesis_asm make plots
#   kernel=4x4_das_thesis_asm make plots_comm
#   config=terapool kernel=4x4_das_thesis_asm trace_jobs=8 make benchmark
#   variant=baseline kernel=4x4_das_thesis_asm make benchmark
#   result_dir=results/matmul_i32_mempool/4x4_das_thesis_asm/das make plots
#   result_dir=results/matmul_i32_mempool/4x4_das_thesis_asm/das make rerun_stall_timeseries force=1
#   result_dir=results/matmul_i32_mempool/4x4_das_thesis_asm/das archive_dir=/scratch/$$USER/mempool/thesis_results/4x4_das_thesis_asm_das make archive_result
#   result_dir=results/matmul_i32_mempool/4x4_das_thesis_asm/das archive_dir=/scratch/$$USER/mempool/thesis_results/4x4_das_thesis_asm_das make freeze
#
# For `app=matmul_i32`, `make benchmark` rebuilds the software binary first so
# the compiled kernel matches `kernel=<name>`.
# ──────────────────────────────────────────────────────────────────────────

SHELL = /usr/bin/env bash
ROOT_DIR := $(patsubst %/,%, $(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
MEMPOOL_DIR := $(shell git rev-parse --show-toplevel 2>/dev/null || echo $$MEMPOOL_DIR)
INSTALL_DIR := $(abspath $(ROOT_DIR)/../install)
TOOLCHAIN_DIR := $(abspath $(ROOT_DIR)/../toolchain)

# Include configuration
config_mk = $(abspath $(ROOT_DIR)/../config/config.mk)
include $(config_mk)

# Build paths
scratch_storage       ?= auto
scratch_storage_root  ?= /scratch/$(USER)/mempool/hardware
scratch_user_dir      := /scratch/$(USER)
scratch_storage_available := $(shell if [ -d "$(scratch_user_dir)" ] && [ -w "$(scratch_user_dir)" ]; then echo 1; else echo 0; fi)
use_scratch_storage   := 0
result_data_storage   ?= auto
result_data_root      ?= $(scratch_storage_root)/results_data
use_scratch_result_data := 0

ifeq ($(scratch_storage),1)
use_scratch_storage := 1
else ifeq ($(scratch_storage),auto)
use_scratch_storage := $(scratch_storage_available)
endif

ifeq ($(result_data_storage),1)
use_scratch_result_data := 1
else ifeq ($(result_data_storage),auto)
use_scratch_result_data := $(scratch_storage_available)
endif

ifeq ($(origin buildpath), undefined)
	ifeq ($(use_scratch_storage),1)
buildpath := $(scratch_storage_root)/build
	else
buildpath := build
	endif
endif

ifeq ($(origin resultpath), undefined)
resultpath := results
endif
# Bender
bender          ?= $(INSTALL_DIR)/bender/bender
# Path to the binaries
app_bin_root    ?= $(abspath $(ROOT_DIR)/../software/bin)
variant         ?= das
app_default_goals := $(filter build_app benchmark plots plots_parallel plots_comm archive_result freeze rerun_stall_timeseries,$(MAKECMDGOALS))
ifneq ($(strip $(app_default_goals)),)
app             ?= matmul_i32
endif

# Top level module to compile
top_level       ?= mempool_tb

# QuestaSim
questa_version  ?= 2022.3-bt
questa_cmd      ?= questa-$(questa_version)
questa_args     ?=
QUESTASIM_HOME  ?= /usr/pack/questa-$(questa_version)/questasim
# Check if the specified QuestaSim version exists
ifeq (, $(shell which $(questa_cmd)))
  # Spaces are needed for indentation here!
  $(warning "Specified QuestaSim version ($(questa_cmd)) not found in PATH $(PATH)")
  questa_cmd =
endif

# VCS Version
vcs_version     ?= 2020.12-bt
vcs_cmd         ?= vcs-$(vcs_version)
VCS_HOME        ?= /usr/pack/vcs-$(vcs_version)/vcs
# Verilator
verilator       ?= $(INSTALL_DIR)/verilator/bin/verilator
verilator_build ?= $(ROOT_DIR)/verilator_build
verilator_files ?= $(verilator_build)/files
verilator_top   ?= mempool_tb_verilator
# Python
python_venv     ?= $(MEMPOOL_DIR)/mempoolvenv/bin/python
python          ?= $(python_venv)
# Enable tracing
snitch_trace    ?= 1
trace           := $(patsubst $(buildpath)/%.dasm,$(buildpath)/%.trace,$(wildcard $(buildpath)/*.dasm))
tracepath       ?= $(buildpath)/traces
traceresult     ?= $(tracepath)/results.csv
path_util_monitor ?= 0
tile_path_monitor ?= $(path_util_monitor)
path_util_start ?=
path_util_end ?=
path_util_load_only ?= 0
tile_path_tile ?=
tile_path_active_only ?= 0
tile_path_load_only ?= 0
tile_path_source_ports_only ?= 0
remote_req_shared_slots ?= 0



# Path to DRAMsys
dramsys_resouces_path ?= $(MEMPOOL_DIR)/hardware/deps/dram_rtl_sim/dramsys_lib/DRAMSys/configs
dramsys_lib_path ?= $(MEMPOOL_DIR)/hardware/deps/dram_rtl_sim/dramsys_lib/DRAMSys/build/lib
questa_args += +DRAMSYS_RES=$(dramsys_resouces_path)
questa_args += -sv_lib $(dramsys_lib_path)/libsystemc -sv_lib $(dramsys_lib_path)/libDRAMSys_Simulator

# Path to the binaries
default_app_path := $(app_bin_root)
ifneq ($(strip $(app)),)
ifneq ($(strip $(config)),)
ifneq ($(strip $(kernel)),)
ifneq ($(strip $(variant)),)
default_app_path := $(app_bin_root)/apps/baremetal/$(app)_$(config)/$(kernel)/$(variant)
endif
endif
endif
endif
app_path ?= $(default_app_path)
result_app ?= $(app)
preload_skip_validation_goals := $(filter build_app benchmark plots plots_parallel plots_comm archive_result freeze rerun_stall_timeseries,$(MAKECMDGOALS))
ifdef app
preload := "$(app_path)/$(app)"
ifeq ($(strip $(preload_skip_validation_goals)),)
ifeq ("$(shell test -f $(preload))","")
# The file doesn't exist, let's search for the app in the bin folder
# since we might just be missing the hierarchy
search_app := $(shell if test -d $(app_path); then find $(app_path) -type f -path "*$(app)"; fi)
num_matches := $(words $(search_app))
ifeq ($(num_matches),0)
$(error "App ($(app)) not found in bin dir $(app_path)")
else ifeq ($(num_matches),1)
preload := "$(search_app)"
$(warning "Running application $(preload)")
else
$(error "Multiple apps ($(app)) found in bin dir $(app_path): $(search_app)")
endif
endif
endif
endif

# QuestaSim args
library     ?= work
dpi_library ?= work-dpi
questa_args += $(if $(preload),+PRELOAD=$(preload),)
questa_args += -sv_lib $(dpi_library)/mempool_dpi
questa_args += -work $(library)
questa_args += -suppress vsim-12070
path_monitor_args :=
ifeq ($(path_util_monitor),1)
path_monitor_args += +path_util_monitor
endif
ifeq ($(tile_path_monitor),1)
path_monitor_args += +tile_path_monitor
endif
ifneq ($(strip $(path_util_start)),)
path_monitor_args += +path_util_start=$(path_util_start)
endif
ifneq ($(strip $(path_util_end)),)
path_monitor_args += +path_util_end=$(path_util_end)
endif
ifeq ($(path_util_load_only),1)
path_monitor_args += +path_util_load_only
endif
ifneq ($(strip $(tile_path_tile)),)
path_monitor_args += +tile_path_tile=$(tile_path_tile)
endif
ifeq ($(tile_path_active_only),1)
path_monitor_args += +tile_path_active_only
endif
ifeq ($(tile_path_load_only),1)
path_monitor_args += +tile_path_load_only
endif
ifeq ($(tile_path_source_ports_only),1)
path_monitor_args += +tile_path_source_ports_only
endif
questa_args += $(path_monitor_args)

# VCS args
vcs_args    += $(if $(preload),+PRELOAD=$(preload),)
vcs_args    += $(path_monitor_args)

# Questasim vlog args
vlog_args   += -suppress vlog-2583
vlog_args   += -suppress vlog-13314
vlog_args   +=-suppress vlog-13233
vlog_args   += -work $(library)

# VCS vlogan args
vlogan_args += -assert svaext
vlogan_args += +v2k
vlogan_args += -override_timescale=1ns/1ps -kdb

# Configuration name
vlog_defs   += -D$(config)
# VCS options
vlogan_args += -assert svaext +v2k -override_timescale=1ns/1ps -kdb
ifdef preload
	vcs_args += +PRELOAD=$(preload)
endif

# DPI source files
dpi     := $(patsubst tb/dpi/%.cpp,$(buildpath)/$(dpi_library)/%.o,$(wildcard tb/dpi/*.cpp))
dpi_vcs := $(patsubst tb/dpi/%.cpp,$(buildpath)/$(dpi_library)/%_vcs.o,$(wildcard tb/dpi/*.cpp))
# System variant (baseline, das, redmule)
# Kernel label for result directory grouping — must be set explicitly
kernel          ?=
valid_variants  := baseline das redmule

ifneq ($(strip $(variant)),)
  ifeq ($(filter $(variant),$(valid_variants)),)
    $(error Unsupported variant=$(variant). Supported values: $(valid_variants))
  endif
endif

# Auto-map variant to das flag: variant=das → das=1, variant=baseline → das=0
ifeq ($(variant),das)
  das := 1
else ifeq ($(variant),baseline)
  das := 0
else ifeq ($(variant),redmule)
  das := 0
endif

ifneq ($(strip $(variant)),)
  ifeq ($(variant),baseline)
    ifneq ($(das),0)
      $(error variant=baseline requires das=0, but got das=$(das))
    endif
  else ifeq ($(variant),das)
    ifneq ($(das),1)
      $(error variant=das requires das=1, but got das=$(das))
    endif
  else ifeq ($(variant),redmule)
    ifneq ($(das),0)
      $(error variant=redmule currently expects das=0, but got das=$(das))
    endif
  endif
endif

ifneq ($(filter $(back2local),0 1),$(back2local))
$(error Unsupported back2local=$(back2local). Supported values: 0 1)
endif
ifneq ($(filter $(remote_req_shared_slots),0 1),$(remote_req_shared_slots))
$(error Unsupported remote_req_shared_slots=$(remote_req_shared_slots). Supported values: 0 1)
endif
# Traces
trace = $(patsubst $(buildpath)/%.dasm,$(buildpath)/%.trace,$(wildcard $(buildpath)/*.dasm))
tracepath ?= $(buildpath)/traces
traceresult ?= $(tracepath)/results.csv
tracepartspath ?= $(tracepath)/results_parts
trace_merge_script ?= $(ROOT_DIR)/scripts/merge_trace_results_csv.py
result_dir_goals := $(filter benchmark plots plots_comm archive_result freeze rerun_stall_timeseries,$(MAKECMDGOALS))
ifneq ($(strip $(result_dir_goals)),)
ifndef result_dir
	ifndef kernel
		$(error kernel is not set. Specify kernel=<name>, e.g. kernel=2x2_xpulpv2)
	endif
  ifndef variant
    $(error variant is not set. Specify variant=baseline, variant=das, or variant=redmule)
  endif
	result_dir := $(resultpath)/$(result_app)_$(config)/$(kernel)/$(variant)
endif
endif

ifdef result_dir
result_dir_abs := $(abspath $(result_dir))
result_data_scratch_key := $(shell printf '%s' "$(result_dir_abs)" | sed 's|^/||; s|[^A-Za-z0-9._/-]|_|g')
result_data_dir := $(result_dir)/data
comm_events_csv := $(result_data_dir)/comm_events_benchmark.csv
stall_timeseries_csv := $(result_data_dir)/stall_timeseries_benchmark.csv
scratch_result_data_dir := $(result_data_root)/$(result_data_scratch_key)/data
scratch_comm_events_csv := $(scratch_result_data_dir)/comm_events_benchmark.csv
scratch_stall_timeseries_csv := $(scratch_result_data_dir)/stall_timeseries_benchmark.csv
endif

vlog_args += -suppress vlog-2583 -suppress vlog-13314 -suppress vlog-13233
vlog_args += -work $(library)
# Defines
vlog_defs += -D$(config)
# Cluster configurations
vlog_defs   += -DNUM_CORES=$(num_cores)
vlog_defs   += -DNUM_CORES_PER_TILE=$(num_cores_per_tile)
vlog_defs   += -DNUM_DIVSQRT_PER_TILE=$(num_divsqrt_per_tile)
vlog_defs   += -DNUM_GROUPS=$(num_groups)
vlog_defs   += -DBANKING_FACTOR=$(banking_factor)
vlog_defs   += -DL2_BASE=32\'d$(l2_base)
vlog_defs   += -DL2_SIZE=32\'d$(l2_size)
vlog_defs   += -DL2_BANKS=$(l2_banks)
vlog_defs   += -DL1_BANK_SIZE=$(l1_bank_size)
vlog_defs   += -DBOOT_ADDR=32\'d$(boot_addr)
# DAS: only define the macro when das=1 (ifdef DAS checks existence, not value)
ifeq ($(das),1)
vlog_defs   += -DDAS
vlog_defs   += -DNUM_DAS_PARTITIONS=$(num_das_partitions)
vlog_defs   += -DDAS_MEM_SIZE=$(das_mem_size)
endif
ifeq ($(back2local),1)
vlog_defs   += -DBACK2LOCAL
endif
ifeq ($(remote_req_shared_slots),1)
vlog_defs   += -DREMOTE_REQ_SHARED_SLOTS
endif
ifneq ($(filter 1,$(path_util_monitor) $(tile_path_monitor)),)
vlog_defs   += -DPATH_UTIL_MONITOR
endif
# Snitch ISA
vlog_defs   += -DXPULPIMG=$(xpulpimg)
vlog_defs   += -DZFINX=$(zfinx)
vlog_defs   += -DZQUARTERINX=$(zquarterinx)
vlog_defs   += -DXDIVSQRT=$(xDivSqrt)
vlog_defs   += -DSNITCH_TRACE=$(snitch_trace)
# AXI & DMA
vlog_defs   += -DAXI_DATA_WIDTH=$(axi_data_width)
vlog_defs   += -DRO_LINE_WIDTH=$(ro_line_width)
vlog_defs   += -DDMAS_PER_GROUP=$(dmas_per_group)
vlog_defs   += -DAXI_HIER_RADIX=$(axi_hier_radix)
vlog_defs   += -DAXI_MASTERS_PER_GROUP=$(axi_masters_per_group)
# Systolic configurations
vlog_defs   += -DSEQ_MEM_SIZE=$(seq_mem_size)
vlog_defs   += -DXQUEUE_SIZE=$(xqueue_size)
# TeraPool configurations
vlog_defs   += -DNUM_SUB_GROUPS_PER_GROUP=$(num_sub_groups_per_group)
vlog_defs   += -DREMOTE_GROUP_LATENCY_CYCLES=$(remote_group_latency_cycles)
# DRAMsys co-simulation
vlog_defs   += -D${l2_sim_type}
vlog_defs   += -DDRAM_AXI_WIDTH_INTERLEAVED=${dram_axi_width_interleaved}

# Verilator CPP defines
cpp_defs    += -DL2_BASE=$(l2_base)
cpp_defs    += -DL2_SIZE=$(l2_size)
cpp_defs    += -DL2_BANKS=$(l2_banks)
cpp_defs    += -DAXI_DATA_WIDTH=$(axi_data_width)

# Enable Traffic Generator
# The Traffic Generator replaces the cores and generates random traffic to the memory system.
# It is useful for testing the memory latency under random traffic patterns.
ifdef tg
	tg_ncycles  ?= 10000
	tg_reqprob  ?= 0.2
	tg_seqprob  ?= 0
	tg_seed     ?= 0

	vlog_defs += -DTRAFFIC_GEN=1
	cpp_defs  += -DTRAFFIC_GEN=1
	cpp_defs  += -DTG_REQ_PROB=$(tg_reqprob)
	cpp_defs  += -DTG_SEQ_PROB=$(tg_seqprob)
	cpp_defs  += -DTG_NCYCLES=$(tg_ncycles)
	cpp_defs  += -DNUM_CORES=$(num_cores)
	cpp_defs  += -DTG_SEED=$(tg_seed)

	# How many cycles should we execute?
	veril_flags := --term-after-cycles=$(tg_ncycles)
	# Pass cycle count to QuestaSim TB via plusarg
	questa_args += +tg_ncycles=$(tg_ncycles)
else
	tg := 0
	verilator_args := --meminit=ram,$(preload)
endif

# DPI source files
dpi     := $(patsubst tb/dpi/%.cpp,$(buildpath)/$(dpi_library)/%.o,$(wildcard tb/dpi/*.cpp))
dpi_vcs := $(patsubst tb/dpi/%.cpp,$(buildpath)/$(dpi_library)/%_vcs.o,$(wildcard tb/dpi/*.cpp))

.DEFAULT_GOAL := compile

# Build path
$(buildpath):
	mkdir -p $(buildpath)

# Bender
$(bender):
	make -C $(MEMPOOL_DIR) bender

################
# Modelsim     #
################

# Library
.PHONY: lib
lib: $(buildpath) $(buildpath)/$(library)
$(buildpath)/$(library):
	cd $(buildpath) && $(questa_cmd) vlib $(library) && chmod +w modelsim.ini; $(questa_cmd) vmap $(library) $(library)

# Compilation
.PHONY: compile
compile: dpi lib $(buildpath) $(buildpath)/compile.tcl update-opcodes
$(buildpath)/compile.tcl: $(bender) $(config_mk) Makefile $(MEMPOOL_DIR)/Bender.yml $(shell find {src,tb,deps} -type f)
	$(bender) script vsim --vlog-arg="$(vlog_args)" $(vlog_defs) -t rtl -t mempool_vsim > $(buildpath)/compile.tcl
	echo "exit" >> $(buildpath)/compile.tcl
	cd $(buildpath) && $(questa_cmd) vsim -work $(library) -c -do compile.tcl

# Simulation
.PHONY: sim clean-path-monitor
sim: clean-dasm clean-path-monitor compile
	rm -f $(buildpath)/transcript
	cd $(buildpath) && \
	$(questa_cmd) vsim -voptargs=+acc $(questa_args) $(library).$(top_level) -do "set config ${config}" -do ../scripts/questa/run.tcl
	./scripts/return_status.sh $(buildpath)/transcript

.PHONY: simc
simc: clean-dasm clean-path-monitor compile
	rm -f $(buildpath)/transcript
	cd $(buildpath) && \
	$(questa_cmd) vsim -c $(questa_args) $(library).$(top_level) -do "run -a"
	./scripts/return_status.sh $(buildpath)/transcript

clean-path-monitor:
	rm -f $(buildpath)/path_util_group*.csv $(buildpath)/tile_path_tile*.csv

# DPIs
.PHONY: dpi
dpi: $(buildpath)/$(dpi_library)/mempool_dpi.so

$(buildpath)/$(dpi_library)/%.o: tb/dpi/%.cpp
	mkdir -p $(buildpath)/$(dpi_library)
	$(CXX) -shared -fPIC -std=c++11 -Bsymbolic $(cpp_defs) -c $< -I$(QUESTASIM_HOME)/include -o $@

$(buildpath)/$(dpi_library)/mempool_dpi.so: $(dpi)
	mkdir -p $(buildpath)/$(dpi_library)
	$(CXX) -shared -m64 -o $(buildpath)/$(dpi_library)/mempool_dpi.so $^

################
# VCS          #
################

# Config stamp — forces VCS rebuild when the effective VCS fingerprint changes.
# The stamp captures config name, core/group topology, subgroup count, and DAS
# mode. If the stamp file is missing or its content differs from the current
# invocation, we delete the stale VCS artefacts so Make sees them as
# out-of-date and re-elaborates / recompiles.
vcs_config_stamp      := $(buildpath)/.vcs_config_stamp
vcs_config_fingerprint := $(config):cores=$(num_cores):groups=$(num_groups):sub=$(num_sub_groups_per_group):das=$(das):back2local=$(back2local):sharedslots=$(remote_req_shared_slots):pathmon=$(path_util_monitor):tilepathmon=$(tile_path_monitor):pathload=$(path_util_load_only):tileactive=$(tile_path_active_only):tileload=$(tile_path_load_only):tilesrcports=$(tile_path_source_ports_only)

$(vcs_config_stamp): FORCE | $(buildpath)
	@if [ ! -f "$@" ] || [ "$$(cat $@)" != "$(vcs_config_fingerprint)" ]; then \
		echo "[VCS] Config changed → $(vcs_config_fingerprint)  (was: $$(cat $@ 2>/dev/null || echo '<none>'))"; \
		rm -f $(buildpath)/compilevcs.sh $(buildpath)/mempool_simv $(buildpath)/mempool_simvopt; \
		echo "$(vcs_config_fingerprint)" > $@; \
	fi
FORCE:

# Elaboration
.PHONY: elabvcs
elabvcs: dpivcs $(buildpath) $(vcs_config_stamp) $(buildpath)/compilevcs.sh update-opcodes
$(buildpath)/compilevcs.sh: $(bender) $(config_mk) Makefile $(MEMPOOL_DIR)/Bender.yml $(shell find {src,tb,deps} -type f)
	$(bender) script vcs --vlogan-bin="$(vcs_cmd) vlogan" --vlog-arg="$(vlogan_args)" $(vlog_defs) -t rtl -t mempool_vsim > $(buildpath)/compilevcs.sh
	echo "exit" >> $(buildpath)/compilevcs.sh
	# Call VCS
	cd $(buildpath) && \
	chmod +x compilevcs.sh && \
	./compilevcs.sh

# Compilation
.PHONY: compile_vcs_simv compile_vcs_simvopt
compile_vcs_simv: elabvcs $(buildpath)/mempool_simv
$(buildpath)/mempool_simv: $(buildpath)/compilevcs.sh $(buildpath)/$(dpi_library)/mempool_vcs_dpi.so
	cd $(buildpath) && \
	$(vcs_cmd) vcs -full64 $(top_level) -cc $(CC) -cpp $(CXX) -ld $(CXX) $(dpi_library)/mempool_vcs_dpi.so -debug_access=r -kdb -assert disable_cover -o mempool_simv

compile_vcs_simvopt: elabvcs $(buildpath)/mempool_simvopt
$(buildpath)/mempool_simvopt: $(buildpath)/compilevcs.sh $(buildpath)/$(dpi_library)/mempool_vcs_dpi.so
	cd $(buildpath) && \
	$(vcs_cmd) vcs -full64 $(top_level) -cc $(CC) -cpp $(CXX) -ld $(CXX) $(dpi_library)/mempool_vcs_dpi.so -assert disable_cover -o mempool_simvopt

# Simulation
simvcs: clean-dasm clean-path-monitor compile_vcs_simv
	rm -f $(buildpath)/transcript
	cd $(buildpath) && \
	./mempool_simv $(vcs_args) -ucli -do ../scripts/vcs/run.tcl -gui

simcvcs: clean-dasm clean-path-monitor compile_vcs_simvopt
	@if [ -f "$(vcs_config_stamp)" ] && [ "$$(cat $(vcs_config_stamp))" != "$(vcs_config_fingerprint)" ]; then \
		echo "ERROR: VCS binary was built for '$$(cat $(vcs_config_stamp))' but you requested '$(vcs_config_fingerprint)'" >&2; \
		echo "       Run 'make compile_vcs_simvopt config=$(config)' first, or delete build/mempool_simvopt." >&2; \
		exit 1; \
	fi
	rm -f $(buildpath)/transcript
	cd $(buildpath) && \
	set -o pipefail; ./mempool_simvopt $(vcs_args) | tee transcript

# DPIs
.PHONY: dpivcs
dpivcs: $(buildpath)/$(dpi_library)/mempool_vcs_dpi.so

$(buildpath)/$(dpi_library)/%_vcs.o: tb/dpi/%.cpp
	mkdir -p $(buildpath)/$(dpi_library)
	$(CXX) -shared -fPIC -std=c++11 -Bsymbolic -c $< -I$(VCS_HOME)/include -o $@

$(buildpath)/$(dpi_library)/mempool_vcs_dpi.so: $(dpi_vcs)
	mkdir -p $(buildpath)/$(dpi_library)
	$(CXX) -shared -m64 -o $(buildpath)/$(dpi_library)/mempool_vcs_dpi.so $^

################
# Verilator    #
################

VERILATOR_SRC   := $(ROOT_DIR)/tb/verilator
VERILATOR_LIBS  := $(shell find $(VERILATOR_SRC) -name "*.cc" -print | sort)
VERILATOR_INCS  := $(shell find $(VERILATOR_SRC) -name "cpp" -print | sort)
VERILATOR_EXE   := $(verilator_build)/V$(verilator_top)
VERILATOR_MK    := $(VERILATOR_EXE).mk
VERILATOR_WAIVE := $(shell find $(VERILATOR_SRC) -name "*.vlt" -print | sort)
VERILATOR_CONF  := $(VERILATOR_SRC)/verilator.flags

VERILATOR_FLAGS += -CFLAGS "-DTOPLEVEL_NAME=$(verilator_top)"
VERILATOR_FLAGS += --Mdir $(verilator_build)
VERILATOR_FLAGS += -f $(verilator_files)
VERILATOR_FLAGS += -f $(VERILATOR_CONF)
VERILATOR_FLAGS += $(VERILATOR_WAIVE)
# Disable noisy Verilator lints
VERILATOR_FLAGS += -Wno-DECLFILENAME
VERILATOR_FLAGS += -Wno-PINCONNECTEMPTY
VERILATOR_FLAGS += -Wno-IMPORTSTAR
VERILATOR_FLAGS += -Wno-EOFNEWLINE
# VERILATOR_FLAGS += --trace --trace-fst --trace-structs --trace-params --trace-max-array 1024
# VERILATOR_FLAGS += --debug

# We need to link the verilated model against LLVM's libc++.
# Define CLANG_PATH to be the path of your Clang installation.
# At IIS, check .gitlab/.gitlab-ci.yml for an example CLANG_PATH.

ifneq (${CLANG_PATH},)
  VERILATOR_FLAGS += -CFLAGS "-nostdinc++ -isystem $(CLANG_PATH)/include/c++/v1"
  VERILATOR_FLAGS += -LDFLAGS "-L $(CLANG_PATH)/lib -Wl,-rpath,$(CLANG_PATH)/lib -lc++ -nostdlib++"
  VERILATOR_FLAGS += -CFLAGS "-Wno-unknown-warning-option"
  # The injected `-nostdlib++` flag is clang-specific; ensure we build/link the
  # generated model with clang++ when CLANG_PATH is used.
  CLANG_CC  := $(CLANG_PATH)/bin/clang
  CLANG_CXX := $(CLANG_PATH)/bin/clang++
else
  CLANG_CC  ?= clang
  CLANG_CXX ?= clang++
endif

$(VERILATOR_MK): $(VERILATOR_CONF) $(VERILATOR_WAIVE) $(MEMPOOL_DIR)/Bender.yml $(shell find {src,tb,deps} -type f) $(bender) $(config_mk) Makefile
	rm -rf $(verilator_build); mkdir -p $(verilator_build)
	# Overwrite Bootaddress to L2 base while we don't have a DPI to write a wake-up
	$(eval boot_addr=$(l2_base))
	# Create Bender script of all RTL files
	$(bender) script verilator $(vlog_defs) -t rtl -t mempool_verilator -DCOMMON_CELLS_ASSERTS_OFF > $(verilator_files)
	# Append the verilator library files
	@echo '' >> $(verilator_files)
	# Append the verilator library files: Includes
	@echo $(addprefix +incdir+,$(VERILATOR_INCS)) | tr ' ' '\n' >> $(verilator_files)
	for i in $(VERILATOR_INCS); do echo "-CFLAGS -I$${i}" >> $(verilator_files); done
	# Append the CPP defines
	@echo "-CFLAGS \"$(cpp_defs)\"" >> $(verilator_files)
	# Append the verilator library files: source files
	@echo $(VERILATOR_LIBS) | tr ' ' '\n'  >> $(verilator_files)
	# Create Verilator Makefile
	$(verilator) $(VERILATOR_FLAGS) --top-module $(verilator_top)

$(VERILATOR_EXE): $(VERILATOR_MK) $(shell find $(VERILATOR_SRC) -type f) Makefile
	$(MAKE) -j10 -C $(verilator_build) -f $< CC="$(CLANG_CC)" CXX="$(CLANG_CXX)" LINK="$(CLANG_CXX)"

verilate: $(VERILATOR_EXE) $(buildpath) Makefile
	cd $(buildpath) && $(VERILATOR_EXE) $(verilator_args) | tee transcript
	# Avoid capturing the return status when running the load-throughput analysis
	if [ $(tg) -ne 1 ]; then ./scripts/return_status.sh $(buildpath)/transcript; fi

#############
# Lint      #
#############

.PHONY: lint spyglass/tmp/files

SNPS_SG ?= spyglass-2022.06
lint: spyglass/tmp/files spyglass/sdc/func.sdc spyglass/scripts/run_lint.tcl
	cd spyglass; $(SNPS_SG) sg_shell -tcl scripts/run_lint.tcl

spyglass/tmp/files: $(bender)
	mkdir -p spyglass/tmp
	$(bender) script verilator $(vlog_defs) -t rtl -t mempool_verilator | \
		sed "/^+define+.*'.*$$/s/.*/\"&\"/" > spyglass/tmp/files

################
# Tracing      #
################

# Give configuration in env
trace_env += NUM_CORES=$(num_cores)
trace_env += NUM_GROUPS=$(num_groups)
trace_env += NUM_CORES_PER_TILE=$(num_cores_per_tile)
trace_env += SEQ_MEM_SIZE=$(seq_mem_size)
trace_env += BANKING_FACTOR=$(banking_factor)
trace_env += L1_BANK_SIZE=$(l1_bank_size)
trace_env += NUM_SUB_GROUPS_PER_GROUP=$(num_sub_groups_per_group)
plot_overview ?= 1
plot_group_details ?= 0
plot_tile_details ?= 1
plot_window ?= 64
plot_section ?= 1
plot_comm ?= 1
plot_jobs ?= 1
plot_port ?= 0
plot_port_id ?= 0
plot_port_cycle_start ?= 7500
plot_port_cycle_end ?= auto
plot_port_window ?= $(plot_window)
plot_port_threshold ?= 2
plot_port_node_point ?= tcdm_remote
plot_port_graph_dir ?= $(result_dir)/analysis/path_graph
plot_port_output_dir ?= $(result_dir)/plots/port_pressure
plot_port_exact_operands ?= 1
plot_port_operand_offsets ?= 1
extract_jobs ?= 16
trace_jobs ?= $(extract_jobs)
benchmark_comm ?= 1
benchmark_stall ?= 1
parallel_kernels ?=
parallel_jobs ?= $(words $(parallel_kernels))
archive_include_traces ?= 0
plot_section_args = $(if $(plot_section),--section $(plot_section),)
plot_overview_args = $(if $(filter 1,$(plot_overview)),--overview,)
plot_group_detail_args = $(if $(filter 1,$(plot_group_details)),--group-details,)
plot_tile_detail_args = $(if $(filter 0,$(plot_tile_details)),--skip-tile-details,)
plot_force_args = $(if $(filter 1,$(force)),--force,)
plot_tiles_args = $(if $(strip $(plot_tiles)),--tiles $(plot_tiles),)
plot_topology_args = $(if $(strip $(plot_topology)),--topology $(plot_topology),)
rerun_force_args = $(if $(filter 1,$(force)),--force,)
plot_port_prefix = port$(plot_port_id)
plot_port_source_target_prefix = $(plot_port_prefix)_source_target
plot_port_source_target_dir = $(plot_port_graph_dir)/$(plot_port_source_target_prefix)_classification
plot_port_source_target_matrix = $(plot_port_source_target_dir)/$(plot_port_source_target_prefix)_source_target_matrix.csv

.PHONY: build_app benchmark plots plots_port plots_parallel plots_comm archive_result freeze rerun_stall_timeseries save_transcript save_path_monitor comm_events_real stall_timeseries_build prepare_result_data_links

prepare_result_data_links:
	@mkdir -p "$(result_data_dir)"
ifeq ($(use_scratch_result_data),1)
	@mkdir -p "$(scratch_result_data_dir)"
	@ln -snf "$(scratch_comm_events_csv)" "$(comm_events_csv)"
	@ln -snf "$(scratch_stall_timeseries_csv)" "$(stall_timeseries_csv)"
endif

build_app:
ifdef app
	$(MAKE) -C $(MEMPOOL_DIR)/software/apps/baremetal \
		app="$(app)" \
		python="$(python)" \
		config="$(config)" \
		kernel="$(kernel)" \
		variant="$(variant)" \
		das="$(das)" \
		clean
	$(MAKE) -C $(MEMPOOL_DIR)/software/apps/baremetal \
		app="$(app)" \
		python="$(python)" \
		config="$(config)" \
		kernel="$(kernel)" \
		variant="$(variant)" \
		das="$(das)" \
		$(app)
endif

benchmark: build_app
	@# Guard: refuse to overwrite existing results unless force=1
	@if { [ -f "$(result_dir)/data/results.csv" ] || \
	     [ -f "$(result_dir)/data/comm_events_benchmark.csv" ] || \
	     [ -f "$(result_dir)/data/stall_timeseries_benchmark.csv" ]; } && [ "$(force)" != "1" ]; then \
		echo "ERROR: result_dir already has benchmark data: $(result_dir)/data/" >&2; \
		echo "       Set force=1 to overwrite." >&2; \
		exit 1; \
	fi
	result_dir="$(result_dir)" app="$(app)" $(MAKE) log
	@sim_status=0; \
	$(MAKE) app="$(app)" simcvcs || sim_status=$$?; \
	result_dir="$(result_dir)" $(MAKE) save_transcript; \
	result_dir="$(result_dir)" $(MAKE) save_path_monitor; \
	if [ "$$sim_status" -ne 0 ]; then \
		exit "$$sim_status"; \
	fi
	# Re-run `trace` after simulation so `$(trace)` expands to the newly created
	# per-hart raw trace files.
	result_dir=$(result_dir) $(MAKE) -j $(trace_jobs) trace trace_jobs="$(trace_jobs)"
	# Extract benchmark CSVs from build intermediates; only the final CSVs are
	# written into result_dir/data.
	@if [ "$(benchmark_comm)" = "1" ]; then \
		result_dir=$(result_dir) $(MAKE) comm_events_real extract_jobs="$(extract_jobs)"; \
	else \
		echo "INFO: skipping communication extraction (benchmark_comm=0)"; \
	fi
	@if [ "$(benchmark_stall)" = "1" ]; then \
		result_dir=$(result_dir) $(MAKE) stall_timeseries_build extract_jobs="$(extract_jobs)"; \
	else \
		echo "INFO: skipping stall time-series extraction (benchmark_stall=0)"; \
	fi
	# Clean build intermediates — CSVs are in result_dir/data/
	rm -f $(buildpath)/trace_hart_*.dasm $(buildpath)/trace_hart_*.trace
	rm -rf $(tracepath)

log:
	mkdir -p "$(result_dir)"
	result_dir="$(result_dir)" $(MAKE) --no-print-directory prepare_result_data_links
	[ -e "$(preload)" ] && cp "$(preload)"* "$(result_dir)" || \
	([ -e "$(buildpath)/transcript" ] && \
	cp "$$(grep -oP '(?<=\+PRELOAD=)[^ \" ]*' $(buildpath)/transcript)"* "$(result_dir)" || \
	echo "No application specified")
	{ \
		printf 'app=%s\nresult_app=%s\nconfig=%s\nkernel=%s\nvariant=%s\nresult_dir=%s\nback2local=%s\nremote_req_shared_slots=%s\npath_util_monitor=%s\ntile_path_monitor=%s\npath_util_start=%s\npath_util_end=%s\npath_util_load_only=%s\ntile_path_tile=%s\ntile_path_active_only=%s\ntile_path_load_only=%s\ntile_path_source_ports_only=%s\n' \
			"$(app)" "$(result_app)" "$(config)" "$(kernel)" "$(variant)" "$(result_dir)" "$(back2local)" "$(remote_req_shared_slots)" "$(path_util_monitor)" "$(tile_path_monitor)" "$(path_util_start)" "$(path_util_end)" "$(path_util_load_only)" "$(tile_path_tile)" "$(tile_path_active_only)" "$(tile_path_load_only)" "$(tile_path_source_ports_only)"; \
		env; \
	} > "$(result_dir)/env"
	printf 'APP=%s\nKERNEL=%s\nVARIANT=%s\nCONFIG=%s\nNUM_CORES=%s\nNUM_GROUPS=%s\nNUM_CORES_PER_TILE=%s\nSEQ_MEM_SIZE=%s\nBANKING_FACTOR=%s\nL1_BANK_SIZE=%s\nNUM_SUB_GROUPS_PER_GROUP=%s\nDAS=%s\nBACK2LOCAL=%s\nREMOTE_REQ_SHARED_SLOTS=%s\nNUM_DAS_PARTITIONS=%s\nDAS_MEM_SIZE=%s\nPATH_UTIL_MONITOR=%s\nTILE_PATH_MONITOR=%s\nPATH_UTIL_START=%s\nPATH_UTIL_END=%s\nPATH_UTIL_LOAD_ONLY=%s\nTILE_PATH_TILE=%s\nTILE_PATH_ACTIVE_ONLY=%s\nTILE_PATH_LOAD_ONLY=%s\nTILE_PATH_SOURCE_PORTS_ONLY=%s\n' \
		"$(app)" "$(kernel)" "$(variant)" "$(config)" "$(num_cores)" "$(num_groups)" "$(num_cores_per_tile)" "$(seq_mem_size)" "$(banking_factor)" "$(l1_bank_size)" "$(num_sub_groups_per_group)" "$(das)" "$(back2local)" "$(remote_req_shared_slots)" "$(num_das_partitions)" "$(das_mem_size)" "$(path_util_monitor)" "$(tile_path_monitor)" "$(path_util_start)" "$(path_util_end)" "$(path_util_load_only)" "$(tile_path_tile)" "$(tile_path_active_only)" "$(tile_path_load_only)" "$(tile_path_source_ports_only)" > "$(result_dir)/topology.env"
	cp $(MEMPOOL_DIR)/config/config.mk $(result_dir)/config
	git rev-parse HEAD > "$(result_dir)/git-info.diff"
	git show --oneline -s >> "$(result_dir)/git-info.diff"
	git diff >> "$(result_dir)/git-info.diff"

save_transcript:
	@mkdir -p "$(result_dir)"
	@if [ -f "$(buildpath)/transcript" ]; then \
		cp "$(buildpath)/transcript" "$(result_dir)/transcript.tmp" && \
		mv "$(result_dir)/transcript.tmp" "$(result_dir)/transcript"; \
	else \
		echo "WARNING: transcript not found: $(buildpath)/transcript" >&2; \
	fi

save_path_monitor:
	@if [ "$(path_util_monitor)" = "1" ] || [ "$(tile_path_monitor)" = "1" ]; then \
		found=0; \
		shopt -s nullglob; \
		for file in "$(buildpath)"/path_util_group*.csv "$(buildpath)"/tile_path_tile*.csv; do \
			if [ "$$found" -eq 0 ]; then \
				mkdir -p "$(result_dir)/monitor"; \
				found=1; \
			fi; \
			cp "$$file" "$(result_dir)/monitor/"; \
		done; \
		if [ "$$found" -eq 0 ]; then \
			echo "WARNING: monitor requested but no monitor CSVs were found in $(buildpath)" >&2; \
		fi; \
	fi

trace: pre_trace $(trace) post_trace

# Ensure individual trace targets only start after pre_trace cleans the directory,
# and post_trace only starts after all traces are done.
$(trace): | pre_trace
post_trace: $(trace)

pre_trace:
	rm -rf $(tracepath)
	rm -f $(buildpath)/trace_hart_*.trace

post_trace:
	mkdir -p "$(result_dir)/data"
	result_dir="$(result_dir)" $(MAKE) save_transcript
	$(python) "$(trace_merge_script)" \
		--folder "$(tracepartspath)" \
		--expected-count "$(num_cores)" \
		--csv "$(traceresult)"
	cp $(traceresult) "$(result_dir)/data/"
	$(python) $(ROOT_DIR)/scripts/gen_avg.py --folder "$(result_dir)/data" | tee $(result_dir)/avg.txt

$(tracepath): | pre_trace
	mkdir -p "$@"

$(tracepartspath): | $(tracepath)
	mkdir -p "$@"

# Read .trace files from build/ during benchmark (no copy needed)
stall_timeseries_build:
	$(trace_env) $(python) $(ROOT_DIR)/scripts/trace_analysis/extract/_gen_stall_timeseries_batch.py \
		--folder "$(buildpath)" \
		--csv "$(result_dir)/data/stall_timeseries_benchmark.csv" \
		--benchmark-only -p --force \
		-j $(extract_jobs)

# Read .trace files from an archived result dir (for rerun_stall_timeseries)
stall_timeseries:
	$(trace_env) $(python) $(ROOT_DIR)/scripts/trace_analysis/extract/_gen_stall_timeseries_batch.py \
		--folder "$(result_dir)/traces_dasm" \
		--csv "$(result_dir)/data/stall_timeseries_benchmark.csv" \
		--benchmark-only -p --force \
		-j $(extract_jobs)

comm_events_real:
	$(trace_env) $(python_venv) $(ROOT_DIR)/scripts/trace_analysis/extract/extract_comm_events_batch.py \
		--folder "$(tracepath)" \
		--csv "$(result_dir)/data/comm_events_benchmark.csv" \
		--benchmark-only -p --force \
		-j $(extract_jobs)

# Public Stage 2 plotting entry point.
# Usage:
#   kernel=4x4_das_thesis_asm make plots
#   result_dir=results/matmul_i32_mempool/4x4_das_thesis_asm/das make plots plot_tiles="0 1" force=1
#   result_dir=results/matmul_i32_mempool/4x4_das_thesis_asm/das make plots plot_comm=0
plots:
	@if [ ! -f "$(result_dir)/data/stall_timeseries_benchmark.csv" ]; then \
		echo "ERROR: stall CSV missing: $(result_dir)/data/stall_timeseries_benchmark.csv" >&2; \
		echo "       Run 'make benchmark' first or point result_dir at an existing run." >&2; \
		exit 1; \
	fi
	@if [ "$(plot_comm)" = "1" ] && [ ! -f "$(result_dir)/data/comm_events_benchmark.csv" ]; then \
		echo "ERROR: communication CSV missing: $(result_dir)/data/comm_events_benchmark.csv" >&2; \
		echo "       The official plotting workflow is CSV-only." >&2; \
		echo "       Regenerate the CSV explicitly before running 'make plots'." >&2; \
		exit 1; \
	fi
	$(python) $(ROOT_DIR)/scripts/trace_analysis/plot/plot_all_tiles.py \
		"$(result_dir)" \
		$(plot_section_args) \
		$(plot_overview_args) \
		$(plot_group_detail_args) \
		$(plot_tile_detail_args) \
		--window $(plot_window) \
		$(plot_tiles_args) \
		$(plot_topology_args) \
		$(plot_force_args) \
		-j $(plot_jobs)
	@if [ "$(plot_comm)" = "1" ]; then \
		echo "INFO: using existing communication CSV: $(result_dir)/data/comm_events_benchmark.csv"; \
		if [ "$(force)" = "1" ]; then \
			echo "INFO: force=1 affects plot outputs only; it does not rebuild communication CSVs."; \
		fi; \
			set -e; \
			$(MAKE) --no-print-directory \
				result_dir="$(result_dir)" \
				plot_section="$(plot_section)" \
				python_venv="$(python_venv)" \
				plots_comm; \
	fi
	@if [ "$(plot_port)" = "1" ]; then \
		$(MAKE) --no-print-directory \
			result_dir="$(result_dir)" \
			plot_section="$(plot_section)" \
			plot_port_id="$(plot_port_id)" \
			plot_port_cycle_start="$(plot_port_cycle_start)" \
			plot_port_cycle_end="$(plot_port_cycle_end)" \
			plot_port_window="$(plot_port_window)" \
			plot_port_threshold="$(plot_port_threshold)" \
			plot_port_node_point="$(plot_port_node_point)" \
			plot_port_graph_dir="$(plot_port_graph_dir)" \
			plot_port_output_dir="$(plot_port_output_dir)" \
			plot_port_exact_operands="$(plot_port_exact_operands)" \
			plot_port_operand_offsets="$(plot_port_operand_offsets)" \
			python="$(python)" \
			force="$(force)" \
			plots_port; \
	fi

plots_port:
	@if [ ! -d "$(result_dir)/monitor" ]; then \
		echo "ERROR: monitor directory missing: $(result_dir)/monitor" >&2; \
		echo "       Re-run the benchmark with tile_path_monitor=1 and tile_path_source_ports_only=1." >&2; \
		exit 1; \
	fi
	@if ! ls "$(result_dir)"/monitor/tile_path_tile*.csv >/dev/null 2>&1; then \
		echo "ERROR: no tile path monitor CSVs found in $(result_dir)/monitor" >&2; \
		exit 1; \
	fi
	@if [ "$(plot_port_cycle_end)" = "auto" ] && [ ! -f "$(result_dir)/data/stall_timeseries_benchmark.csv" ]; then \
		echo "ERROR: cannot infer plot_port_cycle_end=auto without $(result_dir)/data/stall_timeseries_benchmark.csv" >&2; \
		exit 1; \
	fi
	@plot_port_cycle_end_resolved="$(plot_port_cycle_end)"; \
	if [ "$$plot_port_cycle_end_resolved" = "auto" ]; then \
		plot_port_cycle_end_resolved="$$( \
			$(python) -c 'import csv,sys; path=sys.argv[1]; section=int(sys.argv[2]); cycles=[]; f=open(path, newline=""); reader=csv.DictReader(f); [cycles.append(int(row["cycle"])) for row in reader if row.get("section") not in (None, "") and int(row["section"]) == section]; f.close(); print(((max(cycles)+499)//500)*500 if cycles else "")' \
				"$(result_dir)/data/stall_timeseries_benchmark.csv" "$(plot_section)" \
		)"; \
	fi; \
	if [ -z "$$plot_port_cycle_end_resolved" ]; then \
		echo "ERROR: no cycles found for plot_section=$(plot_section); cannot infer plot_port_cycle_end" >&2; \
		exit 1; \
	fi; \
	if [ "$(plot_port_exact_operands)" = "1" ]; then \
		plot_port_operand_args="--require-exact-operands"; \
		plot_port_classified_operand_args=""; \
	else \
		plot_port_operand_args="--allow-legacy-route-operands"; \
		plot_port_classified_operand_args="--allow-legacy-route-operands"; \
	fi; \
	plot_port_force_args=""; \
	if [ "$(force)" = "1" ]; then \
		plot_port_force_args="--force"; \
	fi; \
	echo "INFO: port plots use cycles $(plot_port_cycle_start)..$$plot_port_cycle_end_resolved"; \
	$(python) $(ROOT_DIR)/scripts/trace_analysis/path_graph_dataset.py \
		"$(result_dir)" \
		--output-dir "$(plot_port_graph_dir)" \
		--point "$(plot_port_node_point)" \
		--cycle-start "$(plot_port_cycle_start)" \
		--cycle-end "$$plot_port_cycle_end_resolved" \
		--window-size "$(plot_port_window)" \
		--no-html; \
	$(python) $(ROOT_DIR)/scripts/trace_analysis/plot/plot_port_pressure.py \
		"$(plot_port_graph_dir)" \
		--metric-source node-state \
		--node-point "$(plot_port_node_point)" \
		--cycle-start "$(plot_port_cycle_start)" \
		--cycle-end "$$plot_port_cycle_end_resolved" \
		--average-section "$(plot_section)" \
		--window "$(plot_port_window)" \
		--output-dir "$(plot_port_output_dir)" \
		--prefix all_ports_pressure \
		$$plot_port_force_args; \
	$(python) $(ROOT_DIR)/scripts/trace_analysis/plot/plot_port_fanin_heatmap.py \
		"$(plot_port_graph_dir)" \
		--cycle-start "$(plot_port_cycle_start)" \
		--cycle-end "$$plot_port_cycle_end_resolved" \
		--window "$(plot_port_window)" \
		--port "$(plot_port_id)" \
		--node-point "$(plot_port_node_point)" \
		--output-dir "$(plot_port_output_dir)" \
		--prefix "$(plot_port_prefix)_tile_fanin" \
		$$plot_port_operand_args \
		$$plot_port_force_args; \
	$(python) $(ROOT_DIR)/scripts/trace_analysis/plot/plot_fanin_flow_overlay.py \
		"$(plot_port_graph_dir)" \
		--cycle-start "$(plot_port_cycle_start)" \
		--cycle-end "$$plot_port_cycle_end_resolved" \
		--window "$(plot_port_window)" \
		--port "$(plot_port_id)" \
		--node-point "$(plot_port_node_point)" \
		--threshold "$(plot_port_threshold)" \
		--output-dir "$(plot_port_output_dir)" \
		--prefix "$(plot_port_prefix)_fanin_flow" \
		$$plot_port_operand_args \
		$$plot_port_force_args; \
	$(python) $(ROOT_DIR)/scripts/trace_analysis/classify_source_targets.py \
		"$(plot_port_graph_dir)" \
		--cycle-start "$(plot_port_cycle_start)" \
		--cycle-end "$$plot_port_cycle_end_resolved" \
		--port "$(plot_port_id)" \
		--node-point "$(plot_port_node_point)" \
		--high-fanin-threshold "$(plot_port_threshold)" \
		--prefix "$(plot_port_source_target_prefix)" \
		$$plot_port_operand_args \
		$$plot_port_force_args; \
	$(python) $(ROOT_DIR)/scripts/trace_analysis/plot/plot_source_target_classification.py \
		"$(plot_port_source_target_matrix)" \
		--output-dir "$(plot_port_output_dir)" \
		--prefix "$(plot_port_source_target_prefix)" \
		$$plot_port_classified_operand_args \
		$$plot_port_force_args; \
	if [ "$(plot_port_operand_offsets)" = "1" ]; then \
		$(python) $(ROOT_DIR)/scripts/trace_analysis/plot/plot_source_target_operand_offsets.py \
			"$(plot_port_source_target_matrix)" \
			--port "$(plot_port_id)" \
			--output-dir "$(plot_port_output_dir)" \
			--prefix "$(plot_port_prefix)_operand_offsets" \
			$$plot_port_classified_operand_args \
			$$plot_port_force_args; \
	fi

# Public Stage 2 parallel plotting entry point.
# Usage:
#   config=mempool parallel_kernels="4x4_das_thesis_asm" make plots_parallel
#   config=terapool parallel_kernels="4x4_das_thesis_asm" make plots_parallel
plots_parallel:
	@if [ -n "$(result_dir)" ]; then \
		echo "ERROR: plots_parallel does not accept result_dir; use parallel_kernels instead." >&2; \
		exit 1; \
	fi
	@if [ -z "$(strip $(parallel_kernels))" ]; then \
		echo "ERROR: parallel_kernels is empty. Example: parallel_kernels=\"4x4_das_thesis_asm\"" >&2; \
		exit 1; \
	fi
	@if [ -z "$(app)" ]; then \
		echo "ERROR: app is not set. Example: app=matmul_i32" >&2; \
		exit 1; \
	fi
	@if [ -z "$(variant)" ]; then \
		echo "ERROR: variant is not set. Example: variant=das" >&2; \
		exit 1; \
	fi
	@printf '%s\n' $(parallel_kernels) | xargs -I{} -P $(parallel_jobs) sh -c '\
		$(MAKE) --no-print-directory \
			app="$(app)" \
			config="$(config)" \
			variant="$(variant)" \
			kernel="$$1" \
			python="$(python)" \
			python_venv="$(python_venv)" \
			plot_section="$(plot_section)" \
			plot_overview="$(plot_overview)" \
			plot_group_details="$(plot_group_details)" \
			plot_tile_details="$(plot_tile_details)" \
			plot_window="$(plot_window)" \
			plot_comm="$(plot_comm)" \
			plot_tiles="$(plot_tiles)" \
			plot_topology="$(plot_topology)" \
			force="$(force)" \
			plots' sh {}

# Public communication-analysis plotting entry point.
# Generates thesis-quality communication figures (PNG + PDF).
# Usually invoked automatically by `make plots`; can also be run directly
# after extracting communication events.
# Usage:
#   kernel=4x4_das_thesis_asm make plots_comm
#   result_dir=results/matmul_i32_mempool/4x4_das_thesis_asm/das make plots_comm
plots_comm:
	@if [ ! -f "$(result_dir)/data/comm_events_benchmark.csv" ]; then \
		echo "ERROR: comm_events_benchmark.csv missing: $(result_dir)/data/comm_events_benchmark.csv" >&2; \
		echo "       Run extract_comm_events.py first." >&2; \
		exit 1; \
	fi
	$(python_venv) $(ROOT_DIR)/scripts/trace_analysis/plot/plot_comm_thesis.py \
		"$(result_dir)" \
		$(plot_section_args)

# Freeze a selected finished result directory for thesis/reference use.
# Usage:
#   result_dir=results/matmul_i32_mempool/4x4_das_thesis_asm/das \
#   archive_dir=/scratch/$$USER/mempool/thesis_results/4x4_das_thesis_asm_das \
#   make archive_result
archive_result:
	@if [ -z "$(archive_dir)" ]; then \
		echo "ERROR: archive_dir is not set." >&2; \
		echo "       Example: archive_dir=/scratch/$$USER/mempool/thesis_results/4x4_das_thesis_asm_das" >&2; \
		exit 1; \
	fi
	@if [ ! -d "$(result_dir)" ]; then \
		echo "ERROR: result_dir not found: $(result_dir)" >&2; \
		exit 1; \
	fi
	@if [ ! -f "$(result_dir)/data/stall_timeseries_benchmark.csv" ]; then \
		echo "ERROR: result_dir does not look like a finished benchmark result: $(result_dir)" >&2; \
		echo "       Missing: $(result_dir)/data/stall_timeseries_benchmark.csv" >&2; \
		exit 1; \
	fi
	@if [ ! -d "$(result_dir)/plots" ]; then \
		echo "ERROR: plots directory missing: $(result_dir)/plots" >&2; \
		echo "       Run 'make plots' before archiving a thesis result." >&2; \
		exit 1; \
	fi
	@if [ -e "$(archive_dir)" ] && [ "$(force)" != "1" ]; then \
		echo "ERROR: archive_dir already exists: $(archive_dir)" >&2; \
		echo "       Set force=1 to overwrite." >&2; \
		exit 1; \
	fi
	@tmp_dir="$(archive_dir).tmp.$$PPID"; \
		rm -rf "$$tmp_dir"; \
		mkdir -p "$$(dirname "$(archive_dir)")"; \
		mkdir -p "$$tmp_dir"; \
		cp -aL "$(result_dir)/." "$$tmp_dir/"; \
		if [ "$(archive_include_traces)" != "1" ]; then \
			rm -rf "$$tmp_dir/traces_raw" "$$tmp_dir/traces_dasm"; \
		fi; \
		{ \
			printf 'SOURCE_RESULT_DIR=%s\n' "$(abspath $(result_dir))"; \
			printf 'ARCHIVED_AT=%s\n' "$$(date -Iseconds)"; \
			printf 'ARCHIVE_INCLUDE_TRACES=%s\n' "$(archive_include_traces)"; \
			if [ -n "$(archive_note)" ]; then \
				printf 'NOTE=%s\n' "$(archive_note)"; \
			fi; \
		} > "$$tmp_dir/archive.meta"; \
		if [ -e "$(archive_dir)" ]; then \
			rm -rf "$(archive_dir)"; \
		fi; \
		mv "$$tmp_dir" "$(archive_dir)"; \
		echo "Archived result to $(archive_dir)"; \
		echo "NOTE: For reproducibility, commit and push the exact source state that produced this result."; \
		echo "      If the benchmark was generated from uncommitted local changes, freezing alone is not a full source snapshot.";

freeze: archive_result

# Public reprocessing entry point.
# Usage:
#   result_dir=results/matmul_i32_mempool/4x4_das_thesis_asm/das make rerun_stall_timeseries force=1
# This reuses saved traces and metadata; it does not re-run simulation.
rerun_stall_timeseries:
	@if [ ! -d "$(result_dir)/traces_dasm" ]; then \
		echo "ERROR: traces directory missing: $(result_dir)/traces_dasm" >&2; \
		echo "       Point result_dir at an existing benchmark run." >&2; \
		exit 1; \
	fi
	$(python) $(ROOT_DIR)/scripts/trace_analysis/extract/rerun_stall_timeseries.py \
		"$(result_dir)" \
		--jobs $(extract_jobs) \
		--benchmark-only -p \
		$(rerun_force_args)

$(buildpath)/%.trace: $(buildpath)/%.dasm | $(tracepath) $(tracepartspath)
	mkdir -p "$(tracepath)" "$(tracepartspath)"
	$(INSTALL_DIR)/riscv-isa-sim/bin/spike-dasm < $< > $(tracepath)/$*
	$(trace_env) $(python) $(ROOT_DIR)/scripts/gen_trace.py -p --csv $(tracepartspath)/$*.csv $(tracepath)/$* > $@

tracevis:
	$(MEMPOOL_DIR)/scripts/tracevis.py $(preload) $(buildpath)/*.trace -o $(buildpath)/tracevis.json

############################
# Unit tests simulation    #
############################

TESTS_DIR := $(abspath $(ROOT_DIR)/../software/riscv-tests/isa)
include $(TESTS_DIR)/snitch_isa.mk

test_result_dir_vsim := $(resultpath)/test_result_vsim
test_result_dir_verilate := $(resultpath)/test_result_verilate

tests_vsim := $(addsuffix .out,$(addprefix $(test_result_dir_vsim)/,$(rtl_mempool_tests)))
tests_verilate := $(addsuffix .out,$(addprefix $(test_result_dir_verilate)/,$(rtl_mempool_tests)))

simc_test: clean-dasm compile $(buildpath) $(tests_vsim)
verilate_test: clean-dasm $(VERILATOR_EXE) $(buildpath) $(tests_verilate)

$(tests_vsim): $(test_result_dir_vsim)/%.out : $(app_path)/%
	mkdir -p $(test_result_dir_vsim)
	cd $(buildpath) && \
	$(questa_cmd) vsim -c $(questa_args) +PRELOAD=$< $(library).$(top_level) -do "run -a"
	./scripts/return_status.sh $(buildpath)/transcript > $@

$(tests_verilate): $(test_result_dir_verilate)/%.out : $(app_path)/%
	mkdir -p $(test_result_dir_verilate)
	cd $(buildpath) && $(VERILATOR_EXE) --meminit=ram,$< | tee transcript
	./scripts/return_status.sh $(buildpath)/transcript > $@

################
# Helper       #
################

# Bootrom
src/bootrom.sv: $(MEMPOOL_DIR)/software/runtime/bootrom.img $(config_mk) Makefile
	$(python) ./scripts/generate_bootrom.py --sv --output src/bootrom --datawidth $(axi_data_width) $<

$(MEMPOOL_DIR)/software/runtime/bootrom.img:
	make -C $(MEMPOOL_DIR)/software runtime/bootrom.img

# Control Register
CONTROL_REG_DIR := src/control_registers
$(CONTROL_REG_DIR)/control_registers_reg_top.sv: $(CONTROL_REG_DIR)/control_registers.hjson
	make -C $(CONTROL_REG_DIR) all

# Clean targets
.PHONY: clean clean-dasm clean-trace update-opcodes

update-opcodes:
	make -C $(MEMPOOL_DIR) update-opcodes

clean:
	@rm -rf $(buildpath)
	@rm -rf $(verilator_build)

clean-dasm:
	rm -rf $(buildpath)/*.dasm

clean-trace:
	rm -rf $(buildpath)/*.trace
