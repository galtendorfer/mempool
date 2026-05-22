# Copyright 2021 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Author: Matheus Cavalcante, ETH Zurich
#         Samuel Riedel, ETH Zurich

SHELL = /usr/bin/env bash
ROOT_DIR := $(patsubst %/,%, $(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
MEMPOOL_DIR := $(shell git rev-parse --show-toplevel 2>/dev/null || echo $$MEMPOOL_DIR)
SOFTWARE_DIR := $(abspath $(ROOT_DIR)/../..)
APPS_DIR := $(ROOT_DIR)
BIN_ROOT_DIR := $(abspath $(SOFTWARE_DIR)/bin/$(subst $(SOFTWARE_DIR),,$(APPS_DIR)))
BIN_DIR := $(BIN_ROOT_DIR)
ifneq ($(strip $(app)),)
ifneq ($(strip $(config)),)
ifneq ($(strip $(kernel)),)
ifneq ($(strip $(variant)),)
BIN_DIR := $(BIN_ROOT_DIR)/$(app)_$(config)/$(kernel)/$(variant)
endif
endif
endif
endif
DATA_DIR := $(abspath $(SOFTWARE_DIR)/data)
RUNTIME_DIR := $(abspath $(SOFTWARE_DIR)/runtime)
# This will overwrite the ROOT_DIR variable from the included makefile
include $(RUNTIME_DIR)/runtime.mk

MATMUL_I32_APPS := matmul_i32

ifneq ($(filter $(MATMUL_I32_APPS),$(MAKECMDGOALS)),)
        # Skip verification by default (saves huge simulation time). Use verify=1 to enable.
        ifneq ($(verify),1)
          DEFINES += -DSKIP_VERIFY
        endif
	ifeq ($(kernel),2x2_xpulpv2)
		DEFINES += -DMATMUL_I32_KERNEL_2X2_XPULPV2
		ifneq ($(XPULPIMG),1)
$(error kernel=2x2_xpulpv2 requires xpulpimg=1)
		endif
	else ifeq ($(kernel),2x2_rv32im)
		DEFINES += -DMATMUL_I32_KERNEL_2X2_RV32IM
	else ifeq ($(kernel),4x4)
		DEFINES += -DMATMUL_I32_KERNEL_4X4
	else ifeq ($(kernel),4x4_conflict_opt)
		DEFINES += -DMATMUL_I32_KERNEL_4X4_CONFLICT_OPT
	else ifeq ($(kernel),4x4_asm)
		DEFINES += -DMATMUL_I32_KERNEL_4X4_ASM
		ifneq ($(XPULPIMG),1)
$(error kernel=4x4_asm requires xpulpimg=1)
		endif
	else ifeq ($(kernel),4x4_conflict_opt_asm)
		DEFINES += -DMATMUL_I32_KERNEL_4X4_CONFLICT_OPT_ASM
		ifneq ($(XPULPIMG),1)
$(error kernel=4x4_conflict_opt_asm requires xpulpimg=1)
		endif
	else ifneq ($(strip $(kernel)),)
$(error Unsupported kernel=$(kernel) for the matmul_i32 family. Supported values: 2x2_xpulpv2 2x2_rv32im 4x4 4x4_conflict_opt 4x4_asm 4x4_conflict_opt_asm)
	endif
endif

ifneq ($(filter matmul_i32,$(MAKECMDGOALS)),)
ifeq ($(das),1)
ifneq ($(filter 4x4_asm 4x4_conflict_opt_asm,$(kernel)),$(kernel))
$(error DAS-enabled matmul_i32 only supports kernel=4x4_asm or kernel=4x4_conflict_opt_asm)
endif
endif
endif

APPS := $(patsubst $(APPS_DIR)/%/main.c,%,$(shell find $(APPS_DIR) -name "main.c"))
BINARIES := $(addprefix $(BIN_DIR)/,$(APPS))
ALL := $(APPS)

FP_SUFFIXES := f16 f32 f8
I_SUFFIXES := q16 q32 i16 i32 i8
I_APPS := $(foreach suf, $(FP_SUFFIXES), $(filter %_$(suf), $(ALL)))
FP_APPS := $(foreach suf, $(I_SUFFIXES), $(filter %_$(suf), $(ALL)))
# Filter out applications
ALL_GCC := $(filter-out $(I_APPS), $(ALL))
ALL_LLVM := $(filter-out $(FP_APPS), $(ALL))

# Make all applications
all: $(ALL_GCC)

# Make only applications with full llvm support
all_llvm: $(ALL_LLVM)

$(APPS): % : $(BIN_DIR)/% $(APPS_DIR)/Makefile $(shell find $(RUNTIME_DIR)/**.{S,c,h,ld} -type f)

.PHONY: $(BINARIES)
$(BINARIES): $(BIN_DIR)/%: %/main.c.o $(RUNTIME) $(LINKER_SCRIPT) data_%.h update-opcodes
	mkdir -p $(dir $@)
	$(RISCV_CC) -Iinclude -o $@ $< $(RUNTIME) $(RISCV_LDFLAGS) -T$(RUNTIME_DIR)/link.ld
	$(RISCV_OBJDUMP) $(RISCV_OBJDUMP_FLAGS) -D $@ > $@.dump

# Helper targets
update-opcodes:
	make -C $(MEMPOOL_DIR) update-opcodes

.PHONY: clean
clean:
	rm -vf $(BINARIES)
	rm -vf $(addsuffix .dump,$(BINARIES))
	rm -vf $(addsuffix /main.c.o,$(APPS))
	rm -vf $(RUNTIME)
	rm -vf $(LINKER_SCRIPT)
	rm -vf $(wildcard $(DATA_DIR)/data_*.h)

.INTERMEDIATE: $(addsuffix /main.c.o,$(APPS))