// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Samuel Riedel, ETH Zurich

#include <stdint.h>
#include <string.h>

#include "alloc.h"
#include "dma.h"
#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "data_matmul_i32_das.h"

#include "baremetal/mempool_checks.h"
#include "baremetal/mempool_matmul_i32p.h"

#ifndef NUM_DAS_PARTITIONS
#error "matmul_i32_das requires das=1 so NUM_DAS_PARTITIONS is defined."
#endif

#define NUM_TILES (NUM_CORES / NUM_CORES_PER_TILE)

#ifndef TILES_PER_PARTITION
#define TILES_PER_PARTITION (NUM_TILES / NUM_DAS_PARTITIONS)
#endif

#if (TILES_PER_PARTITION < 1) || (TILES_PER_PARTITION > NUM_TILES)
#error "TILES_PER_PARTITION must be within [1, NUM_TILES]."
#endif

#if (NUM_TILES % TILES_PER_PARTITION) != 0
#error "TILES_PER_PARTITION must divide NUM_TILES exactly."
#endif

#if (TILES_PER_PARTITION & (TILES_PER_PARTITION - 1)) != 0
#error "TILES_PER_PARTITION must be a power of two."
#endif

#if (defined(MATMUL_I32_KERNEL_2X2_XPULPV2) + \
     defined(MATMUL_I32_KERNEL_2X2_RV32IM) + \
     defined(MATMUL_I32_KERNEL_4X4) + \
     defined(MATMUL_I32_KERNEL_4X4_CONFLICT_OPT) + \
     defined(MATMUL_I32_KERNEL_4X4_ASM) + \
     defined(MATMUL_I32_KERNEL_4X4_CONFLICT_OPT_ASM)) > 1
#error "Select exactly one MATMUL_I32 kernel."
#endif

int32_t l1_B[matrix_N * matrix_P] __attribute__((section(".l1_prio")));

int32_t *volatile shared_a __attribute__((section(".l1")));
int32_t *volatile shared_c __attribute__((section(".l1")));
uint32_t volatile init_status __attribute__((section(".l1")));

static uint32_t active_matmul_cores(uint32_t available_cores) {
  uint32_t active_cores = available_cores;

#if defined(MATMUL_I32_KERNEL_4X4) || \
    defined(MATMUL_I32_KERNEL_4X4_CONFLICT_OPT) || \
    defined(MATMUL_I32_KERNEL_4X4_ASM) || \
    defined(MATMUL_I32_KERNEL_4X4_CONFLICT_OPT_ASM)
  uint32_t max_tiles = (matrix_M / 4) * (matrix_P / 4);
#else
  uint32_t max_tiles = (matrix_M / 2) * (matrix_P / 2);
#endif

  if (max_tiles > 0 && active_cores > max_tiles) {
    active_cores = max_tiles;
  }

  return active_cores;
}

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  uint32_t kernel_cores = active_matmul_cores(num_cores);
  uint32_t a_size = matrix_M * matrix_N * sizeof(int32_t);
  uint32_t b_size = matrix_N * matrix_P * sizeof(int32_t);
  uint32_t c_size = matrix_M * matrix_P * sizeof(int32_t);

  mempool_init(core_id);
  mempool_barrier_init(core_id);

  if (core_id == 0) {
    alloc_t *das_alloc;

    init_status = 0;
    mempool_dynamic_heap_alloc_init(core_id);
    das_alloc = get_dynamic_heap_alloc();

    shared_a = (int32_t *)partition_malloc(das_alloc, a_size);
    shared_c = (int32_t *)partition_malloc(das_alloc, c_size);

    if (!shared_a || !shared_c) {
      printf("ERROR: matmul_i32_das allocation failed\n");
      init_status = 1;
    } else {
      // Keep B interleaved while A and C use DAS-managed placement.
      das_config(0, TILES_PER_PARTITION, (uint32_t)shared_a, a_size);
      das_config(1, TILES_PER_PARTITION, (uint32_t)shared_c, c_size);

      dma_memcpy_blocking(shared_a, l2_A, a_size);
      dma_memcpy_blocking(l1_B, l2_B, b_size);
    }
  }
  mempool_barrier(num_cores);

  if (init_status != 0) {
    return 1;
  }

  mempool_start_benchmark();

#if defined(MATMUL_I32_KERNEL_2X2_XPULPV2)
  #ifndef __XPULPIMG
  #error "MATMUL_I32_KERNEL_2X2_XPULPV2 requires __XPULPIMG."
  #endif
  if (core_id < kernel_cores) {
    matmul_unrolled_2x2_parallel_i32_xpulpv2(shared_a, l1_B, shared_c,
                                             matrix_M, matrix_N, matrix_P,
                                             core_id, kernel_cores);
  }
#elif defined(MATMUL_I32_KERNEL_2X2_RV32IM)
  if (core_id < kernel_cores) {
    matmul_unrolled_2x2_parallel_i32_rv32im(shared_a, l1_B, shared_c, matrix_M,
                                            matrix_N, matrix_P, core_id,
                                            kernel_cores);
  }
#elif defined(MATMUL_I32_KERNEL_4X4)
  if (core_id < kernel_cores) {
    mat_mul_unrolled_4x4_parallel(shared_a, l1_B, shared_c, matrix_M, matrix_N,
                                  matrix_P, core_id, kernel_cores);
  }
#elif defined(MATMUL_I32_KERNEL_4X4_CONFLICT_OPT)
  if (core_id < kernel_cores) {
    mat_mul_unrolled_4x4_conflict_opt_parallel(
        shared_a, l1_B, shared_c, matrix_M, matrix_N, matrix_P, core_id,
        kernel_cores);
  }
#elif defined(MATMUL_I32_KERNEL_4X4_ASM)
  if (core_id < kernel_cores) {
    mat_mul_unrolled_4x4_parallel_asm(shared_a, l1_B, shared_c, matrix_M,
                                      matrix_N, matrix_P, core_id,
                                      kernel_cores);
  }
#elif defined(MATMUL_I32_KERNEL_4X4_CONFLICT_OPT_ASM)
  if (core_id < kernel_cores) {
    mat_mul_unrolled_4x4_conflict_opt_parallel_asm(
        shared_a, l1_B, shared_c, matrix_M, matrix_N, matrix_P, core_id,
        kernel_cores);
  }
#else
  #ifdef __XPULPIMG
  if (core_id < kernel_cores) {
    matmul_unrolled_2x2_parallel_i32_xpulpv2(shared_a, l1_B, shared_c,
                                             matrix_M, matrix_N, matrix_P,
                                             core_id, kernel_cores);
  }
  #else
  if (core_id < kernel_cores) {
    matmul_unrolled_2x2_parallel_i32_rv32im(shared_a, l1_B, shared_c, matrix_M,
                                            matrix_N, matrix_P, core_id,
                                            kernel_cores);
  }
  #endif
#endif

  mempool_stop_benchmark();
  mempool_log_barrier(2, core_id);

#ifndef SKIP_VERIFY
  mempool_check_i32(shared_c, l2_C, matrix_M * matrix_P, 0, 0);
  mempool_barrier(num_cores);
#endif

  if (core_id == 0) {
    alloc_t *das_alloc = get_dynamic_heap_alloc();
    partition_free(das_alloc, shared_c);
    partition_free(das_alloc, shared_a);
  }

  return 0;
}