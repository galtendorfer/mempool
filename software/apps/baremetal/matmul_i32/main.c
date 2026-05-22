// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Samuel Riedel, ETH Zurich

#include <stdint.h>
#include <string.h>

#ifdef NUM_DAS_PARTITIONS
#include "alloc.h"
#endif
#include "dma.h"
#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "data_matmul_i32.h"

#include "baremetal/mempool_checks.h"
#include "baremetal/mempool_matmul_i32p.h"

/******************************************************************************/
/*                                                                            */
/*                    Compile-Time Configuration                              */
/*                                                                            */
/******************************************************************************/

#if (defined(MATMUL_I32_KERNEL_2X2_XPULPV2) + \
  defined(MATMUL_I32_KERNEL_2X2_RV32IM) + \
  defined(MATMUL_I32_KERNEL_4X4) + \
  defined(MATMUL_I32_KERNEL_4X4_CONFLICT_OPT) + \
  defined(MATMUL_I32_KERNEL_4X4_ASM) + \
  defined(MATMUL_I32_KERNEL_4X4_CONFLICT_OPT_ASM) + \
  defined(MATMUL_I32_KERNEL_4X4_DAS_THESIS_ASM)) > 1
#error "Select exactly one MATMUL_I32 kernel."
#endif

#ifdef NUM_DAS_PARTITIONS
#if !defined(MATMUL_I32_KERNEL_4X4_ASM) && \
  !defined(MATMUL_I32_KERNEL_4X4_CONFLICT_OPT_ASM) && \
  !defined(MATMUL_I32_KERNEL_4X4_DAS_THESIS_ASM)
#error "DAS-enabled matmul_i32 only supports 4x4_asm, 4x4_conflict_opt_asm, and 4x4_das_thesis_asm."
#endif

#define NUM_TILES (NUM_CORES / NUM_CORES_PER_TILE)

// Paper DAS placement: localize A/C and keep B fully interleaved.
#define A_TILES_PER_PARTITION 1
#define B_TILES_PER_PARTITION NUM_TILES
#define C_TILES_PER_PARTITION 1

#if (A_TILES_PER_PARTITION < 1) || (A_TILES_PER_PARTITION > NUM_TILES)
#error "A_TILES_PER_PARTITION must be within [1, NUM_TILES]."
#endif

#if (B_TILES_PER_PARTITION < 1) || (B_TILES_PER_PARTITION > NUM_TILES)
#error "B_TILES_PER_PARTITION must be within [1, NUM_TILES]."
#endif

#if (C_TILES_PER_PARTITION < 1) || (C_TILES_PER_PARTITION > NUM_TILES)
#error "C_TILES_PER_PARTITION must be within [1, NUM_TILES]."
#endif

#if (NUM_TILES % A_TILES_PER_PARTITION) != 0
#error "A_TILES_PER_PARTITION must divide NUM_TILES exactly."
#endif

#if (NUM_TILES % B_TILES_PER_PARTITION) != 0
#error "B_TILES_PER_PARTITION must divide NUM_TILES exactly."
#endif

#if (NUM_TILES % C_TILES_PER_PARTITION) != 0
#error "C_TILES_PER_PARTITION must divide NUM_TILES exactly."
#endif

#if (A_TILES_PER_PARTITION & (A_TILES_PER_PARTITION - 1)) != 0
#error "A_TILES_PER_PARTITION must be a power of two."
#endif

#if (B_TILES_PER_PARTITION & (B_TILES_PER_PARTITION - 1)) != 0
#error "B_TILES_PER_PARTITION must be a power of two."
#endif

#if (C_TILES_PER_PARTITION & (C_TILES_PER_PARTITION - 1)) != 0
#error "C_TILES_PER_PARTITION must be a power of two."
#endif
#endif

/******************************************************************************/
/*                                                                            */
/*                         L1 Working Buffers                                 */
/*                                                                            */
/******************************************************************************/

#ifdef NUM_DAS_PARTITIONS
int32_t *volatile l1_A __attribute__((section(".l1")));
int32_t *volatile l1_B __attribute__((section(".l1")));
int32_t *volatile l1_C __attribute__((section(".l1")));
#else
int32_t l1_A[matrix_M * matrix_N] __attribute__((section(".l1_prio")));
int32_t l1_B[matrix_N * matrix_P] __attribute__((section(".l1_prio")));
int32_t l1_C[matrix_M * matrix_P] __attribute__((section(".l1_prio")));
#endif

/******************************************************************************/
/*                                                                            */
/*                      Core Participation Helper                             */
/*                                                                            */
/******************************************************************************/

static uint32_t active_matmul_cores(uint32_t available_cores) {
  uint32_t active_cores = available_cores;

#if defined(MATMUL_I32_KERNEL_4X4) || \
    defined(MATMUL_I32_KERNEL_4X4_CONFLICT_OPT) || \
    defined(MATMUL_I32_KERNEL_4X4_ASM) || \
    defined(MATMUL_I32_KERNEL_4X4_CONFLICT_OPT_ASM) || \
    defined(MATMUL_I32_KERNEL_4X4_DAS_THESIS_ASM)
  uint32_t max_tiles = (matrix_M / 4) * (matrix_P / 4);
#else
  uint32_t max_tiles = (matrix_M / 2) * (matrix_P / 2);
#endif

  if (max_tiles > 0 && active_cores > max_tiles) {
    active_cores = max_tiles;
  }

  return active_cores;
}

/******************************************************************************/
/*                                                                            */
/*                        Benchmark Entry Point                               */
/*                                                                            */
/******************************************************************************/

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  uint32_t kernel_cores = active_matmul_cores(num_cores);

/****************************************************************************/
/*                                                                          */
/*                     Mode-Specific Buffer Setup                            */
/*                                                                          */
/****************************************************************************/

#ifdef NUM_DAS_PARTITIONS
  uint32_t a_size = matrix_M * matrix_N * sizeof(int32_t);
  uint32_t b_size = matrix_N * matrix_P * sizeof(int32_t);
  uint32_t c_size = matrix_M * matrix_P * sizeof(int32_t);

  mempool_init(core_id);
#endif

mempool_barrier_init(core_id);

#ifdef NUM_DAS_PARTITIONS
  if (core_id == 0) {
    alloc_t *das_alloc;

    mempool_dynamic_heap_alloc_init(core_id);
    das_alloc = get_dynamic_heap_alloc();

    l1_A = (int32_t *)partition_malloc(das_alloc, a_size);
    l1_B = (int32_t *)partition_malloc(das_alloc, b_size);
    l1_C = (int32_t *)partition_malloc(das_alloc, c_size);

    das_config(0, A_TILES_PER_PARTITION, (uint32_t)l1_A, a_size);
    das_config(1, B_TILES_PER_PARTITION, (uint32_t)l1_B, b_size);
    das_config(2, C_TILES_PER_PARTITION, (uint32_t)l1_C, c_size);

    dma_memcpy_blocking(l1_A, l2_A, a_size);
    dma_memcpy_blocking(l1_B, l2_B, b_size);
  }
  mempool_barrier(num_cores);
#else
  // Initialize data
  if (core_id == 0) {
    dma_memcpy_blocking(l1_A, l2_A, matrix_M * matrix_N * sizeof(int32_t));
    dma_memcpy_blocking(l1_B, l2_B, matrix_N * matrix_P * sizeof(int32_t));
  }
  mempool_barrier(num_cores);
#endif

  // Snapshot buffer bases before timing to avoid a startup herd on shared DAS pointers.
  int32_t *const kernel_l1_A = l1_A;
  int32_t *const kernel_l1_B = l1_B;
  int32_t *const kernel_l1_C = l1_C;

  // Realign cores after the shared DAS pointer snapshot so benchmark timing
  // starts after the last post-setup shared-memory read.
  mempool_barrier(num_cores);

  mempool_start_benchmark();


/****************************************************************************/
/*                                                                          */
/*                     Kernel Selection and Execution                       */
/*                                                                          */
/****************************************************************************/
#if defined(MATMUL_I32_KERNEL_2X2_XPULPV2)
    #ifndef __XPULPIMG
    #error "MATMUL_I32_KERNEL_2X2_XPULPV2 requires __XPULPIMG."
    #endif
    if (core_id < kernel_cores) {
      matmul_unrolled_2x2_parallel_i32_xpulpv2(kernel_l1_A, kernel_l1_B, kernel_l1_C, matrix_M,
                                              matrix_N, matrix_P, core_id,
                                              kernel_cores);
    }
#elif defined(MATMUL_I32_KERNEL_2X2_RV32IM)
    if (core_id < kernel_cores) {
      matmul_unrolled_2x2_parallel_i32_rv32im(kernel_l1_A, kernel_l1_B, kernel_l1_C, matrix_M,
                                              matrix_N, matrix_P, core_id,
                                              kernel_cores);
    }
#elif defined(MATMUL_I32_KERNEL_4X4)
    if (core_id < kernel_cores) {
      mat_mul_unrolled_4x4_parallel(kernel_l1_A, kernel_l1_B, kernel_l1_C, matrix_M, matrix_N,
                                    matrix_P, core_id, kernel_cores);
    }
#elif defined(MATMUL_I32_KERNEL_4X4_CONFLICT_OPT)
    if (core_id < kernel_cores) {
      mat_mul_unrolled_4x4_conflict_opt_parallel(kernel_l1_A, kernel_l1_B, kernel_l1_C, matrix_M,
                                                matrix_N, matrix_P, core_id,
                                                kernel_cores);
    }
#elif defined(MATMUL_I32_KERNEL_4X4_ASM)
    if (core_id < kernel_cores) {
      mat_mul_unrolled_4x4_parallel_asm(kernel_l1_A, kernel_l1_B, kernel_l1_C, matrix_M, matrix_N,
                                        matrix_P, core_id, kernel_cores);
    }
#elif defined(MATMUL_I32_KERNEL_4X4_CONFLICT_OPT_ASM)
    if (core_id < kernel_cores) {
      mat_mul_unrolled_4x4_conflict_opt_parallel_asm(
          kernel_l1_A, kernel_l1_B, kernel_l1_C, matrix_M, matrix_N, matrix_P, core_id,
          kernel_cores);
    }
#elif defined(MATMUL_I32_KERNEL_4X4_DAS_THESIS_ASM)
  if (core_id < kernel_cores) {
    mat_mul_unrolled_4x4_das_thesis_parallel_asm(
        kernel_l1_A, kernel_l1_B, kernel_l1_C, matrix_M, matrix_N, matrix_P,
        core_id, kernel_cores);
  }
#else
    // Preserve the historical default when no explicit kernel is requested.
    #ifdef __XPULPIMG
    if (core_id < kernel_cores) {
      matmul_unrolled_2x2_parallel_i32_xpulpv2(kernel_l1_A, kernel_l1_B, kernel_l1_C, matrix_M,
                                              matrix_N, matrix_P, core_id,
                                              kernel_cores);
    }
    #else
    if (core_id < kernel_cores) {
      matmul_unrolled_2x2_parallel_i32_rv32im(kernel_l1_A, kernel_l1_B, kernel_l1_C, matrix_M,
                                              matrix_N, matrix_P, core_id,
                                              kernel_cores);
    }
    #endif
#endif
  




  mempool_stop_benchmark();
  mempool_log_barrier(2, core_id);




    /****************************************************************************/
    /*                                                                          */
    /*                      Verification And Cleanup                             */
    /*                                                                          */
    /****************************************************************************/

#ifndef SKIP_VERIFY
  // Verify results
  mempool_check_i32(l1_C, l2_C, matrix_M * matrix_P, 0, 0);
  mempool_barrier(num_cores);
#endif


//free 
#ifdef NUM_DAS_PARTITIONS
  if (core_id == 0) {
    alloc_t *das_alloc = get_dynamic_heap_alloc();
    partition_free(das_alloc, l1_C);
    partition_free(das_alloc, l1_B);
    partition_free(das_alloc, l1_A);
  }
#endif

  return 0;
}
