// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Samuel Riedel, ETH Zurich

#include <stdint.h>
#include <string.h>

#include "dma.h"
#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "data_matmul_i32.h"

#include "baremetal/mempool_checks.h"
#include "baremetal/mempool_matmul_i32p.h"

#if (defined(MATMUL_I32_KERNEL_2X2_XPULPV2) +                                  \
  defined(MATMUL_I32_KERNEL_2X2_RV32IM) +                                   \
  defined(MATMUL_I32_KERNEL_4X4) +                                           \
  defined(MATMUL_I32_KERNEL_4X4_CONFLICT_OPT) +                              \
  defined(MATMUL_I32_KERNEL_4X4_ASM) +                                       \
  defined(MATMUL_I32_KERNEL_4X4_CONFLICT_OPT_ASM)) > 1
#error "Select exactly one MATMUL_I32 kernel."
#endif

int32_t l1_A[matrix_M * matrix_N] __attribute__((section(".l1_prio")));
int32_t l1_B[matrix_N * matrix_P] __attribute__((section(".l1_prio")));
int32_t l1_C[matrix_M * matrix_P] __attribute__((section(".l1_prio")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  mempool_barrier_init(core_id);

  // Initialize data
  if (core_id == 0) {
    dma_memcpy_blocking(l1_A, l2_A, matrix_M * matrix_N * sizeof(int32_t));
    dma_memcpy_blocking(l1_B, l2_B, matrix_N * matrix_P * sizeof(int32_t));
  }
  mempool_barrier(num_cores);

  // Benchmark
  mempool_start_benchmark();
#if defined(MATMUL_I32_KERNEL_2X2_XPULPV2)
#ifndef __XPULPIMG
#error "MATMUL_I32_KERNEL_2X2_XPULPV2 requires __XPULPIMG."
#endif
  matmul_unrolled_2x2_parallel_i32_xpulpv2(l1_A, l1_B, l1_C, matrix_M, matrix_N,
                                           matrix_P, core_id, num_cores);
#elif defined(MATMUL_I32_KERNEL_2X2_RV32IM)
  matmul_unrolled_2x2_parallel_i32_rv32im(l1_A, l1_B, l1_C, matrix_M, matrix_N,
                                          matrix_P, core_id, num_cores);
#elif defined(MATMUL_I32_KERNEL_4X4)
  mat_mul_unrolled_4x4_parallel(l1_A, l1_B, l1_C, matrix_M, matrix_N, matrix_P,
                                core_id, num_cores);
#elif defined(MATMUL_I32_KERNEL_4X4_CONFLICT_OPT)
  mat_mul_unrolled_4x4_conflict_opt_parallel(l1_A, l1_B, l1_C, matrix_M,
                                             matrix_N, matrix_P, core_id,
                                             num_cores);
#elif defined(MATMUL_I32_KERNEL_4X4_ASM)
  mat_mul_unrolled_4x4_parallel_asm(l1_A, l1_B, l1_C, matrix_M, matrix_N,
                                    matrix_P, core_id, num_cores);
#elif defined(MATMUL_I32_KERNEL_4X4_CONFLICT_OPT_ASM)
  mat_mul_unrolled_4x4_conflict_opt_parallel_asm(
      l1_A, l1_B, l1_C, matrix_M, matrix_N, matrix_P, core_id, num_cores);
#else
  // Preserve the historical default when no explicit kernel is requested.
#ifdef __XPULPIMG
  matmul_unrolled_2x2_parallel_i32_xpulpv2(l1_A, l1_B, l1_C, matrix_M, matrix_N,
                                           matrix_P, core_id, num_cores);
#else
  matmul_unrolled_2x2_parallel_i32_rv32im(l1_A, l1_B, l1_C, matrix_M, matrix_N,
                                          matrix_P, core_id, num_cores);
#endif
#endif
  mempool_stop_benchmark();
  mempool_barrier(num_cores);

  // Verify results
  mempool_check_i32(l1_C, l2_C, matrix_M * matrix_P, 0, 0);
  mempool_barrier(num_cores);
  return 0;
}
