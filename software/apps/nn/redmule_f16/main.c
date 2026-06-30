// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Marco Bertuletti, ETH Zurich

#include <stdint.h>
#include <string.h>

#include "dma.h"
#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "baremetal/mempool_checks.h"
#include "baremetal/mempool_redmule_f16.h"
#include "data_gemm_f16.h"

#define ELEMENTS_PER_ROW (NUM_BANKS * sizeof(int32_t) / sizeof(int16_t))
#define PORT_WIDTH (REDMULE_H * (REDMULE_P + 1))
#define SHIFT (true)

__fp16 l1_X[(matrix_M * matrix_N) + PORT_WIDTH * NUM_REDMULE_TILES]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_W[matrix_N * matrix_P]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_Y[matrix_M * matrix_P]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  mempool_barrier_init(core_id);

#ifdef SINGLE

  // Transfer
  if (core_id == 0) {
    dma_memcpy_blocking(l1_X, l2_X, (matrix_M * matrix_N) * sizeof(int16_t));
    dma_memcpy_blocking(l1_W, l2_W, (matrix_N * matrix_P) * sizeof(int16_t));
    dma_memcpy_blocking(l1_Y, l2_Y, (matrix_M * matrix_P) * sizeof(int16_t));
  }
  mempool_barrier(num_cores);

  // Compute
  redmule_synch_single(l1_X, l1_W, l1_Y, matrix_M, matrix_N, matrix_P, GEMM, 0);

#endif

#ifdef PARALLEL

  uint32_t num_redmules = mempool_get_redmule_count();

  // Transfer
  if (core_id == 0) {
    for (uint32_t i = 0; i < num_redmules; i++) {
      uint32_t X_shift = SHIFT ? (i * PORT_WIDTH) % matrix_N : 0;
      __fp16 *Xsrc = l2_X + i * (matrix_M * matrix_N / num_redmules);
      __fp16 *Xdst = l1_X + i * (matrix_M * matrix_N / num_redmules) + X_shift;
      dma_memcpy_blocking(
          Xdst, Xsrc, (matrix_M * matrix_N / num_redmules) * sizeof(int16_t));
    }
    dma_memcpy_blocking(l1_W, l2_W, (matrix_N * matrix_P) * sizeof(int16_t));
    dma_memcpy_blocking(l1_Y, l2_Y, (matrix_M * matrix_P) * sizeof(int16_t));
  }
  mempool_barrier(num_cores);

  // Compute
  redmule_synch_parallel(l1_X, l1_Y, l1_W, matrix_M, matrix_N, matrix_P, GEMM,
                         SHIFT, PORT_WIDTH);
  mempool_barrier(num_cores);

#endif

  mempool_check_f16(l1_Y, l2_Z, matrix_M * matrix_P, 0.05f, 0);
  mempool_barrier(num_cores);
  return 0;
}
