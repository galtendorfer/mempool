// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Marco Bertuletti, ETH Zurich

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Mempool runtime libraries */
#include "builtins_v2.h"
#include "dma.h"
#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "data_cfft_radix4_f16.h"

/*
======================
Parameters and defines

SINGLE: When defined runs single-core FFT.
PARALLEL: When defined runs parallel FFT.

*/

#define PARALLEL
#define N_FFTs 32

#include "baremetal/mempool_cfft_q16_bitreversal.h"
#include "baremetal/mempool_checks.h"
#include "baremetal/mempool_radix4_cfft_butterfly_f16.h"
#include "baremetal/mempool_radix4_cfft_f16p.h"

__fp16 l1_pSrc[2 * N_CSAMPLES * N_FFTs]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_twiddleCoef_f16[2 * N_TWIDDLES]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
uint16_t l1_BitRevIndexTable[BITREVINDEXTABLE_LENGTH]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  __fp16 *pRes = (__fp16 *)0;
  mempool_barrier_init(core_id);

  /* INITIALIZATION */

  if (core_id == 0) {
    for (uint32_t idx = 0; idx < N_FFTs; idx++) {
      dma_memcpy_blocking(l1_pSrc + 2 * idx * N_CSAMPLES, l2_pSrc,
                          N_CSAMPLES * sizeof(int32_t));
    }
    dma_memcpy_blocking(l1_twiddleCoef_f16, l2_twiddleCoef_f16,
                        N_TWIDDLES * sizeof(int32_t));
    dma_memcpy_blocking(l1_BitRevIndexTable, l2_BitRevIndexTable,
                        BITREVINDEXTABLE_LENGTH * sizeof(int16_t));
    printf("01: END INITIALIZATION\n");
  }
  mempool_barrier(num_cores);

  /* COMPUTATION */

  mempool_start_benchmark();
  mempool_radix4_cfft_f16p(l1_pSrc, N_CSAMPLES, l1_twiddleCoef_f16, 1, N_FFTs,
                           num_cores);
  mempool_bitrevtable_q16p_xpulpimg((int16_t *)l1_pSrc, N_CSAMPLES,
                                    BITREVINDEXTABLE_LENGTH,
                                    l1_BitRevIndexTable, N_FFTs, num_cores);
  mempool_stop_benchmark();
  pRes = l1_pSrc;
  mempool_barrier(num_cores);

  if (core_id == 0) {
    printf("02: END COMPUTATION\n");
  }

  // mempool_check_f16(pRes, l2_pRes, 2 * N_CSAMPLES, (float)TOLERANCE, 0);
  mempool_barrier(num_cores);
  return 0;
}
