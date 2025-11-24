// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Marco Bertuletti, ETH Zurich

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "builtins_v2.h"
#include "dma.h"
#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "data_cfft_radix4_q16.h"

/*
======================
Parameters and defines

SINGLE: When defined runs single-core FFT.
PARALLEL: When defined runs parallel FFT.

*/

#define FOLDED
#define FOLDED_TWIDDLES
#define N_FFTs 1

#include "baremetal/mempool_cfft_q16_bitreversal.h"
#include "baremetal/mempool_checks.h"
#include "baremetal/mempool_radix4_cfft_butterfly_q16.h"
#include "baremetal/mempool_radix4_cfft_q16p.h"
#include "baremetal/mempool_radix4_cfft_q16s.h"

#if defined(SINGLE) || defined(PARALLEL)

int16_t l1_pSrc[2 * N_CSAMPLES * N_FFTs]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
int16_t l1_twiddleCoef_q16[2 * N_TWIDDLES]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
uint16_t l1_BitRevIndexTable[BITREVINDEXTABLE_LENGTH]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  int16_t *pRes; // Result pointer
  mempool_barrier_init(core_id);

  /* INITIALIZATION */

  if (core_id == 0) {
    for (uint32_t idx = 0; idx < N_FFTs; idx++) {
      dma_memcpy_blocking(l1_pSrc + 2 * idx * N_CSAMPLES, l2_pSrc,
                          N_CSAMPLES * sizeof(int32_t));
    }
    dma_memcpy_blocking(l1_twiddleCoef_q16, l2_twiddleCoef_q16,
                        N_TWIDDLES * sizeof(int32_t));
    dma_memcpy_blocking(l1_BitRevIndexTable, l2_BitRevIndexTable,
                        BITREVINDEXTABLE_LENGTH * sizeof(int16_t));
    printf("01: END INITIALIZATION\n");
  }
  mempool_barrier(num_cores);

/* COMPUTATION */

// 01: SINGLE
// A single core executes the FFT and the bitreversal.
#if defined(SINGLE)
  if (core_id == 0) {
    mempool_start_benchmark();
    if (LOG2 % 2 == 0) {
      mempool_radix4_cfft_q16s_riscv32(l1_pSrc, N_CSAMPLES, l1_twiddleCoef_q16,
                                       1);
    } else {
      mempool_radix4by2_cfft_q16s_riscv32(l1_pSrc, N_CSAMPLES,
                                          l1_twiddleCoef_q16);
    }
    mempool_bitrevtable_q16s_riscv32(l1_pSrc, BITREVINDEXTABLE_LENGTH,
                                     l1_BitRevIndexTable);
    pRes = l1_pSrc;
    mempool_stop_benchmark();
  }
  mempool_barrier(num_cores);
#endif

// 02: PARALLEL
// All the cores execute the FFT and the bitreversal.
#if defined(PARALLEL)
  mempool_start_benchmark();
  if (LOG2 % 2 == 0) {
    mempool_radix4_cfft_q16p_xpulpimg(l1_pSrc, N_CSAMPLES, l1_twiddleCoef_q16,
                                      1, N_FFTs, num_cores);
  } else {
    mempool_radix4by2_cfft_q16p_xpulpimg(l1_pSrc, N_CSAMPLES,
                                         l1_twiddleCoef_q16, N_FFTs, num_cores);
  }
  mempool_bitrevtable_q16p_xpulpimg(l1_pSrc, N_CSAMPLES,
                                    BITREVINDEXTABLE_LENGTH,
                                    l1_BitRevIndexTable, N_FFTs, num_cores);
  mempool_stop_benchmark();
  pRes = l1_pSrc;
#endif

  mempool_barrier(num_cores);
  if (core_id == 0) {
    printf("02: END COMPUTATION\n");
  }
  for (uint32_t idx = 0; idx < N_FFTs; idx++) {
    mempool_check_i16(pRes + 2 * idx * N_CSAMPLES, l2_pRes, 2 * N_CSAMPLES,
                      TOLERANCE, 0);
  }
  mempool_barrier(num_cores);
  return 0;
}

#endif

// 03: FOLDED
// All the cores execute the FFT and the bitreversal. The butterfly combine
// samples from each quarter of the input. Each quarter of the input vector is
// therefore folded in the local memory of cores, occupying different memory
// rows. Twiddles can also be folded.
#if defined(FOLDED)

int16_t l1_pSrc[8 * NUM_BANKS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
int16_t l1_pDst[8 * NUM_BANKS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
int16_t l1_twiddleCoef_q16_src[8 * NUM_BANKS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
int16_t l1_twiddleCoef_q16_dst[8 * NUM_BANKS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
uint16_t l1_BitRevIndexTable[BITREVINDEXTABLE_LENGTH]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  int16_t *pRes; // Result pointer
  mempool_barrier_init(core_id);

  if (core_id == 0) {
    dma_memcpy_blocking(l1_pSrc, l2_pSrc, N_CSAMPLES * sizeof(int32_t));
    dma_memcpy_blocking(l1_BitRevIndexTable, l2_BitRevIndexTable,
                        BITREVINDEXTABLE_LENGTH * sizeof(int32_t));
  }
  mempool_barrier(num_cores);

#ifdef FOLDED_TWIDDLES
  for (uint32_t i = core_id; i < (N_CSAMPLES >> 2); i += num_cores) {
    *(v2s *)&l1_twiddleCoef_q16_src[2 * i] = *(v2s *)&l2_twiddleCoef_q16[2 * i];
    *(v2s *)&l1_twiddleCoef_q16_src[2 * i + 2 * NUM_BANKS] =
        *(v2s *)&l2_twiddleCoef_q16[2 * (i * 2U)];
    *(v2s *)&l1_twiddleCoef_q16_src[2 * i + 4 * NUM_BANKS] =
        *(v2s *)&l2_twiddleCoef_q16[2 * (i * 3U)];
  }
#else
  if (core_id == 0) {
    dma_memcpy_blocking(l1_twiddleCoef_q16_src, l2_twiddleCoef_q16,
                        3 * (N_CSAMPLES / 4) * sizeof(int32_t));
  }
#endif
  mempool_barrier(num_cores);

  if (core_id == 0) {
    printf("01: END INITIALIZATION\n");
  }
  mempool_barrier(num_cores);

  if (core_id < (N_CSAMPLES / 16)) {
    mempool_start_benchmark();
    mempool_radix4_cfft_q16p_folded(l1_pSrc, l1_pDst, N_CSAMPLES,
                                    l1_twiddleCoef_q16_src,
                                    l1_twiddleCoef_q16_dst, (N_CSAMPLES / 16));
    pRes = ((LOG2 / 2) % 2) == 0 ? l1_pSrc : l1_pDst;
    mempool_bitrevtable_q16p_xpulpimg(pRes, N_CSAMPLES, BITREVINDEXTABLE_LENGTH,
                                      l1_BitRevIndexTable, 1,
                                      (N_CSAMPLES / 16));
    mempool_stop_benchmark();
  }

  mempool_check_i16(pRes, l2_pRes, 2 * N_CSAMPLES, TOLERANCE, 0);
  mempool_barrier(num_cores);
  return 0;
}

#endif
