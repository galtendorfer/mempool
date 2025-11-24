// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Marco Bertuletti, ETH Zurich

#pragma once
#include "builtins_v2.h"
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// CoSi: (Si, Co) -> C: (Co, -Si)
#define SHUFFLE_TWIDDLEFACT                                                    \
  asm volatile("pv.extract.h  %[t1],%[CoSi1],0;"                               \
               "pv.extract.h  %[t3],%[CoSi2],0;"                               \
               "pv.extract.h  %[t5],%[CoSi3],0;"                               \
               "pv.extract.h  %[t0],%[CoSi1],1;"                               \
               "pv.extract.h  %[t2],%[CoSi2],1;"                               \
               "pv.extract.h  %[t4],%[CoSi3],1;"                               \
               "xor           %[t1],%[t1],%[neg_mask];"                        \
               "xor           %[t3],%[t3],%[neg_mask];"                        \
               "xor           %[t5],%[t5],%[neg_mask];"                        \
               "pv.pack   %[C1],%[t1],%[t0];"                                  \
               "pv.pack   %[C2],%[t3],%[t2];"                                  \
               "pv.pack   %[C3],%[t5],%[t4];"                                  \
               : [C1] "=r"(C1), [C2] "=r"(C2), [C3] "=r"(C3), [t0] "=&r"(t0),  \
                 [t1] "=&r"(t1), [t2] "=&r"(t2), [t3] "=&r"(t3),               \
                 [t4] "=&r"(t4), [t5] "=&r"(t5)                                \
               : [CoSi1] "r"(CoSi1), [CoSi2] "r"(CoSi2), [CoSi3] "r"(CoSi3),   \
                 [neg_mask] "r"(0x00008000)                                    \
               :);

#ifdef FOLDED_TWIDDLES
#define LOAD_STORE_TWIDDLEFACT                                                 \
  CoSi1 = *(v2h *)&pCoef_src[2U * ic];                                         \
  CoSi2 = *(v2h *)&pCoef_src[2U * (ic + 1 * NUM_BANKS)];                       \
  CoSi3 = *(v2h *)&pCoef_src[2U * (ic + 2 * NUM_BANKS)];                       \
  if (ic % 4 == 0) {                                                           \
    *((v2h *)&pCoef_dst[2U * (ic_store)]) = CoSi1;                             \
    *((v2h *)&pCoef_dst[2U * (n2_store * 1 + ic_store)]) = CoSi1;              \
    *((v2h *)&pCoef_dst[2U * (n2_store * 2 + ic_store)]) = CoSi1;              \
    *((v2h *)&pCoef_dst[2U * (n2_store * 3 + ic_store)]) = CoSi1;              \
    ic_store += NUM_BANKS;                                                     \
    *((v2h *)&pCoef_dst[2U * (ic_store)]) = CoSi2;                             \
    *((v2h *)&pCoef_dst[2U * (n2_store * 1 + ic_store)]) = CoSi2;              \
    *((v2h *)&pCoef_dst[2U * (n2_store * 2 + ic_store)]) = CoSi2;              \
    *((v2h *)&pCoef_dst[2U * (n2_store * 3 + ic_store)]) = CoSi2;              \
    ic_store += NUM_BANKS;                                                     \
    *((v2h *)&pCoef_dst[2U * (ic_store)]) = CoSi3;                             \
    *((v2h *)&pCoef_dst[2U * (n2_store * 1 + ic_store)]) = CoSi3;              \
    *((v2h *)&pCoef_dst[2U * (n2_store * 2 + ic_store)]) = CoSi3;              \
    *((v2h *)&pCoef_dst[2U * (n2_store * 3 + ic_store)]) = CoSi3;              \
  }

#else
#define LOAD_STORE_TWIDDLEFACT                                                 \
  CoSi1 = *(v2h *)&pCoef_src[2U * ic];                                         \
  CoSi2 = *(v2h *)&pCoef_src[2U * (ic * 2U)];                                  \
  CoSi3 = *(v2h *)&pCoef_src[2U * (ic * 3U)];
#endif

void mempool_radix4_cfft_f16p(__fp16 *pSrc16, const uint32_t fftLen,
                              const __fp16 *pCoef16, uint32_t twidCoefModifier,
                              uint32_t nffts, uint32_t nPE) {

  uint32_t absolute_core_id = mempool_get_core_id();
  uint32_t core_id = absolute_core_id % nPE;
  v2h CoSi1, CoSi2, CoSi3;
  v2h C1, C2, C3;
  __fp16 t0, t1, t2, t3, t4, t5;
  uint32_t n1, n2, ic, i0, j, k;
  uint32_t step, steps;
  uint32_t idx;

  /* START OF FIRST STAGE PROCESSING */
  n1 = fftLen;
  n2 = n1 >> 2U;
  step = (n2 + nPE - 1) / nPE;
  for (i0 = core_id * step; i0 < MIN(core_id * step + step, n2); i0++) {
    /*  Twiddle coefficients index modifier */
    ic = i0 * twidCoefModifier;
    CoSi1 = *(v2h *)&pCoef16[ic * 2U];
    CoSi2 = *(v2h *)&pCoef16[2U * (ic * 2U)];
    CoSi3 = *(v2h *)&pCoef16[3U * (ic * 2U)];
    SHUFFLE_TWIDDLEFACT;
    for (idx = 0; idx < nffts; idx++) {
      radix4_butterfly_first(pSrc16 + 2U * idx * fftLen,
                             pSrc16 + 2U * idx * fftLen, i0, n2, CoSi1, CoSi2,
                             CoSi3, C1, C2, C3);
    }
  }
  mempool_log_barrier(2, absolute_core_id);
  /* END OF FIRST STAGE PROCESSING */

  /* START OF MIDDLE STAGE PROCESSING */
  twidCoefModifier <<= 2U;
  for (k = fftLen / 4U; k > 4U; k >>= 2U) {
    uint32_t offset, butt_id;
    n1 = n2;
    n2 >>= 2U;
    step = (n2 + nPE - 1) / nPE;
    butt_id = core_id % n2;
    offset = (core_id / n2) * n1;
    for (j = butt_id * step; j < MIN(butt_id * step + step, n2); j++) {
      /*  Twiddle coefficients index modifier */
      ic = twidCoefModifier * j;
      CoSi1 = *(v2h *)&pCoef16[ic * 2U];
      CoSi2 = *(v2h *)&pCoef16[2U * (ic * 2U)];
      CoSi3 = *(v2h *)&pCoef16[3U * (ic * 2U)];
      SHUFFLE_TWIDDLEFACT;
      /*  Butterfly implementation */
      for (i0 = offset + j; i0 < fftLen; i0 += ((nPE + n2 - 1) / n2) * n1) {
        for (idx = 0; idx < nffts; idx++) {
          radix4_butterfly_middle(pSrc16 + 2U * idx * fftLen,
                                  pSrc16 + 2U * idx * fftLen, i0, n2, CoSi1,
                                  CoSi2, CoSi3, C1, C2, C3);
        }
      }
    }
    twidCoefModifier <<= 2U;
    mempool_log_barrier(2, absolute_core_id);
  }
  /* END OF MIDDLE STAGE PROCESSING */

  /* START OF LAST STAGE PROCESSING */
  n1 = n2;
  n2 >>= 2U;
  steps = fftLen / n1;
  step = (steps + nPE - 1) / nPE;
  /*  Butterfly implementation */
  for (i0 = core_id * step * n1; i0 < MIN((core_id * step + step) * n1, fftLen);
       i0 += n1) {
    for (idx = 0; idx < nffts; idx++) {
      radix4_butterfly_last(pSrc16 + 2U * idx * fftLen,
                            pSrc16 + 2U * idx * fftLen, i0);
    }
  }
  mempool_log_barrier(2, absolute_core_id);
  /* END OF LAST STAGE PROCESSING */
  return;
}
