// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Marco Bertuletti, ETH Zurich

#pragma once
#define ASM
#include "builtins_v2.h"
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#ifndef ASM
#define SHUFFLE_TWIDDLEFACT                                                    \
  t1 = (int16_t)CoSi1[0];                                                      \
  t3 = (int16_t)CoSi2[0];                                                      \
  t5 = (int16_t)CoSi3[0];                                                      \
  t0 = (int16_t)CoSi1[1];                                                      \
  t2 = (int16_t)CoSi2[1];                                                      \
  t4 = (int16_t)CoSi3[1];                                                      \
  C1 = __PACK2(t1, -t0);                                                       \
  C2 = __PACK2(t3, -t2);                                                       \
  C3 = __PACK2(t5, -t4);
#else
#define SHUFFLE_TWIDDLEFACT                                                    \
  asm volatile("pv.extract.h  %[t1],%[CoSi1],0;"                               \
               "pv.extract.h  %[t3],%[CoSi2],0;"                               \
               "pv.extract.h  %[t5],%[CoSi3],0;"                               \
               "pv.extract.h  %[t0],%[CoSi1],1;"                               \
               "pv.extract.h  %[t2],%[CoSi2],1;"                               \
               "pv.extract.h  %[t4],%[CoSi3],1;"                               \
               "sub           %[t0],zero,%[t0];"                               \
               "sub           %[t2],zero,%[t2];"                               \
               "sub           %[t4],zero,%[t4];"                               \
               "pv.pack %[C1],%[t1],%[t0];"                                    \
               "pv.pack %[C2],%[t3],%[t2];"                                    \
               "pv.pack %[C3],%[t5],%[t4];"                                    \
               : [C1] "=r"(C1), [C2] "=r"(C2), [C3] "=r"(C3), [t0] "=&r"(t0),  \
                 [t1] "=&r"(t1), [t2] "=&r"(t2), [t3] "=&r"(t3),               \
                 [t4] "=&r"(t4), [t5] "=&r"(t5)                                \
               : [CoSi1] "r"(CoSi1), [CoSi2] "r"(CoSi2), [CoSi3] "r"(CoSi3)    \
               :);
#endif

#ifdef FOLDED_TWIDDLES

#define LOAD_STORE_TWIDDLEFACT                                                 \
  CoSi1 = *(v2s *)&pCoef_src[2U * ic];                                         \
  CoSi2 = *(v2s *)&pCoef_src[2U * (ic + 1 * NUM_BANKS)];                       \
  CoSi3 = *(v2s *)&pCoef_src[2U * (ic + 2 * NUM_BANKS)];                       \
  if (ic % 4 == 0) {                                                           \
    *((v2s *)&pCoef_dst[2U * (ic_store)]) = CoSi1;                             \
    *((v2s *)&pCoef_dst[2U * (n2_store * 1 + ic_store)]) = CoSi1;              \
    *((v2s *)&pCoef_dst[2U * (n2_store * 2 + ic_store)]) = CoSi1;              \
    *((v2s *)&pCoef_dst[2U * (n2_store * 3 + ic_store)]) = CoSi1;              \
    ic_store += NUM_BANKS;                                                     \
    *((v2s *)&pCoef_dst[2U * (ic_store)]) = CoSi2;                             \
    *((v2s *)&pCoef_dst[2U * (n2_store * 1 + ic_store)]) = CoSi2;              \
    *((v2s *)&pCoef_dst[2U * (n2_store * 2 + ic_store)]) = CoSi2;              \
    *((v2s *)&pCoef_dst[2U * (n2_store * 3 + ic_store)]) = CoSi2;              \
    ic_store += NUM_BANKS;                                                     \
    *((v2s *)&pCoef_dst[2U * (ic_store)]) = CoSi3;                             \
    *((v2s *)&pCoef_dst[2U * (n2_store * 1 + ic_store)]) = CoSi3;              \
    *((v2s *)&pCoef_dst[2U * (n2_store * 2 + ic_store)]) = CoSi3;              \
    *((v2s *)&pCoef_dst[2U * (n2_store * 3 + ic_store)]) = CoSi3;              \
  }

#else
#define LOAD_STORE_TWIDDLEFACT                                                 \
  CoSi1 = *(v2s *)&pCoef_src[ic * 2U];                                         \
  CoSi2 = *(v2s *)&pCoef_src[2U * (ic * 2U)];                                  \
  CoSi3 = *(v2s *)&pCoef_src[3U * (ic * 2U)];
#endif

void mempool_radix4_cfft_q16p_xpulpimg(int16_t *pSrc16, const uint32_t fftLen,
                                       const int16_t *pCoef16,
                                       uint32_t twidCoefModifier,
                                       uint32_t nffts, uint32_t nPE) {

  uint32_t absolute_core_id = mempool_get_core_id();
  uint32_t core_id = absolute_core_id % nPE;
  v2s CoSi1, CoSi2, CoSi3;
  v2s C1, C2, C3;
  int16_t t0, t1, t2, t3, t4, t5;
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
    CoSi1 = *(v2s *)&pCoef16[ic * 2U];
    CoSi2 = *(v2s *)&pCoef16[2U * (ic * 2U)];
    CoSi3 = *(v2s *)&pCoef16[3U * (ic * 2U)];
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
      CoSi1 = *(v2s *)&pCoef16[ic * 2U];
      CoSi2 = *(v2s *)&pCoef16[2U * (ic * 2U)];
      CoSi3 = *(v2s *)&pCoef16[3U * (ic * 2U)];
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

void mempool_radix4by2_cfft_q16p_xpulpimg(int16_t *pSrc, const uint32_t fftLen,
                                          const int16_t *pCoef, uint32_t nffts,
                                          uint32_t nPE) {

  uint32_t i, idx;
  uint32_t n2, step;
  int16_t pa, pb, pc, pd;
  int16_t *p;

  v2s CoSi;
  v2s a, b, t;
  int16_t testa, testb;
  uint32_t core_id = mempool_get_core_id();

  n2 = fftLen >> 1;
  step = (n2 + nPE - 1) / nPE;
  for (i = core_id * step; i < MIN(core_id * step + step, n2); i++) {
    for (idx = 0; idx < nffts; idx++) {
      p = pSrc + 2 * idx * fftLen;
      CoSi = *(v2s *)&pCoef[i * 2];
      a = __SRA2(*(v2s *)&p[2 * i], ((v2s){1, 1}));
      b = __SRA2(*(v2s *)&p[2 * (i + n2)], ((v2s){1, 1}));
      t = __SUB2(a, b);
      testa = (int16_t)(__DOTP2(t, CoSi) >> 16);
      testb = (int16_t)(__DOTP2(t, __PACK2(CoSi[0], -CoSi[1])) >> 16);
      *((v2s *)&p[i * 2]) = __SRA2(__ADD2(a, b), ((v2s){1, 1}));
      *((v2s *)&p[(i + n2) * 2]) = __PACK2(testb, testa);
    }
  }
  mempool_log_barrier(2, core_id);

  for (idx = 0; idx < nffts; idx++) {
    p = pSrc + 2 * idx * fftLen;
    // first half
    mempool_radix4_cfft_q16p_xpulpimg(p, n2, (int16_t *)pCoef, 2U, 1, nPE);
    // second half
    mempool_radix4_cfft_q16p_xpulpimg(p + fftLen, n2, (int16_t *)pCoef, 2U, 1,
                                      nPE);
  }

  for (i = core_id * step; i < MIN(core_id * step + step, n2); i++) {
    for (idx = 0; idx < nffts; idx++) {
      p = pSrc + 2 * idx * fftLen;
      pa = *(int16_t *)&p[4 * i + 0];
      pb = *(int16_t *)&p[4 * i + 1];
      pc = *(int16_t *)&p[4 * i + 2];
      pd = *(int16_t *)&p[4 * i + 3];
      pa = (int16_t)(pa << 1U);
      pb = (int16_t)(pb << 1U);
      pc = (int16_t)(pc << 1U);
      pd = (int16_t)(pd << 1U);
      p[4 * i + 0] = pa;
      p[4 * i + 1] = pb;
      p[4 * i + 2] = pc;
      p[4 * i + 3] = pd;
    }
  }
  mempool_log_barrier(2, core_id);
  return;
}

/**
  @brief         Full FFT butterfly
  @param[in]     pSrc16  input buffer of 16b data, Re/Im interleaved
  @param[out]    pDst16  output buffer of 16b data, Re/Im interleaved
  @param[in]     fftLen  Length of the complex input vector
  @param[in]     pCoef_src Twiddle coefficients vector
  @param[in]     pCoef_dst Auxiliary twiddle coefficients vector
  @param[in]     nPE Number of PE
  @return        pointer to output vector
*/
void mempool_radix4_cfft_q16p_folded(int16_t *pSrc16, int16_t *pDst16,
                                     uint32_t fftLen, int16_t *pCoef_src,
                                     int16_t __attribute__((unused)) *
                                         pCoef_dst,
                                     uint32_t nPE) {

  uint32_t core_id = mempool_get_core_id();
  int16_t t0, t1, t2, t3, t4, t5;
  v2s CoSi1, CoSi2, CoSi3;
  v2s C1, C2, C3;

#ifdef FOLDED_TWIDDLES
  uint32_t n1, n2, n2_store;
  uint32_t i0, k, ic, ic_store;
  int16_t *pTmp;
#else
  uint32_t n1, n2;
  uint32_t i0, k, ic;
  int16_t *pTmp;
  uint32_t twidCoefModifier = 1U;
#endif

  /* START OF FIRST STAGE PROCESSING */
  n1 = fftLen;
  n2 = n1 >> 2U;
  for (i0 = core_id * 4; i0 < MIN(core_id * 4 + 4, n2); i0++) {
#ifdef FOLDED_TWIDDLES
    ic = i0;
    ic_store = ic >> 2U;
    n2_store = n2 >> 2U;
#else
    ic = i0 * twidCoefModifier;
#endif
    LOAD_STORE_TWIDDLEFACT;
    SHUFFLE_TWIDDLEFACT;
    radix4_butterfly_first(pSrc16, pDst16, i0, n2, CoSi1, CoSi2, CoSi3, C1, C2,
                           C3);
  }
  pTmp = pSrc16;
  pSrc16 = pDst16;
  pDst16 = pTmp;
#ifdef FOLDED_TWIDDLES
  pTmp = pCoef_src;
  pCoef_src = pCoef_dst;
  pCoef_dst = pTmp;
#else
  twidCoefModifier <<= 2U;
#endif
  mempool_log_partial_barrier(2, core_id, nPE);
  /* END OF FIRST STAGE PROCESSING */

  /* START OF MIDDLE STAGE PROCESSING */
  for (k = fftLen / 4U; k > 4U; k >>= 2U) {
    n1 = n2;
    n2 >>= 2U;
    for (i0 = core_id * 4; i0 < core_id * 4 + 4; i0++) {
#ifdef FOLDED_TWIDDLES
      ic = i0;
      // (ic % n2) / 4 take only every 4th index in the wing
      // (ic / n2) * n2 shift of the wing size
      ic_store = ((ic % n2) >> 2) + (ic / n2) * n2;
      n2_store = n2 >> 2U;
#else
      ic = (i0 % n2) * twidCoefModifier;
#endif
      LOAD_STORE_TWIDDLEFACT;
      SHUFFLE_TWIDDLEFACT;
      radix4_butterfly_middle(pSrc16, pDst16, i0, n2, CoSi1, CoSi2, CoSi3, C1,
                              C2, C3);
    }
    pTmp = pSrc16;
    pSrc16 = pDst16;
    pDst16 = pTmp;
#ifdef FOLDED_TWIDDLES
    pTmp = pCoef_src;
    pCoef_src = pCoef_dst;
    pCoef_dst = pTmp;
#else
    twidCoefModifier <<= 2U;
#endif
    mempool_log_partial_barrier(2, core_id, nPE);
  }
  /* END OF MIDDLE STAGE PROCESSING */

  /* START OF LAST STAGE PROCESSING */
  for (i0 = core_id * 4; i0 < MIN(core_id * 4 + 4, fftLen >> 2U); i0++) {
    radix4_butterfly_last(pSrc16, pDst16, i0);
  }
  mempool_log_partial_barrier(2, core_id, nPE);
  /* END OF LAST STAGE PROCESSING */

  return;
}
