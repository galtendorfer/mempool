// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Note: This test is only for Terapool dynamic heap allocation
// Author: Bowen Wang

#include <stdint.h>
#include <string.h>

#include "alloc.h"
#include "dma.h"
#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#define NUM_TILES (NUM_CORES / NUM_CORES_PER_TILE)
uint32_t l2_array[2 * NUM_BANKS] __attribute__((section(".l2")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();

  // Initialization
  mempool_init(core_id);
  mempool_barrier_init(core_id);

  if (core_id == 0) {

    // --------------------------------------------
    // Initialize
    // --------------------------------------------
    uint32_t num_tiles_per_partition = 4;
    uint32_t array_size =
        2 * num_tiles_per_partition * BANKING_FACTOR * NUM_CORES_PER_TILE;
    // Initialize L2 array
    for (uint32_t i = 0; i < array_size; i++) {
      l2_array[i] = i;
    }

    // --------------------------------------------
    // Verify DMA transfers in DAS region
    // --------------------------------------------
    printf("Verify DMA transfers in DAS region\n\n");

    // 1. Init dynamic heap allocator
    mempool_dynamic_heap_alloc_init(core_id);

    // 2. Set which partition write to.
    uint32_t part_id = 0;

    // 3. Get the allocator
    alloc_t *dynamic_heap_alloc = get_dynamic_heap_alloc();
    alloc_dump(dynamic_heap_alloc);

    // 4. Allocate memory
    uint32_t *array = (uint32_t *)partition_malloc(
        dynamic_heap_alloc, array_size * sizeof(uint32_t));

    // 5. Config the hardware registers
    partition_config(part_id, num_tiles_per_partition);
    start_addr_scheme_config(part_id, (uint32_t)(*array),
                             array_size * sizeof(uint32_t));

    // 6. Move data
    dma_memcpy_blocking(array, l2_array, array_size * sizeof(uint32_t));

    // 7. Change addressing scheme (to fully interleaved)
    partition_config(part_id, NUM_TILES);

    // 8. check
    for (uint32_t i = 0; i < array_size; i++) {
      uint32_t *fetch_address = &array[0] +
          (i % (num_tiles_per_partition * NUM_CORES_PER_TILE * BANKING_FACTOR)) +
          (i / (num_tiles_per_partition * NUM_CORES_PER_TILE * BANKING_FACTOR)) * NUM_BANKS;
//      if (l2_array[i] != *fetch_address) {
//        printf("%4d != %4d at address %8X.\n", i, *fetch_address,
//               fetch_address);
//      }
    }

    // 9. Free array
    partition_free(dynamic_heap_alloc, array);
    printf("SUCCESS on partition %d \n\n", part_id);

    printf("All correct!\n");
  }

  mempool_barrier(num_cores);
  return 0;
}
