// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Note: This test is only for Terapool dynamic heap allocation
// Author: Bowen Wang

#include <stdint.h>
#include <string.h>

#include "alloc.h"
#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#define NUM_TILES (NUM_CORES / NUM_CORES_PER_TILE)
#define NUM_TILES_PER_PARTITION (4)
#define ARRAY_SIZE (2 * NUM_TILES_PER_PARTITION * BANKING_FACTOR * NUM_CORES_PER_TILE)

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();

  // Initialization
  mempool_init(core_id);
  mempool_barrier_init(core_id);

  // --------------------------------------------
  // Verify partition
  // --------------------------------------------

  if (core_id == 0) {
    printf("Verify partition\n");

    // 1. Init dynamic heap allocator
    mempool_dynamic_heap_alloc_init(core_id);

    // 2. Set which partition write to.
    uint32_t part_id = 0;  // set to allocate in the penultimate partition

    // 3. Get the allocator
    alloc_t *dynamic_heap_alloc = get_dynamic_heap_alloc();
    alloc_dump(dynamic_heap_alloc);
    // 4. Allocate memory
    uint32_t *array = (uint32_t *)partition_malloc(dynamic_heap_alloc, ARRAY_SIZE*sizeof(uint32_t));

    // 5. Config the hardware registers
    partition_config(part_id, NUM_TILES_PER_PARTITION);
    start_addr_scheme_config(part_id, (uint32_t)(*array), ARRAY_SIZE*sizeof(uint32_t));

    // 6. Move data
    for (uint32_t i = 0; i < ARRAY_SIZE; i++) {
      array[i] = i;
    }

    // 7. Change addressing scheme (to fully interleaved)
    partition_config(part_id, NUM_TILES);

    // 8. check
    for (uint32_t i = 0; i < ARRAY_SIZE; i++) {
      uint32_t *fetch_address = &array[0] + \
        (i % (NUM_TILES_PER_PARTITION * NUM_CORES_PER_TILE * BANKING_FACTOR)) + \
        (i / (NUM_TILES_PER_PARTITION * NUM_CORES_PER_TILE * BANKING_FACTOR)) * NUM_BANKS;
      if (i != *fetch_address) {
        printf("%4d != %4d at address %8X.\n", i, *fetch_address, fetch_address);
      }
    }

    // 9. Free array
    partition_free(dynamic_heap_alloc, array);
    printf("All correct!\n");
  }

  mempool_barrier(num_cores);


  return 0;
}
