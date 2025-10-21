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

#define TILES_PER_PARTITION (2)
#define ARRAY_SIZE (2 * TILES_PER_PARTITION * BANKING_FACTOR * NUM_CORES_PER_TILE)


int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();

  // Initialization
  mempool_init(core_id);
  mempool_barrier_init(core_id);

  // --------------------------------------------
  // Runtime Partition Selection
  // --------------------------------------------

  if (core_id == 0) {
    printf("Initialize\n");
    // 1. Init dynamic heap allocator
    partition_config(0, TILES_PER_PARTITION);
    mempool_dynamic_heap_alloc_init(core_id, TILES_PER_PARTITION);
    // 2. Set which partition write to.
    uint32_t part_id = 0;  // set to allocate in the first partition
    // 3. Get the allocator and starting address to this region
    alloc_t *dynamic_heap_alloc = get_dynamic_heap_alloc(part_id);
    alloc_dump(dynamic_heap_alloc);
    // 4. Allocate memory
    uint32_t *array = (uint32_t *)domain_malloc(dynamic_heap_alloc, ARRAY_SIZE);
    // 5. Move data
    for (uint32_t i = 0; i < ARRAY_SIZE; ++i) {
        array[i] = i;
    }
    // 6. Free array
    domain_free(dynamic_heap_alloc, array);
    // 7. Free dynamic allocator
    free_dynamic_heap_alloc();
    printf("Done!\n");
  }

  mempool_barrier(num_cores);

  // --------------------------------------------
  // Verify partition
  // --------------------------------------------

  if (core_id == 0) {
    printf("Verify partition\n");
    // 1. Init dynamic heap allocator
    partition_config(0, TILES_PER_PARTITION);
    mempool_dynamic_heap_alloc_init(core_id, TILES_PER_PARTITION);
    // 2. Set which partition write to.
    uint32_t num_partition = mempool_get_tile_count() / TILES_PER_PARTITION;
    uint32_t part_id = 0;  // set to allocate in the penultimate partition
    // 3. Get the allocator and starting address to this region
    alloc_t *dynamic_heap_alloc = get_dynamic_heap_alloc(part_id);
    alloc_dump(dynamic_heap_alloc);
    // 4. Allocate memory
    uint32_t *array = (uint32_t *)domain_malloc(dynamic_heap_alloc, ARRAY_SIZE * TILES_PER_PARTITION);
    // 5. Move data
    for (uint32_t i = 0; i < ARRAY_SIZE; i++) {
      array[i] = i;
    }
    // 6. Change addressing scheme
    partition_config(0, NUM_CORES / NUM_CORES_PER_TILE);
    for (uint32_t i = 0; i < ARRAY_SIZE; i++) {
      uint32_t *fetch_address = &array[0] + \
        (i % (TILES_PER_PARTITION * NUM_CORES_PER_TILE * BANKING_FACTOR)) + \
        (i / (TILES_PER_PARTITION * NUM_CORES_PER_TILE * BANKING_FACTOR)) * NUM_BANKS;
      if (i != *fetch_address) {
        printf("%4d != %4d at address %8X.\n", i, *fetch_address, fetch_address);
      }
    }
    // 7. Free array
    domain_free(dynamic_heap_alloc, array);
    // 8. Free dynamic allocator
    free_dynamic_heap_alloc();
    printf("Done!\n");
  }

  mempool_barrier(num_cores);


  return 0;
}
