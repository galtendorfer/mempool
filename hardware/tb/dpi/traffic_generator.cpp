// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Matheus Cavalcante, ETH Zurich

// Includes
#include <cstdlib>
#include <iostream>
#include <limits.h>
#include <map>
#include <mutex>
#include <queue>
#include <random>
#include <stdint.h>

// Typedefs
typedef uint32_t addr_t;
typedef uint32_t req_id_t;
typedef uint32_t core_id_t;

// Function declarations
extern "C" {
void create_request(const core_id_t *core_id, const uint32_t *cycle,
                    const addr_t *tcdm_base_addr, const addr_t *tcdm_mask,
                    const addr_t *tile_mask, // Indicates the bits of the addr
                                             // who identify the tile
                    const addr_t *seq_mask,  // Indicates the bits that have to
                                             // be set for a local request
                    bool *req_valid, req_id_t *req_id, addr_t *req_addr);
void probe_response(const core_id_t *core_id, const uint32_t *cycle,
                    const bool req_ready, const bool resp_valid,
                    const req_id_t *resp_id);
void print_histogram();
}

// Request probabilities
#ifndef TG_REQ_PROB
#define TG_REQ_PROB 0.2
#endif

#ifndef TG_SEQ_PROB
#define TG_SEQ_PROB 0
#endif

// Number of cycles the simulation has ran
#ifndef TG_NCYCLES
#define TG_NCYCLES 10000
#endif

// Number of cores
#ifndef NUM_CORES
#define NUM_CORES 256
#endif

// Deterministic seed (0 = use std::random_device, non-deterministic)
#ifndef TG_SEED
#define TG_SEED 0
#endif

// Runtime parameters (override compile-time defaults via environment variables)
static int    tg_tile_range = 0;
static float  tg_req_prob = TG_REQ_PROB;
static float  tg_seq_prob = TG_SEQ_PROB;
static int    tg_ncycles  = TG_NCYCLES;
static int    num_cores   = NUM_CORES;
static int    params_initialized = 0;

static void init_params() {
  if (params_initialized) return;
  const char *env;
  if ((env = std::getenv("TG_REQ_PROB"))) tg_req_prob = std::atof(env);
  if ((env = std::getenv("TG_SEQ_PROB"))) tg_seq_prob = std::atof(env);
  if ((env = std::getenv("TG_NCYCLES")))  tg_ncycles  = std::atoi(env);
  if ((env = std::getenv("NUM_CORES")))   num_cores   = std::atoi(env);
  if ((env = std::getenv("TG_TILE_RANGE"))) {
    tg_tile_range = std::atoi(env);
  }
  params_initialized = 1;
}

// Randomizer (deterministic seed if TG_SEED != 0)
std::default_random_engine e1(TG_SEED == 0 ? std::random_device{}() : TG_SEED);
std::uniform_int_distribution<addr_t> addr_dist(0, INT_MAX);
std::uniform_real_distribution<float> real_dist(0, 1);

// Mutexes
std::mutex g_mutex;

// Request struct
typedef struct {
  addr_t addr;
  req_id_t id;
} request_t;

// Map the starting cycle of each request
std::map<std::pair<core_id_t, req_id_t>, uint32_t> starting_cycle;
// Latency histogram
std::map<uint32_t, uint32_t> latency_histogram;
// Request queues
std::map<core_id_t, std::queue<request_t>> requests;

// Transaction IDs
uint32_t tran_id_initialized = 0;
std::map<core_id_t, std::queue<req_id_t>> tran_id;

extern "C" void create_request(const core_id_t *core_id, const uint32_t *cycle,
                               const addr_t *tcdm_base_addr,
                               const addr_t *tcdm_mask, const addr_t *tile_mask,
                               const addr_t *seq_mask, bool *req_valid,
                               req_id_t *req_id, addr_t *req_addr) {
  // Lock the function
  std::lock_guard<std::mutex> guard(g_mutex);

  // Initialize runtime parameters and transaction ID queues
  init_params();
  if (!tran_id_initialized) {
    for (int c = 0; c < num_cores; c++)
      for (int id = 0; id < 2048; id++)
        tran_id[c].push(id);
    tran_id_initialized = 1;
  }

  // Generate new request
  if (!tran_id[*core_id].empty()) {
    if (real_dist(e1) < tg_req_prob) {
      // Generate new address
      request_t next_request;

      // Transaction id
      req_id_t req_id = tran_id[*core_id].front();
      tran_id[*core_id].pop();

      next_request.id = req_id;
      next_request.addr = addr_dist(e1);
      // Make sure the request is in the TCDM region
      next_request.addr =
          (next_request.addr & ~(*tcdm_mask)) | (*tcdm_base_addr & *tcdm_mask);

      const addr_t local_tile = *seq_mask & *tile_mask;
      if (tg_tile_range > 0) {
        const addr_t tile_lsb = *tile_mask & (~(*tile_mask) + 1u);
        const addr_t partition_mask =
            (static_cast<addr_t>(tg_tile_range) - 1u) * tile_lsb;
        addr_t target_tile = local_tile;
        if ((real_dist(e1) >= tg_seq_prob) && tg_tile_range > 1) {
          std::uniform_int_distribution<uint32_t> partition_dist(
              0, tg_tile_range - 2);
          addr_t target_offset =
              static_cast<addr_t>(partition_dist(e1)) * tile_lsb;
          const addr_t local_offset = local_tile & partition_mask;
          if (target_offset >= local_offset) {
            target_offset += tile_lsb;
          }
          target_tile = (local_tile & ~partition_mask) | target_offset;
        }

        next_request.addr =
            (next_request.addr & ~(*tile_mask)) | (target_tile & *tile_mask);
      } else if (real_dist(e1) < tg_seq_prob) {
        next_request.addr = (next_request.addr & ~(*tile_mask)) | local_tile;
      }

      // Address is aligned to 32 bits
      next_request.addr = (next_request.addr >> 2) << 2;

      // Push the request
      starting_cycle[std::make_pair(*core_id, req_id)] = *cycle;
      requests[*core_id].push(next_request);
    }
  } else {
    std::cerr
        << "[traffic_generator] No more available transaction identifiers!"
        << std::endl;
  }

  // Is there a request to be sent?
  if (!requests[*core_id].empty()) {
    *req_valid = true;
    *req_id = requests[*core_id].front().id;
    *req_addr = requests[*core_id].front().addr;
  } else {
    *req_valid = false;
    *req_id = 0;
    *req_addr = 0;
  }
}

extern "C" void probe_response(const core_id_t *core_id, const uint32_t *cycle,
                               const bool req_ready, const bool resp_valid,
                               const req_id_t *resp_id) {
  // Lock the function
  std::lock_guard<std::mutex> guard(g_mutex);

  // Acknowledged request
  if (req_ready && !requests[*core_id].empty()) {
    // Pop the request
    requests[*core_id].pop();
  }

  // Acknowledged response
  if (resp_valid) {
    // Free the request ID
    tran_id[*core_id].push(*resp_id);

    // Account for the latency
    uint32_t latency =
        *cycle - starting_cycle[std::make_pair(*core_id, *resp_id)];
    if (latency_histogram.count(latency) != 0)
      latency_histogram[latency]++;
    else
      latency_histogram[latency] = 1;
  }
}

extern "C" void print_histogram() {
  uint32_t latency = 0;
  uint32_t tran_counter = 0;

  std::cout << "Latency\tCount" << std::endl;
  for (const auto &it : latency_histogram) {
    tran_counter += it.second;
    latency += it.first * it.second;
    std::cout << it.first << "\t" << it.second << std::endl;
  }

  std::cout << "Average latency: " << (1.0 * latency) / tran_counter
            << std::endl;
  std::cout << "Throughput: " << (1.0 * tran_counter) / (tg_ncycles * num_cores)
            << std::endl;
}
