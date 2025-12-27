/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Test program for NUMA binding functionality
// Compile and run: g++ -o test_numa test_numa_binding.cpp -I. -lnuma -lglog
// -lpthread

#include <glog/logging.h>

#include <iostream>

#include "xllm/core/util/numa_utils.h"

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  std::cout << "=== NUMA Binding Test ===" << std::endl;

  // Test 1: Check NUMA availability
  std::cout << "\nTest 1: Checking NUMA availability..." << std::endl;
  bool numa_available = xllm::numa::is_numa_available();
  std::cout << "NUMA available: " << (numa_available ? "YES" : "NO")
            << std::endl;

  if (!numa_available) {
    std::cout << "NUMA not available on this system. Exiting." << std::endl;
    return 0;
  }

  // Test 2: Get number of NUMA nodes
  std::cout << "\nTest 2: Getting number of NUMA nodes..." << std::endl;
  int32_t num_nodes = xllm::numa::get_num_numa_nodes();
  std::cout << "Number of NUMA nodes: " << num_nodes << std::endl;

  // Test 3: Log NUMA topology
  std::cout << "\nTest 3: NUMA topology information:" << std::endl;
  xllm::numa::log_numa_topology();

  // Test 4: Get device NUMA node
  std::cout << "\nTest 4: Getting NUMA node for device 0..." << std::endl;
  int32_t device_numa_node = xllm::numa::get_device_numa_node(0);
  std::cout << "Device 0 is on NUMA node: " << device_numa_node << std::endl;

  // Test 5: Get current NUMA node before binding
  std::cout << "\nTest 5: Getting current NUMA node..." << std::endl;
  int32_t current_node = xllm::numa::get_current_numa_node();
  std::cout << "Current process is running on NUMA node: " << current_node
            << std::endl;

  // Test 6: Bind process to NUMA node 0
  if (num_nodes > 0) {
    std::cout << "\nTest 6: Binding process to NUMA node 0..." << std::endl;
    int ret = xllm::numa::bind_process_to_numa_node(0);
    if (ret == 0) {
      std::cout << "Successfully bound to NUMA node 0" << std::endl;

      // Verify binding
      int32_t new_node = xllm::numa::get_current_numa_node();
      std::cout << "After binding, process is on NUMA node: " << new_node
                << std::endl;

      if (new_node == 0) {
        std::cout << "✓ NUMA binding verified successfully!" << std::endl;
      } else {
        std::cout << "✗ NUMA binding verification failed!" << std::endl;
      }
    } else {
      std::cout << "Failed to bind to NUMA node 0" << std::endl;
    }
  }

  // Test 7: Get CPUs for each NUMA node
  std::cout << "\nTest 7: CPUs per NUMA node:" << std::endl;
  for (int32_t node = 0; node < num_nodes; ++node) {
    std::vector<int32_t> cpus = xllm::numa::get_numa_node_cpus(node);
    std::cout << "NUMA node " << node << " has " << cpus.size() << " CPUs: [";
    for (size_t i = 0; i < cpus.size() && i < 10; ++i) {
      if (i > 0) std::cout << ", ";
      std::cout << cpus[i];
    }
    if (cpus.size() > 10) {
      std::cout << ", ... (total: " << cpus.size() << ")";
    }
    std::cout << "]" << std::endl;
  }

  std::cout << "\n=== All NUMA tests completed ===" << std::endl;
  return 0;
}