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

#include "numa_utils.h"

#include <glog/logging.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>

// Check if NUMA is available at compile time
#ifdef __has_include
#if __has_include(<numa.h>)
#define HAS_NUMA 1
#include <numa.h>
#else
#define HAS_NUMA 0
#endif
#else
#define HAS_NUMA 0
#endif

#if defined(USE_NPU)
#include "acl/acl.h"
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

namespace xllm {
namespace numa {

bool is_numa_available() {
#if HAS_NUMA
  static bool checked = false;
  static bool available = false;

  if (!checked) {
    available = (numa_available() >= 0);
    checked = true;
    if (!available) {
      LOG(WARNING) << "NUMA is not available on this system";
    }
  }
  return available;
#else
  return false;
#endif
}

int32_t get_num_numa_nodes() {
#if HAS_NUMA
  if (!is_numa_available()) {
    return -1;
  }
  return numa_num_configured_nodes();
#else
  return -1;
#endif
}

int32_t get_device_numa_node(int32_t device_index) {
  if (!is_numa_available()) {
    return -1;
  }

#if defined(USE_NPU)
  // For NPU devices, read NUMA node from sysfs
  std::string numa_path = "/sys/class/accel/accel" +
                          std::to_string(device_index) + "/device/numa_node";

  std::ifstream numa_file(numa_path);
  if (numa_file.is_open()) {
    int32_t numa_node;
    numa_file >> numa_node;
    numa_file.close();
    if (numa_node >= 0) {
      return numa_node;
    }
  }

  // Fallback: Try alternative path
  numa_path =
      "/sys/class/npu/npu" + std::to_string(device_index) + "/device/numa_node";
  numa_file.open(numa_path);
  if (numa_file.is_open()) {
    int32_t numa_node;
    numa_file >> numa_node;
    numa_file.close();
    if (numa_node >= 0) {
      return numa_node;
    }
  }

  LOG(WARNING) << "Unable to determine NUMA node for NPU device "
               << device_index << ", using default node 0";
  return 0;

#elif defined(USE_CUDA)
  // For CUDA devices, read from sysfs
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, device_index) == cudaSuccess) {
    std::string pci_bus_id(prop.pciBusID);
    std::string numa_path = "/sys/bus/pci/devices/" + pci_bus_id + "/numa_node";

    std::ifstream numa_file(numa_path);
    if (numa_file.is_open()) {
      int32_t numa_node;
      numa_file >> numa_node;
      numa_file.close();
      if (numa_node >= 0) {
        return numa_node;
      }
    }
  }

  LOG(WARNING) << "Unable to determine NUMA node for CUDA device "
               << device_index << ", using default node 0";
  return 0;

#else
  int32_t num_nodes = get_num_numa_nodes();
  if (num_nodes > 0) {
    return device_index % num_nodes;
  }
  return 0;
#endif
}

int bind_process_to_numa_node(int32_t numa_node) {
#if HAS_NUMA
  if (!is_numa_available()) {
    LOG(WARNING) << "NUMA not available, skipping process binding";
    return -1;
  }

  int32_t num_nodes = get_num_numa_nodes();
  if (numa_node < 0 || numa_node >= num_nodes) {
    LOG(ERROR) << "Invalid NUMA node " << numa_node << ", valid range is [0, "
               << num_nodes - 1 << "]";
    return -1;
  }

  struct bitmask* cpu_mask = numa_allocate_cpumask();
  if (numa_node_to_cpus(numa_node, cpu_mask) < 0) {
    LOG(ERROR) << "Failed to get CPU mask for NUMA node " << numa_node;
    numa_free_cpumask(cpu_mask);
    return -1;
  }

  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);

  int nr_possible_cpus = numa_num_possible_cpus();
  int nr_cpus = 0;
  for (int cpu = 0; cpu < nr_possible_cpus; ++cpu) {
    if (numa_bitmask_isbitset(cpu_mask, cpu) &&
        numa_bitmask_isbitset(numa_all_cpus_ptr, cpu)) {
      CPU_SET(cpu, &cpu_set);
      nr_cpus++;
    }
  }

  numa_free_cpumask(cpu_mask);

  if (nr_cpus == 0) {
    LOG(ERROR) << "No CPUs available on NUMA node " << numa_node;
    return -1;
  }

  pid_t pid = getpid();
  if (sched_setaffinity(pid, sizeof(cpu_set_t), &cpu_set) != 0) {
    LOG(ERROR) << "Failed to bind process to NUMA node " << numa_node << ": "
               << strerror(errno);
    return -1;
  }

  numa_set_preferred(numa_node);

  LOG(INFO) << "Successfully bound process " << pid << " to NUMA node "
            << numa_node << " with " << nr_cpus << " CPUs";

  return 0;
#else
  LOG(WARNING) << "NUMA support not compiled in, skipping process binding";
  return -1;
#endif
}

int bind_thread_to_numa_node(int32_t numa_node) {
#if HAS_NUMA
  if (!is_numa_available()) {
    LOG(WARNING) << "NUMA not available, skipping thread binding";
    return -1;
  }

  int32_t num_nodes = get_num_numa_nodes();
  if (numa_node < 0 || numa_node >= num_nodes) {
    LOG(ERROR) << "Invalid NUMA node " << numa_node << ", valid range is [0, "
               << num_nodes - 1 << "]";
    return -1;
  }

  struct bitmask* cpu_mask = numa_allocate_cpumask();
  if (numa_node_to_cpus(numa_node, cpu_mask) < 0) {
    LOG(ERROR) << "Failed to get CPU mask for NUMA node " << numa_node;
    numa_free_cpumask(cpu_mask);
    return -1;
  }

  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);

  int nr_possible_cpus = numa_num_possible_cpus();
  int nr_cpus = 0;
  for (int cpu = 0; cpu < nr_possible_cpus; ++cpu) {
    if (numa_bitmask_isbitset(cpu_mask, cpu) &&
        numa_bitmask_isbitset(numa_all_cpus_ptr, cpu)) {
      CPU_SET(cpu, &cpu_set);
      nr_cpus++;
    }
  }

  numa_free_cpumask(cpu_mask);

  if (nr_cpus == 0) {
    LOG(ERROR) << "No CPUs available on NUMA node " << numa_node;
    return -1;
  }

  pthread_t thread = pthread_self();
  if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpu_set) != 0) {
    LOG(ERROR) << "Failed to bind thread to NUMA node " << numa_node << ": "
               << strerror(errno);
    return -1;
  }

  LOG(INFO) << "Successfully bound thread to NUMA node " << numa_node
            << " with " << nr_cpus << " CPUs";

  return 0;
#else
  LOG(WARNING) << "NUMA support not compiled in, skipping thread binding";
  return -1;
#endif
}

int32_t get_current_numa_node() {
#if HAS_NUMA
  if (!is_numa_available()) {
    return -1;
  }

  int cpu = sched_getcpu();
  if (cpu < 0) {
    LOG(WARNING) << "Failed to get current CPU";
    return -1;
  }

  return numa_node_of_cpu(cpu);
#else
  return -1;
#endif
}

std::vector<int32_t> get_numa_node_cpus(int32_t numa_node) {
  std::vector<int32_t> cpus;

#if HAS_NUMA
  if (!is_numa_available()) {
    return cpus;
  }

  int32_t num_nodes = get_num_numa_nodes();
  if (numa_node < 0 || numa_node >= num_nodes) {
    LOG(ERROR) << "Invalid NUMA node " << numa_node;
    return cpus;
  }

  struct bitmask* cpu_mask = numa_allocate_cpumask();
  if (numa_node_to_cpus(numa_node, cpu_mask) < 0) {
    LOG(ERROR) << "Failed to get CPU mask for NUMA node " << numa_node;
    numa_free_cpumask(cpu_mask);
    return cpus;
  }

  int nr_possible_cpus = numa_num_possible_cpus();
  for (int cpu = 0; cpu < nr_possible_cpus; ++cpu) {
    if (numa_bitmask_isbitset(cpu_mask, cpu) &&
        numa_bitmask_isbitset(numa_all_cpus_ptr, cpu)) {
      cpus.push_back(cpu);
    }
  }

  numa_free_cpumask(cpu_mask);
#endif
  return cpus;
}

void log_numa_topology() {
#if HAS_NUMA
  if (!is_numa_available()) {
    LOG(INFO) << "NUMA topology: NUMA is not available on this system";
    return;
  }

  int32_t num_nodes = get_num_numa_nodes();
  LOG(INFO) << "NUMA topology: " << num_nodes << " NUMA nodes detected";

  for (int32_t node = 0; node < num_nodes; ++node) {
    std::vector<int32_t> cpus = get_numa_node_cpus(node);

    std::stringstream cpu_list;
    for (size_t i = 0; i < cpus.size(); ++i) {
      if (i > 0) cpu_list << ",";
      cpu_list << cpus[i];
    }

    long long mem_size = numa_node_size64(node, nullptr);

    LOG(INFO) << "  NUMA node " << node << ": " << cpus.size() << " CPUs ["
              << cpu_list.str() << "], "
              << "Memory: " << (mem_size / (1024 * 1024)) << " MB";
  }
#else
  LOG(INFO) << "NUMA topology: NUMA support not compiled in";
#endif
}

}  // namespace numa
}  // namespace xllm