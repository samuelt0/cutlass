/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <type_traits>

#include "cute/tensor.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder_decl.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "helper.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

struct GemmShape {
  int M, N, K;
};

struct BenchmarkResult {
  std::string config_name;
  std::string precision;
  int M, N, K;
  std::string mma_tile_shape;
  std::string cluster_shape;
  std::string mainloop_schedule;
  std::string epilogue_schedule;
  std::string stage_count;
  std::string tile_scheduler;
  double kernel_ms;      // Pure kernel time (run only)
  double init_run_ms;    // initialize() + run() time
  double init_ms;        // initialize() only time
  double gflops;
  bool passed;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Type-to-string helpers for structured CSV metadata
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline std::string mainloop_schedule_name() {
  if constexpr (std::is_same_v<T, cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>)
    return "TmaWarpSpecialized1Sm";
  else if constexpr (std::is_same_v<T, cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>)
    return "TmaWarpSpecialized2Sm";
  else if constexpr (std::is_same_v<T, cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100>)
    return "TmaWarpSpecialized1SmNvf4";
  else if constexpr (std::is_same_v<T, cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100>)
    return "TmaWarpSpecialized2SmNvf4";
  else if constexpr (std::is_same_v<T, cutlass::gemm::collective::KernelScheduleAuto>)
    return "Auto";
  else
    return "Unknown";
}

template <typename T>
inline std::string epilogue_schedule_name() {
  if constexpr (std::is_same_v<T, cutlass::epilogue::NoSmemWarpSpecialized1Sm>)
    return "NoSmem1Sm";
  else if constexpr (std::is_same_v<T, cutlass::epilogue::NoSmemWarpSpecialized2Sm>)
    return "NoSmem2Sm";
  else if constexpr (std::is_same_v<T, cutlass::epilogue::TmaWarpSpecialized1Sm>)
    return "Tma1Sm";
  else if constexpr (std::is_same_v<T, cutlass::epilogue::TmaWarpSpecialized2Sm>)
    return "Tma2Sm";
  else if constexpr (std::is_same_v<T, cutlass::epilogue::TmaWarpSpecialized1SmNvf4>)
    return "Tma1SmNvf4";
  else if constexpr (std::is_same_v<T, cutlass::epilogue::TmaWarpSpecialized2SmNvf4>)
    return "Tma2SmNvf4";
  else if constexpr (std::is_same_v<T, cutlass::epilogue::collective::EpilogueScheduleAuto>)
    return "Auto";
  else
    return "Unknown";
}

template <typename T>
inline std::string tile_scheduler_name() {
  if constexpr (std::is_same_v<T, void>)
    return "CLC";
  else if constexpr (std::is_same_v<T, cutlass::gemm::StaticPersistentScheduler>)
    return "StaticPersistent";
  else
    return "Unknown";
}

template <typename T>
inline std::string stage_count_name() {
  if constexpr (std::is_same_v<T, void>)
    return "AutoCarveout";
  else if constexpr (cute::is_same_v<T, cute::Int<2>> ||
                     cute::is_same_v<T, cutlass::gemm::collective::StageCount<2>>)
    return "2";
  else if constexpr (cute::is_same_v<T, cute::Int<3>> ||
                     cute::is_same_v<T, cutlass::gemm::collective::StageCount<3>>)
    return "3";
  else if constexpr (cute::is_same_v<T, cute::Int<4>> ||
                     cute::is_same_v<T, cutlass::gemm::collective::StageCount<4>>)
    return "4";
  else if constexpr (cute::is_same_v<T, cute::Int<5>> ||
                     cute::is_same_v<T, cutlass::gemm::collective::StageCount<5>>)
    return "5";
  else
    return "Unknown";
}

template <typename ShapeT>
inline std::string shape_to_string() {
  return std::to_string(cute::size<0>(ShapeT{})) + "x" +
         std::to_string(cute::size<1>(ShapeT{})) + "x" +
         std::to_string(cute::size<2>(ShapeT{}));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

struct BenchmarkOptions {

  bool help = false;
  std::vector<GemmShape> shapes;
  int iterations = 100;
  int warmup = 5;
  bool verify = false;
  bool csv = false;
  float alpha = 1.0f;
  float beta = 0.0f;
  int swizzle = 0;

  BenchmarkOptions() {
    // Default shapes
    shapes = {{256, 256, 64}, {1024, 1024, 256}, {2048, 2048, 2048}};
  }

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("warmup", warmup, 5);
    cmd.get_cmd_line_argument("alpha", alpha, 1.0f);
    cmd.get_cmd_line_argument("beta", beta, 0.0f);
    cmd.get_cmd_line_argument("swizzle", swizzle, 0);
    verify = cmd.check_cmd_line_flag("verify");
    csv = cmd.check_cmd_line_flag("csv");

    std::string shapes_str;
    cmd.get_cmd_line_argument("shapes", shapes_str);
    if (!shapes_str.empty()) {
      shapes.clear();
      std::stringstream ss(shapes_str);
      std::string token;
      while (std::getline(ss, token, ',')) {
        GemmShape s;
        if (parse_shape(token, s)) {
          shapes.push_back(s);
        }
      }
    }
  }

  static bool parse_shape(const std::string& str, GemmShape& s) {
    // Parse "MxNxK" format
    int m, n, k;
    char x1, x2;
    std::istringstream iss(str);
    if (iss >> m >> x1 >> n >> x2 >> k && x1 == 'x' && x2 == 'x') {
      s = {m, n, k};
      return true;
    }
    return false;
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "Blackwell GEMM Benchmark\n\n"
        << "Options:\n"
        << "  --help                      Display this usage statement\n"
        << "  --shapes=MxNxK[,MxNxK...]   Comma-separated problem shapes (default: 256x256x64,1024x1024x256,2048x2048x2048)\n"
        << "  --iterations=<int>          Timing iterations (default: 100)\n"
        << "  --warmup=<int>              Warmup iterations (default: 5)\n"
        << "  --verify                    Run correctness check\n"
        << "  --csv                       Output CSV format\n"
        << "  --alpha=<f32>               Epilogue scalar alpha (default: 1.0)\n"
        << "  --beta=<f32>                Epilogue scalar beta (default: 0.0)\n"
        << "  --swizzle=<int>             Cluster rasterization swizzle (default: 0)\n\n";
    return out;
  }

  static double compute_gflops(int M, int N, int K, double runtime_ms) {
    double flop = 2.0 * M * N * K;
    return flop / (runtime_ms / 1000.0) / 1.0e9;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void print_csv_header() {
  std::cout << "precision,config,M,N,K,mma_tile_shape,cluster_shape,mainloop_schedule,epilogue_schedule,stage_count,tile_scheduler,kernel_ms,init_run_ms,init_ms,GFLOPS,verified" << std::endl;
}

inline void print_csv_row(const BenchmarkResult& r) {
  std::cout << r.precision << ","
            << r.config_name << ","
            << r.M << "," << r.N << "," << r.K << ","
            << r.mma_tile_shape << ","
            << r.cluster_shape << ","
            << r.mainloop_schedule << ","
            << r.epilogue_schedule << ","
            << r.stage_count << ","
            << r.tile_scheduler << ","
            << r.kernel_ms << ","
            << r.init_run_ms << ","
            << r.init_ms << ","
            << r.gflops << ","
            << (r.passed ? "true" : "false")
            << std::endl;
}

inline void print_result_row(const BenchmarkResult& r) {
  printf("  %-40s  %4dx%4dx%4d  kernel=%8.4f ms  init+run=%8.4f ms  init=%8.4f ms  %10.1f GFLOPS  %s\n",
         r.config_name.c_str(), r.M, r.N, r.K,
         r.kernel_ms, r.init_run_ms, r.init_ms, r.gflops,
         r.passed ? "PASS" : "FAIL");
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Three-phase benchmark runner for a single GEMM configuration.
/// Measures: kernel-only, init+run, init-only timings.
/// The result_template should have config metadata (config_name, precision,
/// mma_tile_shape, cluster_shape, mainloop_schedule, etc.) pre-populated.
template <typename Gemm>
bool benchmark_gemm(
    const BenchmarkResult& result_template,
    const BenchmarkOptions& options,
    const cutlass::KernelHardwareInfo& hw_info,
    typename Gemm::Arguments& arguments,
    cutlass::device_memory::allocation<uint8_t>& workspace,
    std::vector<BenchmarkResult>& results,
    bool do_verify = false)
{
  Gemm gemm;

  auto [M, N, K, L] = arguments.problem_shape;

  // Check if this configuration can implement the problem
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    BenchmarkResult r = result_template;
    r.M = M; r.N = N; r.K = K;
    r.kernel_ms = -1; r.init_run_ms = -1; r.init_ms = -1;
    r.gflops = 0; r.passed = false;
    results.push_back(r);
    if (options.csv) {
      print_csv_row(r);
    } else {
      printf("  %-40s  %4dx%4dx%4d  N/A (cannot implement)\n",
             r.config_name.c_str(), M, N, K);
    }
    return false;
  }

  // Initialize and run once for correctness/warmup
  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    return false;
  }

  status = gemm.run();
  if (status != cutlass::Status::kSuccess) {
    return false;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  bool passed = true; // default if not verifying

  // Warmup
  for (int i = 0; i < options.warmup; ++i) {
    gemm.initialize(arguments, workspace.get());
    gemm.run();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Phase 1: Kernel-only timing (run() only, no initialize)
  double kernel_ms = 0;
  {
    // Re-initialize once before kernel-only timing
    gemm.initialize(arguments, workspace.get());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    for (int i = 0; i < options.iterations; ++i) {
      gemm.run();
    }
    timer.stop();
    kernel_ms = double(timer.elapsed_millis()) / double(options.iterations);
  }

  // Phase 2: Init+Run timing
  double init_run_ms = 0;
  {
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < options.iterations; ++i) {
      gemm.initialize(arguments, workspace.get());
      gemm.run();
    }
    timer.stop();
    init_run_ms = double(timer.elapsed_millis()) / double(options.iterations);
  }

  // Phase 3: Init-only timing
  double init_ms = 0;
  {
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < options.iterations; ++i) {
      gemm.initialize(arguments, workspace.get());
    }
    timer.stop();
    init_ms = double(timer.elapsed_millis()) / double(options.iterations);
  }

  double gflops = BenchmarkOptions::compute_gflops(M, N, K, kernel_ms);

  BenchmarkResult r = result_template;
  r.M = M; r.N = N; r.K = K;
  r.kernel_ms = kernel_ms;
  r.init_run_ms = init_run_ms;
  r.init_ms = init_ms;
  r.gflops = gflops;
  r.passed = passed;
  results.push_back(r);

  if (options.csv) {
    print_csv_row(r);
  } else {
    print_result_row(r);
  }

  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
