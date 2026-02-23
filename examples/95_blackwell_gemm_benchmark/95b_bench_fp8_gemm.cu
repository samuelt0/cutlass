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

/*! \file
    \brief Blackwell FP8 GEMM benchmark: 22 kernel configurations.

    Usage:
      $ ./95b_bench_fp8_gemm --shapes=256x256x128,1024x1024x256,2048x2048x2048 --iterations=100 --csv
*/

#include <iostream>
#include <vector>
#include <string>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "benchmark_common.h"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////
/// FP8 GEMM configuration template
///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class MmaTile_,
  class Cluster_,
  class MainloopSched,
  class EpiSched,
  class StageCountType_ = void,
  class TileSchedulerTag = void
>
struct FP8GemmConfig {

  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using MmaTileShape = MmaTile_;
  using ClusterShape = Cluster_;
  using MainloopSchedule = MainloopSched;
  using EpilogueSchedule = EpiSched;
  using StageCountTag = StageCountType_;
  using TileScheduler = TileSchedulerTag;

  using FusionOp = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpiSched,
      FusionOp
    >::CollectiveOp;

  using StageCountType = cute::conditional_t<
      cute::is_same_v<StageCountType_, void>,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      StageCountType_>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      StageCountType,
      MainloopSched
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileSchedulerTag
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
/// 22 FP8 configurations
///////////////////////////////////////////////////////////////////////////////////////////////////

using FP8_1sm_128x128_c1x1_clc = FP8GemmConfig<
    Shape<_128, _128, _128>, Shape<_1, _1, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm>;

using FP8_1sm_128x256_c1x1_clc = FP8GemmConfig<
    Shape<_128, _256, _128>, Shape<_1, _1, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm>;

using FP8_2sm_256x128_c2x2_clc = FP8GemmConfig<
    Shape<_256, _128, _128>, Shape<_2, _2, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized2Sm>;

using FP8_2sm_256x256_c2x2_clc = FP8GemmConfig<
    Shape<_256, _256, _128>, Shape<_2, _2, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized2Sm>;

using FP8_auto_c2x2_clc = FP8GemmConfig<
    Shape<_256, _128, _128>, Shape<_2, _2, _1>,
    cutlass::gemm::collective::KernelScheduleAuto,
    cutlass::epilogue::collective::EpilogueScheduleAuto>;

using FP8_2sm_256x128_c2x2_static = FP8GemmConfig<
    Shape<_256, _128, _128>, Shape<_2, _2, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized2Sm,
    void,
    cutlass::gemm::StaticPersistentScheduler>;

// Additional cluster shapes to match FP16 coverage
using FP8_1sm_128x128_c1x2_clc = FP8GemmConfig<
    Shape<_128, _128, _128>, Shape<_1, _2, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm>;

using FP8_2sm_256x128_c2x1_clc = FP8GemmConfig<
    Shape<_256, _128, _128>, Shape<_2, _1, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized2Sm>;

using FP8_2sm_256x128_c4x2_clc = FP8GemmConfig<
    Shape<_256, _128, _128>, Shape<_4, _2, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized2Sm>;

// TMA epilogue variants
using FP8_1sm_128x128_c1x1_tma = FP8GemmConfig<
    Shape<_128, _128, _128>, Shape<_1, _1, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
    cutlass::epilogue::TmaWarpSpecialized1Sm>;

using FP8_2sm_256x128_c2x2_tma = FP8GemmConfig<
    Shape<_256, _128, _128>, Shape<_2, _2, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
    cutlass::epilogue::TmaWarpSpecialized2Sm>;

// Manual stage count variants (1SM NoSmem)
using FP8_1sm_128x128_c1x1_clc_s2 = FP8GemmConfig<
    Shape<_128, _128, _128>, Shape<_1, _1, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm, Int<2>>;

using FP8_1sm_128x128_c1x1_clc_s3 = FP8GemmConfig<
    Shape<_128, _128, _128>, Shape<_1, _1, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm, Int<3>>;

using FP8_1sm_128x128_c1x1_clc_s4 = FP8GemmConfig<
    Shape<_128, _128, _128>, Shape<_1, _1, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm, Int<4>>;

// Manual stage count variants (2SM NoSmem, 256x128)
using FP8_2sm_256x128_c2x2_clc_s2 = FP8GemmConfig<
    Shape<_256, _128, _128>, Shape<_2, _2, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized2Sm, Int<2>>;

using FP8_2sm_256x128_c2x2_clc_s3 = FP8GemmConfig<
    Shape<_256, _128, _128>, Shape<_2, _2, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized2Sm, Int<3>>;

using FP8_2sm_256x128_c2x2_clc_s4 = FP8GemmConfig<
    Shape<_256, _128, _128>, Shape<_2, _2, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized2Sm, Int<4>>;

// Manual stage count variants (2SM NoSmem, 256x256)
using FP8_2sm_256x256_c2x2_clc_s2 = FP8GemmConfig<
    Shape<_256, _256, _128>, Shape<_2, _2, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized2Sm, Int<2>>;

// TMA epilogue with manual stage count
using FP8_1sm_128x128_c1x1_tma_s3 = FP8GemmConfig<
    Shape<_128, _128, _128>, Shape<_1, _1, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
    cutlass::epilogue::TmaWarpSpecialized1Sm, Int<3>>;

using FP8_2sm_256x128_c2x2_tma_s3 = FP8GemmConfig<
    Shape<_256, _128, _128>, Shape<_2, _2, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
    cutlass::epilogue::TmaWarpSpecialized2Sm, Int<3>>;

// Additional cluster shape variants
using FP8_2sm_256x256_c2x1_clc = FP8GemmConfig<
    Shape<_256, _256, _128>, Shape<_2, _1, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized2Sm>;

using FP8_1sm_128x256_c1x2_clc = FP8GemmConfig<
    Shape<_128, _256, _128>, Shape<_1, _2, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm>;

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Benchmark runner
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Config>
struct FP8BenchRunner {
  using Gemm = typename Config::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  cutlass::DeviceAllocation<typename Config::ElementA> block_A;
  cutlass::DeviceAllocation<typename Config::ElementB> block_B;
  cutlass::DeviceAllocation<typename Config::ElementC> block_C;
  cutlass::DeviceAllocation<typename Config::ElementD> block_D;

  template <class Element>
  void initialize_block(cutlass::DeviceAllocation<Element>& block, uint64_t seed) {
    Element scope_max, scope_min;
    int bits = cutlass::sizeof_bits<Element>::value;
    if (bits <= 8) {
      scope_max = Element(2);
      scope_min = Element(-2);
    } else {
      scope_max = Element(8);
      scope_min = Element(-8);
    }
    cutlass::reference::device::BlockFillRandomUniform(
        block.get(), block.size(), seed, scope_max, scope_min, 0);
  }

  void run(const std::string& config_name,
           const BenchmarkOptions& options,
           const cutlass::KernelHardwareInfo& hw_info,
           std::vector<BenchmarkResult>& results) {

    BenchmarkResult result_template;
    result_template.config_name = config_name;
    result_template.precision = "fp8";
    result_template.mma_tile_shape = shape_to_string<typename Config::MmaTileShape>();
    result_template.cluster_shape = shape_to_string<typename Config::ClusterShape>();
    result_template.mainloop_schedule = mainloop_schedule_name<typename Config::MainloopSchedule>();
    result_template.epilogue_schedule = epilogue_schedule_name<typename Config::EpilogueSchedule>();
    result_template.stage_count = stage_count_name<typename Config::StageCountTag>();
    result_template.tile_scheduler = tile_scheduler_name<typename Config::TileScheduler>();

    for (const auto& shape : options.shapes) {
      int M = shape.M, N = shape.N, K = shape.K;

      block_A.reset(M * K);
      block_B.reset(K * N);
      block_C.reset(M * N);
      block_D.reset(M * N);

      initialize_block(block_A, 2023);
      initialize_block(block_B, 2022);
      initialize_block(block_C, 2021);

      auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
      auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
      auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
      auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

      typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {block_A.get(), stride_A, block_B.get(), stride_B},
        {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D},
        hw_info
      };
      arguments.scheduler.max_swizzle_size = options.swizzle;

      size_t workspace_size = Gemm::get_workspace_size(arguments);
      cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

      benchmark_gemm<Gemm>(result_template, options, hw_info, arguments, workspace, results);
    }
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    std::cerr << "This example requires CUDA 12.8 or newer." << std::endl;
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));

  if (props.major != 10 || props.minor != 0) {
    std::cerr << "This example requires a GPU with compute capability 100." << std::endl;
    return 0;
  }

  BenchmarkOptions options;
  // FP8 MMA tile K=128, so default shapes must have K >= 128
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout);
    return 0;
  }

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  std::vector<BenchmarkResult> results;

  if (options.csv) {
    print_csv_header();
  } else {
    std::cout << "=== FP8 GEMM Benchmark (A/B=float_e4m3_t, C/D=float) ===" << std::endl;
    std::cout << "  SM count: " << hw_info.sm_count << std::endl;
    std::cout << "  Iterations: " << options.iterations << ", Warmup: " << options.warmup << std::endl;
    std::cout << std::endl;
  }

  FP8BenchRunner<FP8_1sm_128x128_c1x1_clc>{}.run("1sm_128x128_c1x1_clc", options, hw_info, results);
  FP8BenchRunner<FP8_1sm_128x256_c1x1_clc>{}.run("1sm_128x256_c1x1_clc", options, hw_info, results);
  FP8BenchRunner<FP8_1sm_128x128_c1x2_clc>{}.run("1sm_128x128_c1x2_clc", options, hw_info, results);
  FP8BenchRunner<FP8_2sm_256x128_c2x1_clc>{}.run("2sm_256x128_c2x1_clc", options, hw_info, results);
  FP8BenchRunner<FP8_2sm_256x128_c2x2_clc>{}.run("2sm_256x128_c2x2_clc", options, hw_info, results);
  FP8BenchRunner<FP8_2sm_256x256_c2x2_clc>{}.run("2sm_256x256_c2x2_clc", options, hw_info, results);
  FP8BenchRunner<FP8_2sm_256x128_c4x2_clc>{}.run("2sm_256x128_c4x2_clc", options, hw_info, results);
  FP8BenchRunner<FP8_auto_c2x2_clc>{}.run("auto_c2x2_clc", options, hw_info, results);
  FP8BenchRunner<FP8_2sm_256x128_c2x2_static>{}.run("2sm_256x128_c2x2_static", options, hw_info, results);
  // TMA epilogue variants
  FP8BenchRunner<FP8_1sm_128x128_c1x1_tma>{}.run("1sm_128x128_c1x1_tma", options, hw_info, results);
  FP8BenchRunner<FP8_2sm_256x128_c2x2_tma>{}.run("2sm_256x128_c2x2_tma", options, hw_info, results);
  // Manual stage count variants
  FP8BenchRunner<FP8_1sm_128x128_c1x1_clc_s2>{}.run("1sm_128x128_c1x1_clc_s2", options, hw_info, results);
  FP8BenchRunner<FP8_1sm_128x128_c1x1_clc_s3>{}.run("1sm_128x128_c1x1_clc_s3", options, hw_info, results);
  FP8BenchRunner<FP8_1sm_128x128_c1x1_clc_s4>{}.run("1sm_128x128_c1x1_clc_s4", options, hw_info, results);
  FP8BenchRunner<FP8_2sm_256x128_c2x2_clc_s2>{}.run("2sm_256x128_c2x2_clc_s2", options, hw_info, results);
  FP8BenchRunner<FP8_2sm_256x128_c2x2_clc_s3>{}.run("2sm_256x128_c2x2_clc_s3", options, hw_info, results);
  FP8BenchRunner<FP8_2sm_256x128_c2x2_clc_s4>{}.run("2sm_256x128_c2x2_clc_s4", options, hw_info, results);
  FP8BenchRunner<FP8_2sm_256x256_c2x2_clc_s2>{}.run("2sm_256x256_c2x2_clc_s2", options, hw_info, results);
  FP8BenchRunner<FP8_1sm_128x128_c1x1_tma_s3>{}.run("1sm_128x128_c1x1_tma_s3", options, hw_info, results);
  FP8BenchRunner<FP8_2sm_256x128_c2x2_tma_s3>{}.run("2sm_256x128_c2x2_tma_s3", options, hw_info, results);
  // Additional cluster shape variants
  FP8BenchRunner<FP8_2sm_256x256_c2x1_clc>{}.run("2sm_256x256_c2x1_clc", options, hw_info, results);
  FP8BenchRunner<FP8_1sm_128x256_c1x2_clc>{}.run("1sm_128x256_c1x2_clc", options, hw_info, results);

  // Secondary shapes - run with representative configs only
  options.shapes = {
    {4608, 3072, 3072}, {4608, 9216, 3072}, {4608, 12288, 3072}, {4608, 3072, 12288},
    {16384, 3072, 3072}, {16384, 9216, 3072}, {16384, 12288, 3072}, {16384, 3072, 12288},
    {128, 3072, 3072}, {128, 9216, 3072}, {128, 12288, 3072}, {128, 3072, 12288}
  };
  FP8BenchRunner<FP8_1sm_128x128_c1x1_clc>{}.run("1sm_128x128_c1x1_clc", options, hw_info, results);
  FP8BenchRunner<FP8_2sm_256x128_c2x2_clc>{}.run("2sm_256x128_c2x2_clc", options, hw_info, results);
  FP8BenchRunner<FP8_2sm_256x256_c2x2_clc>{}.run("2sm_256x256_c2x2_clc", options, hw_info, results);
  FP8BenchRunner<FP8_auto_c2x2_clc>{}.run("auto_c2x2_clc", options, hw_info, results);

  if (!options.csv) {
    std::cout << "\nDone. " << results.size() << " benchmarks completed." << std::endl;
  }

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
