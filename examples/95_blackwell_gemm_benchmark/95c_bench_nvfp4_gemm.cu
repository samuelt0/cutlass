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
    \brief Blackwell NVFP4 BlockScaled GEMM benchmark: 20 kernel configurations.
           Uses nv_float4_t<float_e2m1_t> inputs with scale factors, float output.

    Note: NVFP4 MMA tile K=256, so minimum problem K must be 256.
    Default shapes: 256x256x256, 1024x1024x256, 2048x2048x2048

    Usage:
      $ ./95c_bench_nvfp4_gemm --shapes=256x256x256,1024x1024x256,2048x2048x2048 --iterations=100 --csv
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
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "benchmark_common.h"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////
/// NVFP4 GEMM configuration template
///   A/B = nv_float4_t<float_e2m1_t>, C/D = float, OpClass = BlockScaledTensorOp
///   Outputs to float (no SFD generation) - simpler for benchmarking MMA throughput.
///   Pattern follows 72a_blackwell_nvfp4_bf16_gemm.cu
///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class MmaTile_,
  class Cluster_,
  class MainloopSched = cutlass::gemm::collective::KernelScheduleAuto,
  class EpiSched = cutlass::epilogue::collective::EpilogueScheduleAuto,
  class StageCountType_ = void,
  class TileSchedulerTag = void
>
struct NVFP4GemmConfig {

  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;

  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;

  static constexpr int AlignmentA = 32;
  static constexpr int AlignmentB = 32;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using MmaTileShape = MmaTile_;
  using ClusterShape = Cluster_;
  using MainloopSchedule = MainloopSched;
  using EpilogueSchedule = EpiSched;
  using StageCountTag = StageCountType_;
  using TileScheduler = TileSchedulerTag;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      EpiSched
    >::CollectiveOp;

  using StageCountType = cute::conditional_t<
      cute::is_same_v<StageCountType_, void>,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      StageCountType_>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB,
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

  // Scale factor layout types from mainloop
  using LayoutSFA = typename GemmKernel::CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename GemmKernel::CollectiveMainloop::LayoutSFB;
  using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
/// 20 NVFP4 configurations
///////////////////////////////////////////////////////////////////////////////////////////////////

// Auto schedule configs (CLC scheduler = void)
using NVFP4_1sm_128x128_c1x1 = NVFP4GemmConfig<Shape<_128, _128, _256>, Shape<_1, _1, _1>>;
using NVFP4_1sm_128x128_c1x2 = NVFP4GemmConfig<Shape<_128, _128, _256>, Shape<_1, _2, _1>>;
using NVFP4_2sm_256x128_c2x1 = NVFP4GemmConfig<Shape<_256, _128, _256>, Shape<_2, _1, _1>>;
using NVFP4_2sm_256x128_c2x2 = NVFP4GemmConfig<Shape<_256, _128, _256>, Shape<_2, _2, _1>>;
using NVFP4_2sm_256x128_c4x2 = NVFP4GemmConfig<Shape<_256, _128, _256>, Shape<_4, _2, _1>>;

// Explicit NVF4 mainloop + NoSmem epilogue
using NVFP4_1sm_128x128_c1x1_nvf4 = NVFP4GemmConfig<
    Shape<_128, _128, _256>, Shape<_1, _1, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm>;

using NVFP4_2sm_256x128_c2x2_nvf4 = NVFP4GemmConfig<
    Shape<_256, _128, _256>, Shape<_2, _2, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100,
    cutlass::epilogue::NoSmemWarpSpecialized2Sm>;

// Auto schedule with StaticPersistent tile scheduler
using NVFP4_2sm_256x128_c2x2_static = NVFP4GemmConfig<
    Shape<_256, _128, _256>, Shape<_2, _2, _1>,
    cutlass::gemm::collective::KernelScheduleAuto,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    void,
    cutlass::gemm::StaticPersistentScheduler>;

// Explicit NVF4 mainloop + TMA NVF4 epilogue
using NVFP4_1sm_128x128_c1x1_tma = NVFP4GemmConfig<
    Shape<_128, _128, _256>, Shape<_1, _1, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100,
    cutlass::epilogue::TmaWarpSpecialized1SmNvf4>;

using NVFP4_2sm_256x128_c2x2_tma = NVFP4GemmConfig<
    Shape<_256, _128, _256>, Shape<_2, _2, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100,
    cutlass::epilogue::TmaWarpSpecialized2SmNvf4>;

// Auto mainloop + explicit NoSmem epilogue
using NVFP4_1sm_128x128_c1x1_nosmem = NVFP4GemmConfig<
    Shape<_128, _128, _256>, Shape<_1, _1, _1>,
    cutlass::gemm::collective::KernelScheduleAuto,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm>;

using NVFP4_2sm_256x128_c2x2_nosmem = NVFP4GemmConfig<
    Shape<_256, _128, _256>, Shape<_2, _2, _1>,
    cutlass::gemm::collective::KernelScheduleAuto,
    cutlass::epilogue::NoSmemWarpSpecialized2Sm>;

// Manual stage count variants (1SM, explicit NVF4 mainloop + NoSmem epilogue)
// Note: blockscaled builder requires StageCount<N>, not cute::Int<N>
using NVFP4_1sm_128x128_c1x1_nvf4_s2 = NVFP4GemmConfig<
    Shape<_128, _128, _256>, Shape<_1, _1, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm,
    cutlass::gemm::collective::StageCount<2>>;

using NVFP4_1sm_128x128_c1x1_nvf4_s3 = NVFP4GemmConfig<
    Shape<_128, _128, _256>, Shape<_1, _1, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm,
    cutlass::gemm::collective::StageCount<3>>;

using NVFP4_1sm_128x128_c1x1_nvf4_s4 = NVFP4GemmConfig<
    Shape<_128, _128, _256>, Shape<_1, _1, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm,
    cutlass::gemm::collective::StageCount<4>>;

// Manual stage count variants (2SM, explicit NVF4 mainloop + NoSmem epilogue)
using NVFP4_2sm_256x128_c2x2_nvf4_s2 = NVFP4GemmConfig<
    Shape<_256, _128, _256>, Shape<_2, _2, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100,
    cutlass::epilogue::NoSmemWarpSpecialized2Sm,
    cutlass::gemm::collective::StageCount<2>>;

using NVFP4_2sm_256x128_c2x2_nvf4_s3 = NVFP4GemmConfig<
    Shape<_256, _128, _256>, Shape<_2, _2, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100,
    cutlass::epilogue::NoSmemWarpSpecialized2Sm,
    cutlass::gemm::collective::StageCount<3>>;

using NVFP4_2sm_256x128_c2x2_nvf4_s4 = NVFP4GemmConfig<
    Shape<_256, _128, _256>, Shape<_2, _2, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100,
    cutlass::epilogue::NoSmemWarpSpecialized2Sm,
    cutlass::gemm::collective::StageCount<4>>;

// TMA epilogue with manual stage count
using NVFP4_1sm_128x128_c1x1_tma_s3 = NVFP4GemmConfig<
    Shape<_128, _128, _256>, Shape<_1, _1, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100,
    cutlass::epilogue::TmaWarpSpecialized1SmNvf4,
    cutlass::gemm::collective::StageCount<3>>;

using NVFP4_2sm_256x128_c2x2_tma_s3 = NVFP4GemmConfig<
    Shape<_256, _128, _256>, Shape<_2, _2, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100,
    cutlass::epilogue::TmaWarpSpecialized2SmNvf4,
    cutlass::gemm::collective::StageCount<3>>;

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Benchmark runner for NVFP4
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Config>
struct NVFP4BenchRunner {
  using Gemm = typename Config::Gemm;
  using GemmKernel = typename Config::GemmKernel;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  using LayoutSFA = typename Config::LayoutSFA;
  using LayoutSFB = typename Config::LayoutSFB;

  using ElementAData = typename Config::ElementA::DataType;
  using ElementASF = typename Config::ElementA::ScaleFactorType;
  using ElementBData = typename Config::ElementB::DataType;
  using ElementBSF = typename Config::ElementB::ScaleFactorType;

  cutlass::HostTensor<ElementAData, cutlass::layout::PackedVectorLayout> block_A;
  cutlass::HostTensor<ElementASF, cutlass::layout::PackedVectorLayout> block_SFA;
  cutlass::HostTensor<ElementBData, cutlass::layout::PackedVectorLayout> block_B;
  cutlass::HostTensor<ElementBSF, cutlass::layout::PackedVectorLayout> block_SFB;
  cutlass::HostTensor<typename Config::ElementC, cutlass::layout::PackedVectorLayout> block_C;
  cutlass::HostTensor<typename Config::ElementD, cutlass::layout::PackedVectorLayout> block_D;

  template <typename Element, typename Layout>
  void initialize_block(cutlass::TensorView<Element, Layout> view, uint64_t seed) {
    double scope_max, scope_min;
    constexpr int bits = cutlass::sizeof_bits<Element>::value;
    if constexpr (bits <= 6) {
      scope_max = 2; scope_min = -2;
    } else if constexpr (bits <= 8) {
      if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t>) {
        scope_max = 4; scope_min = 1;
      } else {
        scope_max = 1; scope_min = -1;
      }
    } else {
      scope_max = 4; scope_min = -4;
    }
    cutlass::reference::host::TensorFillRandomUniform(view, seed, scope_max, scope_min, 0);
  }

  void run(const std::string& config_name,
           const BenchmarkOptions& options,
           const cutlass::KernelHardwareInfo& hw_info,
           std::vector<BenchmarkResult>& results) {

    BenchmarkResult result_template;
    result_template.config_name = config_name;
    result_template.precision = "nvfp4";
    result_template.mma_tile_shape = shape_to_string<typename Config::MmaTileShape>();
    result_template.cluster_shape = shape_to_string<typename Config::ClusterShape>();
    result_template.mainloop_schedule = mainloop_schedule_name<typename Config::MainloopSchedule>();
    result_template.epilogue_schedule = epilogue_schedule_name<typename Config::EpilogueSchedule>();
    result_template.stage_count = stage_count_name<typename Config::StageCountTag>();
    result_template.tile_scheduler = tile_scheduler_name<typename Config::TileScheduler>();

    for (const auto& shape : options.shapes) {
      int M = shape.M, N = shape.N, K = shape.K;

      auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
      auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
      auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
      auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

      auto layout_A = cute::make_layout(cute::make_shape(M, K, 1), stride_A);
      auto layout_B = cute::make_layout(cute::make_shape(N, K, 1), stride_B);
      auto layout_C = cute::make_layout(cute::make_shape(M, N, 1), stride_C);
      auto layout_D = cute::make_layout(cute::make_shape(M, N, 1), stride_D);

      auto layout_SFA = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
      auto layout_SFB = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

      block_A.reset(cutlass::make_Coord(size(layout_A)));
      block_B.reset(cutlass::make_Coord(size(layout_B)));
      block_C.reset(cutlass::make_Coord(size(layout_C)));
      block_D.reset(cutlass::make_Coord(size(layout_D)));
      block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
      block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));

      initialize_block(block_A.host_view(), 2021);
      initialize_block(block_B.host_view(), 2022);
      initialize_block(block_C.host_view(), 2023);
      initialize_block(block_SFA.host_view(), 2024);
      initialize_block(block_SFB.host_view(), 2025);

      block_A.sync_device();
      block_B.sync_device();
      block_C.sync_device();
      block_SFA.sync_device();
      block_SFB.sync_device();

      typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        { // Mainloop arguments
          block_A.device_data(), stride_A,
          block_B.device_data(), stride_B,
          block_SFA.device_data(), layout_SFA,
          block_SFB.device_data(), layout_SFB
        },
        { // Epilogue arguments
          {options.alpha, options.beta},
          block_C.device_data(), stride_C,
          block_D.device_data(), stride_D
        },
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

  if (props.major != 10 || (props.minor != 0 && props.minor != 1 && props.minor != 3)) {
    std::cerr << "This example requires a GPU with compute capability 100, 101, or 103." << std::endl;
    return 0;
  }

  // NVFP4 requires K >= 256 (MMA tile K=256)
  BenchmarkOptions options;
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
    std::cout << "=== NVFP4 BlockScaled GEMM Benchmark (A/B=nv_float4_t<e2m1>, C/D=float) ===" << std::endl;
    std::cout << "  SM count: " << hw_info.sm_count << std::endl;
    std::cout << "  Iterations: " << options.iterations << ", Warmup: " << options.warmup << std::endl;
    std::cout << std::endl;
  }

  NVFP4BenchRunner<NVFP4_1sm_128x128_c1x1>{}.run("1sm_128x128_c1x1", options, hw_info, results);
  NVFP4BenchRunner<NVFP4_1sm_128x128_c1x2>{}.run("1sm_128x128_c1x2", options, hw_info, results);
  NVFP4BenchRunner<NVFP4_2sm_256x128_c2x1>{}.run("2sm_256x128_c2x1", options, hw_info, results);
  NVFP4BenchRunner<NVFP4_2sm_256x128_c2x2>{}.run("2sm_256x128_c2x2", options, hw_info, results);
  NVFP4BenchRunner<NVFP4_2sm_256x128_c4x2>{}.run("2sm_256x128_c4x2", options, hw_info, results);

  NVFP4BenchRunner<NVFP4_1sm_128x128_c1x1_nvf4>{}.run("1sm_128x128_c1x1_nvf4", options, hw_info, results);
  NVFP4BenchRunner<NVFP4_2sm_256x128_c2x2_nvf4>{}.run("2sm_256x128_c2x2_nvf4", options, hw_info, results);
  NVFP4BenchRunner<NVFP4_2sm_256x128_c2x2_static>{}.run("2sm_256x128_c2x2_static", options, hw_info, results);
  NVFP4BenchRunner<NVFP4_1sm_128x128_c1x1_tma>{}.run("1sm_128x128_c1x1_tma", options, hw_info, results);
  NVFP4BenchRunner<NVFP4_2sm_256x128_c2x2_tma>{}.run("2sm_256x128_c2x2_tma", options, hw_info, results);
  NVFP4BenchRunner<NVFP4_1sm_128x128_c1x1_nosmem>{}.run("1sm_128x128_c1x1_nosmem", options, hw_info, results);
  NVFP4BenchRunner<NVFP4_2sm_256x128_c2x2_nosmem>{}.run("2sm_256x128_c2x2_nosmem", options, hw_info, results);
  // Manual stage count variants
  NVFP4BenchRunner<NVFP4_1sm_128x128_c1x1_nvf4_s2>{}.run("1sm_128x128_c1x1_nvf4_s2", options, hw_info, results);
  NVFP4BenchRunner<NVFP4_1sm_128x128_c1x1_nvf4_s3>{}.run("1sm_128x128_c1x1_nvf4_s3", options, hw_info, results);
  NVFP4BenchRunner<NVFP4_1sm_128x128_c1x1_nvf4_s4>{}.run("1sm_128x128_c1x1_nvf4_s4", options, hw_info, results);
  NVFP4BenchRunner<NVFP4_2sm_256x128_c2x2_nvf4_s2>{}.run("2sm_256x128_c2x2_nvf4_s2", options, hw_info, results);
  NVFP4BenchRunner<NVFP4_2sm_256x128_c2x2_nvf4_s3>{}.run("2sm_256x128_c2x2_nvf4_s3", options, hw_info, results);
  NVFP4BenchRunner<NVFP4_2sm_256x128_c2x2_nvf4_s4>{}.run("2sm_256x128_c2x2_nvf4_s4", options, hw_info, results);
  NVFP4BenchRunner<NVFP4_1sm_128x128_c1x1_tma_s3>{}.run("1sm_128x128_c1x1_tma_s3", options, hw_info, results);
  NVFP4BenchRunner<NVFP4_2sm_256x128_c2x2_tma_s3>{}.run("2sm_256x128_c2x2_tma_s3", options, hw_info, results);

  // Secondary shapes - run with representative configs only
  options.shapes = {
    {4608, 3072, 3072}, {4608, 9216, 3072}, {4608, 12288, 3072}, {4608, 3072, 12288},
    {16384, 3072, 3072}, {16384, 9216, 3072}, {16384, 12288, 3072}, {16384, 3072, 12288},
    {128, 3072, 3072}, {128, 9216, 3072}, {128, 12288, 3072}, {128, 3072, 12288}
  };
  NVFP4BenchRunner<NVFP4_1sm_128x128_c1x1>{}.run("1sm_128x128_c1x1", options, hw_info, results);
  NVFP4BenchRunner<NVFP4_2sm_256x128_c2x2>{}.run("2sm_256x128_c2x2", options, hw_info, results);
  NVFP4BenchRunner<NVFP4_1sm_128x128_c1x1_nvf4>{}.run("1sm_128x128_c1x1_nvf4", options, hw_info, results);
  NVFP4BenchRunner<NVFP4_2sm_256x128_c2x2_nvf4>{}.run("2sm_256x128_c2x2_nvf4", options, hw_info, results);

  if (!options.csv) {
    std::cout << "\nDone. " << results.size() << " benchmarks completed." << std::endl;
  }

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
