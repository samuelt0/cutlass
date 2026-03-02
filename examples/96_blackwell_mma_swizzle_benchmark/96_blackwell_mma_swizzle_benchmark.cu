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

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// tcgen05.mma Swizzle Pattern Benchmark for Blackwell (SM100)
//
// Investigates how different SMEM layouts (swizzle patterns) and descriptor
// configurations affect tcgen05.mma performance.
//
// Correctness: validates all 4 swizzle modes x 3 K_PER_STAGE values (12 tests).
// Performance: sweeps 3 descriptor configs (K_PER_STAGE=64/128/256), 4 swizzle
// modes, and 3 wait patterns for 36 rows of benchmark output with both
// clock64() SM-local cycles and cudaEvent wall-clock timing.
//
// Config: atype=btype=bf16, accumulator=fp32, MMA shape 128x256x16, CTA_group=1, dense.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cutlass/bfloat16.h>
#include <cutlass/arch/barrier.h>

#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/copy_sm90_desc.hpp>        // initialize_barrier, wait_barrier
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/swizzle.hpp>

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int MMA_M = 128;
static constexpr int MMA_N = 256;
static constexpr int MMA_K = 16;            // bf16 K-dim per MMA instruction

// Swizzle mode enum
enum class SwizzleMode { SW_NONE = 0, SW_32B = 1, SW_64B = 2, SW_128B = 3 };

static const char* swizzle_mode_name(SwizzleMode mode) {
  switch (mode) {
    case SwizzleMode::SW_NONE: return "SW_NONE";
    case SwizzleMode::SW_32B:  return "SW_32B";
    case SwizzleMode::SW_64B:  return "SW_64B";
    case SwizzleMode::SW_128B: return "SW_128B";
  }
  return "UNKNOWN";
}

// UMMA descriptor layout_type values
static uint8_t swizzle_mode_to_layout_type(SwizzleMode mode) {
  switch (mode) {
    case SwizzleMode::SW_NONE: return 0;
    case SwizzleMode::SW_128B: return 2;
    case SwizzleMode::SW_64B:  return 4;
    case SwizzleMode::SW_32B:  return 6;
  }
  return 0;
}

// Wait pattern enum
enum class WaitPattern { COMMIT_WAIT_EACH = 0, COMMIT_EACH_WAIT_END = 1, COMMIT_WAIT_END = 2 };

///////////////////////////////////////////////////////////////////////////////////////////////////
// Host swizzle reference implementation (unused — kept for documentation)
//
// The device-side swizzle_store() function below is used instead, because CuTe's
// Swizzle<B,4,3> operates on absolute SMEM byte addresses which are unknown at host time.
// This host implementation was the original approach; it correctly implements the same
// swizzle logic but requires knowing the SMEM base address, making it impractical for
// host-side pre-swizzling.
//
// CuTe swizzle atoms (bits): SW128=Swizzle<3,4,3>, SW64=Swizzle<2,4,3>, SW32=Swizzle<1,4,3>
// For bf16 (16 bits/elem), atom sizes: (8,64), (8,32), (8,16) respectively.
///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////
// Step 2: TMEM-to-linear reindex for tcgen05.ld output
///////////////////////////////////////////////////////////////////////////////////////////////////

// tcgen05.ld.sync.aligned.32x32b.x1 loads from TMEM:
//   Thread t (0..31) reads data-path (base_dp + t), gets 1 x fp32 value per call
//
// The 128x256 accumulator is organized as:
//   4 DP-groups x 256 columns
//   Thread t owns M-rows: {t, t+32, t+64, t+96}
//   Each thread stores 4 * 256 = 1024 fp32 values
//
// Register layout per thread: reg[dp_group * 256 + col] = C[dp_group*32 + t][col]

void tmem_to_linear(
    float const* thread_data,  // [128 threads][1024 values] - flat array
    float* matrix,             // [128][256] output
    int num_threads)
{
  for (int t = 0; t < num_threads; ++t) {
    for (int dp_group = 0; dp_group < 4; ++dp_group) {
      int m = dp_group * 32 + (t % 32);
      for (int col = 0; col < MMA_N; ++col) {
        matrix[m * MMA_N + col] = thread_data[t * 1024 + dp_group * MMA_N + col];
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Step 3 & 4: Benchmark Kernel
///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Shared memory structure: dynamically sized
// Layout: [stage 0 A][pad][stage 0 B][pad][stage 1 A][pad][stage 1 B][pad]...
// Each tile is 1KB-aligned

// Build SMEM descriptor for one MMA K-block
static __device__ uint64_t make_smem_desc(
    uint32_t smem_addr,      // SMEM address of the start of this k-block's data
    uint8_t  layout_type,    // Swizzle layout type
    uint16_t lbo,            // Leading byte offset in uint128_t units
    uint16_t sbo)            // Stride byte offset in uint128_t units
{
  UMMA::SmemDescriptor desc;
  desc.desc_ = 0;
  desc.start_address_ = static_cast<uint16_t>(smem_addr >> 4);
  desc.leading_byte_offset_ = lbo;
  desc.stride_byte_offset_ = sbo;
  desc.layout_type_ = layout_type;
  desc.version_ = 1;       // Blackwell
  desc.base_offset_ = 0;
  desc.lbo_mode_ = 0;      // Legacy mode
  return desc.desc_;
}

// Device-side helper: copy one element from GMEM to SMEM with swizzle applied.
// For swizzled modes (mask != 0): atom-based layout + XOR swizzle on byte addresses.
//   Atom dimensions for K-major bf16: SW_128B=(8,64), SW_64B=(8,32), SW_32B=(8,16).
//   Atoms are tiled: K-direction first within each 8-row MN-group, then MN-groups.
// For INTERLEAVE (mask == 0): row-interleaved reordering, no XOR.
static __device__ void swizzle_store(
    cutlass::bfloat16_t* smem_tile,  // SMEM tile base pointer
    int m, int k, int K,             // logical position and K dimension
    cutlass::bfloat16_t val,         // value to store
    uint32_t swizzle_mask)           // 0x380 for SW128, 0x180 for SW64, 0x80 for SW32, 0 for INTERLEAVE
{
  if (swizzle_mask != 0) {
    // Determine atom width from swizzle mask. The hardware expects data organized
    // in atoms whose row stride matches the layout_type:
    //   SW_128B: 128-byte rows → atom (8, 64) bf16
    //   SW_64B:   64-byte rows → atom (8, 32) bf16
    //   SW_32B:   32-byte rows → atom (8, 16) bf16
    int atom_cols = (swizzle_mask == 0x380) ? 64 :
                    (swizzle_mask == 0x180) ? 32 : 16;

    int mn_group    = m / 8;
    int local_m     = m % 8;
    int atom_k      = k / atom_cols;
    int local_k     = k % atom_cols;
    int num_atoms_k = K / atom_cols;
    int atom_size_bytes = 8 * atom_cols * 2;

    uint32_t tile_base = cute::cast_smem_ptr_to_uint(smem_tile);
    uint32_t atom_base = tile_base +
        static_cast<uint32_t>((mn_group * num_atoms_k + atom_k) * atom_size_bytes);
    uint32_t byte_in_atom = static_cast<uint32_t>(local_m * atom_cols * 2 + local_k * 2);
    uint32_t addr = atom_base + byte_in_atom;
    uint32_t swizzled = addr ^ ((addr & swizzle_mask) >> 3);
    smem_tile[(swizzled - tile_base) / 2] = val;
  } else {
    // INTERLEAVE: ((8,n),2):((1,SBO),LBO) — rows within 8-row group are contiguous,
    // K-chunks of 8 bf16 elements are interleaved.
    int mn_group  = m / 8;
    int local_row = m % 8;
    int k_chunk   = k / 8;
    int k_in_chunk = k % 8;
    int dst_idx = mn_group * (K * 8) + k_chunk * 64 + local_row * 8 + k_in_chunk;
    smem_tile[dst_idx] = val;
  }
}

// Kernel template parameterized by number of stages, wait pattern, and K per stage
template <int NUM_STAGES, int WAIT_PATTERN, int K_PER_STAGE_T>
__global__ void
__launch_bounds__(128, 1)
mma_swizzle_benchmark_kernel(
    cutlass::bfloat16_t const* __restrict__ gA,  // Linear K-major A tiles [NUM_STAGES][MMA_M * K_PER_STAGE_T]
    cutlass::bfloat16_t const* __restrict__ gB,  // Linear K-major B tiles [NUM_STAGES][MMA_N * K_PER_STAGE_T]
    float*    __restrict__ gD,                    // Output [128][256]
    int64_t*  __restrict__ gCycles,               // Output: MMA cycle count
    int64_t*  __restrict__ gFillCycles,           // Output: SMEM fill cycle count
    uint8_t   layout_type,                        // Swizzle layout type for descriptors
    uint16_t  lbo,                                // Leading byte offset (uint128_t units)
    uint16_t  sbo,                                // Stride byte offset (uint128_t units)
    uint32_t  swizzle_mask,                       // Byte-address swizzle mask (0 for INTERLEAVE)
    int       k_iters)                            // Number of mainloop iterations
{
  // Derive tile dimensions from template parameters
  constexpr int K_BLOCKS = K_PER_STAGE_T / MMA_K;
  constexpr int A_TILE_ELEMS = MMA_M * K_PER_STAGE_T;
  constexpr int B_TILE_ELEMS = MMA_N * K_PER_STAGE_T;
  constexpr int A_TILE_BYTES = MMA_M * K_PER_STAGE_T * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int B_TILE_BYTES = MMA_N * K_PER_STAGE_T * static_cast<int>(sizeof(cutlass::bfloat16_t));

  // Prologue: set up SMEM, TMEM, barriers, descriptors

  extern __shared__ char smem_buf[];

  // Stage layout in SMEM (1KB aligned)
  constexpr int A_ALLOC = (A_TILE_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC = (B_TILE_BYTES + 1023) & ~1023;
  constexpr int STAGE_ALLOC = A_ALLOC + B_ALLOC;

  // Pointers to each stage's A and B tiles
  cutlass::bfloat16_t* sA[NUM_STAGES];
  cutlass::bfloat16_t* sB[NUM_STAGES];
  for (int s = 0; s < NUM_STAGES; ++s) {
    sA[s] = reinterpret_cast<cutlass::bfloat16_t*>(smem_buf + s * STAGE_ALLOC);
    sB[s] = reinterpret_cast<cutlass::bfloat16_t*>(smem_buf + s * STAGE_ALLOC + A_ALLOC);
  }

  // Barrier and TMEM base in SMEM (after all stages)
  constexpr int META_OFFSET = NUM_STAGES * STAGE_ALLOC;
  uint64_t* mma_barrier = reinterpret_cast<uint64_t*>(smem_buf + META_OFFSET);
  uint32_t* tmem_base   = reinterpret_cast<uint32_t*>(smem_buf + META_OFFSET + 16);

  // Load tiles from GMEM to SMEM with device-side swizzle/interleave.
  // gA/gB contain LINEAR K-major data; the swizzle is applied here using
  // the actual SMEM byte address (required because Swizzle<B,4,3> operates
  // on the absolute address, not just the offset within the tile).
  int64_t t_fill_start = 0, t_fill_end = 0;
  if (threadIdx.x == 0) {
    t_fill_start = clock64();
  }
  for (int s = 0; s < NUM_STAGES; ++s) {
    for (int i = threadIdx.x; i < A_TILE_ELEMS; i += blockDim.x) {
      int m = i / K_PER_STAGE_T;
      int k = i % K_PER_STAGE_T;
      swizzle_store(sA[s], m, k, K_PER_STAGE_T,
                    gA[s * A_TILE_ELEMS + i], swizzle_mask);
    }
    for (int i = threadIdx.x; i < B_TILE_ELEMS; i += blockDim.x) {
      int n = i / K_PER_STAGE_T;
      int k = i % K_PER_STAGE_T;
      swizzle_store(sB[s], n, k, K_PER_STAGE_T,
                    gB[s * B_TILE_ELEMS + i], swizzle_mask);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    t_fill_end = clock64();
  }

  // TMEM allocation (warp 0 only)
  uint32_t elect_one_thr  = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  if (elect_one_warp) {
    tmem_allocator.allocate(512, tmem_base);
  }
  __syncthreads();
  uint32_t tmem_addr = *tmem_base;

  // Build instruction descriptor for bf16 x bf16 -> fp32, 128x256, K-major, K-major
  uint64_t idescE = UMMA::make_runtime_instr_desc<
      cutlass::bfloat16_t, cutlass::bfloat16_t, float,
      MMA_M, MMA_N, UMMA::Major::K, UMMA::Major::K>();

  // Build SMEM descriptors for each stage and K-block
  uint64_t desc_a[NUM_STAGES][K_BLOCKS];
  uint64_t desc_b[NUM_STAGES][K_BLOCKS];

  for (int s = 0; s < NUM_STAGES; ++s) {
    uint32_t base_a = cute::cast_smem_ptr_to_uint(sA[s]);
    uint32_t base_b = cute::cast_smem_ptr_to_uint(sB[s]);

    for (int kb = 0; kb < K_BLOCKS; ++kb) {
      uint32_t offset_bytes;
      if (swizzle_mask != 0) {
        // Swizzled modes: data is organized in atoms of (8, atom_cols) bf16.
        // Within an atom, k-blocks are at stride 2 uint128_t = 32 bytes.
        // Crossing atom boundaries requires jumping to the next atom base.
        int atom_cols = (swizzle_mask == 0x380) ? 64 :
                        (swizzle_mask == 0x180) ? 32 : 16;
        int kblocks_per_atom = atom_cols / MMA_K;
        int atom_size_bytes  = 8 * atom_cols * 2;
        int atom_idx    = kb / kblocks_per_atom;
        int kb_in_atom  = kb % kblocks_per_atom;
        offset_bytes = atom_idx * atom_size_bytes + kb_in_atom * 32;
      } else {
        // INTERLEAVE: each k-block advances by 2 * LBO uint128_t = 32 * lbo bytes
        offset_bytes = kb * 32 * lbo;
      }
      desc_a[s][kb] = make_smem_desc(base_a + offset_bytes, layout_type, lbo, sbo);
      desc_b[s][kb] = make_smem_desc(base_b + offset_bytes, layout_type, lbo, sbo);
    }
  }

  // Initialize mbarrier with wait-pattern-dependent arrival count
  //   Pattern (a) COMMIT_WAIT_EACH: 1 arrival per phase (arrive+wait toggles each stage)
  //   Pattern (b) COMMIT_EACH_WAIT_END: all arrives aggregate into one phase flip
  //   Pattern (c) COMMIT_WAIT_END: single arrive at end
  if (elect_one_warp && elect_one_thr) {
    uint32_t arrive_count;
    if constexpr (WAIT_PATTERN == 0) {
      arrive_count = 1;
    }
    else if constexpr (WAIT_PATTERN == 1) {
      arrive_count = NUM_STAGES * k_iters;
    }
    else {
      arrive_count = 1;
    }
    cute::initialize_barrier(*mma_barrier, arrive_count);
  }
  int phase_bit = 0;

  // Mainloop — timed with clock64()
  __syncthreads();

  int64_t t_start = 0, t_end = 0;

  if (elect_one_warp) {
    t_start = clock64();

    uint32_t scaleC = 0;  // First MMA clears accumulator

    for (int i = 0; i < k_iters; ++i) {
      #pragma unroll
      for (int s = 0; s < NUM_STAGES; ++s) {
        #pragma unroll
        for (int kb = 0; kb < K_BLOCKS; ++kb) {
          SM100_MMA_F16BF16_SS<
            cutlass::bfloat16_t, cutlass::bfloat16_t, float,
            MMA_M, MMA_N, UMMA::Major::K, UMMA::Major::K>::fma(
              desc_a[s][kb], desc_b[s][kb], tmem_addr, scaleC, idescE);
          scaleC = 1;  // Accumulate after first MMA
        }

        // Wait pattern handling after each stage
        // Note: SMEM is static (never reloaded), so commit/wait is only needed
        // to measure synchronization overhead — not for SMEM protection.
        if constexpr (WAIT_PATTERN == 0) {
          // Pattern (a): commit + wait after each stage (most conservative)
          cutlass::arch::umma_arrive(mma_barrier);
          cute::wait_barrier(*mma_barrier, phase_bit);
          phase_bit ^= 1;
        }
        else if constexpr (WAIT_PATTERN == 1) {
          // Pattern (b): commit after each stage, no wait in loop
          cutlass::arch::umma_arrive(mma_barrier);
        }
        // Pattern (c): nothing inside loop
      }
    }

    // Post-loop: ensure all MMA operations are complete before reading
    if constexpr (WAIT_PATTERN == 1) {
      // Pattern (b): all commits issued in loop, wait for all to complete
      cute::wait_barrier(*mma_barrier, phase_bit);
    }
    else if constexpr (WAIT_PATTERN == 2) {
      // Pattern (c): single commit + wait after entire loop
      cutlass::arch::umma_arrive(mma_barrier);
      cute::wait_barrier(*mma_barrier, phase_bit);
    }

    t_end = clock64();
  }
  __syncthreads();

  // Epilogue: load TMEM -> registers -> GMEM

  // Fence TMEM stores before reading
  cutlass::arch::fence_view_async_tmem_store();
  cutlass::arch::fence_view_async_tmem_load();

  int tid = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;

  constexpr uint32_t TMEM_DP_STRIDE = (1u << 16);

  float reg_buf[MMA_N];  // 256 fp32 values per thread (one per column)

  // Batched TMEM load: 8 columns per call (32 calls vs 256 with 1x variant)
  {
    uint32_t dp_base = tmem_addr + warp_id * 32 * TMEM_DP_STRIDE;
    for (int col = 0; col < MMA_N; col += 8) {
      uint32_t src_addr = dp_base + col;
      SM100_TMEM_LOAD_32dp32b8x::copy(
          src_addr,
          reinterpret_cast<uint32_t&>(reg_buf[col + 0]),
          reinterpret_cast<uint32_t&>(reg_buf[col + 1]),
          reinterpret_cast<uint32_t&>(reg_buf[col + 2]),
          reinterpret_cast<uint32_t&>(reg_buf[col + 3]),
          reinterpret_cast<uint32_t&>(reg_buf[col + 4]),
          reinterpret_cast<uint32_t&>(reg_buf[col + 5]),
          reinterpret_cast<uint32_t&>(reg_buf[col + 6]),
          reinterpret_cast<uint32_t&>(reg_buf[col + 7]));
    }
  }

  // Store registers to GMEM
  // Thread t in warp w owns M-row = w*32 + t, all N columns
  {
    int m = warp_id * 32 + tid;
    for (int col = 0; col < MMA_N; ++col) {
      gD[m * MMA_N + col] = reg_buf[col];
    }
  }

  // Write cycle counts (thread 0 only)
  if (threadIdx.x == 0) {
    gCycles[0]     = t_end - t_start;
    gFillCycles[0] = t_fill_end - t_fill_start;
  }

  __syncthreads();

  // Free TMEM
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(*tmem_base, 512);
  }
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

///////////////////////////////////////////////////////////////////////////////////////////////////
// Host-side helpers
///////////////////////////////////////////////////////////////////////////////////////////////////

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t err = (call);                                                    \
    if (err != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                           \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

// CPU reference GEMM: C[m][n] = sum_k(A[m][k] * B[n][k])  (B is NxK, K-major)
void reference_gemm_bf16(
    cutlass::bfloat16_t const* A,  // [M][K] K-major
    cutlass::bfloat16_t const* B,  // [N][K] K-major
    float* C,                      // [M][N]
    int M, int N, int K)
{
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += float(A[m * K + k]) * float(B[n * K + k]);
      }
      C[m * N + n] = acc;
    }
  }
}

// Compute the byte-address swizzle mask for Swizzle<B,4,3>:
//   mask = ((1 << B) - 1) << 7, matching yyy_msk of Swizzle<B,4,3>
static uint32_t swizzle_mode_to_mask(SwizzleMode mode) {
  switch (mode) {
    case SwizzleMode::SW_128B: return 0x380;  // bits [7,8,9]
    case SwizzleMode::SW_64B:  return 0x180;  // bits [7,8]
    case SwizzleMode::SW_32B:  return 0x080;  // bit [7]
    case SwizzleMode::SW_NONE: return 0;      // no XOR
  }
  return 0;
}

// Compute descriptor parameters for a given swizzle mode and K per stage
void compute_desc_params(SwizzleMode mode, int k_per_stage,
                         uint8_t& layout_type, uint16_t& lbo, uint16_t& sbo) {
  layout_type = swizzle_mode_to_layout_type(mode);

  // For K-major bf16:
  // SBO = stride to next group of 8 MN-rows in uint128_t units
  //     = 8 rows * k_per_stage * sizeof(bf16) / 16 = k_per_stage
  // LBO = leading byte offset (stride for one K step in uint128_t units)
  //   Swizzled: 1 (contiguous in K)
  //   Interleave: 8 (8 rows interleaved)
  switch (mode) {
    case SwizzleMode::SW_128B:
    case SwizzleMode::SW_64B:
    case SwizzleMode::SW_32B:
      lbo = 1;
      sbo = 8 * (k_per_stage * 2 / 16);  // = k_per_stage
      break;
    case SwizzleMode::SW_NONE:
      lbo = 8;                             // 8 rows interleaved
      sbo = (k_per_stage / 8) * 8;        // = k_per_stage
      break;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Step 5: Correctness Test
///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

struct BenchResult {
  int64_t mma_cycles;
  int64_t fill_cycles;
  float wall_clock_us;
};

template <int K_PER_STAGE_T>
bool run_correctness_test_k(SwizzleMode mode) {
  printf("=== Correctness test: %s, K=%d ===\n", swizzle_mode_name(mode), K_PER_STAGE_T);

  constexpr int K = K_PER_STAGE_T;
  int M = MMA_M, N = MMA_N;
  int k_iters = 1;

  // Generate random A(128,K) and B(256,K) in bf16 — LINEAR K-major
  std::vector<cutlass::bfloat16_t> h_A(M * K), h_B(N * K);
  srand(42);
  for (int i = 0; i < M * K; ++i) {
    h_A[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));
  }
  for (int i = 0; i < N * K; ++i) {
    h_B[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));
  }

  // CPU reference
  std::vector<float> h_ref(M * N);
  reference_gemm_bf16(h_A.data(), h_B.data(), h_ref.data(), M, N, K);

  // Allocate device memory
  cutlass::bfloat16_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(cutlass::bfloat16_t)));
  CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(cutlass::bfloat16_t)));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  // Send LINEAR K-major data — swizzle is applied on-device using actual SMEM addresses
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, M * N * sizeof(float)));

  // Compute descriptor params
  uint8_t layout_type;
  uint16_t lbo, sbo;
  compute_desc_params(mode, K, layout_type, lbo, sbo);
  uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

  // Launch kernel: 1 stage, wait pattern 0 (commit+wait each)
  constexpr int A_BYTES = MMA_M * K_PER_STAGE_T * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int B_BYTES = MMA_N * K_PER_STAGE_T * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int A_ALLOC = (A_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC = (B_BYTES + 1023) & ~1023;
  constexpr int SMEM_SIZE = A_ALLOC + B_ALLOC + 256;
  auto kernel = mma_swizzle_benchmark_kernel<1, 0, K_PER_STAGE_T>;

  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

  dim3 grid(1);
  dim3 block(128);

  kernel<<<grid, block, SMEM_SIZE>>>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Read back results
  std::vector<float> h_D(M * N);
  CUDA_CHECK(cudaMemcpy(h_D.data(), d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // Compare
  float max_err = 0.0f;
  float max_rel_err = 0.0f;
  int errors = 0;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float ref = h_ref[m * N + n];
      float got = h_D[m * N + n];
      float err = std::fabs(ref - got);
      float rel = (std::fabs(ref) > 1e-6f) ? err / std::fabs(ref) : err;
      max_err = std::max(max_err, err);
      max_rel_err = std::max(max_rel_err, rel);
      if (rel > 0.05f && err > 0.5f) {
        if (errors < 10) {
          printf("  MISMATCH at [%d][%d]: ref=%.4f got=%.4f err=%.4f\n", m, n, ref, got, err);
        }
        errors++;
      }
    }
  }

  printf("  Max absolute error: %.6f\n", max_err);
  printf("  Max relative error: %.6f\n", max_rel_err);
  printf("  Mismatches: %d / %d\n", errors, M * N);

  bool pass = (errors == 0);
  printf("  Result: %s\n\n", pass ? "PASS" : "FAIL");

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_cycles));
  CUDA_CHECK(cudaFree(d_fill_cycles));

  return pass;
}

bool run_correctness_test(SwizzleMode mode, int k_per_stage) {
  switch (k_per_stage) {
    case 64:  return run_correctness_test_k<64>(mode);
    case 128: return run_correctness_test_k<128>(mode);
    case 256: return run_correctness_test_k<256>(mode);
    default:
      printf("Unsupported K_PER_STAGE=%d for correctness test\n", k_per_stage);
      return false;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Step 6: Performance Benchmark
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int NUM_STAGES, int WAIT_PATTERN, int K_PER_STAGE_T>
BenchResult run_benchmark_config(
    cutlass::bfloat16_t* d_A,
    cutlass::bfloat16_t* d_B,
    float* d_D,
    int64_t* d_cycles,
    int64_t* d_fill_cycles,
    uint8_t layout_type,
    uint16_t lbo,
    uint16_t sbo,
    uint32_t swizzle_mask,
    int k_iters)
{
  constexpr int A_BYTES = MMA_M * K_PER_STAGE_T * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int B_BYTES = MMA_N * K_PER_STAGE_T * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int A_ALLOC = (A_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC = (B_BYTES + 1023) & ~1023;
  constexpr int STAGE_ALLOC = A_ALLOC + B_ALLOC;
  int smem_size = NUM_STAGES * STAGE_ALLOC + 256;

  auto kernel = mma_swizzle_benchmark_kernel<NUM_STAGES, WAIT_PATTERN, K_PER_STAGE_T>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  dim3 grid(1);
  dim3 block(128);

  // Warmup
  kernel<<<grid, block, smem_size>>>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark run with cudaEvent wall-clock timing
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  CUDA_CHECK(cudaEventRecord(ev_start));
  kernel<<<grid, block, smem_size>>>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
  CUDA_CHECK(cudaEventRecord(ev_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));

  BenchResult result;
  CUDA_CHECK(cudaMemcpy(&result.mma_cycles, d_cycles, sizeof(int64_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&result.fill_cycles, d_fill_cycles, sizeof(int64_t), cudaMemcpyDeviceToHost));
  result.wall_clock_us = elapsed_ms * 1000.0f;
  return result;
}

struct BenchConfig {
  int k_per_stage;
  int num_stages;
  const char* label;
};

void run_performance_sweep(int clock_rate_khz, int k_iters) {
  printf("=== Descriptor Configuration Sweep ===\n");
  printf("  MMA shape: %dx%dx%d (bf16), all configs: K_PER_STAGE x stages = 256\n",
         MMA_M, MMA_N, MMA_K);
  printf("  k_iters: %d, total MMAs per row: 16 x %d = %d\n\n",
         k_iters, k_iters, 16 * k_iters);

  BenchConfig configs[] = {
    {  64, 4, "K=64,S=4"  },
    { 128, 2, "K=128,S=2" },
    { 256, 1, "K=256,S=1" },
  };

  SwizzleMode modes[] = {
    SwizzleMode::SW_NONE, SwizzleMode::SW_32B, SwizzleMode::SW_64B, SwizzleMode::SW_128B
  };

  printf("%5s  %4s  %-8s  %-24s  %10s  %8s  %11s  %9s  %10s\n",
         "K/Stg", "Stgs", "Swizzle", "Wait Pattern", "Cycles", "Cyc/MMA", "Latency(us)", "Wall(us)", "Fill_Cyc");
  printf("%5s  %4s  %-8s  %-24s  %10s  %8s  %11s  %9s  %10s\n",
         "-----", "----", "--------", "------------------------", "----------", "--------", "-----------", "---------", "----------");

  for (const auto& cfg : configs) {
    int M = MMA_M, N = MMA_N;

    // Allocate device memory for this config's tile sizes
    int total_A_elems = cfg.num_stages * M * cfg.k_per_stage;
    int total_B_elems = cfg.num_stages * N * cfg.k_per_stage;

    cutlass::bfloat16_t *d_A, *d_B;
    float *d_D;
    int64_t *d_cycles;

    int64_t *d_fill_cycles;

    CUDA_CHECK(cudaMalloc(&d_A, total_A_elems * sizeof(cutlass::bfloat16_t)));
    CUDA_CHECK(cudaMalloc(&d_B, total_B_elems * sizeof(cutlass::bfloat16_t)));
    CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

    // Generate LINEAR K-major data (swizzle applied on-device)
    std::vector<cutlass::bfloat16_t> h_A(total_A_elems), h_B(total_B_elems);
    srand(123);
    for (int i = 0; i < total_A_elems; ++i) h_A[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));
    for (int i = 0; i < total_B_elems; ++i) h_B[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), total_A_elems * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), total_B_elems * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));

    for (SwizzleMode mode : modes) {
      uint8_t layout_type;
      uint16_t lbo, sbo;
      compute_desc_params(mode, cfg.k_per_stage, layout_type, lbo, sbo);
      uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

      int k_blocks = cfg.k_per_stage / MMA_K;
      int total_mmas = cfg.num_stages * k_blocks * k_iters;

      auto run_for_pattern = [&](int wp_id, const char* wp_name) {
        BenchResult result = {};
        int ns = cfg.num_stages;
        int kps = cfg.k_per_stage;

        #define DISPATCH_BENCH(NS, WP, KPS) \
          if (ns == NS && wp_id == WP && kps == KPS) { \
            result = run_benchmark_config<NS, WP, KPS>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters); \
          }

        DISPATCH_BENCH(4, 0, 64)  DISPATCH_BENCH(4, 1, 64)  DISPATCH_BENCH(4, 2, 64)
        DISPATCH_BENCH(2, 0, 128) DISPATCH_BENCH(2, 1, 128) DISPATCH_BENCH(2, 2, 128)
        DISPATCH_BENCH(1, 0, 256) DISPATCH_BENCH(1, 1, 256) DISPATCH_BENCH(1, 2, 256)
        #undef DISPATCH_BENCH

        double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

        printf("%5d  %4d  %-8s  %-24s  %10ld  %8.1f  %11.1f  %9.1f  %10ld\n",
               cfg.k_per_stage, cfg.num_stages,
               swizzle_mode_name(mode), wp_name,
               (long)result.mma_cycles,
               (double)result.mma_cycles / total_mmas,
               latency_us,
               (double)result.wall_clock_us,
               (long)result.fill_cycles);
      };

      run_for_pattern(0, "commit+wait each");
      run_for_pattern(1, "commit each, wait end");
      run_for_pattern(2, "commit+wait end");
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_D));
    CUDA_CHECK(cudaFree(d_cycles));
    CUDA_CHECK(cudaFree(d_fill_cycles));
  }

  printf("\n");
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

///////////////////////////////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  cudaDeviceProp props;
  int current_device_id;
  cudaGetDevice(&current_device_id);
  cudaGetDeviceProperties(&props, current_device_id);

  if (props.major != 10 || props.minor > 1) {
    std::cerr << "This example requires NVIDIA Blackwell (SM100a)." << std::endl;
    std::cerr << "  Found SM " << props.major << "." << props.minor << std::endl;
    return -1;
  }

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  // Parse args: single optional argument for k_iters
  int k_iters = 5000;

  if (argc >= 2) k_iters = atoi(argv[1]);

  int clock_rate_khz = 0;
  cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, current_device_id);

  printf("tcgen05.mma Swizzle Pattern Benchmark\n");
  printf("GPU: %s (SM %d.%d)\n", props.name, props.major, props.minor);
  printf("SMEM: %zu KB, Clock: %d MHz\n", props.sharedMemPerBlockOptin / 1024, clock_rate_khz / 1000);
  printf("\n");

  // Correctness tests: all 4 swizzle modes x 3 K_PER_STAGE values = 12 tests
  bool all_pass = true;
  int k_values[] = {64, 128, 256};
  for (int k : k_values) {
    all_pass &= run_correctness_test(SwizzleMode::SW_128B, k);
    all_pass &= run_correctness_test(SwizzleMode::SW_64B, k);
    all_pass &= run_correctness_test(SwizzleMode::SW_32B, k);
    all_pass &= run_correctness_test(SwizzleMode::SW_NONE, k);
  }

  if (!all_pass) {
    printf("CORRECTNESS TESTS FAILED. Aborting benchmark.\n");
    return 1;
  }

  printf("All correctness tests passed.\n\n");

  // Performance sweep — all 3 descriptor configs run automatically
  run_performance_sweep(clock_rate_khz, k_iters);

#else
  std::cout << "CUTLASS_ARCH_MMA_SM100_SUPPORTED must be enabled. Test is waived.\n" << std::endl;
#endif

  return 0;
}
