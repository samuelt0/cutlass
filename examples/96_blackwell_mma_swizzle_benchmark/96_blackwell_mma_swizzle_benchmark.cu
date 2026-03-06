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
// bf16 K-major: Correctness validates 4 swizzle modes x 3 K_PER_STAGE (12 tests).
//   Performance sweeps 3 configs x 4 swizzles x 3 wait patterns (36 rows).
//   Config: bf16, MMA 128x256x16, CTA_group=1, dense, A/B K-contiguous.
//
// bf16 MN-major: Same shape, but A is M-contiguous [K][M] and B is N-contiguous [K][N].
//   Correctness: 4 swizzle modes x 3 K_PER_STAGE (12 tests).
//   Performance: 3 configs x 4 swizzles x 3 wait patterns (36 rows).
//
// nvfp4: Correctness validates 4 swizzle modes (4 tests).
//   Performance sweeps 4 swizzles x 3 wait patterns (12 rows).
//   Config: mxf4nvf4 block_scale block16, MMA 128x128x64, K_PER_STAGE=256.
//   (MN-major not supported for E2M1/fp4 data type.)
//
// fp8 K-major: Correctness validates 4 swizzle modes (4 tests).
//   Performance sweeps 4 swizzles x 3 wait patterns (12 rows).
//   Config: f8f6f4 dense, MMA 128x256x32, K_PER_STAGE=256, A/B K-contiguous.
//
// fp8 MN-major: Same shape, but A is M-contiguous [K][M] and B is N-contiguous [K][N].
//   Correctness: 4 swizzle modes (4 tests).
//   Performance: 4 swizzles x 3 wait patterns (12 rows).
//
// bf16_2sm K-major: Correctness validates 4 swizzle modes (4 tests).
//   Performance sweeps 4 swizzles x 3 wait patterns (12 rows).
//   Config: bf16 cta_group::2, MMA 256x256x16, K_PER_STAGE=256, cluster(2,1,1).
//
// bf16_2sm MN-major: Same shape, A/B MN-contiguous.
//   Correctness: 4 swizzle modes (4 tests). Performance: 4 swizzles x 3 waits (12 rows).
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
#include <cutlass/float_subbyte.h>   // float_e2m1_t
#include <cutlass/float8.h>          // float_ue4m3_t
#include <cutlass/arch/barrier.h>

#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/copy_sm90_desc.hpp>        // initialize_barrier, wait_barrier
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/swizzle.hpp>
#include <cute/arch/cluster_sm90.hpp>       // cluster_sync, block_rank_in_cluster
#include <cutlass/cluster_launch.hpp>        // ClusterLaunchParams, launch_kernel_on_cluster

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int MMA_M = 128;
static constexpr int MMA_N = 256;
static constexpr int MMA_K = 16;            // bf16 K-dim per MMA instruction

// nvfp4 (mxf4nvf4) MMA parameters
static constexpr int FP4_MMA_M = 128;
static constexpr int FP4_MMA_N = 256;       // N=256, max supported (TMEM: 256 + 8 SF cols = 264 < 512)
static constexpr int FP4_MMA_K = 64;        // 256 bits / 4 bits per element
static constexpr int FP4_VS = 16;           // block16 scale vector size
static constexpr int FP4_K_PER_STAGE = 256; // fp4 elements per stage
static constexpr int FP4_K_BLOCKS = FP4_K_PER_STAGE / FP4_MMA_K;  // = 4
static constexpr int FP4_NUM_SF = FP4_K_PER_STAGE / FP4_VS;       // = 16 SFs per row
static constexpr int FP4_K_BYTES = FP4_K_PER_STAGE / 2;           // = 128 bytes per row

// fp8 (f8f6f4) dense MMA parameters
static constexpr int FP8_MMA_M = 128;
static constexpr int FP8_MMA_N = 256;
static constexpr int FP8_MMA_K = 32;         // 256 bits / 8 bits per element
static constexpr int FP8_K_PER_STAGE = 256;  // fp8 elements per stage
static constexpr int FP8_K_BLOCKS = FP8_K_PER_STAGE / FP8_MMA_K;  // = 8
static constexpr int FP8_K_BYTES = FP8_K_PER_STAGE;                // = 256 bytes per row

// bf16 2SM (cta_group::2) MMA parameters
static constexpr int BF16_2SM_MMA_M = 256;
static constexpr int BF16_2SM_MMA_N = 256;
static constexpr int BF16_2SM_MMA_K = 16;
static constexpr int BF16_2SM_M_PER_CTA = 128;   // each CTA holds 128 M-rows of A
static constexpr int BF16_2SM_N_PER_CTA = 128;   // each CTA holds 128 N-rows of B
static constexpr int BF16_2SM_K_PER_STAGE = 256;
static constexpr int BF16_2SM_K_BLOCKS = BF16_2SM_K_PER_STAGE / BF16_2SM_MMA_K;  // = 16

// fp8 2SM (cta_group::2) MMA parameters
static constexpr int FP8_2SM_MMA_M = 256;
static constexpr int FP8_2SM_MMA_N = 256;
static constexpr int FP8_2SM_MMA_K = 32;        // same as 1SM fp8
static constexpr int FP8_2SM_M_PER_CTA = 128;   // 256/2
static constexpr int FP8_2SM_N_PER_CTA = 128;   // 256/2
static constexpr int FP8_2SM_K_PER_STAGE = 256;
static constexpr int FP8_2SM_K_BLOCKS = 8;      // 256/32

// fp4 2SM (cta_group::2) MMA parameters
static constexpr int FP4_2SM_MMA_M = 256;
static constexpr int FP4_2SM_MMA_N = 256;       // N=256, max supported (TMEM: 256 + 8 SF cols = 264 < 512)
static constexpr int FP4_2SM_MMA_K = 64;        // same as 1SM fp4
static constexpr int FP4_2SM_M_PER_CTA = 128;   // 256/2
static constexpr int FP4_2SM_N_PER_CTA = 128;   // 256/2
static constexpr int FP4_2SM_K_PER_STAGE = 256;
static constexpr int FP4_2SM_K_BLOCKS = 4;      // 256/64

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

// Device-side helper: copy one byte from GMEM to SMEM with swizzle applied (for fp4 data).
// Same logic as swizzle_store() but operates on bytes (uint8_t) instead of bf16 elements.
// k_byte is the byte offset within the row, K_bytes is the total row width in bytes.
static __device__ void swizzle_store_byte(
    uint8_t* smem_tile,
    int m, int k_byte, int K_bytes,
    uint8_t val,
    uint32_t swizzle_mask)
{
  if (swizzle_mask != 0) {
    int atom_cols_bytes = (swizzle_mask == 0x380) ? 128 :
                          (swizzle_mask == 0x180) ? 64 : 32;

    int mn_group    = m / 8;
    int local_m     = m % 8;
    int atom_k      = k_byte / atom_cols_bytes;
    int local_k     = k_byte % atom_cols_bytes;
    int num_atoms_k = K_bytes / atom_cols_bytes;
    int atom_size_bytes = 8 * atom_cols_bytes;

    uint32_t tile_base = cute::cast_smem_ptr_to_uint(smem_tile);
    uint32_t atom_base = tile_base +
        static_cast<uint32_t>((mn_group * num_atoms_k + atom_k) * atom_size_bytes);
    uint32_t byte_in_atom = static_cast<uint32_t>(local_m * atom_cols_bytes + local_k);
    uint32_t addr = atom_base + byte_in_atom;
    uint32_t swizzled = addr ^ ((addr & swizzle_mask) >> 3);
    smem_tile[(swizzled - tile_base)] = val;
  } else {
    // INTERLEAVE: 8-row groups with 16-byte interleave chunks
    int mn_group   = m / 8;
    int local_row  = m % 8;
    int k_chunk    = k_byte / 16;
    int k_in_chunk = k_byte % 16;
    int dst_idx = mn_group * (K_bytes * 8) + k_chunk * 128 + local_row * 16 + k_in_chunk;
    smem_tile[dst_idx] = val;
  }
}

// Device-side helper: copy one bf16 element from GMEM to MN-major SMEM with swizzle.
// For MN-major: atoms have (atom_mn_elems MN, 8 K) layout with MN contiguous.
// Atoms tile K-outer (groups of 8 K-rows), MN-inner.
//   SW_128B: atom (64 MN, 8 K), SW_64B: atom (32 MN, 8 K), SW_32B: atom (16 MN, 8 K).
static __device__ void swizzle_store_mn(
    cutlass::bfloat16_t* smem_tile,
    int mn, int k, int MN,
    cutlass::bfloat16_t val,
    uint32_t swizzle_mask)
{
  if (swizzle_mask != 0) {
    int atom_mn_elems = (swizzle_mask == 0x380) ? 64 :
                        (swizzle_mask == 0x180) ? 32 : 16;

    int k_group      = k / 8;
    int local_k      = k % 8;
    int mn_atom      = mn / atom_mn_elems;
    int local_mn     = mn % atom_mn_elems;
    int num_mn_atoms = MN / atom_mn_elems;
    int atom_size_bytes = atom_mn_elems * 8 * 2;

    uint32_t tile_base = cute::cast_smem_ptr_to_uint(smem_tile);
    uint32_t atom_base = tile_base +
        static_cast<uint32_t>((k_group * num_mn_atoms + mn_atom) * atom_size_bytes);
    uint32_t byte_in_atom = static_cast<uint32_t>(local_k * atom_mn_elems * 2 + local_mn * 2);
    uint32_t addr = atom_base + byte_in_atom;
    uint32_t swizzled = addr ^ ((addr & swizzle_mask) >> 3);
    smem_tile[(swizzled - tile_base) / 2] = val;
  } else {
    // INTERLEAVE MN-major: 8-K-row groups, MN chunks of 8 bf16 elements
    int k_group     = k / 8;
    int local_k     = k % 8;
    int mn_chunk    = mn / 8;
    int mn_in_chunk = mn % 8;
    int dst_idx = k_group * (MN * 8) + mn_chunk * 64 + local_k * 8 + mn_in_chunk;
    smem_tile[dst_idx] = val;
  }
}

// Device-side helper: copy one byte from GMEM to MN-major SMEM with swizzle (for fp8 data).
// Same logic as swizzle_store_mn() but operates on bytes.
static __device__ void swizzle_store_mn_byte(
    uint8_t* smem_tile,
    int mn_byte, int k, int MN_bytes,
    uint8_t val,
    uint32_t swizzle_mask)
{
  if (swizzle_mask != 0) {
    int atom_mn_bytes = (swizzle_mask == 0x380) ? 128 :
                        (swizzle_mask == 0x180) ? 64 : 32;

    int k_group      = k / 8;
    int local_k      = k % 8;
    int mn_atom      = mn_byte / atom_mn_bytes;
    int local_mn     = mn_byte % atom_mn_bytes;
    int num_mn_atoms = MN_bytes / atom_mn_bytes;
    int atom_size_bytes = atom_mn_bytes * 8;

    uint32_t tile_base = cute::cast_smem_ptr_to_uint(smem_tile);
    uint32_t atom_base = tile_base +
        static_cast<uint32_t>((k_group * num_mn_atoms + mn_atom) * atom_size_bytes);
    uint32_t byte_in_atom = static_cast<uint32_t>(local_k * atom_mn_bytes + local_mn);
    uint32_t addr = atom_base + byte_in_atom;
    uint32_t swizzled = addr ^ ((addr & swizzle_mask) >> 3);
    smem_tile[swizzled - tile_base] = val;
  } else {
    // INTERLEAVE MN-major: 8-K-row groups, MN chunks of 16 bytes
    int k_group     = k / 8;
    int local_k     = k % 8;
    int mn_chunk    = mn_byte / 16;
    int mn_in_chunk = mn_byte % 16;
    int dst_idx = k_group * (MN_bytes * 8) + mn_chunk * 128 + local_k * 16 + mn_in_chunk;
    smem_tile[dst_idx] = val;
  }
}

// Kernel template parameterized by number of stages, wait pattern, K per stage, and layout major
template <int NUM_STAGES, int WAIT_PATTERN, int K_PER_STAGE_T, int IS_MN_MAJOR = 0,
          int MMA_M_T = 128, int MMA_N_T = 256>
__global__ void
__launch_bounds__(128, 1)
mma_swizzle_benchmark_kernel(
    cutlass::bfloat16_t const* __restrict__ gA,  // K-major: [NUM_STAGES][MMA_M * K], MN-major: [NUM_STAGES][K * MMA_M]
    cutlass::bfloat16_t const* __restrict__ gB,  // K-major: [NUM_STAGES][MMA_N * K], MN-major: [NUM_STAGES][K * MMA_N]
    float*    __restrict__ gD,                    // Output [128][256]
    int64_t*  __restrict__ gCycles,               // Output: MMA cycle count
    int64_t*  __restrict__ gFillCycles,           // Output: SMEM fill cycle count
    uint8_t   layout_type,                        // Swizzle layout type for descriptors
    uint16_t  lbo_a,                              // Leading byte offset for A (uint128_t units)
    uint16_t  sbo_a,                              // Stride byte offset for A (uint128_t units)
    uint16_t  lbo_b,                              // Leading byte offset for B (uint128_t units)
    uint16_t  sbo_b,                              // Stride byte offset for B (uint128_t units)
    uint32_t  swizzle_mask,                       // Byte-address swizzle mask (0 for INTERLEAVE)
    int       k_iters)                            // Number of mainloop iterations
{
  // Derive tile dimensions from template parameters
  constexpr int K_BLOCKS = K_PER_STAGE_T / MMA_K;
  constexpr int A_TILE_ELEMS = MMA_M_T * K_PER_STAGE_T;
  constexpr int B_TILE_ELEMS = MMA_N_T * K_PER_STAGE_T;
  constexpr int A_TILE_BYTES = MMA_M_T * K_PER_STAGE_T * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int B_TILE_BYTES = MMA_N_T * K_PER_STAGE_T * static_cast<int>(sizeof(cutlass::bfloat16_t));

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
    if constexpr (IS_MN_MAJOR) {
      // MN-major: input is [K_PER_STAGE][MN] with MN contiguous
      for (int i = threadIdx.x; i < A_TILE_ELEMS; i += blockDim.x) {
        int k = i / MMA_M_T;
        int m = i % MMA_M_T;
        swizzle_store_mn(sA[s], m, k, MMA_M_T,
                         gA[s * A_TILE_ELEMS + i], swizzle_mask);
      }
      for (int i = threadIdx.x; i < B_TILE_ELEMS; i += blockDim.x) {
        int k = i / MMA_N_T;
        int n = i % MMA_N_T;
        swizzle_store_mn(sB[s], n, k, MMA_N_T,
                         gB[s * B_TILE_ELEMS + i], swizzle_mask);
      }
    } else {
      // K-major: input is [MN][K_PER_STAGE] with K contiguous
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

  // Build instruction descriptor
  uint64_t idescE;
  if constexpr (IS_MN_MAJOR) {
    idescE = UMMA::make_runtime_instr_desc<
        cutlass::bfloat16_t, cutlass::bfloat16_t, float,
        MMA_M_T, MMA_N_T, UMMA::Major::MN, UMMA::Major::MN>();
  } else {
    idescE = UMMA::make_runtime_instr_desc<
        cutlass::bfloat16_t, cutlass::bfloat16_t, float,
        MMA_M_T, MMA_N_T, UMMA::Major::K, UMMA::Major::K>();
  }

  // Build SMEM descriptors for each stage and K-block
  uint64_t desc_a[NUM_STAGES][K_BLOCKS];
  uint64_t desc_b[NUM_STAGES][K_BLOCKS];

  for (int s = 0; s < NUM_STAGES; ++s) {
    uint32_t base_a = cute::cast_smem_ptr_to_uint(sA[s]);
    uint32_t base_b = cute::cast_smem_ptr_to_uint(sB[s]);

    for (int kb = 0; kb < K_BLOCKS; ++kb) {
      if constexpr (IS_MN_MAJOR) {
        // MN-major: K-block offset is uniform across swizzle modes
        uint32_t offset_a = kb * MMA_K * MMA_M_T * static_cast<int>(sizeof(cutlass::bfloat16_t));
        uint32_t offset_b = kb * MMA_K * MMA_N_T * static_cast<int>(sizeof(cutlass::bfloat16_t));
        desc_a[s][kb] = make_smem_desc(base_a + offset_a, layout_type, lbo_a, sbo_a);
        desc_b[s][kb] = make_smem_desc(base_b + offset_b, layout_type, lbo_b, sbo_b);
      } else {
        uint32_t offset_bytes;
        if (swizzle_mask != 0) {
          // Swizzled modes: data is organized in atoms of (8, atom_cols) bf16.
          int atom_cols = (swizzle_mask == 0x380) ? 64 :
                          (swizzle_mask == 0x180) ? 32 : 16;
          int kblocks_per_atom = atom_cols / MMA_K;
          int atom_size_bytes  = 8 * atom_cols * 2;
          int atom_idx    = kb / kblocks_per_atom;
          int kb_in_atom  = kb % kblocks_per_atom;
          offset_bytes = atom_idx * atom_size_bytes + kb_in_atom * 32;
        } else {
          offset_bytes = kb * 32 * lbo_a;
        }
        desc_a[s][kb] = make_smem_desc(base_a + offset_bytes, layout_type, lbo_a, sbo_a);
        desc_b[s][kb] = make_smem_desc(base_b + offset_bytes, layout_type, lbo_b, sbo_b);
      }
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
          if constexpr (IS_MN_MAJOR) {
            SM100_MMA_F16BF16_SS<
              cutlass::bfloat16_t, cutlass::bfloat16_t, float,
              MMA_M_T, MMA_N_T, UMMA::Major::MN, UMMA::Major::MN>::fma(
                desc_a[s][kb], desc_b[s][kb], tmem_addr, scaleC, idescE);
          } else {
            SM100_MMA_F16BF16_SS<
              cutlass::bfloat16_t, cutlass::bfloat16_t, float,
              MMA_M_T, MMA_N_T, UMMA::Major::K, UMMA::Major::K>::fma(
                desc_a[s][kb], desc_b[s][kb], tmem_addr, scaleC, idescE);
          }
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
  constexpr int ROWS_PER_WARP = MMA_M_T / 4;
  int m = warp_id * ROWS_PER_WARP + tid;

  constexpr uint32_t TMEM_DP_STRIDE = (1u << 16);

  float reg_buf[MMA_N_T];

  // ALL threads execute warp-level TMEM load (no divergence allowed)
  uint32_t dp_base = tmem_addr + warp_id * 32 * TMEM_DP_STRIDE;
  for (int col = 0; col < MMA_N_T; col += 8) {
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

  // Only valid threads write to GMEM
  if (tid < ROWS_PER_WARP) {
    for (int col = 0; col < MMA_N_T; ++col) {
      gD[m * MMA_N_T + col] = reg_buf[col];
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
// nvfp4 (mxf4nvf4) Benchmark Kernel
///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Kernel for nvfp4 block-scaled MMA with tcgen05.cp scale factor copies.
// Single stage (K_PER_STAGE=256), parameterized by wait pattern.
template <int WAIT_PATTERN, int MMA_M_T = 128, int MMA_N_T = 256>
__global__ void
__launch_bounds__(128, 1)
fp4_mma_swizzle_benchmark_kernel(
    uint8_t const* __restrict__ gA,         // Byte-packed fp4: [FP4_MMA_M][FP4_K_BYTES]
    uint8_t const* __restrict__ gB,         // Byte-packed fp4: [FP4_MMA_N][FP4_K_BYTES]
    uint8_t const* __restrict__ gSFA,       // Scale factors: [32][FP4_NUM_SF]
    uint8_t const* __restrict__ gSFB,       // Scale factors: [32][FP4_NUM_SF]
    float*    __restrict__ gD,              // Output: [FP4_MMA_M][FP4_MMA_N]
    int64_t*  __restrict__ gCycles,
    int64_t*  __restrict__ gFillCycles,
    uint8_t   layout_type,
    uint16_t  lbo,
    uint16_t  sbo,
    uint32_t  swizzle_mask,
    int       k_iters)
{
  constexpr int A_TILE_BYTES  = MMA_M_T * FP4_K_BYTES;
  constexpr int B_TILE_BYTES  = MMA_N_T * FP4_K_BYTES;
  constexpr int SF_TILE_BYTES = 32 * FP4_NUM_SF;           // 32 * 16  = 512
  constexpr int M_BLOCKS = MMA_M_T / 128;                 // 1 for M=128, 2 for M=256
  constexpr int MMA_M_PER_BLOCK = 128;                    // hardware MMA M dimension

  extern __shared__ char smem_buf[];

  // SMEM layout (1KB aligned)
  constexpr int A_ALLOC  = (A_TILE_BYTES  + 1023) & ~1023;  // 16384
  constexpr int B_ALLOC  = (B_TILE_BYTES  + 1023) & ~1023;  // 16384
  constexpr int SF_ALLOC = (SF_TILE_BYTES + 1023) & ~1023;  // 1024

  uint8_t* sA   = reinterpret_cast<uint8_t*>(smem_buf);
  uint8_t* sB   = reinterpret_cast<uint8_t*>(smem_buf + A_ALLOC);
  uint8_t* sSFA = reinterpret_cast<uint8_t*>(smem_buf + A_ALLOC + B_ALLOC);
  uint8_t* sSFB = reinterpret_cast<uint8_t*>(smem_buf + A_ALLOC + B_ALLOC + SF_ALLOC);

  constexpr int META_OFFSET = A_ALLOC + B_ALLOC + SF_ALLOC + SF_ALLOC;
  uint64_t* mma_barrier = reinterpret_cast<uint64_t*>(smem_buf + META_OFFSET);
  uint32_t* tmem_base   = reinterpret_cast<uint32_t*>(smem_buf + META_OFFSET + 16);

  // Load A and B data from GMEM to SMEM with swizzle
  int64_t t_fill_start = 0, t_fill_end = 0;
  if (threadIdx.x == 0) {
    t_fill_start = clock64();
  }

  for (int i = threadIdx.x; i < A_TILE_BYTES; i += blockDim.x) {
    int m      = i / FP4_K_BYTES;
    int k_byte = i % FP4_K_BYTES;
    swizzle_store_byte(sA, m, k_byte, FP4_K_BYTES, gA[i], swizzle_mask);
  }
  for (int i = threadIdx.x; i < B_TILE_BYTES; i += blockDim.x) {
    int n      = i / FP4_K_BYTES;
    int k_byte = i % FP4_K_BYTES;
    swizzle_store_byte(sB, n, k_byte, FP4_K_BYTES, gB[i], swizzle_mask);
  }

  // Load SF data (no swizzle — simple packed copy)
  for (int i = threadIdx.x; i < SF_TILE_BYTES; i += blockDim.x) {
    sSFA[i] = gSFA[i];
  }
  for (int i = threadIdx.x; i < SF_TILE_BYTES; i += blockDim.x) {
    sSFB[i] = gSFB[i];
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

  // SF TMEM addresses: accumulator uses columns 0..N-1, SFs start after
  uint32_t tsfa_addr = tmem_addr + MMA_N_T;         // columns start after accumulator
  uint32_t tsfb_addr = tmem_addr + MMA_N_T + 4;    // 4 columns (16 bytes) after SFA

  // Build instruction descriptor for mxf4nvf4 block-scaled MMA
  uint64_t idescE = UMMA::make_runtime_instr_desc_block_scaled<
      cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue4m3_t,
      MMA_M_PER_BLOCK, MMA_N_T, UMMA::Major::K, UMMA::Major::K>(tsfa_addr, tsfb_addr);

  // Build SMEM descriptors for each K-block (per M-block for A)
  uint64_t desc_a[M_BLOCKS * FP4_K_BLOCKS];
  uint64_t desc_b[FP4_K_BLOCKS];

  uint32_t base_a = cute::cast_smem_ptr_to_uint(sA);
  uint32_t base_b = cute::cast_smem_ptr_to_uint(sB);

  for (int mb = 0; mb < M_BLOCKS; ++mb) {
    uint32_t mb_base_a = base_a + mb * MMA_M_PER_BLOCK * FP4_K_BYTES;
    for (int kb = 0; kb < FP4_K_BLOCKS; ++kb) {
      uint32_t offset_bytes;
      if (swizzle_mask != 0) {
        // Swizzled: atom-based layout, 32 bytes per MMA K-block
        int atom_cols_bytes = (swizzle_mask == 0x380) ? 128 :
                              (swizzle_mask == 0x180) ? 64 : 32;
        int mma_k_bytes      = FP4_MMA_K / 2;  // 64 fp4 / 2 = 32 bytes
        int kblocks_per_atom = atom_cols_bytes / mma_k_bytes;
        int atom_size_bytes  = 8 * atom_cols_bytes;
        int atom_idx    = kb / kblocks_per_atom;
        int kb_in_atom  = kb % kblocks_per_atom;
        offset_bytes = atom_idx * atom_size_bytes + kb_in_atom * 32;
      } else {
        // Interleave: LBO-based stride
        offset_bytes = kb * 32 * lbo;
      }
      desc_a[mb * FP4_K_BLOCKS + kb] = make_smem_desc(mb_base_a + offset_bytes, layout_type, lbo, sbo);
    }
  }
  for (int kb = 0; kb < FP4_K_BLOCKS; ++kb) {
    uint32_t offset_bytes;
    if (swizzle_mask != 0) {
      int atom_cols_bytes = (swizzle_mask == 0x380) ? 128 :
                            (swizzle_mask == 0x180) ? 64 : 32;
      int mma_k_bytes      = FP4_MMA_K / 2;
      int kblocks_per_atom = atom_cols_bytes / mma_k_bytes;
      int atom_size_bytes  = 8 * atom_cols_bytes;
      int atom_idx    = kb / kblocks_per_atom;
      int kb_in_atom  = kb % kblocks_per_atom;
      offset_bytes = atom_idx * atom_size_bytes + kb_in_atom * 32;
    } else {
      offset_bytes = kb * 32 * lbo;
    }
    desc_b[kb] = make_smem_desc(base_b + offset_bytes, layout_type, lbo, sbo);
  }

  // Build SF SMEM descriptors for tcgen05.cp (no swizzle, packed 32x16 bytes)
  // layout_type=0, LBO=1 (16 bytes/row), SBO=8 (128 bytes per 8-row group)
  uint64_t sf_desc_a = make_smem_desc(
      cute::cast_smem_ptr_to_uint(sSFA), 0, 1, 8);
  uint64_t sf_desc_b = make_smem_desc(
      cute::cast_smem_ptr_to_uint(sSFB), 0, 1, 8);

  int tid = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;
  constexpr uint32_t TMEM_DP_STRIDE = (1u << 16);

  int64_t total_mma_cycles = 0;

  for (int mb = 0; mb < M_BLOCKS; ++mb) {
    // Initialize mbarrier for this M-block
    if (elect_one_warp && elect_one_thr) {
      uint32_t arrive_count;
      if constexpr (WAIT_PATTERN == 0) {
        arrive_count = 1;
      }
      else if constexpr (WAIT_PATTERN == 1) {
        arrive_count = k_iters;
      }
      else {
        arrive_count = 1;
      }
      cute::initialize_barrier(*mma_barrier, arrive_count);
    }
    int phase_bit = 0;

    __syncthreads();

    if (elect_one_warp) {
      // Fence TMEM before new tcgen05 operations: ensure prior TMEM loads/stores
      // are fully committed before issuing new CP/MMA to the tcgen05 pipeline
      cutlass::arch::fence_view_async_tmem_store();
      cutlass::arch::fence_view_async_tmem_load();

      // Copy scale factors from SMEM to TMEM (each M-block needs fresh SF data)
      SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc_a, tsfa_addr);
      SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc_b, tsfb_addr);
      if constexpr (MMA_N_T > 128) {
        SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc_b, tsfb_addr + 4);
      }

      int64_t t1 = clock64();

      uint32_t scaleC = 0;  // First MMA clears accumulator

      for (int i = 0; i < k_iters; ++i) {
        #pragma unroll
        for (int kb = 0; kb < FP4_K_BLOCKS; ++kb) {
          SM100_MMA_MXF4_SS<
              cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue4m3_t,
              MMA_M_PER_BLOCK, MMA_N_T, FP4_VS, UMMA::Major::K, UMMA::Major::K>::fma(
                desc_a[mb * FP4_K_BLOCKS + kb], desc_b[kb], tmem_addr, scaleC, idescE, tsfa_addr, tsfb_addr);
          scaleC = 1;
        }

        // Wait pattern handling
        if constexpr (WAIT_PATTERN == 0) {
          cutlass::arch::umma_arrive(mma_barrier);
          cute::wait_barrier(*mma_barrier, phase_bit);
          phase_bit ^= 1;
        }
        else if constexpr (WAIT_PATTERN == 1) {
          cutlass::arch::umma_arrive(mma_barrier);
        }
      }

      // Post-loop completion
      if constexpr (WAIT_PATTERN == 1) {
        cute::wait_barrier(*mma_barrier, phase_bit);
      }
      else if constexpr (WAIT_PATTERN == 2) {
        cutlass::arch::umma_arrive(mma_barrier);
        cute::wait_barrier(*mma_barrier, phase_bit);
      }

      int64_t t2 = clock64();
      total_mma_cycles += t2 - t1;
    }
    __syncthreads();

    // Epilogue: TMEM -> registers -> GMEM
    cutlass::arch::fence_view_async_tmem_store();
    cutlass::arch::fence_view_async_tmem_load();

    float reg_buf[MMA_N_T];

    // ALL threads execute warp-level TMEM load (no divergence allowed)
    uint32_t dp_base = tmem_addr + warp_id * 32 * TMEM_DP_STRIDE;
    for (int col = 0; col < MMA_N_T; col += 8) {
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

    // Fence: ensure TMEM loads complete before next M-block's tcgen05.cp/mma stores
    cutlass::arch::fence_view_async_tmem_load();

    // Write to GMEM with M-block offset
    int m = mb * MMA_M_PER_BLOCK + warp_id * 32 + tid;
    for (int col = 0; col < MMA_N_T; ++col) {
      gD[m * MMA_N_T + col] = reg_buf[col];
    }

    __syncthreads();
  }

  if (threadIdx.x == 0) {
    gCycles[0]     = total_mma_cycles;
    gFillCycles[0] = t_fill_end - t_fill_start;
  }

  __syncthreads();

  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(*tmem_base, 512);
  }
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED (fp4 kernel)

///////////////////////////////////////////////////////////////////////////////////////////////////
// fp8 (f8f6f4) Dense Benchmark Kernel
///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Kernel for fp8 dense MMA (no scale factors). Structurally mirrors bf16 kernel
// but with 1-byte elements and MMA K=32. Same fma() signature as bf16.
template <int WAIT_PATTERN, int IS_MN_MAJOR = 0, int MMA_M_T = 128, int MMA_N_T = 256>
__global__ void
__launch_bounds__(128, 1)
fp8_mma_swizzle_benchmark_kernel(
    uint8_t const* __restrict__ gA,         // K-major: [M][K], MN-major: [K][M]
    uint8_t const* __restrict__ gB,         // K-major: [N][K], MN-major: [K][N]
    float*    __restrict__ gD,              // Output: [FP8_MMA_M][FP8_MMA_N]
    int64_t*  __restrict__ gCycles,
    int64_t*  __restrict__ gFillCycles,
    uint8_t   layout_type,
    uint16_t  lbo_a,
    uint16_t  sbo_a,
    uint16_t  lbo_b,
    uint16_t  sbo_b,
    uint32_t  swizzle_mask,
    int       k_iters)
{
  constexpr int A_TILE_BYTES = MMA_M_T * FP8_K_BYTES;
  constexpr int B_TILE_BYTES = MMA_N_T * FP8_K_BYTES;

  extern __shared__ char smem_buf[];

  // SMEM layout (1KB aligned)
  constexpr int A_ALLOC = (A_TILE_BYTES + 1023) & ~1023;  // 32768
  constexpr int B_ALLOC = (B_TILE_BYTES + 1023) & ~1023;  // 65536

  uint8_t* sA = reinterpret_cast<uint8_t*>(smem_buf);
  uint8_t* sB = reinterpret_cast<uint8_t*>(smem_buf + A_ALLOC);

  constexpr int META_OFFSET = A_ALLOC + B_ALLOC;
  uint64_t* mma_barrier = reinterpret_cast<uint64_t*>(smem_buf + META_OFFSET);
  uint32_t* tmem_base   = reinterpret_cast<uint32_t*>(smem_buf + META_OFFSET + 16);

  // Load A and B data from GMEM to SMEM with swizzle
  int64_t t_fill_start = 0, t_fill_end = 0;
  if (threadIdx.x == 0) {
    t_fill_start = clock64();
  }

  if constexpr (IS_MN_MAJOR) {
    // MN-major: input is [K][MN] with MN contiguous (1 byte per element)
    for (int i = threadIdx.x; i < A_TILE_BYTES; i += blockDim.x) {
      int k       = i / MMA_M_T;
      int mn_byte = i % MMA_M_T;
      swizzle_store_mn_byte(sA, mn_byte, k, MMA_M_T, gA[i], swizzle_mask);
    }
    for (int i = threadIdx.x; i < B_TILE_BYTES; i += blockDim.x) {
      int k       = i / MMA_N_T;
      int mn_byte = i % MMA_N_T;
      swizzle_store_mn_byte(sB, mn_byte, k, MMA_N_T, gB[i], swizzle_mask);
    }
  } else {
    // K-major: input is [MN][K] with K contiguous
    for (int i = threadIdx.x; i < A_TILE_BYTES; i += blockDim.x) {
      int m      = i / FP8_K_BYTES;
      int k_byte = i % FP8_K_BYTES;
      swizzle_store_byte(sA, m, k_byte, FP8_K_BYTES, gA[i], swizzle_mask);
    }
    for (int i = threadIdx.x; i < B_TILE_BYTES; i += blockDim.x) {
      int n      = i / FP8_K_BYTES;
      int k_byte = i % FP8_K_BYTES;
      swizzle_store_byte(sB, n, k_byte, FP8_K_BYTES, gB[i], swizzle_mask);
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

  // Build instruction descriptor
  uint64_t idescE;
  if constexpr (IS_MN_MAJOR) {
    idescE = UMMA::make_runtime_instr_desc<
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
        MMA_M_T, MMA_N_T, UMMA::Major::MN, UMMA::Major::MN>();
  } else {
    idescE = UMMA::make_runtime_instr_desc<
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
        MMA_M_T, MMA_N_T, UMMA::Major::K, UMMA::Major::K>();
  }

  // Build SMEM descriptors for each K-block
  uint64_t desc_a[FP8_K_BLOCKS];
  uint64_t desc_b[FP8_K_BLOCKS];

  uint32_t base_a = cute::cast_smem_ptr_to_uint(sA);
  uint32_t base_b = cute::cast_smem_ptr_to_uint(sB);

  for (int kb = 0; kb < FP8_K_BLOCKS; ++kb) {
    if constexpr (IS_MN_MAJOR) {
      // MN-major: K-block offset is uniform across swizzle modes
      uint32_t offset_a = kb * FP8_MMA_K * MMA_M_T;  // 1 byte per fp8 element
      uint32_t offset_b = kb * FP8_MMA_K * MMA_N_T;
      desc_a[kb] = make_smem_desc(base_a + offset_a, layout_type, lbo_a, sbo_a);
      desc_b[kb] = make_smem_desc(base_b + offset_b, layout_type, lbo_b, sbo_b);
    } else {
      uint32_t offset_bytes;
      if (swizzle_mask != 0) {
        // Swizzled: atom-based layout, 32 bytes per MMA K-block
        int atom_cols_bytes = (swizzle_mask == 0x380) ? 128 :
                              (swizzle_mask == 0x180) ? 64 : 32;
        int mma_k_bytes      = FP8_MMA_K;
        int kblocks_per_atom = atom_cols_bytes / mma_k_bytes;
        int atom_size_bytes  = 8 * atom_cols_bytes;
        int atom_idx    = kb / kblocks_per_atom;
        int kb_in_atom  = kb % kblocks_per_atom;
        offset_bytes = atom_idx * atom_size_bytes + kb_in_atom * 32;
      } else {
        offset_bytes = kb * 32 * lbo_a;
      }
      desc_a[kb] = make_smem_desc(base_a + offset_bytes, layout_type, lbo_a, sbo_a);
      desc_b[kb] = make_smem_desc(base_b + offset_bytes, layout_type, lbo_b, sbo_b);
    }
  }

  // Initialize mbarrier
  if (elect_one_warp && elect_one_thr) {
    uint32_t arrive_count;
    if constexpr (WAIT_PATTERN == 0) {
      arrive_count = 1;
    }
    else if constexpr (WAIT_PATTERN == 1) {
      arrive_count = k_iters;
    }
    else {
      arrive_count = 1;
    }
    cute::initialize_barrier(*mma_barrier, arrive_count);
  }
  int phase_bit = 0;

  __syncthreads();

  int64_t t_start = 0, t_end = 0;

  if (elect_one_warp) {
    t_start = clock64();

    uint32_t scaleC = 0;  // First MMA clears accumulator

    for (int i = 0; i < k_iters; ++i) {
      #pragma unroll
      for (int kb = 0; kb < FP8_K_BLOCKS; ++kb) {
        SM100_MMA_F8F6F4_SS::fma(
            desc_a[kb], desc_b[kb], tmem_addr, scaleC, idescE);
        scaleC = 1;
      }

      // Wait pattern handling
      if constexpr (WAIT_PATTERN == 0) {
        cutlass::arch::umma_arrive(mma_barrier);
        cute::wait_barrier(*mma_barrier, phase_bit);
        phase_bit ^= 1;
      }
      else if constexpr (WAIT_PATTERN == 1) {
        cutlass::arch::umma_arrive(mma_barrier);
      }
    }

    // Post-loop completion
    if constexpr (WAIT_PATTERN == 1) {
      cute::wait_barrier(*mma_barrier, phase_bit);
    }
    else if constexpr (WAIT_PATTERN == 2) {
      cutlass::arch::umma_arrive(mma_barrier);
      cute::wait_barrier(*mma_barrier, phase_bit);
    }

    t_end = clock64();
  }
  __syncthreads();

  // Epilogue: TMEM -> registers -> GMEM
  cutlass::arch::fence_view_async_tmem_store();
  cutlass::arch::fence_view_async_tmem_load();

  int tid = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;
  constexpr int ROWS_PER_WARP = MMA_M_T / 4;
  int m = warp_id * ROWS_PER_WARP + tid;

  constexpr uint32_t TMEM_DP_STRIDE = (1u << 16);
  float reg_buf[MMA_N_T];

  // ALL threads execute warp-level TMEM load (no divergence allowed)
  uint32_t dp_base = tmem_addr + warp_id * 32 * TMEM_DP_STRIDE;
  for (int col = 0; col < MMA_N_T; col += 8) {
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

  // Only valid threads write to GMEM
  if (tid < ROWS_PER_WARP) {
    for (int col = 0; col < MMA_N_T; ++col) {
      gD[m * MMA_N_T + col] = reg_buf[col];
    }
  }

  if (threadIdx.x == 0) {
    gCycles[0]     = t_end - t_start;
    gFillCycles[0] = t_fill_end - t_fill_start;
  }

  __syncthreads();

  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(*tmem_base, 512);
  }
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED (fp8 kernel)

///////////////////////////////////////////////////////////////////////////////////////////////////
// bf16 2SM (cta_group::2) Benchmark Kernel
///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Kernel for bf16 2SM MMA using cta_group::2.
// Two CTAs in a cluster collaborate on a single 256x256x16 MMA instruction.
// Each CTA holds 128 M-rows of A and 128 N-rows of B in its SMEM.
// Only the leader CTA (rank 0) issues the MMA instruction; the hardware
// automatically accesses the peer CTA's SMEM.
template <int WAIT_PATTERN, int IS_MN_MAJOR = 0, int MMA_M_T = 256, int MMA_N_T = 256>
__global__ void
__launch_bounds__(128, 1)
bf16_2sm_mma_swizzle_benchmark_kernel(
    cutlass::bfloat16_t const* __restrict__ gA,  // K-major: [256][K], MN-major: [K][128] x 2 CTAs
    cutlass::bfloat16_t const* __restrict__ gB,  // K-major: [256][K], MN-major: [K][128] x 2 CTAs
    float*    __restrict__ gD,                    // Output [256][256]
    int64_t*  __restrict__ gCycles,
    int64_t*  __restrict__ gFillCycles,
    uint8_t   layout_type,
    uint16_t  lbo_a,
    uint16_t  sbo_a,
    uint16_t  lbo_b,
    uint16_t  sbo_b,
    uint32_t  swizzle_mask,
    int       k_iters)
{
  constexpr int K_PER_STAGE = BF16_2SM_K_PER_STAGE;  // 256
  constexpr int M_PER_CTA   = MMA_M_T / 2;
  constexpr int N_PER_CTA   = MMA_N_T / 2;
  constexpr int K_BLOCKS    = K_PER_STAGE / BF16_2SM_MMA_K;  // 16

  constexpr int A_TILE_ELEMS = M_PER_CTA * K_PER_STAGE;    // 128 * 256 = 32768
  constexpr int B_TILE_ELEMS = N_PER_CTA * K_PER_STAGE;    // 128 * 256 = 32768
  constexpr int A_TILE_BYTES = A_TILE_ELEMS * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int B_TILE_BYTES = B_TILE_ELEMS * static_cast<int>(sizeof(cutlass::bfloat16_t));

  // Determine CTA role within the cluster
  uint32_t cta_rank = cute::block_rank_in_cluster();
  bool is_leader = (cta_rank == 0);

  extern __shared__ char smem_buf[];

  // SMEM layout per CTA (1KB aligned)
  constexpr int A_ALLOC = (A_TILE_BYTES + 1023) & ~1023;  // 64KB
  constexpr int B_ALLOC = (B_TILE_BYTES + 1023) & ~1023;  // 64KB

  cutlass::bfloat16_t* sA = reinterpret_cast<cutlass::bfloat16_t*>(smem_buf);
  cutlass::bfloat16_t* sB = reinterpret_cast<cutlass::bfloat16_t*>(smem_buf + A_ALLOC);

  constexpr int META_OFFSET = A_ALLOC + B_ALLOC;
  uint64_t* mma_barrier = reinterpret_cast<uint64_t*>(smem_buf + META_OFFSET);
  uint32_t* tmem_base   = reinterpret_cast<uint32_t*>(smem_buf + META_OFFSET + 16);

  // Each CTA loads its portion of A and B from GMEM to SMEM
  // CTA 0: A[0:128, :], B[0:128, :]
  // CTA 1: A[128:256, :], B[128:256, :]
  int64_t t_fill_start = 0, t_fill_end = 0;
  if (threadIdx.x == 0) {
    t_fill_start = clock64();
  }

  int a_offset = cta_rank * A_TILE_ELEMS;
  int b_offset = cta_rank * B_TILE_ELEMS;

  if constexpr (IS_MN_MAJOR) {
    // MN-major: per-CTA data is [K_PER_STAGE][M_PER_CTA] with MN contiguous
    for (int i = threadIdx.x; i < A_TILE_ELEMS; i += blockDim.x) {
      int k = i / M_PER_CTA;
      int m = i % M_PER_CTA;
      swizzle_store_mn(sA, m, k, M_PER_CTA,
                       gA[a_offset + i], swizzle_mask);
    }
    for (int i = threadIdx.x; i < B_TILE_ELEMS; i += blockDim.x) {
      int k = i / N_PER_CTA;
      int n = i % N_PER_CTA;
      swizzle_store_mn(sB, n, k, N_PER_CTA,
                       gB[b_offset + i], swizzle_mask);
    }
  } else {
    // K-major: per-CTA data is [M_PER_CTA][K_PER_STAGE] with K contiguous
    for (int i = threadIdx.x; i < A_TILE_ELEMS; i += blockDim.x) {
      int m = i / K_PER_STAGE;
      int k = i % K_PER_STAGE;
      swizzle_store(sA, m, k, K_PER_STAGE,
                    gA[a_offset + i], swizzle_mask);
    }
    for (int i = threadIdx.x; i < B_TILE_ELEMS; i += blockDim.x) {
      int n = i / K_PER_STAGE;
      int k = i % K_PER_STAGE;
      swizzle_store(sB, n, k, K_PER_STAGE,
                    gB[b_offset + i], swizzle_mask);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    t_fill_end = clock64();
  }

  // TMEM allocation with 2SM allocator (warp 0 of both CTAs)
  uint32_t elect_one_thr  = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  using TmemAllocator = cute::TMEM::Allocator2Sm;
  TmemAllocator tmem_allocator{};

  if (elect_one_warp) {
    tmem_allocator.allocate(512, tmem_base);
  }
  __syncthreads();

  // cluster_sync to ensure both CTAs see barrier init + TMEM alloc
  cute::cluster_sync();

  uint32_t tmem_addr = *tmem_base;

  // Build instruction descriptor
  uint64_t idescE;
  if constexpr (IS_MN_MAJOR) {
    idescE = UMMA::make_runtime_instr_desc<
        cutlass::bfloat16_t, cutlass::bfloat16_t, float,
        MMA_M_T, MMA_N_T, UMMA::Major::MN, UMMA::Major::MN>();
  } else {
    idescE = UMMA::make_runtime_instr_desc<
        cutlass::bfloat16_t, cutlass::bfloat16_t, float,
        MMA_M_T, MMA_N_T, UMMA::Major::K, UMMA::Major::K>();
  }

  // Build SMEM descriptors for each K-block (same per-CTA layout as 1SM bf16)
  uint64_t desc_a[K_BLOCKS];
  uint64_t desc_b[K_BLOCKS];

  uint32_t base_a = cute::cast_smem_ptr_to_uint(sA);
  uint32_t base_b = cute::cast_smem_ptr_to_uint(sB);

  for (int kb = 0; kb < K_BLOCKS; ++kb) {
    if constexpr (IS_MN_MAJOR) {
      // MN-major: K-block offset is uniform
      uint32_t offset_a = kb * BF16_2SM_MMA_K * M_PER_CTA * static_cast<int>(sizeof(cutlass::bfloat16_t));
      uint32_t offset_b = kb * BF16_2SM_MMA_K * N_PER_CTA * static_cast<int>(sizeof(cutlass::bfloat16_t));
      desc_a[kb] = make_smem_desc(base_a + offset_a, layout_type, lbo_a, sbo_a);
      desc_b[kb] = make_smem_desc(base_b + offset_b, layout_type, lbo_b, sbo_b);
    } else {
      uint32_t offset_bytes;
      if (swizzle_mask != 0) {
        int atom_cols = (swizzle_mask == 0x380) ? 64 :
                        (swizzle_mask == 0x180) ? 32 : 16;
        int kblocks_per_atom = atom_cols / BF16_2SM_MMA_K;
        int atom_size_bytes  = 8 * atom_cols * 2;
        int atom_idx    = kb / kblocks_per_atom;
        int kb_in_atom  = kb % kblocks_per_atom;
        offset_bytes = atom_idx * atom_size_bytes + kb_in_atom * 32;
      } else {
        offset_bytes = kb * 32 * lbo_a;
      }
      desc_a[kb] = make_smem_desc(base_a + offset_bytes, layout_type, lbo_a, sbo_a);
      desc_b[kb] = make_smem_desc(base_b + offset_bytes, layout_type, lbo_b, sbo_b);
    }
  }

  // Initialize mbarrier on each CTA
  if (elect_one_warp && elect_one_thr) {
    uint32_t arrive_count;
    if constexpr (WAIT_PATTERN == 0) {
      arrive_count = 1;
    }
    else if constexpr (WAIT_PATTERN == 1) {
      arrive_count = k_iters;
    }
    else {
      arrive_count = 1;
    }
    cute::initialize_barrier(*mma_barrier, arrive_count);
  }
  int phase_bit = 0;

  __syncthreads();
  cute::cluster_sync();

  // Mainloop — timed with clock64()
  int64_t t_start = 0, t_end = 0;

  if (elect_one_warp) {
    t_start = clock64();

    uint32_t scaleC = 0;  // First MMA clears accumulator

    for (int i = 0; i < k_iters; ++i) {
      if (is_leader) {
        // Only leader CTA issues the MMA instruction
        #pragma unroll
        for (int kb = 0; kb < K_BLOCKS; ++kb) {
          if constexpr (IS_MN_MAJOR) {
            SM100_MMA_F16BF16_2x1SM_SS<
              cutlass::bfloat16_t, cutlass::bfloat16_t, float,
              MMA_M_T, MMA_N_T, UMMA::Major::MN, UMMA::Major::MN>::fma(
                desc_a[kb], desc_b[kb], tmem_addr, scaleC, idescE);
          } else {
            SM100_MMA_F16BF16_2x1SM_SS<
              cutlass::bfloat16_t, cutlass::bfloat16_t, float,
              MMA_M_T, MMA_N_T, UMMA::Major::K, UMMA::Major::K>::fma(
                desc_a[kb], desc_b[kb], tmem_addr, scaleC, idescE);
          }
          scaleC = 1;  // Accumulate after first MMA
        }

        // Leader issues multicast arrive to both CTAs' barriers
        if constexpr (WAIT_PATTERN == 0) {
          cutlass::arch::umma_arrive_multicast_2x1SM(mma_barrier, 0x3);
        }
        else if constexpr (WAIT_PATTERN == 1) {
          cutlass::arch::umma_arrive_multicast_2x1SM(mma_barrier, 0x3);
        }
      }

      // Both CTAs wait on their local barrier
      if constexpr (WAIT_PATTERN == 0) {
        cute::wait_barrier(*mma_barrier, phase_bit);
        phase_bit ^= 1;
      }
      // Pattern (b) and (c): no wait inside loop
    }

    // Post-loop: ensure all MMA operations are complete
    if constexpr (WAIT_PATTERN == 1) {
      cute::wait_barrier(*mma_barrier, phase_bit);
    }
    else if constexpr (WAIT_PATTERN == 2) {
      if (is_leader) {
        cutlass::arch::umma_arrive_multicast_2x1SM(mma_barrier, 0x3);
      }
      cute::wait_barrier(*mma_barrier, phase_bit);
    }

    t_end = clock64();
  }
  __syncthreads();
  cute::cluster_sync();

  // Epilogue: each CTA reads its M_PER_CTA TMEM rows
  cutlass::arch::fence_view_async_tmem_store();
  cutlass::arch::fence_view_async_tmem_load();

  int tid = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;
  constexpr int ROWS_PER_WARP_2SM = M_PER_CTA / 4;  // 16 for M_PER_CTA=64, 32 for M_PER_CTA=128
  int local_m = warp_id * ROWS_PER_WARP_2SM + tid;

  constexpr uint32_t TMEM_DP_STRIDE = (1u << 16);

  // CTA 1 offsets its TMEM read by M_PER_CTA DP rows
  uint32_t tmem_cta_offset = cta_rank * (M_PER_CTA / 32) * 32 * TMEM_DP_STRIDE;

  float reg_buf[MMA_N_T];

  // ALL threads execute warp-level TMEM load (no divergence allowed)
  uint32_t dp_base = tmem_addr + tmem_cta_offset + warp_id * 32 * TMEM_DP_STRIDE;
  for (int col = 0; col < MMA_N_T; col += 8) {
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

  // Only valid threads write to GMEM
  if (tid < ROWS_PER_WARP_2SM) {
    int m = cta_rank * M_PER_CTA + local_m;
    for (int col = 0; col < MMA_N_T; ++col) {
      gD[m * MMA_N_T + col] = reg_buf[col];
    }
  }

  // Write cycle counts (thread 0 of leader CTA only)
  if (threadIdx.x == 0 && is_leader) {
    gCycles[0]     = t_end - t_start;
    gFillCycles[0] = t_fill_end - t_fill_start;
  }

  __syncthreads();
  cute::cluster_sync();

  // Free TMEM via 2SM allocator
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(*tmem_base, 512);
  }
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED (bf16 2SM kernel)

///////////////////////////////////////////////////////////////////////////////////////////////////
// fp8 2SM (cta_group::2) Benchmark Kernel
///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Kernel for fp8 2SM MMA using cta_group::2.
// Two CTAs in a cluster collaborate on a single 256x256x32 MMA instruction.
// Each CTA holds 128 M-rows of A and 128 N-rows of B in its SMEM.
template <int WAIT_PATTERN, int IS_MN_MAJOR = 0, int MMA_M_T = 256, int MMA_N_T = 256>
__global__ void
__launch_bounds__(128, 1)
fp8_2sm_mma_swizzle_benchmark_kernel(
    uint8_t const* __restrict__ gA,         // K-major: [256][K], MN-major: [K][128] x 2 CTAs
    uint8_t const* __restrict__ gB,         // K-major: [256][K], MN-major: [K][128] x 2 CTAs
    float*    __restrict__ gD,              // Output [256][256]
    int64_t*  __restrict__ gCycles,
    int64_t*  __restrict__ gFillCycles,
    uint8_t   layout_type,
    uint16_t  lbo_a,
    uint16_t  sbo_a,
    uint16_t  lbo_b,
    uint16_t  sbo_b,
    uint32_t  swizzle_mask,
    int       k_iters)
{
  constexpr int K_PER_STAGE = FP8_2SM_K_PER_STAGE;  // 256
  constexpr int M_PER_CTA   = MMA_M_T / 2;
  constexpr int N_PER_CTA   = MMA_N_T / 2;
  constexpr int K_BLOCKS    = K_PER_STAGE / FP8_2SM_MMA_K;  // 8

  constexpr int A_TILE_BYTES = M_PER_CTA * K_PER_STAGE;    // 128 * 256 = 32768 (1 byte/elem)
  constexpr int B_TILE_BYTES = N_PER_CTA * K_PER_STAGE;    // 128 * 256 = 32768

  // Determine CTA role within the cluster
  uint32_t cta_rank = cute::block_rank_in_cluster();
  bool is_leader = (cta_rank == 0);

  extern __shared__ char smem_buf[];

  // SMEM layout per CTA (1KB aligned)
  constexpr int A_ALLOC = (A_TILE_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC = (B_TILE_BYTES + 1023) & ~1023;

  uint8_t* sA = reinterpret_cast<uint8_t*>(smem_buf);
  uint8_t* sB = reinterpret_cast<uint8_t*>(smem_buf + A_ALLOC);

  constexpr int META_OFFSET = A_ALLOC + B_ALLOC;
  uint64_t* mma_barrier = reinterpret_cast<uint64_t*>(smem_buf + META_OFFSET);
  uint32_t* tmem_base   = reinterpret_cast<uint32_t*>(smem_buf + META_OFFSET + 16);

  // Each CTA loads its portion of A and B from GMEM to SMEM
  int64_t t_fill_start = 0, t_fill_end = 0;
  if (threadIdx.x == 0) {
    t_fill_start = clock64();
  }

  int a_offset = cta_rank * A_TILE_BYTES;
  int b_offset = cta_rank * B_TILE_BYTES;

  if constexpr (IS_MN_MAJOR) {
    // MN-major: per-CTA data is [K_PER_STAGE][M_PER_CTA] with MN contiguous (1 byte/elem)
    for (int i = threadIdx.x; i < A_TILE_BYTES; i += blockDim.x) {
      int k       = i / M_PER_CTA;
      int mn_byte = i % M_PER_CTA;
      swizzle_store_mn_byte(sA, mn_byte, k, M_PER_CTA, gA[a_offset + i], swizzle_mask);
    }
    for (int i = threadIdx.x; i < B_TILE_BYTES; i += blockDim.x) {
      int k       = i / N_PER_CTA;
      int mn_byte = i % N_PER_CTA;
      swizzle_store_mn_byte(sB, mn_byte, k, N_PER_CTA, gB[b_offset + i], swizzle_mask);
    }
  } else {
    // K-major: per-CTA data is [M_PER_CTA][K_PER_STAGE] with K contiguous
    for (int i = threadIdx.x; i < A_TILE_BYTES; i += blockDim.x) {
      int m      = i / K_PER_STAGE;
      int k_byte = i % K_PER_STAGE;
      swizzle_store_byte(sA, m, k_byte, K_PER_STAGE, gA[a_offset + i], swizzle_mask);
    }
    for (int i = threadIdx.x; i < B_TILE_BYTES; i += blockDim.x) {
      int n      = i / K_PER_STAGE;
      int k_byte = i % K_PER_STAGE;
      swizzle_store_byte(sB, n, k_byte, K_PER_STAGE, gB[b_offset + i], swizzle_mask);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    t_fill_end = clock64();
  }

  // TMEM allocation with 2SM allocator (warp 0 of both CTAs)
  uint32_t elect_one_thr  = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  using TmemAllocator = cute::TMEM::Allocator2Sm;
  TmemAllocator tmem_allocator{};

  if (elect_one_warp) {
    tmem_allocator.allocate(512, tmem_base);
  }
  __syncthreads();

  // cluster_sync to ensure both CTAs see barrier init + TMEM alloc
  cute::cluster_sync();

  uint32_t tmem_addr = *tmem_base;

  // Build instruction descriptor
  uint64_t idescE;
  if constexpr (IS_MN_MAJOR) {
    idescE = UMMA::make_runtime_instr_desc<
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
        MMA_M_T, MMA_N_T, UMMA::Major::MN, UMMA::Major::MN>();
  } else {
    idescE = UMMA::make_runtime_instr_desc<
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
        MMA_M_T, MMA_N_T, UMMA::Major::K, UMMA::Major::K>();
  }

  // Build SMEM descriptors for each K-block (same per-CTA layout as 1SM fp8)
  uint64_t desc_a[K_BLOCKS];
  uint64_t desc_b[K_BLOCKS];

  uint32_t base_a = cute::cast_smem_ptr_to_uint(sA);
  uint32_t base_b = cute::cast_smem_ptr_to_uint(sB);

  for (int kb = 0; kb < K_BLOCKS; ++kb) {
    if constexpr (IS_MN_MAJOR) {
      // MN-major: K-block offset is uniform
      uint32_t offset_a = kb * FP8_2SM_MMA_K * M_PER_CTA;  // 1 byte per fp8 element
      uint32_t offset_b = kb * FP8_2SM_MMA_K * N_PER_CTA;
      desc_a[kb] = make_smem_desc(base_a + offset_a, layout_type, lbo_a, sbo_a);
      desc_b[kb] = make_smem_desc(base_b + offset_b, layout_type, lbo_b, sbo_b);
    } else {
      uint32_t offset_bytes;
      if (swizzle_mask != 0) {
        int atom_cols_bytes = (swizzle_mask == 0x380) ? 128 :
                              (swizzle_mask == 0x180) ? 64 : 32;
        int mma_k_bytes      = FP8_2SM_MMA_K;  // 32 bytes (1 byte/elem)
        int kblocks_per_atom = atom_cols_bytes / mma_k_bytes;
        int atom_size_bytes  = 8 * atom_cols_bytes;
        int atom_idx    = kb / kblocks_per_atom;
        int kb_in_atom  = kb % kblocks_per_atom;
        offset_bytes = atom_idx * atom_size_bytes + kb_in_atom * 32;
      } else {
        offset_bytes = kb * 32 * lbo_a;
      }
      desc_a[kb] = make_smem_desc(base_a + offset_bytes, layout_type, lbo_a, sbo_a);
      desc_b[kb] = make_smem_desc(base_b + offset_bytes, layout_type, lbo_b, sbo_b);
    }
  }

  // Initialize mbarrier on each CTA
  if (elect_one_warp && elect_one_thr) {
    uint32_t arrive_count;
    if constexpr (WAIT_PATTERN == 0) {
      arrive_count = 1;
    }
    else if constexpr (WAIT_PATTERN == 1) {
      arrive_count = k_iters;
    }
    else {
      arrive_count = 1;
    }
    cute::initialize_barrier(*mma_barrier, arrive_count);
  }
  int phase_bit = 0;

  __syncthreads();
  cute::cluster_sync();

  // Mainloop — timed with clock64()
  int64_t t_start = 0, t_end = 0;

  if (elect_one_warp) {
    t_start = clock64();

    uint32_t scaleC = 0;  // First MMA clears accumulator

    for (int i = 0; i < k_iters; ++i) {
      if (is_leader) {
        // Only leader CTA issues the MMA instruction
        #pragma unroll
        for (int kb = 0; kb < K_BLOCKS; ++kb) {
          SM100_MMA_F8F6F4_2x1SM_SS::fma(
              desc_a[kb], desc_b[kb], tmem_addr, scaleC, idescE);
          scaleC = 1;  // Accumulate after first MMA
        }

        // Leader issues multicast arrive to both CTAs' barriers
        if constexpr (WAIT_PATTERN == 0) {
          cutlass::arch::umma_arrive_multicast_2x1SM(mma_barrier, 0x3);
        }
        else if constexpr (WAIT_PATTERN == 1) {
          cutlass::arch::umma_arrive_multicast_2x1SM(mma_barrier, 0x3);
        }
      }

      // Both CTAs wait on their local barrier
      if constexpr (WAIT_PATTERN == 0) {
        cute::wait_barrier(*mma_barrier, phase_bit);
        phase_bit ^= 1;
      }
      // Pattern (b) and (c): no wait inside loop
    }

    // Post-loop: ensure all MMA operations are complete
    if constexpr (WAIT_PATTERN == 1) {
      cute::wait_barrier(*mma_barrier, phase_bit);
    }
    else if constexpr (WAIT_PATTERN == 2) {
      if (is_leader) {
        cutlass::arch::umma_arrive_multicast_2x1SM(mma_barrier, 0x3);
      }
      cute::wait_barrier(*mma_barrier, phase_bit);
    }

    t_end = clock64();
  }
  __syncthreads();
  cute::cluster_sync();

  // Epilogue: each CTA reads its M_PER_CTA TMEM rows
  cutlass::arch::fence_view_async_tmem_store();
  cutlass::arch::fence_view_async_tmem_load();

  int tid = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;
  constexpr int ROWS_PER_WARP_2SM = M_PER_CTA / 4;
  int local_m = warp_id * ROWS_PER_WARP_2SM + tid;

  constexpr uint32_t TMEM_DP_STRIDE = (1u << 16);

  // CTA 1 offsets its TMEM read by M_PER_CTA DP rows
  uint32_t tmem_cta_offset = cta_rank * (M_PER_CTA / 32) * 32 * TMEM_DP_STRIDE;

  float reg_buf[MMA_N_T];

  // ALL threads execute warp-level TMEM load (no divergence allowed)
  uint32_t dp_base = tmem_addr + tmem_cta_offset + warp_id * 32 * TMEM_DP_STRIDE;
  for (int col = 0; col < MMA_N_T; col += 8) {
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

  // Only valid threads write to GMEM
  if (tid < ROWS_PER_WARP_2SM) {
    int m = cta_rank * M_PER_CTA + local_m;
    for (int col = 0; col < MMA_N_T; ++col) {
      gD[m * MMA_N_T + col] = reg_buf[col];
    }
  }

  // Write cycle counts (thread 0 of leader CTA only)
  if (threadIdx.x == 0 && is_leader) {
    gCycles[0]     = t_end - t_start;
    gFillCycles[0] = t_fill_end - t_fill_start;
  }

  __syncthreads();
  cute::cluster_sync();

  // Free TMEM via 2SM allocator
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(*tmem_base, 512);
  }
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED (fp8 2SM kernel)

///////////////////////////////////////////////////////////////////////////////////////////////////
// fp4 2SM (cta_group::2) Benchmark Kernel
///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Kernel for fp4 2SM MMA using cta_group::2 with block-scaled mxf4nvf4.
// Two CTAs in a cluster collaborate on a single 256x256x64 MMA instruction.
// K-major only (MN-major not supported for E2M1/fp4 data type).
template <int WAIT_PATTERN, int MMA_M_T = 256, int MMA_N_T = 256>
__global__ void
__launch_bounds__(128, 1)
fp4_2sm_mma_swizzle_benchmark_kernel(
    uint8_t const* __restrict__ gA,         // Byte-packed fp4: [256][K/2]
    uint8_t const* __restrict__ gB,         // Byte-packed fp4: [256][K/2]
    uint8_t const* __restrict__ gSFA,       // Scale factors: [32][FP4_NUM_SF]
    uint8_t const* __restrict__ gSFB,       // Scale factors: [32][FP4_NUM_SF]
    float*    __restrict__ gD,              // Output: [256][256]
    int64_t*  __restrict__ gCycles,
    int64_t*  __restrict__ gFillCycles,
    uint8_t   layout_type,
    uint16_t  lbo,
    uint16_t  sbo,
    uint32_t  swizzle_mask,
    int       k_iters)
{
  constexpr int M_PER_CTA   = MMA_M_T / 2;    // 128
  constexpr int N_PER_CTA   = MMA_N_T / 2;    // 64
  constexpr int K_BLOCKS    = FP4_2SM_K_PER_STAGE / FP4_2SM_MMA_K;  // 4

  constexpr int A_TILE_BYTES  = M_PER_CTA * FP4_K_BYTES;    // 128 * 128 = 16384
  constexpr int B_TILE_BYTES  = N_PER_CTA * FP4_K_BYTES;    // 64 * 128 = 8192
  constexpr int SF_TILE_BYTES = 32 * FP4_NUM_SF;             // 32 * 16 = 512

  // Determine CTA role within the cluster
  uint32_t cta_rank = cute::block_rank_in_cluster();
  bool is_leader = (cta_rank == 0);

  extern __shared__ char smem_buf[];

  // SMEM layout per CTA (1KB aligned)
  constexpr int A_ALLOC  = (A_TILE_BYTES  + 1023) & ~1023;
  constexpr int B_ALLOC  = (B_TILE_BYTES  + 1023) & ~1023;
  constexpr int SF_ALLOC = (SF_TILE_BYTES + 1023) & ~1023;

  uint8_t* sA   = reinterpret_cast<uint8_t*>(smem_buf);
  uint8_t* sB   = reinterpret_cast<uint8_t*>(smem_buf + A_ALLOC);
  uint8_t* sSFA = reinterpret_cast<uint8_t*>(smem_buf + A_ALLOC + B_ALLOC);
  uint8_t* sSFB = reinterpret_cast<uint8_t*>(smem_buf + A_ALLOC + B_ALLOC + SF_ALLOC);

  constexpr int META_OFFSET = A_ALLOC + B_ALLOC + SF_ALLOC + SF_ALLOC;
  uint64_t* mma_barrier = reinterpret_cast<uint64_t*>(smem_buf + META_OFFSET);
  uint32_t* tmem_base   = reinterpret_cast<uint32_t*>(smem_buf + META_OFFSET + 16);

  // Each CTA loads its portion of A and B, plus both full SF arrays
  int64_t t_fill_start = 0, t_fill_end = 0;
  if (threadIdx.x == 0) {
    t_fill_start = clock64();
  }

  int a_offset = cta_rank * A_TILE_BYTES;
  int b_offset = cta_rank * B_TILE_BYTES;

  // K-major only for fp4
  for (int i = threadIdx.x; i < A_TILE_BYTES; i += blockDim.x) {
    int m      = i / FP4_K_BYTES;
    int k_byte = i % FP4_K_BYTES;
    swizzle_store_byte(sA, m, k_byte, FP4_K_BYTES, gA[a_offset + i], swizzle_mask);
  }
  for (int i = threadIdx.x; i < B_TILE_BYTES; i += blockDim.x) {
    int n      = i / FP4_K_BYTES;
    int k_byte = i % FP4_K_BYTES;
    swizzle_store_byte(sB, n, k_byte, FP4_K_BYTES, gB[b_offset + i], swizzle_mask);
  }

  // Both CTAs load the same full SF arrays (SFs are broadcast-indexed by m%32/n%32)
  for (int i = threadIdx.x; i < SF_TILE_BYTES; i += blockDim.x) {
    sSFA[i] = gSFA[i];
  }
  for (int i = threadIdx.x; i < SF_TILE_BYTES; i += blockDim.x) {
    sSFB[i] = gSFB[i];
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    t_fill_end = clock64();
  }

  // TMEM allocation with 2SM allocator
  uint32_t elect_one_thr  = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  using TmemAllocator = cute::TMEM::Allocator2Sm;
  TmemAllocator tmem_allocator{};

  if (elect_one_warp) {
    tmem_allocator.allocate(512, tmem_base);
  }
  __syncthreads();

  cute::cluster_sync();

  uint32_t tmem_addr = *tmem_base;

  // SF TMEM addresses: accumulator uses columns 0..N-1, SFs start after
  uint32_t tsfa_addr = tmem_addr + MMA_N_T;         // columns start after accumulator
  uint32_t tsfb_addr = tmem_addr + MMA_N_T + 4;    // 4 columns (16 bytes) after SFA

  // Build instruction descriptor for mxf4nvf4 block-scaled MMA
  uint64_t idescE = UMMA::make_runtime_instr_desc_block_scaled<
      cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue4m3_t,
      MMA_M_T, MMA_N_T, UMMA::Major::K, UMMA::Major::K>(tsfa_addr, tsfb_addr);

  // Build SMEM descriptors for each K-block
  uint64_t desc_a[K_BLOCKS];
  uint64_t desc_b[K_BLOCKS];

  uint32_t base_a = cute::cast_smem_ptr_to_uint(sA);
  uint32_t base_b = cute::cast_smem_ptr_to_uint(sB);

  for (int kb = 0; kb < K_BLOCKS; ++kb) {
    uint32_t offset_bytes;
    if (swizzle_mask != 0) {
      // Swizzled: atom-based layout, 32 bytes per MMA K-block
      int atom_cols_bytes = (swizzle_mask == 0x380) ? 128 :
                            (swizzle_mask == 0x180) ? 64 : 32;
      int mma_k_bytes      = FP4_2SM_MMA_K / 2;  // 64 fp4 / 2 = 32 bytes
      int kblocks_per_atom = atom_cols_bytes / mma_k_bytes;
      int atom_size_bytes  = 8 * atom_cols_bytes;
      int atom_idx    = kb / kblocks_per_atom;
      int kb_in_atom  = kb % kblocks_per_atom;
      offset_bytes = atom_idx * atom_size_bytes + kb_in_atom * 32;
    } else {
      // Interleave: LBO-based stride
      offset_bytes = kb * 32 * lbo;
    }
    desc_a[kb] = make_smem_desc(base_a + offset_bytes, layout_type, lbo, sbo);
    desc_b[kb] = make_smem_desc(base_b + offset_bytes, layout_type, lbo, sbo);
  }

  // Build SF SMEM descriptors for tcgen05.cp
  uint64_t sf_desc_a = make_smem_desc(
      cute::cast_smem_ptr_to_uint(sSFA), 0, 1, 8);
  uint64_t sf_desc_b = make_smem_desc(
      cute::cast_smem_ptr_to_uint(sSFB), 0, 1, 8);

  // Initialize mbarrier on each CTA
  if (elect_one_warp && elect_one_thr) {
    uint32_t arrive_count;
    if constexpr (WAIT_PATTERN == 0) {
      arrive_count = 1;
    }
    else if constexpr (WAIT_PATTERN == 1) {
      arrive_count = k_iters;
    }
    else {
      arrive_count = 1;
    }
    cute::initialize_barrier(*mma_barrier, arrive_count);
  }
  int phase_bit = 0;

  __syncthreads();
  cute::cluster_sync();

  int64_t t_start = 0, t_end = 0;

  if (elect_one_warp) {
    // Copy scale factors from SMEM to TMEM (once, before MMA loop)
    SM100_UTCCP_4x32dp128bit_2cta::copy(sf_desc_a, tsfa_addr);
    SM100_UTCCP_4x32dp128bit_2cta::copy(sf_desc_b, tsfb_addr);
    if constexpr (MMA_N_T > 128) {
      SM100_UTCCP_4x32dp128bit_2cta::copy(sf_desc_b, tsfb_addr + 4);
    }

    t_start = clock64();

    uint32_t scaleC = 0;  // First MMA clears accumulator

    for (int i = 0; i < k_iters; ++i) {
      if (is_leader) {
        #pragma unroll
        for (int kb = 0; kb < K_BLOCKS; ++kb) {
          SM100_MMA_MXF4_2x1SM_SS<
              cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue4m3_t,
              MMA_M_T, MMA_N_T, FP4_VS, UMMA::Major::K, UMMA::Major::K>::fma(
                desc_a[kb], desc_b[kb], tmem_addr, scaleC, idescE, tsfa_addr, tsfb_addr);
          scaleC = 1;
        }

        if constexpr (WAIT_PATTERN == 0) {
          cutlass::arch::umma_arrive_multicast_2x1SM(mma_barrier, 0x3);
        }
        else if constexpr (WAIT_PATTERN == 1) {
          cutlass::arch::umma_arrive_multicast_2x1SM(mma_barrier, 0x3);
        }
      }

      if constexpr (WAIT_PATTERN == 0) {
        cute::wait_barrier(*mma_barrier, phase_bit);
        phase_bit ^= 1;
      }
    }

    // Post-loop completion
    if constexpr (WAIT_PATTERN == 1) {
      cute::wait_barrier(*mma_barrier, phase_bit);
    }
    else if constexpr (WAIT_PATTERN == 2) {
      if (is_leader) {
        cutlass::arch::umma_arrive_multicast_2x1SM(mma_barrier, 0x3);
      }
      cute::wait_barrier(*mma_barrier, phase_bit);
    }

    t_end = clock64();
  }
  __syncthreads();
  cute::cluster_sync();

  // Epilogue: each CTA reads its M_PER_CTA TMEM rows
  cutlass::arch::fence_view_async_tmem_store();
  cutlass::arch::fence_view_async_tmem_load();

  int tid = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;
  constexpr int ROWS_PER_WARP_2SM = M_PER_CTA / 4;
  int local_m = warp_id * ROWS_PER_WARP_2SM + tid;

  constexpr uint32_t TMEM_DP_STRIDE = (1u << 16);

  // CTA 1 offsets its TMEM read by M_PER_CTA DP rows
  uint32_t tmem_cta_offset = cta_rank * (M_PER_CTA / 32) * 32 * TMEM_DP_STRIDE;

  float reg_buf[MMA_N_T];

  // ALL threads execute warp-level TMEM load (no divergence allowed)
  uint32_t dp_base = tmem_addr + tmem_cta_offset + warp_id * 32 * TMEM_DP_STRIDE;
  for (int col = 0; col < MMA_N_T; col += 8) {
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

  // Only valid threads write to GMEM
  if (tid < ROWS_PER_WARP_2SM) {
    int m = cta_rank * M_PER_CTA + local_m;
    for (int col = 0; col < MMA_N_T; ++col) {
      gD[m * MMA_N_T + col] = reg_buf[col];
    }
  }

  if (threadIdx.x == 0 && is_leader) {
    gCycles[0]     = t_end - t_start;
    gFillCycles[0] = t_fill_end - t_fill_start;
  }

  __syncthreads();
  cute::cluster_sync();

  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(*tmem_base, 512);
  }
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED (fp4 2SM kernel)

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

// FP4 E2M1 to float conversion: 1 sign + 2 exponent + 1 mantissa, bias=1
static float fp4_e2m1_to_float(uint8_t bits4) {
  int sign = (bits4 >> 3) & 1;
  int exp  = (bits4 >> 1) & 3;
  int mant = bits4 & 1;

  float val;
  if (exp == 0) {
    // Subnormal: 0.m * 2^(1-bias) = m * 0.5
    val = mant * 0.5f;
  } else {
    // Normal: (1 + m*0.5) * 2^(exp-bias)
    val = (1.0f + mant * 0.5f) * static_cast<float>(1 << (exp - 1));
  }
  return sign ? -val : val;
}

// UE4M3 (unsigned E4M3) to float conversion for scale factors
static float ue4m3_to_float(uint8_t bits) {
  // Use cutlass conversion
  return float(cutlass::float_ue4m3_t::bitcast(bits));
}

// CPU reference GEMM for fp4 with block-scaled MMA:
//   C[m][n] = sum_k( SFA[m%32][k/VS] * A_fp4[m][k] * SFB[n%32][k/VS] * B_fp4[n][k] )
// A_packed/B_packed are byte-packed fp4: low nibble = even k, high nibble = odd k
void reference_gemm_fp4(
    uint8_t const* A_packed,  // [M][K/2] byte-packed fp4 (K-major)
    uint8_t const* B_packed,  // [N][K/2] byte-packed fp4 (K-major)
    uint8_t const* SFA,       // [32][K/VS] scale factors
    uint8_t const* SFB,       // [32][K/VS] scale factors
    float* C,                 // [M][N]
    int M, int N, int K, int VS)
{
  int num_sf = K / VS;
  int K_bytes = K / 2;

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        int sf_idx = k / VS;

        // Extract fp4 nibbles from packed bytes
        int byte_a = m * K_bytes + k / 2;
        int byte_b = n * K_bytes + k / 2;
        uint8_t nibble_a = (k % 2 == 0) ? (A_packed[byte_a] & 0x0F) : (A_packed[byte_a] >> 4);
        uint8_t nibble_b = (k % 2 == 0) ? (B_packed[byte_b] & 0x0F) : (B_packed[byte_b] >> 4);

        float fa = fp4_e2m1_to_float(nibble_a);
        float fb = fp4_e2m1_to_float(nibble_b);

        // Scale factors with m%32 / n%32 sharing (warpx4 broadcast)
        float sfa = ue4m3_to_float(SFA[(m % 32) * num_sf + sf_idx]);
        float sfb = ue4m3_to_float(SFB[(n % 32) * num_sf + sf_idx]);

        acc += sfa * fa * sfb * fb;
      }
      C[m * N + n] = acc;
    }
  }
}

// CPU reference GEMM for dense fp8: C[m][n] = sum_k(A[m][k] * B[n][k])
void reference_gemm_fp8(
    uint8_t const* A,  // [M][K] K-major, 1 byte per fp8 element
    uint8_t const* B,  // [N][K] K-major
    float* C,          // [M][N]
    int M, int N, int K)
{
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        float fa = float(cutlass::float_e4m3_t::bitcast(A[m * K + k]));
        float fb = float(cutlass::float_e4m3_t::bitcast(B[n * K + k]));
        acc += fa * fb;
      }
      C[m * N + n] = acc;
    }
  }
}

// CPU reference GEMM for MN-major bf16: A is [K][M] M-contiguous, B is [K][N] N-contiguous
void reference_gemm_bf16_mn(
    cutlass::bfloat16_t const* A,  // [K][M] M-contiguous
    cutlass::bfloat16_t const* B,  // [K][N] N-contiguous
    float* C,                      // [M][N]
    int M, int N, int K)
{
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += float(A[k * M + m]) * float(B[k * N + n]);
      }
      C[m * N + n] = acc;
    }
  }
}

// CPU reference GEMM for MN-major fp8: A is [K][M] M-contiguous, B is [K][N] N-contiguous
void reference_gemm_fp8_mn(
    uint8_t const* A,  // [K][M] M-contiguous, 1 byte per fp8
    uint8_t const* B,  // [K][N] N-contiguous
    float* C,          // [M][N]
    int M, int N, int K)
{
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        float fa = float(cutlass::float_e4m3_t::bitcast(A[k * M + m]));
        float fb = float(cutlass::float_e4m3_t::bitcast(B[k * N + n]));
        acc += fa * fb;
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

// Compute descriptor parameters for fp4 data (byte-addressed, 4-bit elements)
void compute_desc_params_fp4(SwizzleMode mode, int k_per_stage,
                              uint8_t& layout_type, uint16_t& lbo, uint16_t& sbo) {
  layout_type = swizzle_mode_to_layout_type(mode);
  int row_bytes = k_per_stage / 2;  // fp4: 2 elements per byte
  switch (mode) {
    case SwizzleMode::SW_128B:
    case SwizzleMode::SW_64B:
    case SwizzleMode::SW_32B:
      lbo = 1;
      sbo = 8 * row_bytes / 16;     // 8 rows x row_bytes / 16 bytes per uint128_t
      break;
    case SwizzleMode::SW_NONE:
      lbo = 8;                       // 8 rows interleaved
      sbo = 8 * row_bytes / 16;     // same formula
      break;
  }
}

// Compute descriptor parameters for fp8 data (1 byte per element)
void compute_desc_params_fp8(SwizzleMode mode, int k_per_stage,
                              uint8_t& layout_type, uint16_t& lbo, uint16_t& sbo) {
  layout_type = swizzle_mode_to_layout_type(mode);
  int row_bytes = k_per_stage;  // fp8: 1 byte per element
  switch (mode) {
    case SwizzleMode::SW_128B:
    case SwizzleMode::SW_64B:
    case SwizzleMode::SW_32B:
      lbo = 1;
      sbo = 8 * row_bytes / 16;  // 8 * 256 / 16 = 128
      break;
    case SwizzleMode::SW_NONE:
      lbo = 8;                   // 8 rows interleaved (matches swizzle_store_byte interleave)
      sbo = 8 * row_bytes / 16;  // same
      break;
  }
}

// Compute descriptor parameters for MN-major bf16 data.
// mn_size is the MN dimension (M for A descriptors, N for B descriptors).
void compute_desc_params_mn(SwizzleMode mode, int mn_size,
                             uint8_t& layout_type, uint16_t& lbo, uint16_t& sbo) {
  layout_type = swizzle_mode_to_layout_type(mode);
  // For MN-major bf16:
  // LBO = atom MN width in elements (swizzled), or mn_size (interleave)
  // SBO = 8 K-rows * mn_size * sizeof(bf16) / 16 = mn_size
  switch (mode) {
    case SwizzleMode::SW_128B:
      lbo = 64;        // atom MN width = 64 bf16 elements
      sbo = mn_size;   // 8 * mn_size * 2 / 16
      break;
    case SwizzleMode::SW_64B:
      lbo = 32;
      sbo = mn_size;
      break;
    case SwizzleMode::SW_32B:
      lbo = 16;
      sbo = mn_size;
      break;
    case SwizzleMode::SW_NONE:
      lbo = mn_size;   // stride across 8-K-row groups
      sbo = 8;         // 8 K-rows interleaved
      break;
  }
}

// Compute descriptor parameters for MN-major fp8 data.
void compute_desc_params_mn_fp8(SwizzleMode mode, int mn_size,
                                 uint8_t& layout_type, uint16_t& lbo, uint16_t& sbo) {
  layout_type = swizzle_mode_to_layout_type(mode);
  // For MN-major fp8 (1 byte/element):
  // SBO = 8 * mn_size * 1 / 16 = mn_size / 2
  switch (mode) {
    case SwizzleMode::SW_128B:
      lbo = 64;
      sbo = mn_size / 2;
      break;
    case SwizzleMode::SW_64B:
      lbo = 32;
      sbo = mn_size / 2;
      break;
    case SwizzleMode::SW_32B:
      lbo = 16;
      sbo = mn_size / 2;
      break;
    case SwizzleMode::SW_NONE:
      lbo = mn_size / 2;
      sbo = 8;
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

template <int K_PER_STAGE_T, int MMA_M_T = 128, int MMA_N_T = 256>
bool run_correctness_test_k(SwizzleMode mode) {
  printf("=== Correctness test: %s, K=%d, MMA=%dx%dx%d ===\n",
         swizzle_mode_name(mode), K_PER_STAGE_T, MMA_M_T, MMA_N_T, MMA_K);

  constexpr int K = K_PER_STAGE_T;
  int M = MMA_M_T, N = MMA_N_T;
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
  constexpr int A_BYTES = MMA_M_T * K_PER_STAGE_T * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int B_BYTES = MMA_N_T * K_PER_STAGE_T * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int A_ALLOC = (A_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC = (B_BYTES + 1023) & ~1023;
  constexpr int SMEM_SIZE = A_ALLOC + B_ALLOC + 256;
  auto kernel = mma_swizzle_benchmark_kernel<1, 0, K_PER_STAGE_T, 0, MMA_M_T, MMA_N_T>;

  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

  dim3 grid(1);
  dim3 block(128);

  kernel<<<grid, block, SMEM_SIZE>>>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, lbo, sbo, swizzle_mask, k_iters);
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

template <int NUM_STAGES, int WAIT_PATTERN, int K_PER_STAGE_T, int MMA_M_T = 128, int MMA_N_T = 256>
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
  constexpr int A_BYTES = MMA_M_T * K_PER_STAGE_T * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int B_BYTES = MMA_N_T * K_PER_STAGE_T * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int A_ALLOC = (A_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC = (B_BYTES + 1023) & ~1023;
  constexpr int STAGE_ALLOC = A_ALLOC + B_ALLOC;
  int smem_size = NUM_STAGES * STAGE_ALLOC + 256;

  auto kernel = mma_swizzle_benchmark_kernel<NUM_STAGES, WAIT_PATTERN, K_PER_STAGE_T, 0, MMA_M_T, MMA_N_T>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  dim3 grid(1);
  dim3 block(128);

  // Warmup
  kernel<<<grid, block, smem_size>>>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, lbo, sbo, swizzle_mask, k_iters);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark run with cudaEvent wall-clock timing
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  CUDA_CHECK(cudaEventRecord(ev_start));
  kernel<<<grid, block, smem_size>>>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, lbo, sbo, swizzle_mask, k_iters);
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

void run_performance_sweep(int clock_rate_khz, int k_iters, FILE* csv_fp) {
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

        double cyc_per_mma = (double)result.mma_cycles / total_mmas;
        double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

        printf("%5d  %4d  %-8s  %-24s  %10ld  %8.1f  %11.1f  %9.1f  %10ld\n",
               cfg.k_per_stage, cfg.num_stages,
               swizzle_mode_name(mode), wp_name,
               (long)result.mma_cycles,
               cyc_per_mma,
               latency_us,
               (double)result.wall_clock_us,
               (long)result.fill_cycles);

        if (csv_fp) {
          fprintf(csv_fp, "bf16,%dx%dx%d,%d,%d,%s,%s,%ld,%.1f,%.1f,%.1f,%ld\n",
                  MMA_M, MMA_N, MMA_K, cfg.k_per_stage, cfg.num_stages,
                  swizzle_mode_name(mode), wp_name,
                  (long)result.mma_cycles, cyc_per_mma, latency_us,
                  (double)result.wall_clock_us, (long)result.fill_cycles);
        }
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

///////////////////////////////////////////////////////////////////////////////////////////////////
// bf16 MN-major Correctness & Performance
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int K_PER_STAGE_T, int MMA_M_T = 128, int MMA_N_T = 256>
bool run_correctness_test_mn_k(SwizzleMode mode) {
  printf("=== Correctness test MN-major: %s, K=%d, MMA=%dx%dx%d ===\n",
         swizzle_mode_name(mode), K_PER_STAGE_T, MMA_M_T, MMA_N_T, MMA_K);

  constexpr int K = K_PER_STAGE_T;
  int M = MMA_M_T, N = MMA_N_T;
  int k_iters = 1;

  // Generate random MN-major data: A[K][M] and B[K][N]
  std::vector<cutlass::bfloat16_t> h_A(K * M), h_B(K * N);
  srand(42);
  for (int i = 0; i < K * M; ++i) {
    h_A[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));
  }
  for (int i = 0; i < K * N; ++i) {
    h_B[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));
  }

  // CPU reference with MN-major indexing
  std::vector<float> h_ref(M * N);
  reference_gemm_bf16_mn(h_A.data(), h_B.data(), h_ref.data(), M, N, K);

  // Allocate device memory
  cutlass::bfloat16_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, K * M * sizeof(cutlass::bfloat16_t)));
  CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(cutlass::bfloat16_t)));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), K * M * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, M * N * sizeof(float)));

  // Compute separate descriptor params for A (mn_size=M) and B (mn_size=N)
  uint8_t layout_type;
  uint16_t lbo_a, sbo_a, lbo_b, sbo_b;
  compute_desc_params_mn(mode, M, layout_type, lbo_a, sbo_a);
  compute_desc_params_mn(mode, N, layout_type, lbo_b, sbo_b);
  uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

  // Launch MN-major kernel: 1 stage, wait pattern 0, IS_MN_MAJOR=1
  constexpr int A_BYTES = MMA_M_T * K_PER_STAGE_T * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int B_BYTES = MMA_N_T * K_PER_STAGE_T * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int A_ALLOC = (A_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC = (B_BYTES + 1023) & ~1023;
  constexpr int SMEM_SIZE = A_ALLOC + B_ALLOC + 256;
  auto kernel = mma_swizzle_benchmark_kernel<1, 0, K_PER_STAGE_T, 1, MMA_M_T, MMA_N_T>;

  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

  dim3 grid(1);
  dim3 block(128);

  kernel<<<grid, block, SMEM_SIZE>>>(d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
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

bool run_correctness_test_mn(SwizzleMode mode, int k_per_stage) {
  switch (k_per_stage) {
    case 64:  return run_correctness_test_mn_k<64>(mode);
    case 128: return run_correctness_test_mn_k<128>(mode);
    case 256: return run_correctness_test_mn_k<256>(mode);
    default:
      printf("Unsupported K_PER_STAGE=%d for MN-major correctness test\n", k_per_stage);
      return false;
  }
}

template <int NUM_STAGES, int WAIT_PATTERN, int K_PER_STAGE_T, int MMA_M_T = 128, int MMA_N_T = 256>
BenchResult run_benchmark_config_mn(
    cutlass::bfloat16_t* d_A,
    cutlass::bfloat16_t* d_B,
    float* d_D,
    int64_t* d_cycles,
    int64_t* d_fill_cycles,
    uint8_t layout_type,
    uint16_t lbo_a, uint16_t sbo_a,
    uint16_t lbo_b, uint16_t sbo_b,
    uint32_t swizzle_mask,
    int k_iters)
{
  constexpr int A_BYTES = MMA_M_T * K_PER_STAGE_T * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int B_BYTES = MMA_N_T * K_PER_STAGE_T * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int A_ALLOC = (A_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC = (B_BYTES + 1023) & ~1023;
  constexpr int STAGE_ALLOC = A_ALLOC + B_ALLOC;
  int smem_size = NUM_STAGES * STAGE_ALLOC + 256;

  auto kernel = mma_swizzle_benchmark_kernel<NUM_STAGES, WAIT_PATTERN, K_PER_STAGE_T, 1, MMA_M_T, MMA_N_T>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  dim3 grid(1);
  dim3 block(128);

  // Warmup
  kernel<<<grid, block, smem_size>>>(d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark run with cudaEvent wall-clock timing
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  CUDA_CHECK(cudaEventRecord(ev_start));
  kernel<<<grid, block, smem_size>>>(d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
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

void run_performance_sweep_mn(int clock_rate_khz, int k_iters, FILE* csv_fp) {
  printf("=== bf16 MN-major Descriptor Configuration Sweep ===\n");
  printf("  MMA shape: %dx%dx%d (bf16 MN-major), all configs: K_PER_STAGE x stages = 256\n",
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

    int total_A_elems = cfg.num_stages * M * cfg.k_per_stage;
    int total_B_elems = cfg.num_stages * N * cfg.k_per_stage;

    cutlass::bfloat16_t *d_A, *d_B;
    float *d_D;
    int64_t *d_cycles, *d_fill_cycles;

    CUDA_CHECK(cudaMalloc(&d_A, total_A_elems * sizeof(cutlass::bfloat16_t)));
    CUDA_CHECK(cudaMalloc(&d_B, total_B_elems * sizeof(cutlass::bfloat16_t)));
    CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

    // Generate MN-major data
    std::vector<cutlass::bfloat16_t> h_A(total_A_elems), h_B(total_B_elems);
    srand(123);
    for (int i = 0; i < total_A_elems; ++i) h_A[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));
    for (int i = 0; i < total_B_elems; ++i) h_B[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), total_A_elems * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), total_B_elems * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));

    for (SwizzleMode mode : modes) {
      uint8_t layout_type;
      uint16_t lbo_a, sbo_a, lbo_b, sbo_b;
      compute_desc_params_mn(mode, M, layout_type, lbo_a, sbo_a);
      compute_desc_params_mn(mode, N, layout_type, lbo_b, sbo_b);
      uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

      int k_blocks = cfg.k_per_stage / MMA_K;
      int total_mmas = cfg.num_stages * k_blocks * k_iters;

      auto run_for_pattern = [&](int wp_id, const char* wp_name) {
        BenchResult result = {};
        int ns = cfg.num_stages;
        int kps = cfg.k_per_stage;

        #define DISPATCH_BENCH_MN(NS, WP, KPS) \
          if (ns == NS && wp_id == WP && kps == KPS) { \
            result = run_benchmark_config_mn<NS, WP, KPS>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters); \
          }

        DISPATCH_BENCH_MN(4, 0, 64)  DISPATCH_BENCH_MN(4, 1, 64)  DISPATCH_BENCH_MN(4, 2, 64)
        DISPATCH_BENCH_MN(2, 0, 128) DISPATCH_BENCH_MN(2, 1, 128) DISPATCH_BENCH_MN(2, 2, 128)
        DISPATCH_BENCH_MN(1, 0, 256) DISPATCH_BENCH_MN(1, 1, 256) DISPATCH_BENCH_MN(1, 2, 256)
        #undef DISPATCH_BENCH_MN

        double cyc_per_mma = (double)result.mma_cycles / total_mmas;
        double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

        printf("%5d  %4d  %-8s  %-24s  %10ld  %8.1f  %11.1f  %9.1f  %10ld\n",
               cfg.k_per_stage, cfg.num_stages,
               swizzle_mode_name(mode), wp_name,
               (long)result.mma_cycles,
               cyc_per_mma,
               latency_us,
               (double)result.wall_clock_us,
               (long)result.fill_cycles);

        if (csv_fp) {
          fprintf(csv_fp, "bf16_mn,%dx%dx%d,%d,%d,%s,%s,%ld,%.1f,%.1f,%.1f,%ld\n",
                  MMA_M, MMA_N, MMA_K, cfg.k_per_stage, cfg.num_stages,
                  swizzle_mode_name(mode), wp_name,
                  (long)result.mma_cycles, cyc_per_mma, latency_us,
                  (double)result.wall_clock_us, (long)result.fill_cycles);
        }
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
// nvfp4 Correctness & Performance
///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <int MMA_M_T = 128, int MMA_N_T = 256>
bool run_fp4_correctness_test(SwizzleMode mode) {
  printf("=== nvfp4 Correctness test: %s, K=%d, MMA=%dx%dx%d ===\n",
         swizzle_mode_name(mode), FP4_K_PER_STAGE, MMA_M_T, MMA_N_T, FP4_MMA_K);

  int M = MMA_M_T, N = MMA_N_T, K = FP4_K_PER_STAGE;
  int K_bytes = K / 2;
  int k_iters = 1;

  // Generate random byte-packed fp4 data (each byte holds 2 fp4 values)
  // Use nibble values 0-7 (positive fp4 values: 0, 0.5, 1, 1.5, 2, 3, 4, 6)
  std::vector<uint8_t> h_A(M * K_bytes), h_B(N * K_bytes);
  srand(42);
  for (auto& v : h_A) v = static_cast<uint8_t>(rand() & 0x77);  // both nibbles in [0,7]
  for (auto& v : h_B) v = static_cast<uint8_t>(rand() & 0x77);

  // Generate scale factors (use small values to keep products reasonable)
  // UE4M3 value 0x38 = 1.0, 0x30 = 0.5, 0x3C = 1.5
  std::vector<uint8_t> h_SFA(32 * FP4_NUM_SF), h_SFB(32 * FP4_NUM_SF);
  uint8_t sf_vals[] = {0x38, 0x30, 0x3C, 0x34};  // ~1.0, ~0.5, ~1.5, ~0.75
  for (size_t i = 0; i < h_SFA.size(); ++i) h_SFA[i] = sf_vals[i % 4];
  for (size_t i = 0; i < h_SFB.size(); ++i) h_SFB[i] = sf_vals[(i + 1) % 4];

  // CPU reference
  std::vector<float> h_ref(M * N);
  reference_gemm_fp4(h_A.data(), h_B.data(), h_SFA.data(), h_SFB.data(),
                     h_ref.data(), M, N, K, FP4_VS);

  // Allocate device memory
  uint8_t *d_A, *d_B, *d_SFA, *d_SFB;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, M * K_bytes));
  CUDA_CHECK(cudaMalloc(&d_B, N * K_bytes));
  CUDA_CHECK(cudaMalloc(&d_SFA, 32 * FP4_NUM_SF));
  CUDA_CHECK(cudaMalloc(&d_SFB, 32 * FP4_NUM_SF));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_SFA, h_SFA.data(), 32 * FP4_NUM_SF, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_SFB, h_SFB.data(), 32 * FP4_NUM_SF, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, M * N * sizeof(float)));

  // Compute descriptor params
  uint8_t layout_type;
  uint16_t lbo, sbo;
  compute_desc_params_fp4(mode, K, layout_type, lbo, sbo);
  uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

  // SMEM size
  constexpr int A_ALLOC  = (MMA_M_T * FP4_K_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC  = (MMA_N_T * FP4_K_BYTES + 1023) & ~1023;
  constexpr int SF_ALLOC = (32 * FP4_NUM_SF + 1023) & ~1023;
  constexpr int SMEM_SIZE = A_ALLOC + B_ALLOC + SF_ALLOC + SF_ALLOC + 256;

  auto kernel = fp4_mma_swizzle_benchmark_kernel<0, MMA_M_T, MMA_N_T>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

  dim3 grid(1);
  dim3 block(128);

  kernel<<<grid, block, SMEM_SIZE>>>(
      d_A, d_B, d_SFA, d_SFB, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo, sbo, swizzle_mask, k_iters);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Read back results
  std::vector<float> h_D(M * N);
  CUDA_CHECK(cudaMemcpy(h_D.data(), d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // Compare with relaxed tolerance (fp4 is low precision)
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
      if (rel > 0.1f && err > 1.0f) {
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
  CUDA_CHECK(cudaFree(d_SFA));
  CUDA_CHECK(cudaFree(d_SFB));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_cycles));
  CUDA_CHECK(cudaFree(d_fill_cycles));

  return pass;
}

template <int WAIT_PATTERN, int MMA_M_T = 128, int MMA_N_T = 256>
BenchResult run_fp4_benchmark_config(
    uint8_t* d_A,
    uint8_t* d_B,
    uint8_t* d_SFA,
    uint8_t* d_SFB,
    float* d_D,
    int64_t* d_cycles,
    int64_t* d_fill_cycles,
    uint8_t layout_type,
    uint16_t lbo,
    uint16_t sbo,
    uint32_t swizzle_mask,
    int k_iters)
{
  constexpr int A_ALLOC  = (MMA_M_T * FP4_K_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC  = (MMA_N_T * FP4_K_BYTES + 1023) & ~1023;
  constexpr int SF_ALLOC = (32 * FP4_NUM_SF + 1023) & ~1023;
  int smem_size = A_ALLOC + B_ALLOC + SF_ALLOC + SF_ALLOC + 256;

  auto kernel = fp4_mma_swizzle_benchmark_kernel<WAIT_PATTERN, MMA_M_T, MMA_N_T>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  dim3 grid(1);
  dim3 block(128);

  // Warmup
  kernel<<<grid, block, smem_size>>>(
      d_A, d_B, d_SFA, d_SFB, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo, sbo, swizzle_mask, k_iters);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark run with wall-clock timing
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  CUDA_CHECK(cudaEventRecord(ev_start));
  kernel<<<grid, block, smem_size>>>(
      d_A, d_B, d_SFA, d_SFB, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo, sbo, swizzle_mask, k_iters);
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

void run_fp4_performance_sweep(int clock_rate_khz, int k_iters, FILE* csv_fp) {
  printf("=== nvfp4 (mxf4nvf4) Descriptor Configuration Sweep ===\n");
  printf("  MMA shape: %dx%dx%d (fp4, block16), K_PER_STAGE=%d (1 stage)\n",
         FP4_MMA_M, FP4_MMA_N, FP4_MMA_K, FP4_K_PER_STAGE);
  printf("  k_iters: %d, total MMAs per row: %d x %d = %d\n\n",
         k_iters, FP4_K_BLOCKS, k_iters, FP4_K_BLOCKS * k_iters);

  SwizzleMode modes[] = {
    SwizzleMode::SW_NONE, SwizzleMode::SW_32B, SwizzleMode::SW_64B, SwizzleMode::SW_128B
  };

  printf("%8s  %-24s  %10s  %8s  %11s  %9s  %10s\n",
         "Swizzle", "Wait Pattern", "Cycles", "Cyc/MMA", "Latency(us)", "Wall(us)", "Fill_Cyc");
  printf("%8s  %-24s  %10s  %8s  %11s  %9s  %10s\n",
         "--------", "------------------------", "----------", "--------", "-----------", "---------", "----------");

  int M = FP4_MMA_M, N = FP4_MMA_N;

  // Allocate device memory
  uint8_t *d_A, *d_B, *d_SFA, *d_SFB;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, M * FP4_K_BYTES));
  CUDA_CHECK(cudaMalloc(&d_B, N * FP4_K_BYTES));
  CUDA_CHECK(cudaMalloc(&d_SFA, 32 * FP4_NUM_SF));
  CUDA_CHECK(cudaMalloc(&d_SFB, 32 * FP4_NUM_SF));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  // Fill with random data
  std::vector<uint8_t> h_A(M * FP4_K_BYTES), h_B(N * FP4_K_BYTES);
  std::vector<uint8_t> h_SFA(32 * FP4_NUM_SF, 0x38), h_SFB(32 * FP4_NUM_SF, 0x38);  // SF=1.0
  srand(123);
  for (auto& v : h_A) v = static_cast<uint8_t>(rand() & 0x77);
  for (auto& v : h_B) v = static_cast<uint8_t>(rand() & 0x77);

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * FP4_K_BYTES, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * FP4_K_BYTES, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_SFA, h_SFA.data(), 32 * FP4_NUM_SF, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_SFB, h_SFB.data(), 32 * FP4_NUM_SF, cudaMemcpyHostToDevice));

  for (SwizzleMode mode : modes) {
    uint8_t layout_type;
    uint16_t lbo, sbo;
    compute_desc_params_fp4(mode, FP4_K_PER_STAGE, layout_type, lbo, sbo);
    uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

    int total_mmas = FP4_K_BLOCKS * k_iters;

    auto run_for_pattern = [&](int wp_id, const char* wp_name) {
      BenchResult result = {};

      if (wp_id == 0)
        result = run_fp4_benchmark_config<0>(d_A, d_B, d_SFA, d_SFB, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else if (wp_id == 1)
        result = run_fp4_benchmark_config<1>(d_A, d_B, d_SFA, d_SFB, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else
        result = run_fp4_benchmark_config<2>(d_A, d_B, d_SFA, d_SFB, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);

      double cyc_per_mma = (double)result.mma_cycles / total_mmas;
      double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

      printf("%-8s  %-24s  %10ld  %8.1f  %11.1f  %9.1f  %10ld\n",
             swizzle_mode_name(mode), wp_name,
             (long)result.mma_cycles,
             cyc_per_mma,
             latency_us,
             (double)result.wall_clock_us,
             (long)result.fill_cycles);

      if (csv_fp) {
        fprintf(csv_fp, "nvfp4,%dx%dx%d,%d,%d,%s,%s,%ld,%.1f,%.1f,%.1f,%ld\n",
                FP4_MMA_M, FP4_MMA_N, FP4_MMA_K, FP4_K_PER_STAGE, 1,
                swizzle_mode_name(mode), wp_name,
                (long)result.mma_cycles, cyc_per_mma, latency_us,
                (double)result.wall_clock_us, (long)result.fill_cycles);
      }
    };

    run_for_pattern(0, "commit+wait each");
    run_for_pattern(1, "commit each, wait end");
    run_for_pattern(2, "commit+wait end");
  }

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_SFA));
  CUDA_CHECK(cudaFree(d_SFB));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_cycles));
  CUDA_CHECK(cudaFree(d_fill_cycles));

  printf("\n");
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED (fp4 tests)

///////////////////////////////////////////////////////////////////////////////////////////////////
// fp8 Correctness & Performance
///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <int MMA_M_T = 128, int MMA_N_T = 256>
bool run_fp8_correctness_test(SwizzleMode mode) {
  printf("=== fp8 Correctness test: %s, K=%d, MMA=%dx%dx%d ===\n",
         swizzle_mode_name(mode), FP8_K_PER_STAGE, MMA_M_T, MMA_N_T, FP8_MMA_K);

  int M = MMA_M_T, N = MMA_N_T, K = FP8_K_PER_STAGE;
  int k_iters = 1;

  // Generate random fp8 data: values in [-2, 2] range
  std::vector<uint8_t> h_A(M * K), h_B(N * K);
  srand(42);
  for (auto& v : h_A) {
    float f = float(rand() % 5 - 2);
    v = cutlass::float_e4m3_t(f).storage;
  }
  for (auto& v : h_B) {
    float f = float(rand() % 5 - 2);
    v = cutlass::float_e4m3_t(f).storage;
  }

  // CPU reference
  std::vector<float> h_ref(M * N);
  reference_gemm_fp8(h_A.data(), h_B.data(), h_ref.data(), M, N, K);

  // Allocate device memory
  uint8_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, M * K));
  CUDA_CHECK(cudaMalloc(&d_B, N * K));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, M * N * sizeof(float)));

  // Compute descriptor params
  uint8_t layout_type;
  uint16_t lbo, sbo;
  compute_desc_params_fp8(mode, K, layout_type, lbo, sbo);
  uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

  // SMEM size
  constexpr int A_ALLOC = (MMA_M_T * FP8_K_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC = (MMA_N_T * FP8_K_BYTES + 1023) & ~1023;
  constexpr int SMEM_SIZE = A_ALLOC + B_ALLOC + 256;

  auto kernel = fp8_mma_swizzle_benchmark_kernel<0, 0, MMA_M_T, MMA_N_T>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

  dim3 grid(1);
  dim3 block(128);

  kernel<<<grid, block, SMEM_SIZE>>>(
      d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo, sbo, lbo, sbo, swizzle_mask, k_iters);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Read back results
  std::vector<float> h_D(M * N);
  CUDA_CHECK(cudaMemcpy(h_D.data(), d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // Compare with tolerance appropriate for fp8 precision
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

template <int WAIT_PATTERN, int MMA_M_T = 128, int MMA_N_T = 256>
BenchResult run_fp8_benchmark_config(
    uint8_t* d_A,
    uint8_t* d_B,
    float* d_D,
    int64_t* d_cycles,
    int64_t* d_fill_cycles,
    uint8_t layout_type,
    uint16_t lbo,
    uint16_t sbo,
    uint32_t swizzle_mask,
    int k_iters)
{
  constexpr int A_ALLOC = (MMA_M_T * FP8_K_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC = (MMA_N_T * FP8_K_BYTES + 1023) & ~1023;
  int smem_size = A_ALLOC + B_ALLOC + 256;

  auto kernel = fp8_mma_swizzle_benchmark_kernel<WAIT_PATTERN, 0, MMA_M_T, MMA_N_T>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  dim3 grid(1);
  dim3 block(128);

  // Warmup
  kernel<<<grid, block, smem_size>>>(
      d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo, sbo, lbo, sbo, swizzle_mask, k_iters);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark run with wall-clock timing
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  CUDA_CHECK(cudaEventRecord(ev_start));
  kernel<<<grid, block, smem_size>>>(
      d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo, sbo, lbo, sbo, swizzle_mask, k_iters);
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

void run_fp8_performance_sweep(int clock_rate_khz, int k_iters, FILE* csv_fp) {
  printf("=== fp8 (f8f6f4) Descriptor Configuration Sweep ===\n");
  printf("  MMA shape: %dx%dx%d (fp8, dense), K_PER_STAGE=%d (1 stage)\n",
         FP8_MMA_M, FP8_MMA_N, FP8_MMA_K, FP8_K_PER_STAGE);
  printf("  k_iters: %d, total MMAs per row: %d x %d = %d\n\n",
         k_iters, FP8_K_BLOCKS, k_iters, FP8_K_BLOCKS * k_iters);

  SwizzleMode modes[] = {
    SwizzleMode::SW_NONE, SwizzleMode::SW_32B, SwizzleMode::SW_64B, SwizzleMode::SW_128B
  };

  printf("%8s  %-24s  %10s  %8s  %11s  %9s  %10s\n",
         "Swizzle", "Wait Pattern", "Cycles", "Cyc/MMA", "Latency(us)", "Wall(us)", "Fill_Cyc");
  printf("%8s  %-24s  %10s  %8s  %11s  %9s  %10s\n",
         "--------", "------------------------", "----------", "--------", "-----------", "---------", "----------");

  int M = FP8_MMA_M, N = FP8_MMA_N;

  // Allocate device memory
  uint8_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, M * FP8_K_BYTES));
  CUDA_CHECK(cudaMalloc(&d_B, N * FP8_K_BYTES));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  // Fill with random data
  std::vector<uint8_t> h_A(M * FP8_K_BYTES), h_B(N * FP8_K_BYTES);
  srand(123);
  for (auto& v : h_A) v = static_cast<uint8_t>(rand() & 0x7F);  // positive fp8 values
  for (auto& v : h_B) v = static_cast<uint8_t>(rand() & 0x7F);

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * FP8_K_BYTES, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * FP8_K_BYTES, cudaMemcpyHostToDevice));

  for (SwizzleMode mode : modes) {
    uint8_t layout_type;
    uint16_t lbo, sbo;
    compute_desc_params_fp8(mode, FP8_K_PER_STAGE, layout_type, lbo, sbo);
    uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

    int total_mmas = FP8_K_BLOCKS * k_iters;

    auto run_for_pattern = [&](int wp_id, const char* wp_name) {
      BenchResult result = {};

      if (wp_id == 0)
        result = run_fp8_benchmark_config<0>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else if (wp_id == 1)
        result = run_fp8_benchmark_config<1>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else
        result = run_fp8_benchmark_config<2>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);

      double cyc_per_mma = (double)result.mma_cycles / total_mmas;
      double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

      printf("%-8s  %-24s  %10ld  %8.1f  %11.1f  %9.1f  %10ld\n",
             swizzle_mode_name(mode), wp_name,
             (long)result.mma_cycles,
             cyc_per_mma,
             latency_us,
             (double)result.wall_clock_us,
             (long)result.fill_cycles);

      if (csv_fp) {
        fprintf(csv_fp, "fp8,%dx%dx%d,%d,%d,%s,%s,%ld,%.1f,%.1f,%.1f,%ld\n",
                FP8_MMA_M, FP8_MMA_N, FP8_MMA_K, FP8_K_PER_STAGE, 1,
                swizzle_mode_name(mode), wp_name,
                (long)result.mma_cycles, cyc_per_mma, latency_us,
                (double)result.wall_clock_us, (long)result.fill_cycles);
      }
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

  printf("\n");
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// fp8 MN-major Correctness & Performance
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMA_M_T = 128, int MMA_N_T = 256>
bool run_fp8_correctness_test_mn(SwizzleMode mode) {
  printf("=== fp8 MN-major Correctness test: %s, K=%d, MMA=%dx%dx%d ===\n",
         swizzle_mode_name(mode), FP8_K_PER_STAGE, MMA_M_T, MMA_N_T, FP8_MMA_K);

  int M = MMA_M_T, N = MMA_N_T, K = FP8_K_PER_STAGE;
  int k_iters = 1;

  // Generate random fp8 MN-major data: A[K][M] and B[K][N]
  std::vector<uint8_t> h_A(K * M), h_B(K * N);
  srand(42);
  for (auto& v : h_A) {
    float f = float(rand() % 5 - 2);
    v = cutlass::float_e4m3_t(f).storage;
  }
  for (auto& v : h_B) {
    float f = float(rand() % 5 - 2);
    v = cutlass::float_e4m3_t(f).storage;
  }

  // CPU reference with MN-major indexing
  std::vector<float> h_ref(M * N);
  reference_gemm_fp8_mn(h_A.data(), h_B.data(), h_ref.data(), M, N, K);

  // Allocate device memory
  uint8_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, K * M));
  CUDA_CHECK(cudaMalloc(&d_B, K * N));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), K * M, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, M * N * sizeof(float)));

  // Compute separate descriptor params for A and B
  uint8_t layout_type;
  uint16_t lbo_a, sbo_a, lbo_b, sbo_b;
  compute_desc_params_mn_fp8(mode, M, layout_type, lbo_a, sbo_a);
  compute_desc_params_mn_fp8(mode, N, layout_type, lbo_b, sbo_b);
  uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

  // SMEM size (same as K-major — total bytes identical)
  constexpr int A_ALLOC = (MMA_M_T * FP8_K_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC = (MMA_N_T * FP8_K_BYTES + 1023) & ~1023;
  constexpr int SMEM_SIZE = A_ALLOC + B_ALLOC + 256;

  auto kernel = fp8_mma_swizzle_benchmark_kernel<0, 1, MMA_M_T, MMA_N_T>;  // IS_MN_MAJOR=1
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

  dim3 grid(1);
  dim3 block(128);

  kernel<<<grid, block, SMEM_SIZE>>>(
      d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
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

template <int WAIT_PATTERN, int MMA_M_T = 128, int MMA_N_T = 256>
BenchResult run_fp8_benchmark_config_mn(
    uint8_t* d_A,
    uint8_t* d_B,
    float* d_D,
    int64_t* d_cycles,
    int64_t* d_fill_cycles,
    uint8_t layout_type,
    uint16_t lbo_a, uint16_t sbo_a,
    uint16_t lbo_b, uint16_t sbo_b,
    uint32_t swizzle_mask,
    int k_iters)
{
  constexpr int A_ALLOC = (MMA_M_T * FP8_K_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC = (MMA_N_T * FP8_K_BYTES + 1023) & ~1023;
  int smem_size = A_ALLOC + B_ALLOC + 256;

  auto kernel = fp8_mma_swizzle_benchmark_kernel<WAIT_PATTERN, 1, MMA_M_T, MMA_N_T>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  dim3 grid(1);
  dim3 block(128);

  // Warmup
  kernel<<<grid, block, smem_size>>>(
      d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark run with wall-clock timing
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  CUDA_CHECK(cudaEventRecord(ev_start));
  kernel<<<grid, block, smem_size>>>(
      d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
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

void run_fp8_performance_sweep_mn(int clock_rate_khz, int k_iters, FILE* csv_fp) {
  printf("=== fp8 MN-major (f8f6f4) Descriptor Configuration Sweep ===\n");
  printf("  MMA shape: %dx%dx%d (fp8 MN-major, dense), K_PER_STAGE=%d (1 stage)\n",
         FP8_MMA_M, FP8_MMA_N, FP8_MMA_K, FP8_K_PER_STAGE);
  printf("  k_iters: %d, total MMAs per row: %d x %d = %d\n\n",
         k_iters, FP8_K_BLOCKS, k_iters, FP8_K_BLOCKS * k_iters);

  SwizzleMode modes[] = {
    SwizzleMode::SW_NONE, SwizzleMode::SW_32B, SwizzleMode::SW_64B, SwizzleMode::SW_128B
  };

  printf("%8s  %-24s  %10s  %8s  %11s  %9s  %10s\n",
         "Swizzle", "Wait Pattern", "Cycles", "Cyc/MMA", "Latency(us)", "Wall(us)", "Fill_Cyc");
  printf("%8s  %-24s  %10s  %8s  %11s  %9s  %10s\n",
         "--------", "------------------------", "----------", "--------", "-----------", "---------", "----------");

  int M = FP8_MMA_M, N = FP8_MMA_N, K = FP8_K_PER_STAGE;

  // Allocate device memory for MN-major data
  uint8_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, K * M));
  CUDA_CHECK(cudaMalloc(&d_B, K * N));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  // Fill with random data
  std::vector<uint8_t> h_A(K * M), h_B(K * N);
  srand(123);
  for (auto& v : h_A) v = static_cast<uint8_t>(rand() & 0x7F);
  for (auto& v : h_B) v = static_cast<uint8_t>(rand() & 0x7F);

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), K * M, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N, cudaMemcpyHostToDevice));

  for (SwizzleMode mode : modes) {
    uint8_t layout_type;
    uint16_t lbo_a, sbo_a, lbo_b, sbo_b;
    compute_desc_params_mn_fp8(mode, M, layout_type, lbo_a, sbo_a);
    compute_desc_params_mn_fp8(mode, N, layout_type, lbo_b, sbo_b);
    uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

    int total_mmas = FP8_K_BLOCKS * k_iters;

    auto run_for_pattern = [&](int wp_id, const char* wp_name) {
      BenchResult result = {};

      if (wp_id == 0)
        result = run_fp8_benchmark_config_mn<0>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
      else if (wp_id == 1)
        result = run_fp8_benchmark_config_mn<1>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
      else
        result = run_fp8_benchmark_config_mn<2>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);

      double cyc_per_mma = (double)result.mma_cycles / total_mmas;
      double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

      printf("%-8s  %-24s  %10ld  %8.1f  %11.1f  %9.1f  %10ld\n",
             swizzle_mode_name(mode), wp_name,
             (long)result.mma_cycles,
             cyc_per_mma,
             latency_us,
             (double)result.wall_clock_us,
             (long)result.fill_cycles);

      if (csv_fp) {
        fprintf(csv_fp, "fp8_mn,%dx%dx%d,%d,%d,%s,%s,%ld,%.1f,%.1f,%.1f,%ld\n",
                FP8_MMA_M, FP8_MMA_N, FP8_MMA_K, FP8_K_PER_STAGE, 1,
                swizzle_mode_name(mode), wp_name,
                (long)result.mma_cycles, cyc_per_mma, latency_us,
                (double)result.wall_clock_us, (long)result.fill_cycles);
      }
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

  printf("\n");
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED (fp8 tests)

///////////////////////////////////////////////////////////////////////////////////////////////////
// bf16 2SM Correctness & Performance
///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <int MMA_M_T = 256, int MMA_N_T = 256>
bool run_bf16_2sm_correctness_test(SwizzleMode mode) {
  printf("=== bf16 2SM Correctness test: %s, K=%d, MMA=%dx%dx%d ===\n",
         swizzle_mode_name(mode), BF16_2SM_K_PER_STAGE, MMA_M_T, MMA_N_T, BF16_2SM_MMA_K);

  int M = MMA_M_T, N = MMA_N_T, K = BF16_2SM_K_PER_STAGE;
  int k_iters = 1;

  // Generate random A[256 x K] and B[256 x K] in bf16, LINEAR K-major
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

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, M * N * sizeof(float)));

  // Compute descriptor params (reuse bf16 1SM params — same per-CTA SMEM layout)
  uint8_t layout_type;
  uint16_t lbo, sbo;
  compute_desc_params(mode, K, layout_type, lbo, sbo);
  uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

  // SMEM size per CTA
  constexpr int M_PER_CTA = MMA_M_T / 2;
  constexpr int N_PER_CTA = MMA_N_T / 2;
  constexpr int A_BYTES = M_PER_CTA * BF16_2SM_K_PER_STAGE * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int B_BYTES = N_PER_CTA * BF16_2SM_K_PER_STAGE * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int A_ALLOC = (A_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC = (B_BYTES + 1023) & ~1023;
  constexpr int SMEM_SIZE = A_ALLOC + B_ALLOC + 256;

  auto kernel = bf16_2sm_mma_swizzle_benchmark_kernel<0, 0, MMA_M_T, MMA_N_T>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

  // Cluster launch: 2 CTAs in a cluster
  cutlass::ClusterLaunchParams params;
  params.grid_dims = dim3(2);
  params.block_dims = dim3(128);
  params.cluster_dims = dim3(2, 1, 1);
  params.smem_size_in_bytes = SMEM_SIZE;

  void const* kernel_ptr = reinterpret_cast<void const*>(kernel);
  auto status = cutlass::launch_kernel_on_cluster(
      params, kernel_ptr,
      d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo, sbo, lbo, sbo, swizzle_mask, k_iters);

  if (status != cutlass::Status::kSuccess) {
    printf("  Cluster launch failed!\n");
    return false;
  }
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

template <int WAIT_PATTERN, int MMA_M_T = 256, int MMA_N_T = 256>
BenchResult run_bf16_2sm_benchmark_config(
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
  constexpr int M_PER_CTA = MMA_M_T / 2;
  constexpr int N_PER_CTA = MMA_N_T / 2;
  constexpr int A_BYTES = M_PER_CTA * BF16_2SM_K_PER_STAGE * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int B_BYTES = N_PER_CTA * BF16_2SM_K_PER_STAGE * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int A_ALLOC = (A_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC = (B_BYTES + 1023) & ~1023;
  int smem_size = A_ALLOC + B_ALLOC + 256;

  auto kernel = bf16_2sm_mma_swizzle_benchmark_kernel<WAIT_PATTERN, 0, MMA_M_T, MMA_N_T>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cutlass::ClusterLaunchParams params;
  params.grid_dims = dim3(2);
  params.block_dims = dim3(128);
  params.cluster_dims = dim3(2, 1, 1);
  params.smem_size_in_bytes = smem_size;

  void const* kernel_ptr = reinterpret_cast<void const*>(kernel);

  // Warmup
  cutlass::launch_kernel_on_cluster(
      params, kernel_ptr,
      d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo, sbo, lbo, sbo, swizzle_mask, k_iters);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark run with cudaEvent wall-clock timing
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  CUDA_CHECK(cudaEventRecord(ev_start));
  cutlass::launch_kernel_on_cluster(
      params, kernel_ptr,
      d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo, sbo, lbo, sbo, swizzle_mask, k_iters);
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

void run_bf16_2sm_performance_sweep(int clock_rate_khz, int k_iters, FILE* csv_fp) {
  printf("=== bf16 2SM (cta_group::2) Descriptor Configuration Sweep ===\n");
  printf("  MMA shape: %dx%dx%d (bf16, 2SM), K_PER_STAGE=%d (1 stage)\n",
         BF16_2SM_MMA_M, BF16_2SM_MMA_N, BF16_2SM_MMA_K, BF16_2SM_K_PER_STAGE);
  printf("  k_iters: %d, total MMAs per row: %d x %d = %d\n\n",
         k_iters, BF16_2SM_K_BLOCKS, k_iters, BF16_2SM_K_BLOCKS * k_iters);

  SwizzleMode modes[] = {
    SwizzleMode::SW_NONE, SwizzleMode::SW_32B, SwizzleMode::SW_64B, SwizzleMode::SW_128B
  };

  printf("%8s  %-24s  %10s  %8s  %11s  %9s  %10s\n",
         "Swizzle", "Wait Pattern", "Cycles", "Cyc/MMA", "Latency(us)", "Wall(us)", "Fill_Cyc");
  printf("%8s  %-24s  %10s  %8s  %11s  %9s  %10s\n",
         "--------", "------------------------", "----------", "--------", "-----------", "---------", "----------");

  int M = BF16_2SM_MMA_M, N = BF16_2SM_MMA_N, K = BF16_2SM_K_PER_STAGE;

  // Allocate device memory (full 256xK for both A and B)
  cutlass::bfloat16_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(cutlass::bfloat16_t)));
  CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(cutlass::bfloat16_t)));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  // Fill with random data
  std::vector<cutlass::bfloat16_t> h_A(M * K), h_B(N * K);
  srand(123);
  for (int i = 0; i < M * K; ++i) h_A[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));
  for (int i = 0; i < N * K; ++i) h_B[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));

  for (SwizzleMode mode : modes) {
    uint8_t layout_type;
    uint16_t lbo, sbo;
    compute_desc_params(mode, K, layout_type, lbo, sbo);
    uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

    int total_mmas = BF16_2SM_K_BLOCKS * k_iters;

    auto run_for_pattern = [&](int wp_id, const char* wp_name) {
      BenchResult result = {};

      if (wp_id == 0)
        result = run_bf16_2sm_benchmark_config<0>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else if (wp_id == 1)
        result = run_bf16_2sm_benchmark_config<1>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else
        result = run_bf16_2sm_benchmark_config<2>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);

      double cyc_per_mma = (double)result.mma_cycles / total_mmas;
      double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

      printf("%-8s  %-24s  %10ld  %8.1f  %11.1f  %9.1f  %10ld\n",
             swizzle_mode_name(mode), wp_name,
             (long)result.mma_cycles,
             cyc_per_mma,
             latency_us,
             (double)result.wall_clock_us,
             (long)result.fill_cycles);

      if (csv_fp) {
        fprintf(csv_fp, "bf16_2sm,%dx%dx%d,%d,%d,%s,%s,%ld,%.1f,%.1f,%.1f,%ld\n",
                BF16_2SM_MMA_M, BF16_2SM_MMA_N, BF16_2SM_MMA_K, BF16_2SM_K_PER_STAGE, 1,
                swizzle_mode_name(mode), wp_name,
                (long)result.mma_cycles, cyc_per_mma, latency_us,
                (double)result.wall_clock_us, (long)result.fill_cycles);
      }
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

  printf("\n");
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// bf16 2SM MN-major Correctness & Performance
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMA_M_T = 256, int MMA_N_T = 256>
bool run_bf16_2sm_correctness_test_mn(SwizzleMode mode) {
  printf("=== bf16 2SM MN-major Correctness test: %s, K=%d, MMA=%dx%dx%d ===\n",
         swizzle_mode_name(mode), BF16_2SM_K_PER_STAGE, MMA_M_T, MMA_N_T, BF16_2SM_MMA_K);

  int M = MMA_M_T, N = MMA_N_T, K = BF16_2SM_K_PER_STAGE;
  int M_PER_CTA = MMA_M_T / 2, N_PER_CTA = MMA_N_T / 2;
  int k_iters = 1;

  // Generate random MN-major data: A[K][M] and B[K][N]
  std::vector<cutlass::bfloat16_t> h_A_full(K * M), h_B_full(K * N);
  srand(42);
  for (int i = 0; i < K * M; ++i) {
    h_A_full[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));
  }
  for (int i = 0; i < K * N; ++i) {
    h_B_full[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));
  }

  // CPU reference with MN-major indexing
  std::vector<float> h_ref(M * N);
  reference_gemm_bf16_mn(h_A_full.data(), h_B_full.data(), h_ref.data(), M, N, K);

  // Split into per-CTA portions: [K][128] for CTA 0 and [K][128] for CTA 1
  std::vector<cutlass::bfloat16_t> h_A_split(K * M), h_B_split(K * N);
  for (int k = 0; k < K; ++k) {
    for (int m = 0; m < M_PER_CTA; ++m) {
      h_A_split[k * M_PER_CTA + m] = h_A_full[k * M + m];                         // CTA 0
      h_A_split[K * M_PER_CTA + k * M_PER_CTA + m] = h_A_full[k * M + M_PER_CTA + m]; // CTA 1
    }
    for (int n = 0; n < N_PER_CTA; ++n) {
      h_B_split[k * N_PER_CTA + n] = h_B_full[k * N + n];
      h_B_split[K * N_PER_CTA + k * N_PER_CTA + n] = h_B_full[k * N + N_PER_CTA + n];
    }
  }

  // Allocate device memory
  cutlass::bfloat16_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, K * M * sizeof(cutlass::bfloat16_t)));
  CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(cutlass::bfloat16_t)));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  CUDA_CHECK(cudaMemcpy(d_A, h_A_split.data(), K * M * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B_split.data(), K * N * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, M * N * sizeof(float)));

  // Compute descriptor params for MN-major (mn_size = M_PER_CTA for A, N_PER_CTA for B)
  uint8_t layout_type;
  uint16_t lbo_a, sbo_a, lbo_b, sbo_b;
  compute_desc_params_mn(mode, M_PER_CTA, layout_type, lbo_a, sbo_a);
  compute_desc_params_mn(mode, N_PER_CTA, layout_type, lbo_b, sbo_b);
  uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

  // SMEM size per CTA
  constexpr int A_BYTES_2 = (MMA_M_T / 2) * BF16_2SM_K_PER_STAGE * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int B_BYTES_2 = (MMA_N_T / 2) * BF16_2SM_K_PER_STAGE * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int A_ALLOC = (A_BYTES_2 + 1023) & ~1023;
  constexpr int B_ALLOC = (B_BYTES_2 + 1023) & ~1023;
  constexpr int SMEM_SIZE = A_ALLOC + B_ALLOC + 256;

  auto kernel = bf16_2sm_mma_swizzle_benchmark_kernel<0, 1, MMA_M_T, MMA_N_T>;  // IS_MN_MAJOR=1
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

  cutlass::ClusterLaunchParams params;
  params.grid_dims = dim3(2);
  params.block_dims = dim3(128);
  params.cluster_dims = dim3(2, 1, 1);
  params.smem_size_in_bytes = SMEM_SIZE;

  void const* kernel_ptr = reinterpret_cast<void const*>(kernel);
  auto status = cutlass::launch_kernel_on_cluster(
      params, kernel_ptr,
      d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);

  if (status != cutlass::Status::kSuccess) {
    printf("  Cluster launch failed!\n");
    return false;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Read back and compare
  std::vector<float> h_D(M * N);
  CUDA_CHECK(cudaMemcpy(h_D.data(), d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost));

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

template <int WAIT_PATTERN, int MMA_M_T = 256, int MMA_N_T = 256>
BenchResult run_bf16_2sm_benchmark_config_mn(
    cutlass::bfloat16_t* d_A,
    cutlass::bfloat16_t* d_B,
    float* d_D,
    int64_t* d_cycles,
    int64_t* d_fill_cycles,
    uint8_t layout_type,
    uint16_t lbo_a, uint16_t sbo_a,
    uint16_t lbo_b, uint16_t sbo_b,
    uint32_t swizzle_mask,
    int k_iters)
{
  constexpr int M_PER_CTA = MMA_M_T / 2;
  constexpr int N_PER_CTA = MMA_N_T / 2;
  constexpr int A_BYTES = M_PER_CTA * BF16_2SM_K_PER_STAGE * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int B_BYTES = N_PER_CTA * BF16_2SM_K_PER_STAGE * static_cast<int>(sizeof(cutlass::bfloat16_t));
  constexpr int A_ALLOC = (A_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC = (B_BYTES + 1023) & ~1023;
  int smem_size = A_ALLOC + B_ALLOC + 256;

  auto kernel = bf16_2sm_mma_swizzle_benchmark_kernel<WAIT_PATTERN, 1, MMA_M_T, MMA_N_T>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cutlass::ClusterLaunchParams params;
  params.grid_dims = dim3(2);
  params.block_dims = dim3(128);
  params.cluster_dims = dim3(2, 1, 1);
  params.smem_size_in_bytes = smem_size;

  void const* kernel_ptr = reinterpret_cast<void const*>(kernel);

  // Warmup
  cutlass::launch_kernel_on_cluster(
      params, kernel_ptr,
      d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark run
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  CUDA_CHECK(cudaEventRecord(ev_start));
  cutlass::launch_kernel_on_cluster(
      params, kernel_ptr,
      d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
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

void run_bf16_2sm_performance_sweep_mn(int clock_rate_khz, int k_iters, FILE* csv_fp) {
  printf("=== bf16 2SM MN-major (cta_group::2) Descriptor Configuration Sweep ===\n");
  printf("  MMA shape: %dx%dx%d (bf16 2SM MN-major), K_PER_STAGE=%d (1 stage)\n",
         BF16_2SM_MMA_M, BF16_2SM_MMA_N, BF16_2SM_MMA_K, BF16_2SM_K_PER_STAGE);
  printf("  k_iters: %d, total MMAs per row: %d x %d = %d\n\n",
         k_iters, BF16_2SM_K_BLOCKS, k_iters, BF16_2SM_K_BLOCKS * k_iters);

  SwizzleMode modes[] = {
    SwizzleMode::SW_NONE, SwizzleMode::SW_32B, SwizzleMode::SW_64B, SwizzleMode::SW_128B
  };

  printf("%8s  %-24s  %10s  %8s  %11s  %9s  %10s\n",
         "Swizzle", "Wait Pattern", "Cycles", "Cyc/MMA", "Latency(us)", "Wall(us)", "Fill_Cyc");
  printf("%8s  %-24s  %10s  %8s  %11s  %9s  %10s\n",
         "--------", "------------------------", "----------", "--------", "-----------", "---------", "----------");

  int M = BF16_2SM_MMA_M, N = BF16_2SM_MMA_N, K = BF16_2SM_K_PER_STAGE;
  int M_PER_CTA = BF16_2SM_M_PER_CTA, N_PER_CTA = BF16_2SM_N_PER_CTA;

  // Allocate device memory for split MN-major data
  cutlass::bfloat16_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, K * M * sizeof(cutlass::bfloat16_t)));
  CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(cutlass::bfloat16_t)));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  // Generate and split MN-major data into per-CTA portions
  std::vector<cutlass::bfloat16_t> h_A_full(K * M), h_B_full(K * N);
  srand(123);
  for (int i = 0; i < K * M; ++i) h_A_full[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));
  for (int i = 0; i < K * N; ++i) h_B_full[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));

  std::vector<cutlass::bfloat16_t> h_A_split(K * M), h_B_split(K * N);
  for (int k = 0; k < K; ++k) {
    for (int m = 0; m < M_PER_CTA; ++m) {
      h_A_split[k * M_PER_CTA + m] = h_A_full[k * M + m];
      h_A_split[K * M_PER_CTA + k * M_PER_CTA + m] = h_A_full[k * M + M_PER_CTA + m];
    }
    for (int n = 0; n < N_PER_CTA; ++n) {
      h_B_split[k * N_PER_CTA + n] = h_B_full[k * N + n];
      h_B_split[K * N_PER_CTA + k * N_PER_CTA + n] = h_B_full[k * N + N_PER_CTA + n];
    }
  }

  CUDA_CHECK(cudaMemcpy(d_A, h_A_split.data(), K * M * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B_split.data(), K * N * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));

  for (SwizzleMode mode : modes) {
    uint8_t layout_type;
    uint16_t lbo_a, sbo_a, lbo_b, sbo_b;
    compute_desc_params_mn(mode, M_PER_CTA, layout_type, lbo_a, sbo_a);
    compute_desc_params_mn(mode, N_PER_CTA, layout_type, lbo_b, sbo_b);
    uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

    int total_mmas = BF16_2SM_K_BLOCKS * k_iters;

    auto run_for_pattern = [&](int wp_id, const char* wp_name) {
      BenchResult result = {};

      if (wp_id == 0)
        result = run_bf16_2sm_benchmark_config_mn<0>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
      else if (wp_id == 1)
        result = run_bf16_2sm_benchmark_config_mn<1>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
      else
        result = run_bf16_2sm_benchmark_config_mn<2>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);

      double cyc_per_mma = (double)result.mma_cycles / total_mmas;
      double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

      printf("%-8s  %-24s  %10ld  %8.1f  %11.1f  %9.1f  %10ld\n",
             swizzle_mode_name(mode), wp_name,
             (long)result.mma_cycles,
             cyc_per_mma,
             latency_us,
             (double)result.wall_clock_us,
             (long)result.fill_cycles);

      if (csv_fp) {
        fprintf(csv_fp, "bf16_2sm_mn,%dx%dx%d,%d,%d,%s,%s,%ld,%.1f,%.1f,%.1f,%ld\n",
                BF16_2SM_MMA_M, BF16_2SM_MMA_N, BF16_2SM_MMA_K, BF16_2SM_K_PER_STAGE, 1,
                swizzle_mode_name(mode), wp_name,
                (long)result.mma_cycles, cyc_per_mma, latency_us,
                (double)result.wall_clock_us, (long)result.fill_cycles);
      }
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

  printf("\n");
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED (bf16 2SM tests)

///////////////////////////////////////////////////////////////////////////////////////////////////
// fp8 2SM Correctness & Performance
///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <int MMA_M_T = 256, int MMA_N_T = 256>
bool run_fp8_2sm_correctness_test(SwizzleMode mode) {
  printf("=== fp8 2SM Correctness test: %s, K=%d, MMA=%dx%dx%d ===\n",
         swizzle_mode_name(mode), FP8_2SM_K_PER_STAGE, MMA_M_T, MMA_N_T, FP8_2SM_MMA_K);

  int M = MMA_M_T, N = MMA_N_T, K = FP8_2SM_K_PER_STAGE;
  int k_iters = 1;

  // Generate random fp8 data: A[256][K] and B[256][K]
  std::vector<uint8_t> h_A(M * K), h_B(N * K);
  srand(42);
  for (auto& v : h_A) {
    float f = float(rand() % 5 - 2);
    v = cutlass::float_e4m3_t(f).storage;
  }
  for (auto& v : h_B) {
    float f = float(rand() % 5 - 2);
    v = cutlass::float_e4m3_t(f).storage;
  }

  // CPU reference
  std::vector<float> h_ref(M * N);
  reference_gemm_fp8(h_A.data(), h_B.data(), h_ref.data(), M, N, K);

  // Allocate device memory
  uint8_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, M * K));
  CUDA_CHECK(cudaMalloc(&d_B, N * K));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, M * N * sizeof(float)));

  // Compute descriptor params (reuse fp8 params — same per-CTA SMEM layout)
  uint8_t layout_type;
  uint16_t lbo, sbo;
  compute_desc_params_fp8(mode, K, layout_type, lbo, sbo);
  uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

  // SMEM size per CTA
  constexpr int M_PER_CTA = MMA_M_T / 2;
  constexpr int N_PER_CTA = MMA_N_T / 2;
  constexpr int A_BYTES = M_PER_CTA * FP8_2SM_K_PER_STAGE;
  constexpr int B_BYTES = N_PER_CTA * FP8_2SM_K_PER_STAGE;
  constexpr int A_ALLOC = (A_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC = (B_BYTES + 1023) & ~1023;
  constexpr int SMEM_SIZE = A_ALLOC + B_ALLOC + 256;

  auto kernel = fp8_2sm_mma_swizzle_benchmark_kernel<0, 0, MMA_M_T, MMA_N_T>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

  // Cluster launch: 2 CTAs in a cluster
  cutlass::ClusterLaunchParams params;
  params.grid_dims = dim3(2);
  params.block_dims = dim3(128);
  params.cluster_dims = dim3(2, 1, 1);
  params.smem_size_in_bytes = SMEM_SIZE;

  void const* kernel_ptr = reinterpret_cast<void const*>(kernel);
  auto status = cutlass::launch_kernel_on_cluster(
      params, kernel_ptr,
      d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo, sbo, lbo, sbo, swizzle_mask, k_iters);

  if (status != cutlass::Status::kSuccess) {
    printf("  Cluster launch failed!\n");
    return false;
  }
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

template <int WAIT_PATTERN, int MMA_M_T = 256, int MMA_N_T = 256>
BenchResult run_fp8_2sm_benchmark_config(
    uint8_t* d_A,
    uint8_t* d_B,
    float* d_D,
    int64_t* d_cycles,
    int64_t* d_fill_cycles,
    uint8_t layout_type,
    uint16_t lbo,
    uint16_t sbo,
    uint32_t swizzle_mask,
    int k_iters)
{
  constexpr int M_PER_CTA = MMA_M_T / 2;
  constexpr int N_PER_CTA = MMA_N_T / 2;
  constexpr int A_BYTES = M_PER_CTA * FP8_2SM_K_PER_STAGE;
  constexpr int B_BYTES = N_PER_CTA * FP8_2SM_K_PER_STAGE;
  constexpr int A_ALLOC = (A_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC = (B_BYTES + 1023) & ~1023;
  int smem_size = A_ALLOC + B_ALLOC + 256;

  auto kernel = fp8_2sm_mma_swizzle_benchmark_kernel<WAIT_PATTERN, 0, MMA_M_T, MMA_N_T>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cutlass::ClusterLaunchParams params;
  params.grid_dims = dim3(2);
  params.block_dims = dim3(128);
  params.cluster_dims = dim3(2, 1, 1);
  params.smem_size_in_bytes = smem_size;

  void const* kernel_ptr = reinterpret_cast<void const*>(kernel);

  // Warmup
  cutlass::launch_kernel_on_cluster(
      params, kernel_ptr,
      d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo, sbo, lbo, sbo, swizzle_mask, k_iters);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark run with cudaEvent wall-clock timing
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  CUDA_CHECK(cudaEventRecord(ev_start));
  cutlass::launch_kernel_on_cluster(
      params, kernel_ptr,
      d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo, sbo, lbo, sbo, swizzle_mask, k_iters);
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

void run_fp8_2sm_performance_sweep(int clock_rate_khz, int k_iters, FILE* csv_fp) {
  printf("=== fp8 2SM (cta_group::2) Descriptor Configuration Sweep ===\n");
  printf("  MMA shape: %dx%dx%d (fp8, 2SM), K_PER_STAGE=%d (1 stage)\n",
         FP8_2SM_MMA_M, FP8_2SM_MMA_N, FP8_2SM_MMA_K, FP8_2SM_K_PER_STAGE);
  printf("  k_iters: %d, total MMAs per row: %d x %d = %d\n\n",
         k_iters, FP8_2SM_K_BLOCKS, k_iters, FP8_2SM_K_BLOCKS * k_iters);

  SwizzleMode modes[] = {
    SwizzleMode::SW_NONE, SwizzleMode::SW_32B, SwizzleMode::SW_64B, SwizzleMode::SW_128B
  };

  printf("%8s  %-24s  %10s  %8s  %11s  %9s  %10s\n",
         "Swizzle", "Wait Pattern", "Cycles", "Cyc/MMA", "Latency(us)", "Wall(us)", "Fill_Cyc");
  printf("%8s  %-24s  %10s  %8s  %11s  %9s  %10s\n",
         "--------", "------------------------", "----------", "--------", "-----------", "---------", "----------");

  int M = FP8_2SM_MMA_M, N = FP8_2SM_MMA_N, K = FP8_2SM_K_PER_STAGE;

  // Allocate device memory (full 256xK for both A and B)
  uint8_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, M * K));
  CUDA_CHECK(cudaMalloc(&d_B, N * K));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  // Fill with random data
  std::vector<uint8_t> h_A(M * K), h_B(N * K);
  srand(123);
  for (auto& v : h_A) v = static_cast<uint8_t>(rand() & 0x7F);
  for (auto& v : h_B) v = static_cast<uint8_t>(rand() & 0x7F);

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K, cudaMemcpyHostToDevice));

  for (SwizzleMode mode : modes) {
    uint8_t layout_type;
    uint16_t lbo, sbo;
    compute_desc_params_fp8(mode, K, layout_type, lbo, sbo);
    uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

    int total_mmas = FP8_2SM_K_BLOCKS * k_iters;

    auto run_for_pattern = [&](int wp_id, const char* wp_name) {
      BenchResult result = {};

      if (wp_id == 0)
        result = run_fp8_2sm_benchmark_config<0>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else if (wp_id == 1)
        result = run_fp8_2sm_benchmark_config<1>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else
        result = run_fp8_2sm_benchmark_config<2>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);

      double cyc_per_mma = (double)result.mma_cycles / total_mmas;
      double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

      printf("%-8s  %-24s  %10ld  %8.1f  %11.1f  %9.1f  %10ld\n",
             swizzle_mode_name(mode), wp_name,
             (long)result.mma_cycles,
             cyc_per_mma,
             latency_us,
             (double)result.wall_clock_us,
             (long)result.fill_cycles);

      if (csv_fp) {
        fprintf(csv_fp, "fp8_2sm,%dx%dx%d,%d,%d,%s,%s,%ld,%.1f,%.1f,%.1f,%ld\n",
                FP8_2SM_MMA_M, FP8_2SM_MMA_N, FP8_2SM_MMA_K, FP8_2SM_K_PER_STAGE, 1,
                swizzle_mode_name(mode), wp_name,
                (long)result.mma_cycles, cyc_per_mma, latency_us,
                (double)result.wall_clock_us, (long)result.fill_cycles);
      }
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

  printf("\n");
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// fp8 2SM MN-major Correctness & Performance
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMA_M_T = 256, int MMA_N_T = 256>
bool run_fp8_2sm_correctness_test_mn(SwizzleMode mode) {
  printf("=== fp8 2SM MN-major Correctness test: %s, K=%d, MMA=%dx%dx%d ===\n",
         swizzle_mode_name(mode), FP8_2SM_K_PER_STAGE, MMA_M_T, MMA_N_T, FP8_2SM_MMA_K);

  int M = MMA_M_T, N = MMA_N_T, K = FP8_2SM_K_PER_STAGE;
  int M_PER_CTA = MMA_M_T / 2, N_PER_CTA = MMA_N_T / 2;
  int k_iters = 1;

  // Generate random fp8 MN-major data: A[K][M] and B[K][N]
  std::vector<uint8_t> h_A_full(K * M), h_B_full(K * N);
  srand(42);
  for (auto& v : h_A_full) {
    float f = float(rand() % 5 - 2);
    v = cutlass::float_e4m3_t(f).storage;
  }
  for (auto& v : h_B_full) {
    float f = float(rand() % 5 - 2);
    v = cutlass::float_e4m3_t(f).storage;
  }

  // CPU reference with MN-major indexing
  std::vector<float> h_ref(M * N);
  reference_gemm_fp8_mn(h_A_full.data(), h_B_full.data(), h_ref.data(), M, N, K);

  // Split into per-CTA portions: [K][M_PER_CTA] for CTA 0 and [K][M_PER_CTA] for CTA 1
  std::vector<uint8_t> h_A_split(K * M), h_B_split(K * N);
  for (int k = 0; k < K; ++k) {
    for (int m = 0; m < M_PER_CTA; ++m) {
      h_A_split[k * M_PER_CTA + m] = h_A_full[k * M + m];
      h_A_split[K * M_PER_CTA + k * M_PER_CTA + m] = h_A_full[k * M + M_PER_CTA + m];
    }
    for (int n = 0; n < N_PER_CTA; ++n) {
      h_B_split[k * N_PER_CTA + n] = h_B_full[k * N + n];
      h_B_split[K * N_PER_CTA + k * N_PER_CTA + n] = h_B_full[k * N + N_PER_CTA + n];
    }
  }

  // Allocate device memory
  uint8_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, K * M));
  CUDA_CHECK(cudaMalloc(&d_B, K * N));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  CUDA_CHECK(cudaMemcpy(d_A, h_A_split.data(), K * M, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B_split.data(), K * N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, M * N * sizeof(float)));

  // Compute descriptor params for MN-major (mn_size = M_PER_CTA for A, N_PER_CTA for B)
  uint8_t layout_type;
  uint16_t lbo_a, sbo_a, lbo_b, sbo_b;
  compute_desc_params_mn_fp8(mode, M_PER_CTA, layout_type, lbo_a, sbo_a);
  compute_desc_params_mn_fp8(mode, N_PER_CTA, layout_type, lbo_b, sbo_b);
  uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

  // SMEM size per CTA
  constexpr int A_BYTES_2 = (MMA_M_T / 2) * FP8_2SM_K_PER_STAGE;
  constexpr int B_BYTES_2 = (MMA_N_T / 2) * FP8_2SM_K_PER_STAGE;
  constexpr int A_ALLOC = (A_BYTES_2 + 1023) & ~1023;
  constexpr int B_ALLOC = (B_BYTES_2 + 1023) & ~1023;
  constexpr int SMEM_SIZE = A_ALLOC + B_ALLOC + 256;

  auto kernel = fp8_2sm_mma_swizzle_benchmark_kernel<0, 1, MMA_M_T, MMA_N_T>;  // IS_MN_MAJOR=1
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

  cutlass::ClusterLaunchParams params;
  params.grid_dims = dim3(2);
  params.block_dims = dim3(128);
  params.cluster_dims = dim3(2, 1, 1);
  params.smem_size_in_bytes = SMEM_SIZE;

  void const* kernel_ptr = reinterpret_cast<void const*>(kernel);
  auto status = cutlass::launch_kernel_on_cluster(
      params, kernel_ptr,
      d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);

  if (status != cutlass::Status::kSuccess) {
    printf("  Cluster launch failed!\n");
    return false;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Read back and compare
  std::vector<float> h_D(M * N);
  CUDA_CHECK(cudaMemcpy(h_D.data(), d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost));

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

template <int WAIT_PATTERN, int MMA_M_T = 256, int MMA_N_T = 256>
BenchResult run_fp8_2sm_benchmark_config_mn(
    uint8_t* d_A,
    uint8_t* d_B,
    float* d_D,
    int64_t* d_cycles,
    int64_t* d_fill_cycles,
    uint8_t layout_type,
    uint16_t lbo_a, uint16_t sbo_a,
    uint16_t lbo_b, uint16_t sbo_b,
    uint32_t swizzle_mask,
    int k_iters)
{
  constexpr int M_PER_CTA = MMA_M_T / 2;
  constexpr int N_PER_CTA = MMA_N_T / 2;
  constexpr int A_BYTES = M_PER_CTA * FP8_2SM_K_PER_STAGE;
  constexpr int B_BYTES = N_PER_CTA * FP8_2SM_K_PER_STAGE;
  constexpr int A_ALLOC = (A_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC = (B_BYTES + 1023) & ~1023;
  int smem_size = A_ALLOC + B_ALLOC + 256;

  auto kernel = fp8_2sm_mma_swizzle_benchmark_kernel<WAIT_PATTERN, 1, MMA_M_T, MMA_N_T>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cutlass::ClusterLaunchParams params;
  params.grid_dims = dim3(2);
  params.block_dims = dim3(128);
  params.cluster_dims = dim3(2, 1, 1);
  params.smem_size_in_bytes = smem_size;

  void const* kernel_ptr = reinterpret_cast<void const*>(kernel);

  // Warmup
  cutlass::launch_kernel_on_cluster(
      params, kernel_ptr,
      d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark run
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  CUDA_CHECK(cudaEventRecord(ev_start));
  cutlass::launch_kernel_on_cluster(
      params, kernel_ptr,
      d_A, d_B, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
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

void run_fp8_2sm_performance_sweep_mn(int clock_rate_khz, int k_iters, FILE* csv_fp) {
  printf("=== fp8 2SM MN-major (cta_group::2) Descriptor Configuration Sweep ===\n");
  printf("  MMA shape: %dx%dx%d (fp8 2SM MN-major), K_PER_STAGE=%d (1 stage)\n",
         FP8_2SM_MMA_M, FP8_2SM_MMA_N, FP8_2SM_MMA_K, FP8_2SM_K_PER_STAGE);
  printf("  k_iters: %d, total MMAs per row: %d x %d = %d\n\n",
         k_iters, FP8_2SM_K_BLOCKS, k_iters, FP8_2SM_K_BLOCKS * k_iters);

  SwizzleMode modes[] = {
    SwizzleMode::SW_NONE, SwizzleMode::SW_32B, SwizzleMode::SW_64B, SwizzleMode::SW_128B
  };

  printf("%8s  %-24s  %10s  %8s  %11s  %9s  %10s\n",
         "Swizzle", "Wait Pattern", "Cycles", "Cyc/MMA", "Latency(us)", "Wall(us)", "Fill_Cyc");
  printf("%8s  %-24s  %10s  %8s  %11s  %9s  %10s\n",
         "--------", "------------------------", "----------", "--------", "-----------", "---------", "----------");

  int M = FP8_2SM_MMA_M, N = FP8_2SM_MMA_N, K = FP8_2SM_K_PER_STAGE;
  int M_PER_CTA = FP8_2SM_M_PER_CTA, N_PER_CTA = FP8_2SM_N_PER_CTA;

  // Allocate device memory for split MN-major data
  uint8_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, K * M));
  CUDA_CHECK(cudaMalloc(&d_B, K * N));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  // Generate and split MN-major data into per-CTA portions
  std::vector<uint8_t> h_A_full(K * M), h_B_full(K * N);
  srand(123);
  for (auto& v : h_A_full) v = static_cast<uint8_t>(rand() & 0x7F);
  for (auto& v : h_B_full) v = static_cast<uint8_t>(rand() & 0x7F);

  std::vector<uint8_t> h_A_split(K * M), h_B_split(K * N);
  for (int k = 0; k < K; ++k) {
    for (int m = 0; m < M_PER_CTA; ++m) {
      h_A_split[k * M_PER_CTA + m] = h_A_full[k * M + m];
      h_A_split[K * M_PER_CTA + k * M_PER_CTA + m] = h_A_full[k * M + M_PER_CTA + m];
    }
    for (int n = 0; n < N_PER_CTA; ++n) {
      h_B_split[k * N_PER_CTA + n] = h_B_full[k * N + n];
      h_B_split[K * N_PER_CTA + k * N_PER_CTA + n] = h_B_full[k * N + N_PER_CTA + n];
    }
  }

  CUDA_CHECK(cudaMemcpy(d_A, h_A_split.data(), K * M, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B_split.data(), K * N, cudaMemcpyHostToDevice));

  for (SwizzleMode mode : modes) {
    uint8_t layout_type;
    uint16_t lbo_a, sbo_a, lbo_b, sbo_b;
    compute_desc_params_mn_fp8(mode, M_PER_CTA, layout_type, lbo_a, sbo_a);
    compute_desc_params_mn_fp8(mode, N_PER_CTA, layout_type, lbo_b, sbo_b);
    uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

    int total_mmas = FP8_2SM_K_BLOCKS * k_iters;

    auto run_for_pattern = [&](int wp_id, const char* wp_name) {
      BenchResult result = {};

      if (wp_id == 0)
        result = run_fp8_2sm_benchmark_config_mn<0>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
      else if (wp_id == 1)
        result = run_fp8_2sm_benchmark_config_mn<1>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
      else
        result = run_fp8_2sm_benchmark_config_mn<2>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);

      double cyc_per_mma = (double)result.mma_cycles / total_mmas;
      double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

      printf("%-8s  %-24s  %10ld  %8.1f  %11.1f  %9.1f  %10ld\n",
             swizzle_mode_name(mode), wp_name,
             (long)result.mma_cycles,
             cyc_per_mma,
             latency_us,
             (double)result.wall_clock_us,
             (long)result.fill_cycles);

      if (csv_fp) {
        fprintf(csv_fp, "fp8_2sm_mn,%dx%dx%d,%d,%d,%s,%s,%ld,%.1f,%.1f,%.1f,%ld\n",
                FP8_2SM_MMA_M, FP8_2SM_MMA_N, FP8_2SM_MMA_K, FP8_2SM_K_PER_STAGE, 1,
                swizzle_mode_name(mode), wp_name,
                (long)result.mma_cycles, cyc_per_mma, latency_us,
                (double)result.wall_clock_us, (long)result.fill_cycles);
      }
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

  printf("\n");
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED (fp8 2SM tests)

///////////////////////////////////////////////////////////////////////////////////////////////////
// fp4 2SM Correctness & Performance
///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <int MMA_M_T = 256, int MMA_N_T = 256>
bool run_fp4_2sm_correctness_test(SwizzleMode mode) {
  printf("=== fp4 2SM Correctness test: %s, K=%d, MMA=%dx%dx%d ===\n",
         swizzle_mode_name(mode), FP4_2SM_K_PER_STAGE, MMA_M_T, MMA_N_T, FP4_2SM_MMA_K);

  int M = MMA_M_T, N = MMA_N_T, K = FP4_2SM_K_PER_STAGE;
  int K_bytes = K / 2;
  int k_iters = 1;

  // Generate random byte-packed fp4 data
  std::vector<uint8_t> h_A(M * K_bytes), h_B(N * K_bytes);
  srand(42);
  for (auto& v : h_A) v = static_cast<uint8_t>(rand() & 0x77);
  for (auto& v : h_B) v = static_cast<uint8_t>(rand() & 0x77);

  // Generate scale factors
  std::vector<uint8_t> h_SFA(32 * FP4_NUM_SF), h_SFB(32 * FP4_NUM_SF);
  uint8_t sf_vals[] = {0x38, 0x30, 0x3C, 0x34};
  for (size_t i = 0; i < h_SFA.size(); ++i) h_SFA[i] = sf_vals[i % 4];
  for (size_t i = 0; i < h_SFB.size(); ++i) h_SFB[i] = sf_vals[(i + 1) % 4];

  // CPU reference
  std::vector<float> h_ref(M * N);
  reference_gemm_fp4(h_A.data(), h_B.data(), h_SFA.data(), h_SFB.data(),
                     h_ref.data(), M, N, K, FP4_VS);

  // Allocate device memory
  uint8_t *d_A, *d_B, *d_SFA, *d_SFB;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, M * K_bytes));
  CUDA_CHECK(cudaMalloc(&d_B, N * K_bytes));
  CUDA_CHECK(cudaMalloc(&d_SFA, 32 * FP4_NUM_SF));
  CUDA_CHECK(cudaMalloc(&d_SFB, 32 * FP4_NUM_SF));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_SFA, h_SFA.data(), 32 * FP4_NUM_SF, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_SFB, h_SFB.data(), 32 * FP4_NUM_SF, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, M * N * sizeof(float)));

  // Compute descriptor params
  uint8_t layout_type;
  uint16_t lbo, sbo;
  compute_desc_params_fp4(mode, K, layout_type, lbo, sbo);
  uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

  // SMEM size per CTA
  constexpr int M_PER_CTA = MMA_M_T / 2;
  constexpr int N_PER_CTA = MMA_N_T / 2;
  constexpr int A_ALLOC  = (M_PER_CTA * FP4_K_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC  = (N_PER_CTA * FP4_K_BYTES + 1023) & ~1023;
  constexpr int SF_ALLOC = (32 * FP4_NUM_SF + 1023) & ~1023;
  constexpr int SMEM_SIZE = A_ALLOC + B_ALLOC + SF_ALLOC + SF_ALLOC + 256;

  auto kernel = fp4_2sm_mma_swizzle_benchmark_kernel<0, MMA_M_T, MMA_N_T>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

  // Cluster launch: 2 CTAs in a cluster
  cutlass::ClusterLaunchParams params;
  params.grid_dims = dim3(2);
  params.block_dims = dim3(128);
  params.cluster_dims = dim3(2, 1, 1);
  params.smem_size_in_bytes = SMEM_SIZE;

  void const* kernel_ptr = reinterpret_cast<void const*>(kernel);
  auto status = cutlass::launch_kernel_on_cluster(
      params, kernel_ptr,
      d_A, d_B, d_SFA, d_SFB, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo, sbo, swizzle_mask, k_iters);

  if (status != cutlass::Status::kSuccess) {
    printf("  Cluster launch failed!\n");
    return false;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Read back results
  std::vector<float> h_D(M * N);
  CUDA_CHECK(cudaMemcpy(h_D.data(), d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // Compare with relaxed tolerance (fp4 is low precision)
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
      if (rel > 0.1f && err > 1.0f) {
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
  CUDA_CHECK(cudaFree(d_SFA));
  CUDA_CHECK(cudaFree(d_SFB));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_cycles));
  CUDA_CHECK(cudaFree(d_fill_cycles));

  return pass;
}

template <int WAIT_PATTERN, int MMA_M_T = 256, int MMA_N_T = 256>
BenchResult run_fp4_2sm_benchmark_config(
    uint8_t* d_A,
    uint8_t* d_B,
    uint8_t* d_SFA,
    uint8_t* d_SFB,
    float* d_D,
    int64_t* d_cycles,
    int64_t* d_fill_cycles,
    uint8_t layout_type,
    uint16_t lbo,
    uint16_t sbo,
    uint32_t swizzle_mask,
    int k_iters)
{
  constexpr int M_PER_CTA = MMA_M_T / 2;
  constexpr int N_PER_CTA = MMA_N_T / 2;
  constexpr int A_ALLOC  = (M_PER_CTA * FP4_K_BYTES + 1023) & ~1023;
  constexpr int B_ALLOC  = (N_PER_CTA * FP4_K_BYTES + 1023) & ~1023;
  constexpr int SF_ALLOC = (32 * FP4_NUM_SF + 1023) & ~1023;
  int smem_size = A_ALLOC + B_ALLOC + SF_ALLOC + SF_ALLOC + 256;

  auto kernel = fp4_2sm_mma_swizzle_benchmark_kernel<WAIT_PATTERN, MMA_M_T, MMA_N_T>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cutlass::ClusterLaunchParams params;
  params.grid_dims = dim3(2);
  params.block_dims = dim3(128);
  params.cluster_dims = dim3(2, 1, 1);
  params.smem_size_in_bytes = smem_size;

  void const* kernel_ptr = reinterpret_cast<void const*>(kernel);

  // Warmup
  cutlass::launch_kernel_on_cluster(
      params, kernel_ptr,
      d_A, d_B, d_SFA, d_SFB, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo, sbo, swizzle_mask, k_iters);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark run
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  CUDA_CHECK(cudaEventRecord(ev_start));
  cutlass::launch_kernel_on_cluster(
      params, kernel_ptr,
      d_A, d_B, d_SFA, d_SFB, d_D, d_cycles, d_fill_cycles,
      layout_type, lbo, sbo, swizzle_mask, k_iters);
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

void run_fp4_2sm_performance_sweep(int clock_rate_khz, int k_iters, FILE* csv_fp) {
  printf("=== fp4 2SM (cta_group::2) Descriptor Configuration Sweep ===\n");
  printf("  MMA shape: %dx%dx%d (fp4, 2SM, block16), K_PER_STAGE=%d (1 stage)\n",
         FP4_2SM_MMA_M, FP4_2SM_MMA_N, FP4_2SM_MMA_K, FP4_2SM_K_PER_STAGE);
  printf("  k_iters: %d, total MMAs per row: %d x %d = %d\n\n",
         k_iters, FP4_2SM_K_BLOCKS, k_iters, FP4_2SM_K_BLOCKS * k_iters);

  SwizzleMode modes[] = {
    SwizzleMode::SW_NONE, SwizzleMode::SW_32B, SwizzleMode::SW_64B, SwizzleMode::SW_128B
  };

  printf("%8s  %-24s  %10s  %8s  %11s  %9s  %10s\n",
         "Swizzle", "Wait Pattern", "Cycles", "Cyc/MMA", "Latency(us)", "Wall(us)", "Fill_Cyc");
  printf("%8s  %-24s  %10s  %8s  %11s  %9s  %10s\n",
         "--------", "------------------------", "----------", "--------", "-----------", "---------", "----------");

  int M = FP4_2SM_MMA_M, N = FP4_2SM_MMA_N, K = FP4_2SM_K_PER_STAGE;
  int K_bytes = K / 2;

  // Allocate device memory
  uint8_t *d_A, *d_B, *d_SFA, *d_SFB;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, M * K_bytes));
  CUDA_CHECK(cudaMalloc(&d_B, N * K_bytes));
  CUDA_CHECK(cudaMalloc(&d_SFA, 32 * FP4_NUM_SF));
  CUDA_CHECK(cudaMalloc(&d_SFB, 32 * FP4_NUM_SF));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  // Fill with random data
  std::vector<uint8_t> h_A(M * K_bytes), h_B(N * K_bytes);
  std::vector<uint8_t> h_SFA(32 * FP4_NUM_SF, 0x38), h_SFB(32 * FP4_NUM_SF, 0x38);
  srand(123);
  for (auto& v : h_A) v = static_cast<uint8_t>(rand() & 0x77);
  for (auto& v : h_B) v = static_cast<uint8_t>(rand() & 0x77);

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_SFA, h_SFA.data(), 32 * FP4_NUM_SF, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_SFB, h_SFB.data(), 32 * FP4_NUM_SF, cudaMemcpyHostToDevice));

  for (SwizzleMode mode : modes) {
    uint8_t layout_type;
    uint16_t lbo, sbo;
    compute_desc_params_fp4(mode, K, layout_type, lbo, sbo);
    uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

    int total_mmas = FP4_2SM_K_BLOCKS * k_iters;

    auto run_for_pattern = [&](int wp_id, const char* wp_name) {
      BenchResult result = {};

      if (wp_id == 0)
        result = run_fp4_2sm_benchmark_config<0>(d_A, d_B, d_SFA, d_SFB, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else if (wp_id == 1)
        result = run_fp4_2sm_benchmark_config<1>(d_A, d_B, d_SFA, d_SFB, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else
        result = run_fp4_2sm_benchmark_config<2>(d_A, d_B, d_SFA, d_SFB, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);

      double cyc_per_mma = (double)result.mma_cycles / total_mmas;
      double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

      printf("%-8s  %-24s  %10ld  %8.1f  %11.1f  %9.1f  %10ld\n",
             swizzle_mode_name(mode), wp_name,
             (long)result.mma_cycles,
             cyc_per_mma,
             latency_us,
             (double)result.wall_clock_us,
             (long)result.fill_cycles);

      if (csv_fp) {
        fprintf(csv_fp, "fp4_2sm,%dx%dx%d,%d,%d,%s,%s,%ld,%.1f,%.1f,%.1f,%ld\n",
                FP4_2SM_MMA_M, FP4_2SM_MMA_N, FP4_2SM_MMA_K, FP4_2SM_K_PER_STAGE, 1,
                swizzle_mode_name(mode), wp_name,
                (long)result.mma_cycles, cyc_per_mma, latency_us,
                (double)result.wall_clock_us, (long)result.fill_cycles);
      }
    };

    run_for_pattern(0, "commit+wait each");
    run_for_pattern(1, "commit each, wait end");
    run_for_pattern(2, "commit+wait end");
  }

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_SFA));
  CUDA_CHECK(cudaFree(d_SFB));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_cycles));
  CUDA_CHECK(cudaFree(d_fill_cycles));

  printf("\n");
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED (fp4 2SM tests)

///////////////////////////////////////////////////////////////////////////////////////////////////
// Shape-parameterized driver functions for additional MMA shapes
///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// bf16 1SM K-major: correctness (4 swizzles) + performance (4 swizzles x 3 waits, K=256, S=1)
template <int M, int N>
void test_bf16_shape_k(bool& all_pass, int clock_rate_khz, int k_iters, FILE* csv_fp) {
  printf("\n=== bf16 K-major shape %dx%dx%d ===\n\n", M, N, MMA_K);

  SwizzleMode modes[] = {
    SwizzleMode::SW_NONE, SwizzleMode::SW_32B, SwizzleMode::SW_64B, SwizzleMode::SW_128B
  };

  // Correctness
  for (SwizzleMode mode : modes) {
    all_pass &= run_correctness_test_k<256, M, N>(mode);
  }

  // Performance: K=256, S=1
  constexpr int K_PER_STAGE = 256;
  int total_A_elems = M * K_PER_STAGE;
  int total_B_elems = N * K_PER_STAGE;

  cutlass::bfloat16_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, total_A_elems * sizeof(cutlass::bfloat16_t)));
  CUDA_CHECK(cudaMalloc(&d_B, total_B_elems * sizeof(cutlass::bfloat16_t)));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  std::vector<cutlass::bfloat16_t> h_A(total_A_elems), h_B(total_B_elems);
  srand(123);
  for (int i = 0; i < total_A_elems; ++i) h_A[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));
  for (int i = 0; i < total_B_elems; ++i) h_B[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), total_A_elems * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), total_B_elems * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));

  for (SwizzleMode mode : modes) {
    uint8_t layout_type;
    uint16_t lbo, sbo;
    compute_desc_params(mode, K_PER_STAGE, layout_type, lbo, sbo);
    uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

    int k_blocks = K_PER_STAGE / MMA_K;
    int total_mmas = k_blocks * k_iters;

    auto run_for_pattern = [&](int wp_id, const char* wp_name) {
      BenchResult result = {};
      if (wp_id == 0)
        result = run_benchmark_config<1, 0, 256, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else if (wp_id == 1)
        result = run_benchmark_config<1, 1, 256, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else
        result = run_benchmark_config<1, 2, 256, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);

      double cyc_per_mma = (double)result.mma_cycles / total_mmas;
      double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

      if (csv_fp) {
        fprintf(csv_fp, "bf16,%dx%dx%d,%d,%d,%s,%s,%ld,%.1f,%.1f,%.1f,%ld\n",
                M, N, MMA_K, K_PER_STAGE, 1,
                swizzle_mode_name(mode), wp_name,
                (long)result.mma_cycles, cyc_per_mma, latency_us,
                (double)result.wall_clock_us, (long)result.fill_cycles);
      }
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

// bf16 1SM MN-major
template <int M, int N>
void test_bf16_shape_mn(bool& all_pass, int clock_rate_khz, int k_iters, FILE* csv_fp) {
  printf("\n=== bf16 MN-major shape %dx%dx%d ===\n\n", M, N, MMA_K);

  // MN-major swizzle filtering: SW_128B atom = 64 bf16 elems
  SwizzleMode all_modes[] = {
    SwizzleMode::SW_NONE, SwizzleMode::SW_32B, SwizzleMode::SW_64B, SwizzleMode::SW_128B
  };
  std::vector<SwizzleMode> modes;
  for (SwizzleMode m : all_modes) {
    if (m == SwizzleMode::SW_128B && N < 64) continue;
    if (m == SwizzleMode::SW_64B  && N < 32) continue;
    if (m == SwizzleMode::SW_32B  && N < 16) continue;
    modes.push_back(m);
  }

  // Correctness
  for (SwizzleMode mode : modes) {
    all_pass &= run_correctness_test_mn_k<256, M, N>(mode);
  }

  // Performance: K=256, S=1
  constexpr int K_PER_STAGE = 256;
  int total_A_elems = M * K_PER_STAGE;
  int total_B_elems = N * K_PER_STAGE;

  cutlass::bfloat16_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, total_A_elems * sizeof(cutlass::bfloat16_t)));
  CUDA_CHECK(cudaMalloc(&d_B, total_B_elems * sizeof(cutlass::bfloat16_t)));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  std::vector<cutlass::bfloat16_t> h_A(total_A_elems), h_B(total_B_elems);
  srand(123);
  for (int i = 0; i < total_A_elems; ++i) h_A[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));
  for (int i = 0; i < total_B_elems; ++i) h_B[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), total_A_elems * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), total_B_elems * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));

  for (SwizzleMode mode : modes) {
    uint8_t layout_type;
    uint16_t lbo_a, sbo_a, lbo_b, sbo_b;
    compute_desc_params_mn(mode, M, layout_type, lbo_a, sbo_a);
    compute_desc_params_mn(mode, N, layout_type, lbo_b, sbo_b);
    uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

    int k_blocks = K_PER_STAGE / MMA_K;
    int total_mmas = k_blocks * k_iters;

    auto run_for_pattern = [&](int wp_id, const char* wp_name) {
      BenchResult result = {};
      if (wp_id == 0)
        result = run_benchmark_config_mn<1, 0, 256, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
      else if (wp_id == 1)
        result = run_benchmark_config_mn<1, 1, 256, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
      else
        result = run_benchmark_config_mn<1, 2, 256, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);

      double cyc_per_mma = (double)result.mma_cycles / total_mmas;
      double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

      if (csv_fp) {
        fprintf(csv_fp, "bf16_mn,%dx%dx%d,%d,%d,%s,%s,%ld,%.1f,%.1f,%.1f,%ld\n",
                M, N, MMA_K, K_PER_STAGE, 1,
                swizzle_mode_name(mode), wp_name,
                (long)result.mma_cycles, cyc_per_mma, latency_us,
                (double)result.wall_clock_us, (long)result.fill_cycles);
      }
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

// fp4 shape driver
template <int M, int N>
void test_fp4_shape(bool& all_pass, int clock_rate_khz, int k_iters, FILE* csv_fp) {
  printf("\n=== nvfp4 shape %dx%dx%d ===\n\n", M, N, FP4_MMA_K);

  SwizzleMode modes[] = {
    SwizzleMode::SW_NONE, SwizzleMode::SW_32B, SwizzleMode::SW_64B, SwizzleMode::SW_128B
  };

  // Correctness
  for (SwizzleMode mode : modes) {
    all_pass &= run_fp4_correctness_test<M, N>(mode);
  }

  // Performance
  int K_bytes = FP4_K_BYTES;

  uint8_t *d_A, *d_B, *d_SFA, *d_SFB;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, M * K_bytes));
  CUDA_CHECK(cudaMalloc(&d_B, N * K_bytes));
  CUDA_CHECK(cudaMalloc(&d_SFA, 32 * FP4_NUM_SF));
  CUDA_CHECK(cudaMalloc(&d_SFB, 32 * FP4_NUM_SF));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  std::vector<uint8_t> h_A(M * K_bytes), h_B(N * K_bytes);
  std::vector<uint8_t> h_SFA(32 * FP4_NUM_SF, 0x38), h_SFB(32 * FP4_NUM_SF, 0x38);
  srand(123);
  for (auto& v : h_A) v = static_cast<uint8_t>(rand() & 0x77);
  for (auto& v : h_B) v = static_cast<uint8_t>(rand() & 0x77);

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_SFA, h_SFA.data(), 32 * FP4_NUM_SF, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_SFB, h_SFB.data(), 32 * FP4_NUM_SF, cudaMemcpyHostToDevice));

  for (SwizzleMode mode : modes) {
    uint8_t layout_type;
    uint16_t lbo, sbo;
    compute_desc_params_fp4(mode, FP4_K_PER_STAGE, layout_type, lbo, sbo);
    uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

    int total_mmas = FP4_K_BLOCKS * k_iters;

    auto run_for_pattern = [&](int wp_id, const char* wp_name) {
      BenchResult result = {};
      if (wp_id == 0)
        result = run_fp4_benchmark_config<0, M, N>(d_A, d_B, d_SFA, d_SFB, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else if (wp_id == 1)
        result = run_fp4_benchmark_config<1, M, N>(d_A, d_B, d_SFA, d_SFB, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else
        result = run_fp4_benchmark_config<2, M, N>(d_A, d_B, d_SFA, d_SFB, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);

      double cyc_per_mma = (double)result.mma_cycles / total_mmas;
      double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

      if (csv_fp) {
        fprintf(csv_fp, "nvfp4,%dx%dx%d,%d,%d,%s,%s,%ld,%.1f,%.1f,%.1f,%ld\n",
                M, N, FP4_MMA_K, FP4_K_PER_STAGE, 1,
                swizzle_mode_name(mode), wp_name,
                (long)result.mma_cycles, cyc_per_mma, latency_us,
                (double)result.wall_clock_us, (long)result.fill_cycles);
      }
    };

    run_for_pattern(0, "commit+wait each");
    run_for_pattern(1, "commit each, wait end");
    run_for_pattern(2, "commit+wait end");
  }

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_SFA));
  CUDA_CHECK(cudaFree(d_SFB));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_cycles));
  CUDA_CHECK(cudaFree(d_fill_cycles));
}

// fp8 K-major shape driver
template <int M, int N>
void test_fp8_shape_k(bool& all_pass, int clock_rate_khz, int k_iters, FILE* csv_fp) {
  printf("\n=== fp8 K-major shape %dx%dx%d ===\n\n", M, N, FP8_MMA_K);

  SwizzleMode modes[] = {
    SwizzleMode::SW_NONE, SwizzleMode::SW_32B, SwizzleMode::SW_64B, SwizzleMode::SW_128B
  };

  for (SwizzleMode mode : modes) {
    all_pass &= run_fp8_correctness_test<M, N>(mode);
  }

  int K = FP8_K_PER_STAGE;

  uint8_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, M * K));
  CUDA_CHECK(cudaMalloc(&d_B, N * K));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  std::vector<uint8_t> h_A(M * K), h_B(N * K);
  srand(123);
  for (auto& v : h_A) v = static_cast<uint8_t>(rand() & 0x7F);
  for (auto& v : h_B) v = static_cast<uint8_t>(rand() & 0x7F);

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K, cudaMemcpyHostToDevice));

  for (SwizzleMode mode : modes) {
    uint8_t layout_type;
    uint16_t lbo, sbo;
    compute_desc_params_fp8(mode, K, layout_type, lbo, sbo);
    uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

    int total_mmas = FP8_K_BLOCKS * k_iters;

    auto run_for_pattern = [&](int wp_id, const char* wp_name) {
      BenchResult result = {};
      if (wp_id == 0)
        result = run_fp8_benchmark_config<0, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else if (wp_id == 1)
        result = run_fp8_benchmark_config<1, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else
        result = run_fp8_benchmark_config<2, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);

      double cyc_per_mma = (double)result.mma_cycles / total_mmas;
      double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

      if (csv_fp) {
        fprintf(csv_fp, "fp8,%dx%dx%d,%d,%d,%s,%s,%ld,%.1f,%.1f,%.1f,%ld\n",
                M, N, FP8_MMA_K, FP8_K_PER_STAGE, 1,
                swizzle_mode_name(mode), wp_name,
                (long)result.mma_cycles, cyc_per_mma, latency_us,
                (double)result.wall_clock_us, (long)result.fill_cycles);
      }
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

// fp8 MN-major shape driver
template <int M, int N>
void test_fp8_shape_mn(bool& all_pass, int clock_rate_khz, int k_iters, FILE* csv_fp) {
  printf("\n=== fp8 MN-major shape %dx%dx%d ===\n\n", M, N, FP8_MMA_K);

  // MN-major swizzle filtering: SW_128B atom = 128 fp8 bytes
  SwizzleMode all_modes[] = {
    SwizzleMode::SW_NONE, SwizzleMode::SW_32B, SwizzleMode::SW_64B, SwizzleMode::SW_128B
  };
  std::vector<SwizzleMode> modes;
  for (SwizzleMode m : all_modes) {
    if (m == SwizzleMode::SW_128B && N < 128) continue;
    if (m == SwizzleMode::SW_64B  && N < 64) continue;
    if (m == SwizzleMode::SW_32B  && N < 32) continue;
    modes.push_back(m);
  }

  int K = FP8_K_PER_STAGE;

  for (SwizzleMode mode : modes) {
    all_pass &= run_fp8_correctness_test_mn<M, N>(mode);
  }

  uint8_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, K * M));
  CUDA_CHECK(cudaMalloc(&d_B, K * N));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  std::vector<uint8_t> h_A(K * M), h_B(K * N);
  srand(123);
  for (auto& v : h_A) v = static_cast<uint8_t>(rand() & 0x7F);
  for (auto& v : h_B) v = static_cast<uint8_t>(rand() & 0x7F);

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), K * M, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N, cudaMemcpyHostToDevice));

  for (SwizzleMode mode : modes) {
    uint8_t layout_type;
    uint16_t lbo_a, sbo_a, lbo_b, sbo_b;
    compute_desc_params_mn_fp8(mode, M, layout_type, lbo_a, sbo_a);
    compute_desc_params_mn_fp8(mode, N, layout_type, lbo_b, sbo_b);
    uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

    int total_mmas = FP8_K_BLOCKS * k_iters;

    auto run_for_pattern = [&](int wp_id, const char* wp_name) {
      BenchResult result = {};
      if (wp_id == 0)
        result = run_fp8_benchmark_config_mn<0, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
      else if (wp_id == 1)
        result = run_fp8_benchmark_config_mn<1, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
      else
        result = run_fp8_benchmark_config_mn<2, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);

      double cyc_per_mma = (double)result.mma_cycles / total_mmas;
      double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

      if (csv_fp) {
        fprintf(csv_fp, "fp8_mn,%dx%dx%d,%d,%d,%s,%s,%ld,%.1f,%.1f,%.1f,%ld\n",
                M, N, FP8_MMA_K, FP8_K_PER_STAGE, 1,
                swizzle_mode_name(mode), wp_name,
                (long)result.mma_cycles, cyc_per_mma, latency_us,
                (double)result.wall_clock_us, (long)result.fill_cycles);
      }
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

// bf16 2SM K-major shape driver
template <int M, int N>
void test_2sm_shape_k(bool& all_pass, int clock_rate_khz, int k_iters, FILE* csv_fp) {
  printf("\n=== bf16 2SM K-major shape %dx%dx%d ===\n\n", M, N, BF16_2SM_MMA_K);

  SwizzleMode modes[] = {
    SwizzleMode::SW_NONE, SwizzleMode::SW_32B, SwizzleMode::SW_64B, SwizzleMode::SW_128B
  };

  for (SwizzleMode mode : modes) {
    all_pass &= run_bf16_2sm_correctness_test<M, N>(mode);
  }

  int K = BF16_2SM_K_PER_STAGE;

  cutlass::bfloat16_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(cutlass::bfloat16_t)));
  CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(cutlass::bfloat16_t)));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  std::vector<cutlass::bfloat16_t> h_A(M * K), h_B(N * K);
  srand(123);
  for (int i = 0; i < M * K; ++i) h_A[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));
  for (int i = 0; i < N * K; ++i) h_B[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));

  for (SwizzleMode mode : modes) {
    uint8_t layout_type;
    uint16_t lbo, sbo;
    compute_desc_params(mode, K, layout_type, lbo, sbo);
    uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

    int k_blocks = K / BF16_2SM_MMA_K;
    int total_mmas = k_blocks * k_iters;

    auto run_for_pattern = [&](int wp_id, const char* wp_name) {
      BenchResult result = {};
      if (wp_id == 0)
        result = run_bf16_2sm_benchmark_config<0, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else if (wp_id == 1)
        result = run_bf16_2sm_benchmark_config<1, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else
        result = run_bf16_2sm_benchmark_config<2, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);

      double cyc_per_mma = (double)result.mma_cycles / total_mmas;
      double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

      if (csv_fp) {
        fprintf(csv_fp, "bf16_2sm,%dx%dx%d,%d,%d,%s,%s,%ld,%.1f,%.1f,%.1f,%ld\n",
                M, N, BF16_2SM_MMA_K, BF16_2SM_K_PER_STAGE, 1,
                swizzle_mode_name(mode), wp_name,
                (long)result.mma_cycles, cyc_per_mma, latency_us,
                (double)result.wall_clock_us, (long)result.fill_cycles);
      }
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

// bf16 2SM MN-major shape driver
template <int M, int N>
void test_2sm_shape_mn(bool& all_pass, int clock_rate_khz, int k_iters, FILE* csv_fp) {
  printf("\n=== bf16 2SM MN-major shape %dx%dx%d ===\n\n", M, N, BF16_2SM_MMA_K);

  constexpr int M_PER_CTA = M / 2;
  constexpr int N_PER_CTA = N / 2;

  // MN-major swizzle filtering: atom = 64 bf16 elems
  SwizzleMode all_modes[] = {
    SwizzleMode::SW_NONE, SwizzleMode::SW_32B, SwizzleMode::SW_64B, SwizzleMode::SW_128B
  };
  std::vector<SwizzleMode> modes;
  for (SwizzleMode m : all_modes) {
    if (m == SwizzleMode::SW_128B && N_PER_CTA < 64) continue;
    if (m == SwizzleMode::SW_64B  && N_PER_CTA < 32) continue;
    if (m == SwizzleMode::SW_32B  && N_PER_CTA < 16) continue;
    modes.push_back(m);
  }

  int K = BF16_2SM_K_PER_STAGE;

  for (SwizzleMode mode : modes) {
    all_pass &= run_bf16_2sm_correctness_test_mn<M, N>(mode);
  }

  cutlass::bfloat16_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, K * M * sizeof(cutlass::bfloat16_t)));
  CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(cutlass::bfloat16_t)));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  // Generate and split MN-major data
  std::vector<cutlass::bfloat16_t> h_A_full(K * M), h_B_full(K * N);
  srand(123);
  for (int i = 0; i < K * M; ++i) h_A_full[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));
  for (int i = 0; i < K * N; ++i) h_B_full[i] = cutlass::bfloat16_t(float(rand() % 5 - 2));

  std::vector<cutlass::bfloat16_t> h_A_split(K * M), h_B_split(K * N);
  for (int k = 0; k < K; ++k) {
    for (int m = 0; m < M_PER_CTA; ++m) {
      h_A_split[k * M_PER_CTA + m] = h_A_full[k * M + m];
      h_A_split[K * M_PER_CTA + k * M_PER_CTA + m] = h_A_full[k * M + M_PER_CTA + m];
    }
    for (int n = 0; n < N_PER_CTA; ++n) {
      h_B_split[k * N_PER_CTA + n] = h_B_full[k * N + n];
      h_B_split[K * N_PER_CTA + k * N_PER_CTA + n] = h_B_full[k * N + N_PER_CTA + n];
    }
  }

  CUDA_CHECK(cudaMemcpy(d_A, h_A_split.data(), K * M * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B_split.data(), K * N * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));

  for (SwizzleMode mode : modes) {
    uint8_t layout_type;
    uint16_t lbo_a, sbo_a, lbo_b, sbo_b;
    compute_desc_params_mn(mode, M_PER_CTA, layout_type, lbo_a, sbo_a);
    compute_desc_params_mn(mode, N_PER_CTA, layout_type, lbo_b, sbo_b);
    uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

    int k_blocks = K / BF16_2SM_MMA_K;
    int total_mmas = k_blocks * k_iters;

    auto run_for_pattern = [&](int wp_id, const char* wp_name) {
      BenchResult result = {};
      if (wp_id == 0)
        result = run_bf16_2sm_benchmark_config_mn<0, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
      else if (wp_id == 1)
        result = run_bf16_2sm_benchmark_config_mn<1, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
      else
        result = run_bf16_2sm_benchmark_config_mn<2, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);

      double cyc_per_mma = (double)result.mma_cycles / total_mmas;
      double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

      if (csv_fp) {
        fprintf(csv_fp, "bf16_2sm_mn,%dx%dx%d,%d,%d,%s,%s,%ld,%.1f,%.1f,%.1f,%ld\n",
                M, N, BF16_2SM_MMA_K, BF16_2SM_K_PER_STAGE, 1,
                swizzle_mode_name(mode), wp_name,
                (long)result.mma_cycles, cyc_per_mma, latency_us,
                (double)result.wall_clock_us, (long)result.fill_cycles);
      }
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

// fp8 2SM K-major shape driver
template <int M, int N>
void test_fp8_2sm_shape_k(bool& all_pass, int clock_rate_khz, int k_iters, FILE* csv_fp) {
  printf("\n=== fp8 2SM K-major shape %dx%dx%d ===\n\n", M, N, FP8_2SM_MMA_K);

  SwizzleMode modes[] = {
    SwizzleMode::SW_NONE, SwizzleMode::SW_32B, SwizzleMode::SW_64B, SwizzleMode::SW_128B
  };

  for (SwizzleMode mode : modes) {
    all_pass &= run_fp8_2sm_correctness_test<M, N>(mode);
  }

  int K = FP8_2SM_K_PER_STAGE;

  uint8_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, M * K));
  CUDA_CHECK(cudaMalloc(&d_B, N * K));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  std::vector<uint8_t> h_A(M * K), h_B(N * K);
  srand(123);
  for (auto& v : h_A) v = static_cast<uint8_t>(rand() & 0x7F);
  for (auto& v : h_B) v = static_cast<uint8_t>(rand() & 0x7F);

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K, cudaMemcpyHostToDevice));

  for (SwizzleMode mode : modes) {
    uint8_t layout_type;
    uint16_t lbo, sbo;
    compute_desc_params_fp8(mode, K, layout_type, lbo, sbo);
    uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

    int k_blocks = K / FP8_2SM_MMA_K;
    int total_mmas = k_blocks * k_iters;

    auto run_for_pattern = [&](int wp_id, const char* wp_name) {
      BenchResult result = {};
      if (wp_id == 0)
        result = run_fp8_2sm_benchmark_config<0, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else if (wp_id == 1)
        result = run_fp8_2sm_benchmark_config<1, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else
        result = run_fp8_2sm_benchmark_config<2, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);

      double cyc_per_mma = (double)result.mma_cycles / total_mmas;
      double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

      if (csv_fp) {
        fprintf(csv_fp, "fp8_2sm,%dx%dx%d,%d,%d,%s,%s,%ld,%.1f,%.1f,%.1f,%ld\n",
                M, N, FP8_2SM_MMA_K, FP8_2SM_K_PER_STAGE, 1,
                swizzle_mode_name(mode), wp_name,
                (long)result.mma_cycles, cyc_per_mma, latency_us,
                (double)result.wall_clock_us, (long)result.fill_cycles);
      }
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

// fp8 2SM MN-major shape driver
template <int M, int N>
void test_fp8_2sm_shape_mn(bool& all_pass, int clock_rate_khz, int k_iters, FILE* csv_fp) {
  printf("\n=== fp8 2SM MN-major shape %dx%dx%d ===\n\n", M, N, FP8_2SM_MMA_K);

  constexpr int M_PER_CTA = M / 2;
  constexpr int N_PER_CTA = N / 2;

  // MN-major swizzle filtering: atom = 128 fp8 bytes
  SwizzleMode all_modes[] = {
    SwizzleMode::SW_NONE, SwizzleMode::SW_32B, SwizzleMode::SW_64B, SwizzleMode::SW_128B
  };
  std::vector<SwizzleMode> modes;
  for (SwizzleMode m : all_modes) {
    if (m == SwizzleMode::SW_128B && N_PER_CTA < 128) continue;
    if (m == SwizzleMode::SW_64B  && N_PER_CTA < 64) continue;
    if (m == SwizzleMode::SW_32B  && N_PER_CTA < 32) continue;
    modes.push_back(m);
  }

  int K = FP8_2SM_K_PER_STAGE;

  for (SwizzleMode mode : modes) {
    all_pass &= run_fp8_2sm_correctness_test_mn<M, N>(mode);
  }

  uint8_t *d_A, *d_B;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, K * M));
  CUDA_CHECK(cudaMalloc(&d_B, K * N));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  // Generate and split MN-major data
  std::vector<uint8_t> h_A_full(K * M), h_B_full(K * N);
  srand(123);
  for (auto& v : h_A_full) v = static_cast<uint8_t>(rand() & 0x7F);
  for (auto& v : h_B_full) v = static_cast<uint8_t>(rand() & 0x7F);

  std::vector<uint8_t> h_A_split(K * M), h_B_split(K * N);
  for (int k = 0; k < K; ++k) {
    for (int m = 0; m < M_PER_CTA; ++m) {
      h_A_split[k * M_PER_CTA + m] = h_A_full[k * M + m];
      h_A_split[K * M_PER_CTA + k * M_PER_CTA + m] = h_A_full[k * M + M_PER_CTA + m];
    }
    for (int n = 0; n < N_PER_CTA; ++n) {
      h_B_split[k * N_PER_CTA + n] = h_B_full[k * N + n];
      h_B_split[K * N_PER_CTA + k * N_PER_CTA + n] = h_B_full[k * N + N_PER_CTA + n];
    }
  }

  CUDA_CHECK(cudaMemcpy(d_A, h_A_split.data(), K * M, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B_split.data(), K * N, cudaMemcpyHostToDevice));

  for (SwizzleMode mode : modes) {
    uint8_t layout_type;
    uint16_t lbo_a, sbo_a, lbo_b, sbo_b;
    compute_desc_params_mn_fp8(mode, M_PER_CTA, layout_type, lbo_a, sbo_a);
    compute_desc_params_mn_fp8(mode, N_PER_CTA, layout_type, lbo_b, sbo_b);
    uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

    int k_blocks = K / FP8_2SM_MMA_K;
    int total_mmas = k_blocks * k_iters;

    auto run_for_pattern = [&](int wp_id, const char* wp_name) {
      BenchResult result = {};
      if (wp_id == 0)
        result = run_fp8_2sm_benchmark_config_mn<0, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
      else if (wp_id == 1)
        result = run_fp8_2sm_benchmark_config_mn<1, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);
      else
        result = run_fp8_2sm_benchmark_config_mn<2, M, N>(d_A, d_B, d_D, d_cycles, d_fill_cycles, layout_type, lbo_a, sbo_a, lbo_b, sbo_b, swizzle_mask, k_iters);

      double cyc_per_mma = (double)result.mma_cycles / total_mmas;
      double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

      if (csv_fp) {
        fprintf(csv_fp, "fp8_2sm_mn,%dx%dx%d,%d,%d,%s,%s,%ld,%.1f,%.1f,%.1f,%ld\n",
                M, N, FP8_2SM_MMA_K, FP8_2SM_K_PER_STAGE, 1,
                swizzle_mode_name(mode), wp_name,
                (long)result.mma_cycles, cyc_per_mma, latency_us,
                (double)result.wall_clock_us, (long)result.fill_cycles);
      }
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

// fp4 2SM K-major shape driver
template <int M, int N>
void test_fp4_2sm_shape_k(bool& all_pass, int clock_rate_khz, int k_iters, FILE* csv_fp) {
  printf("\n=== fp4 2SM K-major shape %dx%dx%d ===\n\n", M, N, FP4_2SM_MMA_K);

  SwizzleMode modes[] = {
    SwizzleMode::SW_NONE, SwizzleMode::SW_32B, SwizzleMode::SW_64B, SwizzleMode::SW_128B
  };

  for (SwizzleMode mode : modes) {
    all_pass &= run_fp4_2sm_correctness_test<M, N>(mode);
  }

  int K_bytes = FP4_K_BYTES;

  uint8_t *d_A, *d_B, *d_SFA, *d_SFB;
  float *d_D;
  int64_t *d_cycles, *d_fill_cycles;

  CUDA_CHECK(cudaMalloc(&d_A, M * K_bytes));
  CUDA_CHECK(cudaMalloc(&d_B, N * K_bytes));
  CUDA_CHECK(cudaMalloc(&d_SFA, 32 * FP4_NUM_SF));
  CUDA_CHECK(cudaMalloc(&d_SFB, 32 * FP4_NUM_SF));
  CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_fill_cycles, sizeof(int64_t)));

  std::vector<uint8_t> h_A(M * K_bytes), h_B(N * K_bytes);
  std::vector<uint8_t> h_SFA(32 * FP4_NUM_SF, 0x38), h_SFB(32 * FP4_NUM_SF, 0x38);
  srand(123);
  for (auto& v : h_A) v = static_cast<uint8_t>(rand() & 0x77);
  for (auto& v : h_B) v = static_cast<uint8_t>(rand() & 0x77);

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_SFA, h_SFA.data(), 32 * FP4_NUM_SF, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_SFB, h_SFB.data(), 32 * FP4_NUM_SF, cudaMemcpyHostToDevice));

  for (SwizzleMode mode : modes) {
    uint8_t layout_type;
    uint16_t lbo, sbo;
    compute_desc_params_fp4(mode, FP4_2SM_K_PER_STAGE, layout_type, lbo, sbo);
    uint32_t swizzle_mask = swizzle_mode_to_mask(mode);

    int total_mmas = FP4_2SM_K_BLOCKS * k_iters;

    auto run_for_pattern = [&](int wp_id, const char* wp_name) {
      BenchResult result = {};
      if (wp_id == 0)
        result = run_fp4_2sm_benchmark_config<0, M, N>(d_A, d_B, d_SFA, d_SFB, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else if (wp_id == 1)
        result = run_fp4_2sm_benchmark_config<1, M, N>(d_A, d_B, d_SFA, d_SFB, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);
      else
        result = run_fp4_2sm_benchmark_config<2, M, N>(d_A, d_B, d_SFA, d_SFB, d_D, d_cycles, d_fill_cycles, layout_type, lbo, sbo, swizzle_mask, k_iters);

      double cyc_per_mma = (double)result.mma_cycles / total_mmas;
      double latency_us = (double)result.mma_cycles * 1000.0 / (double)clock_rate_khz;

      if (csv_fp) {
        fprintf(csv_fp, "fp4_2sm,%dx%dx%d,%d,%d,%s,%s,%ld,%.1f,%.1f,%.1f,%ld\n",
                M, N, FP4_2SM_MMA_K, FP4_2SM_K_PER_STAGE, 1,
                swizzle_mode_name(mode), wp_name,
                (long)result.mma_cycles, cyc_per_mma, latency_us,
                (double)result.wall_clock_us, (long)result.fill_cycles);
      }
    };

    run_for_pattern(0, "commit+wait each");
    run_for_pattern(1, "commit each, wait end");
    run_for_pattern(2, "commit+wait end");
  }

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_SFA));
  CUDA_CHECK(cudaFree(d_SFB));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_cycles));
  CUDA_CHECK(cudaFree(d_fill_cycles));
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED (shape drivers)

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

  // Open CSV file for performance results
  FILE* csv_fp = fopen("benchmark_results.csv", "w");
  if (csv_fp) {
    fprintf(csv_fp, "precision,mma_shape,k_per_stage,num_stages,swizzle,wait_pattern,cycles,cyc_per_mma,latency_us,wall_us,fill_cycles\n");
  }

  // Performance sweep — all 3 descriptor configs run automatically
  run_performance_sweep(clock_rate_khz, k_iters, csv_fp);

  // Additional bf16 K-major shapes
  {
    bool shape_pass = true;
    test_bf16_shape_k<128, 128>(shape_pass, clock_rate_khz, k_iters, csv_fp);
    test_bf16_shape_k<64, 256>(shape_pass, clock_rate_khz, k_iters, csv_fp);
    test_bf16_shape_k<128, 64>(shape_pass, clock_rate_khz, k_iters, csv_fp);
  }

  // ========== bf16 MN-major Tests ==========

  printf("\n=== bf16 MN-major Tests ===\n\n");

  for (int k : k_values) {
    all_pass &= run_correctness_test_mn(SwizzleMode::SW_128B, k);
    all_pass &= run_correctness_test_mn(SwizzleMode::SW_64B, k);
    all_pass &= run_correctness_test_mn(SwizzleMode::SW_32B, k);
    all_pass &= run_correctness_test_mn(SwizzleMode::SW_NONE, k);
  }

  if (!all_pass) {
    printf("bf16 MN-major CORRECTNESS TESTS FAILED. Aborting benchmark.\n");
    if (csv_fp) fclose(csv_fp);
    return 1;
  }

  printf("All bf16 MN-major correctness tests passed.\n\n");

  run_performance_sweep_mn(clock_rate_khz, k_iters, csv_fp);

  // Additional bf16 MN-major shapes
  {
    bool shape_pass = true;
    test_bf16_shape_mn<128, 128>(shape_pass, clock_rate_khz, k_iters, csv_fp);
    test_bf16_shape_mn<64, 256>(shape_pass, clock_rate_khz, k_iters, csv_fp);
    test_bf16_shape_mn<128, 64>(shape_pass, clock_rate_khz, k_iters, csv_fp);
  }

  // ========== nvfp4 (mxf4nvf4) Tests ==========

  printf("\n=== nvfp4 (mxf4nvf4) Tests ===\n\n");

  // Correctness: 4 swizzle modes
  all_pass &= run_fp4_correctness_test(SwizzleMode::SW_128B);
  all_pass &= run_fp4_correctness_test(SwizzleMode::SW_64B);
  all_pass &= run_fp4_correctness_test(SwizzleMode::SW_32B);
  all_pass &= run_fp4_correctness_test(SwizzleMode::SW_NONE);

  if (!all_pass) {
    printf("nvfp4 CORRECTNESS TESTS FAILED. Aborting benchmark.\n");
    if (csv_fp) fclose(csv_fp);
    return 1;
  }

  printf("All nvfp4 correctness tests passed.\n\n");

  // Performance sweep: 4 swizzle modes x 3 wait patterns = 12 rows
  run_fp4_performance_sweep(clock_rate_khz, k_iters, csv_fp);

  // Additional nvfp4 shapes
  {
    bool shape_pass = true;
    test_fp4_shape<256, 256>(shape_pass, clock_rate_khz, k_iters, csv_fp);
    test_fp4_shape<128, 128>(shape_pass, clock_rate_khz, k_iters, csv_fp);
    test_fp4_shape<128, 64>(shape_pass, clock_rate_khz, k_iters, csv_fp);
  }

  // ========== fp8 (f8f6f4) Tests ==========

  printf("\n=== fp8 (f8f6f4) Tests ===\n\n");

  // Correctness: 4 swizzle modes
  all_pass &= run_fp8_correctness_test(SwizzleMode::SW_128B);
  all_pass &= run_fp8_correctness_test(SwizzleMode::SW_64B);
  all_pass &= run_fp8_correctness_test(SwizzleMode::SW_32B);
  all_pass &= run_fp8_correctness_test(SwizzleMode::SW_NONE);

  if (!all_pass) {
    printf("fp8 CORRECTNESS TESTS FAILED. Aborting benchmark.\n");
    if (csv_fp) fclose(csv_fp);
    return 1;
  }

  printf("All fp8 correctness tests passed.\n\n");

  // Performance sweep: 4 swizzle modes x 3 wait patterns = 12 rows
  run_fp8_performance_sweep(clock_rate_khz, k_iters, csv_fp);

  // Additional fp8 K-major shapes
  {
    bool shape_pass = true;
    test_fp8_shape_k<128, 128>(shape_pass, clock_rate_khz, k_iters, csv_fp);
    test_fp8_shape_k<128, 64>(shape_pass, clock_rate_khz, k_iters, csv_fp);
    test_fp8_shape_k<128, 32>(shape_pass, clock_rate_khz, k_iters, csv_fp);
  }

  // ========== fp8 MN-major Tests ==========

  printf("\n=== fp8 MN-major (f8f6f4) Tests ===\n\n");

  all_pass &= run_fp8_correctness_test_mn(SwizzleMode::SW_128B);
  all_pass &= run_fp8_correctness_test_mn(SwizzleMode::SW_64B);
  all_pass &= run_fp8_correctness_test_mn(SwizzleMode::SW_32B);
  all_pass &= run_fp8_correctness_test_mn(SwizzleMode::SW_NONE);

  if (!all_pass) {
    printf("fp8 MN-major CORRECTNESS TESTS FAILED. Aborting benchmark.\n");
    if (csv_fp) fclose(csv_fp);
    return 1;
  }

  printf("All fp8 MN-major correctness tests passed.\n\n");

  run_fp8_performance_sweep_mn(clock_rate_khz, k_iters, csv_fp);

  // Additional fp8 MN-major shapes
  {
    bool shape_pass = true;
    test_fp8_shape_mn<128, 128>(shape_pass, clock_rate_khz, k_iters, csv_fp);
    test_fp8_shape_mn<128, 64>(shape_pass, clock_rate_khz, k_iters, csv_fp);
    test_fp8_shape_mn<128, 32>(shape_pass, clock_rate_khz, k_iters, csv_fp);
  }

  // ========== bf16 2SM (cta_group::2) Tests ==========

  printf("\n=== bf16 2SM (cta_group::2) Tests ===\n\n");

  // Correctness: 4 swizzle modes
  all_pass &= run_bf16_2sm_correctness_test(SwizzleMode::SW_128B);
  all_pass &= run_bf16_2sm_correctness_test(SwizzleMode::SW_64B);
  all_pass &= run_bf16_2sm_correctness_test(SwizzleMode::SW_32B);
  all_pass &= run_bf16_2sm_correctness_test(SwizzleMode::SW_NONE);

  if (!all_pass) {
    printf("bf16 2SM CORRECTNESS TESTS FAILED. Aborting benchmark.\n");
    if (csv_fp) fclose(csv_fp);
    return 1;
  }

  printf("All bf16 2SM correctness tests passed.\n\n");

  // Performance sweep: 4 swizzle modes x 3 wait patterns = 12 rows
  run_bf16_2sm_performance_sweep(clock_rate_khz, k_iters, csv_fp);

  // Additional bf16 2SM K-major shapes
  {
    bool shape_pass = true;
    test_2sm_shape_k<256, 128>(shape_pass, clock_rate_khz, k_iters, csv_fp);
    test_2sm_shape_k<256, 64>(shape_pass, clock_rate_khz, k_iters, csv_fp);
  }

  // ========== bf16 2SM MN-major Tests ==========

  printf("\n=== bf16 2SM MN-major (cta_group::2) Tests ===\n\n");

  all_pass &= run_bf16_2sm_correctness_test_mn(SwizzleMode::SW_128B);
  all_pass &= run_bf16_2sm_correctness_test_mn(SwizzleMode::SW_64B);
  all_pass &= run_bf16_2sm_correctness_test_mn(SwizzleMode::SW_32B);
  all_pass &= run_bf16_2sm_correctness_test_mn(SwizzleMode::SW_NONE);

  if (!all_pass) {
    printf("bf16 2SM MN-major CORRECTNESS TESTS FAILED. Aborting benchmark.\n");
    if (csv_fp) fclose(csv_fp);
    return 1;
  }

  printf("All bf16 2SM MN-major correctness tests passed.\n\n");

  run_bf16_2sm_performance_sweep_mn(clock_rate_khz, k_iters, csv_fp);

  // Additional bf16 2SM MN-major shapes
  {
    bool shape_pass = true;
    test_2sm_shape_mn<256, 128>(shape_pass, clock_rate_khz, k_iters, csv_fp);
    test_2sm_shape_mn<256, 64>(shape_pass, clock_rate_khz, k_iters, csv_fp);
  }

  // ========== fp8 2SM (cta_group::2) K-major Tests ==========

  printf("\n=== fp8 2SM (cta_group::2) K-major Tests ===\n\n");

  all_pass &= run_fp8_2sm_correctness_test(SwizzleMode::SW_128B);
  all_pass &= run_fp8_2sm_correctness_test(SwizzleMode::SW_64B);
  all_pass &= run_fp8_2sm_correctness_test(SwizzleMode::SW_32B);
  all_pass &= run_fp8_2sm_correctness_test(SwizzleMode::SW_NONE);

  if (!all_pass) {
    printf("fp8 2SM K-major CORRECTNESS TESTS FAILED. Aborting benchmark.\n");
    if (csv_fp) fclose(csv_fp);
    return 1;
  }

  printf("All fp8 2SM K-major correctness tests passed.\n\n");

  run_fp8_2sm_performance_sweep(clock_rate_khz, k_iters, csv_fp);

  // Additional fp8 2SM K-major shapes
  {
    bool shape_pass = true;
    test_fp8_2sm_shape_k<256, 128>(shape_pass, clock_rate_khz, k_iters, csv_fp);
    test_fp8_2sm_shape_k<256, 64>(shape_pass, clock_rate_khz, k_iters, csv_fp);
  }

  // ========== fp8 2SM MN-major Tests ==========

  printf("\n=== fp8 2SM MN-major (cta_group::2) Tests ===\n\n");

  all_pass &= run_fp8_2sm_correctness_test_mn(SwizzleMode::SW_128B);
  all_pass &= run_fp8_2sm_correctness_test_mn(SwizzleMode::SW_64B);
  all_pass &= run_fp8_2sm_correctness_test_mn(SwizzleMode::SW_32B);
  all_pass &= run_fp8_2sm_correctness_test_mn(SwizzleMode::SW_NONE);

  if (!all_pass) {
    printf("fp8 2SM MN-major CORRECTNESS TESTS FAILED. Aborting benchmark.\n");
    if (csv_fp) fclose(csv_fp);
    return 1;
  }

  printf("All fp8 2SM MN-major correctness tests passed.\n\n");

  run_fp8_2sm_performance_sweep_mn(clock_rate_khz, k_iters, csv_fp);

  // Additional fp8 2SM MN-major shapes
  {
    bool shape_pass = true;
    test_fp8_2sm_shape_mn<256, 128>(shape_pass, clock_rate_khz, k_iters, csv_fp);
    test_fp8_2sm_shape_mn<256, 64>(shape_pass, clock_rate_khz, k_iters, csv_fp);
  }

  // ========== fp4 2SM (cta_group::2) K-major Tests ==========

  printf("\n=== fp4 2SM (cta_group::2) K-major Tests ===\n\n");

  all_pass &= run_fp4_2sm_correctness_test(SwizzleMode::SW_128B);
  all_pass &= run_fp4_2sm_correctness_test(SwizzleMode::SW_64B);
  all_pass &= run_fp4_2sm_correctness_test(SwizzleMode::SW_32B);
  all_pass &= run_fp4_2sm_correctness_test(SwizzleMode::SW_NONE);

  if (!all_pass) {
    printf("fp4 2SM K-major CORRECTNESS TESTS FAILED. Aborting benchmark.\n");
    if (csv_fp) fclose(csv_fp);
    return 1;
  }

  printf("All fp4 2SM K-major correctness tests passed.\n\n");

  run_fp4_2sm_performance_sweep(clock_rate_khz, k_iters, csv_fp);

  // Additional fp4 2SM K-major shapes
  {
    bool shape_pass = true;
    test_fp4_2sm_shape_k<256, 128>(shape_pass, clock_rate_khz, k_iters, csv_fp);
    test_fp4_2sm_shape_k<256, 64>(shape_pass, clock_rate_khz, k_iters, csv_fp);
  }

  if (csv_fp) {
    fclose(csv_fp);
    printf("Performance results written to benchmark_results.csv\n");
  }

#else
  std::cout << "CUTLASS_ARCH_MMA_SM100_SUPPORTED must be enabled. Test is waived.\n" << std::endl;
#endif

  return 0;
}
