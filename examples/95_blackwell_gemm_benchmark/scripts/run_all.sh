#!/bin/bash
# Run all benchmarks and collect CSV output
# Usage: bash scripts/run_all.sh [BUILD_DIR] [SHAPES] [ITERATIONS]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUTLASS_DIR="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
BUILD_DIR="${1:-${CUTLASS_DIR}/build}"
SHAPES="${2:-256x256x64,1024x1024x256,2048x2048x2048}"
ITERATIONS="${3:-100}"
OUTPUT_DIR="${SCRIPT_DIR}/../results"

mkdir -p "${OUTPUT_DIR}"

find_binary() {
  find "${BUILD_DIR}" -name "$1" -type f -executable 2>/dev/null | head -1
}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Blackwell GEMM Benchmark Suite ==="
echo "Shapes: ${SHAPES}"
echo "Iterations: ${ITERATIONS}"
echo "Output: ${OUTPUT_DIR}/"
echo ""

# FP16 benchmark
BIN=$(find_binary "95a_bench_fp16_gemm")
if [ -n "${BIN}" ]; then
  echo "--- Running FP16 benchmark ---"
  "${BIN}" --shapes="${SHAPES}" --iterations="${ITERATIONS}" --csv \
    > "${OUTPUT_DIR}/fp16_${TIMESTAMP}.csv"
  echo "  Saved to ${OUTPUT_DIR}/fp16_${TIMESTAMP}.csv"
else
  echo "  SKIP: 95a_bench_fp16_gemm not found"
fi

# FP8 benchmark (K >= 128 required for FP8 MMA tiles)
FP8_SHAPES="256x256x128,1024x1024x256,2048x2048x2048"
BIN=$(find_binary "95b_bench_fp8_gemm")
if [ -n "${BIN}" ]; then
  echo "--- Running FP8 benchmark ---"
  "${BIN}" --shapes="${FP8_SHAPES}" --iterations="${ITERATIONS}" --csv \
    > "${OUTPUT_DIR}/fp8_${TIMESTAMP}.csv"
  echo "  Saved to ${OUTPUT_DIR}/fp8_${TIMESTAMP}.csv"
else
  echo "  SKIP: 95b_bench_fp8_gemm not found"
fi

# NVFP4 benchmark (uses K>=256 shapes)
NVFP4_SHAPES="256x256x256,1024x1024x256,2048x2048x2048"
BIN=$(find_binary "95c_bench_nvfp4_gemm")
if [ -n "${BIN}" ]; then
  echo "--- Running NVFP4 benchmark ---"
  "${BIN}" --shapes="${NVFP4_SHAPES}" --iterations="${ITERATIONS}" --csv \
    > "${OUTPUT_DIR}/nvfp4_${TIMESTAMP}.csv"
  echo "  Saved to ${OUTPUT_DIR}/nvfp4_${TIMESTAMP}.csv"
else
  echo "  SKIP: 95c_bench_nvfp4_gemm not found"
fi

# Combine all CSVs
COMBINED="${OUTPUT_DIR}/all_results_${TIMESTAMP}.csv"
echo "precision,config,M,N,K,mma_tile_shape,cluster_shape,mainloop_schedule,epilogue_schedule,stage_count,tile_scheduler,kernel_ms,init_run_ms,init_ms,GFLOPS,verified" > "${COMBINED}"
for f in "${OUTPUT_DIR}/fp16_${TIMESTAMP}.csv" \
         "${OUTPUT_DIR}/fp8_${TIMESTAMP}.csv" \
         "${OUTPUT_DIR}/nvfp4_${TIMESTAMP}.csv"; do
  if [ -f "$f" ]; then
    # Skip header line from each file
    tail -n +2 "$f" >> "${COMBINED}"
  fi
done
echo ""
echo "Combined results: ${COMBINED}"
echo "Done."
