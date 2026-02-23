#!/bin/bash
# Run all benchmarks and collect CSV output
# Usage: bash scripts/run_all.sh [BUILD_DIR] [ITERATIONS]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUTLASS_DIR="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
BUILD_DIR="${1:-${CUTLASS_DIR}/build}"
ITERATIONS="${2:-100}"
OUTPUT_DIR="${SCRIPT_DIR}/../results"

mkdir -p "${OUTPUT_DIR}"

find_binary() {
  find "${BUILD_DIR}" -name "$1" -type f -executable 2>/dev/null | head -1
}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Blackwell GEMM Benchmark Suite ==="
echo "Iterations: ${ITERATIONS}"
echo "Output: ${OUTPUT_DIR}/"
echo ""

# FP16 benchmark
BIN=$(find_binary "95a_bench_fp16_gemm")
if [ -n "${BIN}" ]; then
  echo "--- Running FP16 benchmark ---"
  "${BIN}" --iterations="${ITERATIONS}" --csv \
    > "${OUTPUT_DIR}/llm_proj_fp16_${TIMESTAMP}.csv"
  echo "  Saved to ${OUTPUT_DIR}/llm_proj_fp16_${TIMESTAMP}.csv"
else
  echo "  SKIP: 95a_bench_fp16_gemm not found"
fi

# FP8 benchmark
BIN=$(find_binary "95b_bench_fp8_gemm")
if [ -n "${BIN}" ]; then
  echo "--- Running FP8 benchmark ---"
  "${BIN}" --iterations="${ITERATIONS}" --csv \
    > "${OUTPUT_DIR}/llm_proj_fp8_${TIMESTAMP}.csv"
  echo "  Saved to ${OUTPUT_DIR}/llm_proj_fp8_${TIMESTAMP}.csv"
else
  echo "  SKIP: 95b_bench_fp8_gemm not found"
fi

# NVFP4 benchmark
BIN=$(find_binary "95c_bench_nvfp4_gemm")
if [ -n "${BIN}" ]; then
  echo "--- Running NVFP4 benchmark ---"
  "${BIN}" --iterations="${ITERATIONS}" --csv \
    > "${OUTPUT_DIR}/llm_proj_nvfp4_${TIMESTAMP}.csv"
  echo "  Saved to ${OUTPUT_DIR}/llm_proj_nvfp4_${TIMESTAMP}.csv"
else
  echo "  SKIP: 95c_bench_nvfp4_gemm not found"
fi

# Combine all CSVs
COMBINED="${OUTPUT_DIR}/llm_proj_all_results_${TIMESTAMP}.csv"
echo "precision,config,M,N,K,mma_tile_shape,cluster_shape,mainloop_schedule,epilogue_schedule,stage_count,tile_scheduler,kernel_ms,init_run_ms,init_ms,GFLOPS,verified" > "${COMBINED}"
for f in "${OUTPUT_DIR}/llm_proj_fp16_${TIMESTAMP}.csv" \
         "${OUTPUT_DIR}/llm_proj_fp8_${TIMESTAMP}.csv" \
         "${OUTPUT_DIR}/llm_proj_nvfp4_${TIMESTAMP}.csv"; do
  if [ -f "$f" ]; then
    # Skip header line from each file
    tail -n +2 "$f" >> "${COMBINED}"
  fi
done
echo ""
echo "Combined results: ${COMBINED}"
echo ""
echo "Done."
