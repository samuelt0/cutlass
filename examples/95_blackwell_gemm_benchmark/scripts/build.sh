#!/bin/bash
# Build all benchmark targets for example 95
# Usage: bash scripts/build.sh [BUILD_DIR]
#   BUILD_DIR defaults to ./build

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUTLASS_DIR="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
BUILD_DIR="${1:-${CUTLASS_DIR}/build}"

if [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
  echo "Build directory not configured. Run cmake first:"
  echo "  mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}"
  echo "  cmake ${CUTLASS_DIR} -DCUTLASS_NVCC_ARCHS=100a"
  exit 1
fi

echo "Building benchmark targets in ${BUILD_DIR}..."
cmake --build "${BUILD_DIR}" \
  --target 95a_bench_fp16_gemm 95b_bench_fp8_gemm 95c_bench_nvfp4_gemm \
  -j"$(nproc)"

echo "Build complete."
echo "Binaries:"
for target in 95a_bench_fp16_gemm 95b_bench_fp8_gemm 95c_bench_nvfp4_gemm; do
  bin=$(find "${BUILD_DIR}" -name "${target}" -type f -executable 2>/dev/null | head -1)
  if [ -n "${bin}" ]; then
    echo "  ${bin}"
  else
    echo "  ${target}: NOT FOUND"
  fi
done
