#!/bin/bash
# Profile a benchmark target with ncu for detailed kernel metrics
# Usage: bash scripts/run_ncu.sh <TARGET> [SHAPES] [BUILD_DIR]
#   e.g.: bash scripts/run_ncu.sh 95a_bench_fp16_gemm 256x256x64

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUTLASS_DIR="$(cd "${SCRIPT_DIR}/../../../" && pwd)"

TARGET="${1:?Usage: run_ncu.sh <TARGET> [SHAPES] [BUILD_DIR]}"
SHAPES="${2:-256x256x64}"
BUILD_DIR="${3:-${CUTLASS_DIR}/build}"
OUTPUT_DIR="${SCRIPT_DIR}/../results"

mkdir -p "${OUTPUT_DIR}"

BIN=$(find "${BUILD_DIR}" -name "${TARGET}" -type f -executable 2>/dev/null | head -1)
if [ -z "${BIN}" ]; then
  echo "Error: ${TARGET} not found in ${BUILD_DIR}"
  exit 1
fi

OUTPUT="${OUTPUT_DIR}/${TARGET}_ncu"
echo "Profiling ${TARGET} with ncu..."
echo "  Shapes: ${SHAPES}"
echo "  Output: ${OUTPUT}"

ncu \
  --set full \
  --launch-skip 5 \
  --launch-count 3 \
  -o "${OUTPUT}" \
  "${BIN}" --shapes="${SHAPES}" --iterations=10

echo "Done. View with: ncu-ui ${OUTPUT}.ncu-rep"
