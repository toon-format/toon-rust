#!/bin/bash
# CUDA Kernel Build Script for Geoshi Polynomial Transformation
# Target: RTX 4080 Laptop GPU (Compute Capability 8.9)
# 
# © 2025 ArcMoon Studios • SPDX-License-Identifier: MIT OR Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
KERNELS_DIR="${PROJECT_ROOT}/crates/geoshi/kernels"
BUILD_DIR="${PROJECT_ROOT}/target/cuda-build"

echo "🚀 Building CUDA Kernels for Geoshi"
echo "📍 Project Root: ${PROJECT_ROOT}"
echo "📁 Kernels Dir:  ${KERNELS_DIR}"
echo "🏗️  Build Dir:   ${BUILD_DIR}"
echo ""

# Create build directory
mkdir -p "${BUILD_DIR}"

# Check for nvcc
if ! command -v nvcc &> /dev/null; then
    echo "❌ Error: nvcc not found in PATH"
    echo "   Please install NVIDIA CUDA Toolkit"
    echo "   Download: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

echo "✅ Found nvcc: $(nvcc --version | head -n1)"
echo ""

echo "🔧 Compiling mgsk.cu (Geoshi ops bundle)..."
nvcc -ptx \
    -arch=sm_89 \
    -O3 \
    -use_fast_math \
    --generate-line-info \
    -Xptxas -v \
    --maxrregcount=32 \
    -I/usr/local/cuda/include \
    "${KERNELS_DIR}/mgsk.cu" \
    -o "${BUILD_DIR}/mgsk.ptx"

if [ $? -eq 0 ]; then
    echo "✅ mgsk.cu compiled successfully"
    echo "📄 Output: ${BUILD_DIR}/mgsk.ptx"
else
    echo "❌ Failed to compile mgsk.cu"
    exit 1
fi

# Export environment variables for Rust integration
export MGSK_PTX="${BUILD_DIR}/mgsk.ptx"

echo ""
echo "🎯 Environment Variables Set:"
echo "   MGSK_PTX=${MGSK_PTX}"
echo ""
echo "💡 For Rust integration, set these environment variables:"
echo "   export MGSK_PTX=${MGSK_PTX}"
echo ""
echo "🏗️  Build completed successfully!"
echo "🎯 Target: RTX 4080 Laptop GPU (Ada Lovelace)"
echo "⚡ Bundle: distance, lattice, convolution, mesh kernels"
