#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ $(uname -s) != Darwin || $(uname -m) != x86_64 ]]; then
    echo 'Run this script on the Intel Mac with Xcode command-line tools installed.' >&2
    exit 1
fi
build=${EDGE0_BUILD_DIR:-"$root/build-edge0-metal"}
cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_NATIVE=OFF \
    -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple -DGGML_BACKEND_DL=OFF \
    -DBUILD_SHARED_LIBS=OFF -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=ON "$@"
cmake --build "$build" --config Release --target \
    llama-completion llama-edge0 llama-bench test-moe-stream -j "${EDGE0_JOBS:-4}"
echo "Built $build/bin"
echo "List GPUs: $build/bin/llama-completion --list-devices"
echo "Check streaming: $build/bin/test-moe-stream MTL0"
