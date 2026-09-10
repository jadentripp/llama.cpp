#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
backend=${1:-cpu}
if [[ $# -gt 0 ]]; then shift; fi
if [[ $backend != cpu && $backend != vulkan ]]; then
    echo 'Usage: build-android.sh [cpu|vulkan] [extra CMake arguments]' >&2
    exit 1
fi
: "${ANDROID_NDK:?Set ANDROID_NDK to the unpacked Android NDK directory}"
build=${EDGE0_BUILD_DIR:-"$root/build-edge0-android-$backend"}
extra=()
if [[ $backend == vulkan ]]; then
    extra+=(-DGGML_VULKAN=ON)
    source_headers=''
    if [[ -n ${VULKAN_SDK:-} && -f $VULKAN_SDK/include/vulkan/vulkan.hpp ]]; then
        source_headers=$VULKAN_SDK/include
    elif [[ -f /usr/include/vulkan/vulkan.hpp ]]; then
        source_headers=/usr/include
    fi
    if [[ -n $source_headers ]]; then
        # CMake can remove /usr/include as an implicit host path during an NDK
        # cross-build. Copy the platform-neutral Vulkan/SPIR-V headers so Android
        # never searches the host's libc headers and still finds vulkan.hpp.
        mkdir -p "$build/vulkan-headers"
        cp -R "$source_headers/vulkan" "$build/vulkan-headers/"
        if [[ -d $source_headers/vk_video ]]; then
            cp -R "$source_headers/vk_video" "$build/vulkan-headers/"
        fi
        if [[ -d $source_headers/spirv ]]; then
            cp -R "$source_headers/spirv" "$build/vulkan-headers/"
        fi
        extra+=("-DVulkan_INCLUDE_DIR=$build/vulkan-headers")
    fi
    if command -v glslc >/dev/null; then
        extra+=("-DVulkan_GLSLC_EXECUTABLE=$(command -v glslc)")
    fi
    if [[ -d /usr/share/cmake/SPIRV-Headers ]]; then
        extra+=(-DSPIRV-Headers_DIR=/usr/share/cmake/SPIRV-Headers)
    fi
else
    extra+=(-DGGML_VULKAN=OFF)
fi
cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_static \
    -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_BACKEND_DL=OFF \
    -DBUILD_SHARED_LIBS=OFF -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=ON "${extra[@]}" "$@"
cmake --build "$build" --config Release --target \
    llama-completion llama-edge0 llama-bench test-moe-stream -j "${EDGE0_JOBS:-4}"
strip_tools=("$ANDROID_NDK"/toolchains/llvm/prebuilt/*/bin/llvm-strip)
if [[ ${#strip_tools[@]} == 1 && -x ${strip_tools[0]} ]]; then
    "${strip_tools[0]}" --strip-unneeded "$build/bin/llama-completion" \
        "$build/bin/llama-edge0" "$build/bin/llama-bench" "$build/bin/test-moe-stream"
fi
echo "Built Android arm64 executables in $build/bin"
