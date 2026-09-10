#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo 'Usage: run-gguf.sh BUILD_DIR MODEL.gguf [extra llama-completion arguments]' >&2
    exit 1
fi
build=$1
model=$2
shift 2
export GGML_MOE_STREAM=1
export GGML_MOE_CACHE_MIB=${GGML_MOE_CACHE_MIB:-256}
export GGML_MOE_MAX_EXPERTS=${GGML_MOE_MAX_EXPERTS:-32}
export GGML_MOE_TRACE=${GGML_MOE_TRACE:-1}
extra=()
if [[ ${EDGE0_GPU_LAYERS:-0} == 0 ]]; then
    extra+=(--no-op-offload)
fi
exec "$build/bin/llama-completion" -m "$model" -ngl "${EDGE0_GPU_LAYERS:-0}" \
    -c 2048 -b 128 -ub 128 -n 64 -t "${EDGE0_THREADS:-4}" \
    --temp 0 -p 'The capital of France is' "${extra[@]}" "$@"
