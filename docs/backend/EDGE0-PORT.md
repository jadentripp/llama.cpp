# Experimental Edge0 port: Intel Mac and Android

This fork includes opt-in compact MoE expert streaming and an experimental runner for matched Edge0 base, Recover-LoRA, and prerouter checkpoints. Existing GGUF models can use the streaming path without Edge0 adapters. The trained runner is a separate, single-sequence command, `llama-edge0`.

## Revisions

- Branch: `work/edge0-port-20260910` in `jadentripp/llama.cpp`.
- Upstream merged: `d344123fe2de081a72e02d6869360dfcbc0b528b` (276 commits beyond the original fork).
- Edge0 reference: `a38d3dae7ed9c24d44f62c455f3c7ac67345d844`.
- Validated 8B release: `Edge0/Edge0-8B-A1B-preview`, revision `0bf17abed23b7de4b66e265a3848e80447ee7b41`.
- 35B profile: `Edge0/Edge0-35B-A3B-preview`, revision `1ff9f4478890faec0368c5463b621d1036d5b518`.

The merge retains this fork's Intel/AMD Metal private buffers, bounded 16 MiB transfer staging, queue serialization, discrete-device operator fallbacks, and recurrent-state test improvements. `AGENTS.md` now authorizes autonomous development, dependency setup, tests, upstream merges, commits, and pushes within this fork.

## What has been verified

| Check | Result |
| --- | --- |
| Linux x86_64 build | Passed for completion, benchmark, trained runner, and streaming tests |
| Compact expert execution | 18 CPU cases passed: F32, Q4_K, Q4_1; cache off/on; single-token and batched execution; strided IDs; duplicate IDs; graph reuse; eviction; oversized-batch fallback |
| Vulkan API execution | 18 cases passed on Mesa llvmpipe software Vulkan, including cache copies/eviction; this does not emulate the Pixel GPU |
| Conversion math | 6 numerical tests passed for affine int4/int8, exact coefficient checks, Qwen value-head layouts, and split MLA LoRA |
| Real Edge0 8B release | Converted base plus all released LoRA targets and 16 routing heads; generated a correct short answer to a capital-of-France prompt |
| Streamed versus resident 8B | All 1,257,472 logits from 8 generation steps were equal in the Linux CPU check |
| Checkpoint replay | Model state plus routing state reproduced the next-step logits exactly |
| Android arm64 CPU | Cross-compiled with NDK r28c, API 28; system-library-only executable dependencies |
| Android arm64 Vulkan | Cross-compiled with NDK r28c, host glslc, Vulkan and SPIR-V headers |
| Intel Mac / Pixel execution | Not run on physical devices in this environment |
| Qwen 35B metadata and adapters | 733 base tensor names mapped; 620 LoRA factors and 99 head matrices converted; tokenizer metadata prepared |
| Full 35B model generation | Not run; full-model conversion and inference remain experimental |
| Parity with MLX / quality benchmark | Not established; CPU equivalence above compares two paths in this fork |

These checks establish a testable prototype. They do not establish target-device speed, thermal behavior, or equality with Edge0's MLX activations. The runner uses GGML backbone arithmetic and fp16 head cast boundaries; ties and rounding can differ from MLX.

## Build and try an existing GGUF on the Intel Mac

Run in your checkout on the Intel Mac. CMake and Xcode command-line tools are required.

```sh
git fetch origin
git switch work/edge0-port-20260910
bash scripts/edge0/build-mac.sh
build-edge0-metal/bin/llama-completion --list-devices
build-edge0-metal/bin/test-moe-stream MTL0
EDGE0_GPU_LAYERS=99 bash scripts/edge0/run-gguf.sh build-edge0-metal /absolute/path/model.gguf
```

Use the backend name printed by `--list-devices` if it differs from `MTL0`. The Mac build uses the existing Metal path and Accelerate; the compact scheduler is backend-independent. The Radeon Pro 5500M's 8 GB VRAM and the machine's system RAM are separate budgets. Start with a small context and inspect actual allocations before increasing them.

`run-gguf.sh` defaults to CPU unless `EDGE0_GPU_LAYERS` is set. For a direct resident comparison, invoke `llama-completion` with the same model and generation arguments and `GGML_MOE_STREAM=0`. Do not attach Edge0's learned adapters to an unrelated GGUF or fine-tune.

## Convert a matched Edge0 release

Convert on a desktop, then copy the three GGUFs to the target device. The converter needs Python 3.11 or newer, NumPy, and the normal Hugging Face conversion dependencies. This does not require MLX, Apple Silicon, or downloading remote Python model code.

```sh
python3 -m venv .venv-edge0
. .venv-edge0/bin/activate
python -m pip install -r requirements/requirements-convert_hf_to_gguf.txt
hf download Edge0/Edge0-8B-A1B-preview --revision 0bf17abed23b7de4b66e265a3848e80447ee7b41 --local-dir models/edge0-8b-mlx
python convert_edge0_to_gguf.py models/edge0-8b-mlx --output-dir models/edge0-8b-gguf
python scripts/edge0/render-prompt.py models/edge0-8b-mlx 'What is the capital of France?' > prompt.txt
```

Outputs are `base.gguf`, `lora.gguf`, and `prerouter.gguf`. The tested 8B conversion produced approximately 5.42 GiB, 18.55 MiB, and 37.00 MiB respectively. Allow space for both the downloaded source and the output. Conversion reads tensors twice to avoid a full dequantized intermediate model.

The 35B profile uses the same command with `Edge0/Edge0-35B-A3B-preview` and its pinned revision above. It is not a device-tested 35B release of this fork.

MLX affine group-64 int4 is repacked to GGUF Q4_1 by preserving all codes and duplicating coefficients for its 32-value blocks. This is not Q4_K_M. Mixed int8 router weights are dequantized to F32. When an original coefficient cannot round-trip through fp16, the complete tensor is stored in F32 with a warning. The tested 8B embedding required this fallback; expert tensors remained int4. MLA key/value projections that change quantization axes are also preserved in F32.

The converter applies the appropriate Qwen value-head permutations, avoids adding the HF norm offset twice to sanitized MLX weights, and transforms the split Ling MLA LoRA factors without merging them into the base. Original unused base pregate tensors are omitted in favor of the supplied trained heads. The SHA-256 bundle identifier covers the actual model and adapter files, not merely their dimensions. The runner rejects mismatched base/LoRA/head sets.

## Run the trained pipeline on the Mac

```sh
GGML_MOE_STREAM=1 GGML_MOE_CACHE_MIB=512 GGML_MOE_TRACE=1 \
  build-edge0-metal/bin/llama-edge0 \
  --bundle models/edge0-8b-gguf --prompt-file prompt.txt \
  --gpu-layers 99 --threads 4 --context 2048 --batch 128 --tokens 64
```

The command uses greedy generation and accepts a raw prompt. `render-prompt.py` applies the release's own chat template with thinking disabled. `--verify-replay` additionally tests save/restore of the model and routing state before generating. Use `--help` for the small set of supported options.

The Qwen profile uses K=4, heads owned by layers 6..38, and executed current/previous-token one-hot features. The pinned Edge0 installer patches owner blocks only, so its last layer retains the native router; this runner preserves that behavior. Multi-token Qwen prefill uses native routing. Ling uses K=8, owner layers 7..22, executed one-hot features, and unbiased sigmoid/group-limited predicted selection. Its prefill tail primes the first decode prediction. Comments in parts of the reference describe older owner layouts; this port follows the active release configuration and execution code.

## Build Android CPU and Vulkan executables

On a desktop with Android NDK r28c unpacked:

```sh
export ANDROID_NDK=/absolute/path/android-ndk-r28c
bash scripts/edge0/build-android.sh cpu
bash scripts/edge0/build-android.sh vulkan
```

Vulkan additionally needs a host `glslc`, Vulkan C/C++ headers, and SPIRV-Headers CMake package. On Ubuntu 24.04 these are supplied by `glslc`, `libvulkan-dev`, and `spirv-headers`. With the Vulkan SDK on macOS, source its environment setup first. Extra CMake arguments can be appended, for example `-DVulkan_INCLUDE_DIR=/path/to/include` or `-DSPIRV-Headers_DIR=/path/to/cmake/SPIRV-Headers`.

The build targets ordinary `arm64-v8a` Android. It does not depend on Qualcomm-specific kernels or Google's TPU. The fork's **Edge0 portable builds** GitHub Actions workflow also builds Intel Mac, Android CPU, and Android Vulkan artifacts when this branch is pushed. Build artifacts do not establish device execution results.

## Try it on the Pixel 9 Pro

Use USB debugging and `adb` from the desktop. Start with the CPU build and the 8B bundle. Run executables from `/data/local/tmp`, not Android shared storage.

```sh
adb shell mkdir -p /data/local/tmp/edge0/bin
adb push build-edge0-android-cpu/bin/llama-edge0 /data/local/tmp/edge0/bin/
adb push build-edge0-android-cpu/bin/llama-completion /data/local/tmp/edge0/bin/
adb push build-edge0-android-cpu/bin/llama-bench /data/local/tmp/edge0/bin/
adb push build-edge0-android-cpu/bin/test-moe-stream /data/local/tmp/edge0/bin/
adb push models/edge0-8b-gguf /data/local/tmp/edge0/
adb push prompt.txt /data/local/tmp/edge0/
adb shell
```

In the Android shell:

```sh
cd /data/local/tmp/edge0
chmod +x bin/*
./bin/test-moe-stream
GGML_MOE_STREAM=1 GGML_MOE_TRACE=1 ./bin/llama-edge0 \
  --bundle edge0-8b-gguf --prompt-file prompt.txt \
  --gpu-layers 0 --threads 4 --context 2048 --batch 128 --tokens 64
```

Then copy the four executables from `build-edge0-android-vulkan/bin` into a separate directory on the phone, preserving the CPU binaries for comparison. Run `llama-completion --list-devices`, then `test-moe-stream Vulkan0` using the printed device name. If that passes, try the trained runner with `--gpu-layers 99`, `GGML_MOE_STREAM=1`, and a 128 or 256 MiB expert cache. GPU fallback, memory pressure, and speed depend on the Pixel's actual driver and workload.

For an existing compatible GGUF, use `llama-completion` with `-m`, `-ngl`, `-c 2048`, `-b 128`, `-ub 128`, and `GGML_MOE_STREAM=1`. CPU-only runs can use `-ngl 0 --no-op-offload`. A Termux CPU build can use the normal CMake build below; NDK/ADB avoids installing the build toolchain on the phone.

Android's available shared memory must cover the OS, dense weights, KV/recurrent state, staging, and cache. File mappings and the OS page cache are additional accounting concerns. The expert-cache setting is not a process-memory limit. No full 35B fit or throughput claim is made for the Pixel.

## Streaming controls and implementation

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `GGML_MOE_STREAM` | `0` | Opt in to lazy expert mappings and compact `MUL_MAT_ID` execution |
| `GGML_MOE_MAX_EXPERTS` | `32` | Maximum possible active slots for one expert matrix operation; K times microbatch tokens must fit |
| `GGML_MOE_CACHE_MIB` | `256` | Device expert LRU capacity; `0` disables caching |
| `GGML_MOE_TRACE` | `0` | Print hits, misses, evictions, upload bytes, and cache allocation on teardown |
| `GGML_MOE_STREAM_CPU` | `0` | Force compact copies on CPU for correctness tests; usually slower than native mapped CPU weights |

Selected experts are sorted/deduplicated, copied into compact staging tensors, and addressed by locally remapped IDs. Other consumers keep the original router IDs and weights. The LRU distinguishes source tensor, expert, layout, and backend. Transfers are synchronized before shared staging reuse or cache eviction. Large prefill operations fall back to the CPU without dropping experts or allocating a full GPU expert layer.

The limit bounds cache allocations, separately from dense weights, activations, KV/recurrent state, and allocator-owned staging. This implementation retains the fork's synchronous Metal staging guarantees. It does not yet provide Edge0's asynchronous SSD prefetch and transfer/compute overlap, and it does not promise an MLX-equivalent speedup. CPU execution normally reads mapped experts directly rather than caching duplicate CPU copies.

The experimental trained runner owns one sequence and its routing state. It supports local snapshot replay for testing. The learned mode is not wired into `llama-server`, multi-user slot scheduling, speculative decoding, persistent sessions, or an Android GUI. Exact streaming without trained routing works through the regular scheduler used by existing applications.

## Reproduce the portable checks

```sh
cmake -S . -B build-port-check -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
  -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-port-check --target llama-edge0 llama-completion llama-bench test-moe-stream -j 4
build-port-check/bin/test-moe-stream
python tests/test-edge0-conversion.py
python scripts/edge0/check-streaming.py build-port-check/bin/llama-edge0 \
  models/edge0-8b-gguf --prompt-file prompt.txt --output comparison
```

`check-streaming.py` saves stdout, logs, raw F32 logits, and `comparison.json`. It performs model/routing checkpoint replay in both runs. `--gpu-layers` enables a device comparison, but its resident baseline must fit on the device. For a small synthetic GPU comparison use `test-moe-stream BACKEND` instead.

## Reference material

- [Pinned Edge0 repository](https://github.com/Edge0-AI/Edge0/tree/a38d3dae7ed9c24d44f62c455f3c7ac67345d844) (Apache-2.0): checkpoint conventions, routing features, and streaming design.
- [Existing Intel/AMD Metal notes](INTEL-AMD-METAL.md).
- [Upstream Android instructions](../android.md).
