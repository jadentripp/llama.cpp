# Intel Mac AMD Metal support

This fork keeps current llama.cpp model support while fixing the private-buffer path used by Intel Macs with discrete AMD Radeon GPUs. It was developed and tested on a MacBookPro16,1 with an AMD Radeon Pro 5500M 8 GB.

## What changes

- Model weights use private GPU buffers on discrete Metal devices instead of mapped system-memory buffers.
- Private-buffer uploads and downloads use owned shared staging buffers, so unaligned host pointers and non-page-sized transfers are safe.
- Readback uses the synchronized generic path instead of wrapping arbitrary caller memory in a Metal bytes-no-copy buffer.
- Concurrent command encoding stays enabled on unified-memory devices and is disabled on legacy discrete devices, where multi-node graphs can race.
- Incorrect legacy discrete kernels use another backend: `TIMESTEP_EMBEDDING`, `ARGSORT` above 8192 columns, and `TOP_K` above 4096 columns.
- Unified-memory Metal devices keep the upstream mapped-buffer and concurrency behavior.

The fallback decisions are based on device memory behavior rather than model names, so they apply to every model architecture supported by the upstream revision. An unsupported Metal operation can run on the CPU through the normal backend scheduler.

## Build

Metal is enabled by default on macOS.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build --target llama-cli llama-bench test-backend-ops test-metal-private-buffer -j 8
```

## Verify the private-buffer path

```sh
build/bin/test-metal-private-buffer
build/bin/test-backend-ops test -b MTL0
```

The focused test covers unaligned host offsets, odd transfer sizes, nonzero tensor offsets, partial fills, sentinels, and strided asynchronous copies. It prints the detected Metal device or skips cleanly on machines without a private Metal buffer type.

To verify model residency, run with info logging and look for `MTL0_Private model buffer size`:

```sh
build/bin/llama-cli -m /path/to/model.gguf -c 4096 -ngl 99 -lv 4
```

## Qwen3.5 example

```sh
build/bin/llama-cli -m /path/to/Qwen3.5-0.8B-Q4_K_M.gguf -c 4096 -ngl 99 -fa off -cnv
build/bin/llama-bench -m /path/to/Qwen3.5-0.8B-Q4_K_M.gguf -p 512 -n 128 -r 5 -ngl 99 -fa off
```

The full Qwen3.5 0.8B Q4_K_M file is 542.31 MiB. On the test machine, 521.57 MiB of model data, a 48.00 MiB KV buffer, a 19.27 MiB recurrent-state buffer, and a 94.02 MiB compute buffer were allocated as `MTL0_Private` at a 4096-token context.

A thermally stable five-run benchmark of the same Metal patch at build 10576 measured 256.89 +/- 1.43 tokens/s for prompt processing at 512 tokens and 66.83 +/- 1.33 tokens/s for generation at 128 tokens. The final fast-forward to build 10580 did not change Metal code. Performance and allocation sizes vary with the model, context, prompt, hardware, and device temperature.

## Validation snapshot

Validation used upstream commit `54ee5ee643f29abba6852903ddfdb688c2361b5b` (build 10580):

- Private-buffer regression test: passed.
- Full supported Metal backend operator suite: 9138/9138 passed.
- Qwen-related operator suite: 3388/3388 passed.
- Qwen3.5 0.8B Q4_K_M deterministic generation: returned `VRAM_OK` with full GPU offload.

The large sort and timestep cases listed above are intentionally excluded from Metal support on legacy discrete devices and remain available through CPU fallback. This fork does not claim that the affected GPU kernels are repaired; it prevents them from producing incorrect results.
