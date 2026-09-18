# Bonsai 2 27B throughput vs context (Radeon Pro 5500M)

Measured on a MacBookPro16,1 (Intel, 64 GB RAM) with an AMD Radeon Pro 5500M 8 GB. Model: `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5.53 GiB, 26.90 B, PTQ1_0). Runtime: this fork at `afb43d062` (PrismML `prism` plus discrete Metal private-buffer support), Metal build with embedded library.

This card thermally throttles. Numbers are single-run unless noted. They are not a substitute for a cooled, multi-rep bench.

There are two different “context” numbers. Mixing them is how 7 t/s at `-c 32000` gets mistaken for 7 t/s after a 32k-token prompt.

## Allocated context, short prompt

`llama-cli` with a ~18-token prompt. `-c` reserves the KV window. Attention only runs over the tokens actually present, so decode barely moves until the cache leaves the GPU or the process overcommits VRAM.

| `-c` | KV | Flash attn | Prompt t/s | Decode t/s | GPU residency |
| ---: | --- | --- | ---: | ---: | --- |
| 512 | f16 GPU | off | ~8 | 7.4–7.8 | fits |
| 1024 | q4_0 GPU | on | 7.3 | 6.3 | fits |
| 32000 | f16 GPU | off | 7.7 | 6.9 | fits (`--fit`, `-ub 64`) |
| 65792 | q8_0 GPU | on | 7.5 | 6.3 | fits |
| 124672 | q4_0 GPU | on | 7.4 | 6.6 | fits (~7750 / 8176 MiB) |
| 262144 | q4_0 GPU | on | 1.5 | 0.23 | **overcommit** (10177 MiB self on 8176 MiB card) |
| 262144 | q4_0 CPU (`-nkvo`) | on | 6.2 | 1.9 | GPU 5419 MiB weights; 4757 MiB KV+recurrent on host |

`--fit on -fitt 256` is what produced the 32k / 66k / 125k rows. Forcing `-c 262144 --fit off` on GPU does load, then pages. Do not use it.

Quantized V requires flash attention. `-fa off` cannot use `-ctv q4_0` / `q8_0`.

## Filled context

`llama-bench` on GPU, q4_0 K/V, flash attn on, `-ub 64`, one repetition. `-d N` stuffs N tokens into the cache, then measures generation.

| Test | t/s |
| --- | ---: |
| pp64 | 10.02 |
| tg16 | 7.31 |
| pp64 @ d128 | 7.75 |
| tg16 @ d128 | 6.12 |
| tg16 @ d256 (isolated rerun) | 4.57 |

A stacked run that also measured d256 in the same process dropped to 3.11 t/s decode / 4.10 t/s pp64. Treat that as heat, not a second architecture point.

A d32 run in a warm GPU reported 3.38 t/s decode, which is inconsistent with the cold 0→128→256 series and is discarded.

**32k filled was not measured.** Flash-attn prefill batches of 2048 and 512 tokens at `n_ctx=32000` failed with `llama_decode` status -3 (`GGML_STATUS_FAILED`). An 8-token-batch fill was still running after ~85 minutes and was killed. Do not restart that job; attention cost grows with depth, so the remaining fill only gets slower.

### Rough extrapolation (not measured)

From isolated tg16 at depth 0 (7.31 t/s ≈ 137 ms/token) and depth 256 (4.57 t/s ≈ 219 ms/token):

`ms/token ≈ 137 + 0.32 × n_filled`

| Filled tokens | Estimated decode t/s |
| ---: | ---: |
| 512 | ~3.3 |
| 2048 | ~1.3 |
| 8192 | ~0.4 |
| 32000 | ~0.10 |
| 124672 | ~0.025 |
| 262144 host KV | well below the empty-cache 1.9 t/s |

Those last rows are a straight line through two short points on a throttling laptop. They are only here so 32k is not confused with the 6.9 t/s empty-window number.

## What fits in 8 GB

Weights take 5395 MiB private Metal. Hybrid attention keeps KV smaller than a dense 27B, but 262144 × q4 KV is still 4608 MiB and does not fit next to the weights.

| KV | Max `-c` that `--fit` chose | KV buffer |
| --- | ---: | ---: |
| f16 | 32000 | 2000 MiB |
| q8_0 | 65792 | 2184 MiB |
| q4_0 | 124672 | 2191 MiB |

q4 K mean-centering (`--kv-mean-center`) was calibrated on this machine (2×256-token chunks, K rotation on) and loads for 16/16 layers. It was not quality-benched.

## Commands

Empty window, 32k f16:

```sh
build-metal/bin/llama-cli \
  -m /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 99 --fit on -fitt 256 -b 512 -ub 64 \
  -fa off --reasoning off -p "..."
```

Max in-VRAM window, q4:

```sh
build-metal/bin/llama-cli \
  -m /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --kv-mean-center /path/to/kv-mean-center.gguf \
  -ngl 99 --fit on -fitt 256 -b 512 -ub 64 \
  -ctk q4_0 -ctv q4_0 -fa on --reasoning off -p "..."
```

262k with KV in system RAM (slow decode, no GPU paging):

```sh
build-metal/bin/llama-cli \
  -m /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --kv-mean-center /path/to/kv-mean-center.gguf \
  -ngl 99 -c 262144 --fit off -nkvo \
  -ctk q4_0 -ctv q4_0 -fa on -ub 64 --reasoning off -p "..."
```

Filled-depth bench (keep `-d` small on this GPU):

```sh
build-metal/bin/llama-bench \
  -m /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 99 -ctk q4_0 -ctv q4_0 -fa on -ub 64 -b 64 \
  -p 64 -n 16 -d 0,128,256 -r 1
```
