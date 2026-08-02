# Qwen3.6 MoE vs dense on M1 Max

Measured 2026-08-01/02 with [`bench.py`][bench]. Raw data in `results/`.

## Setup

| | |
|---|---|
| Machine | Mac Studio, Apple M1 Max — 32 GPU cores, 64 GB unified memory (~400 GB/s) |
| Runtime | LM Studio, MLX (safetensors), OpenAI-compatible endpoint on `localhost:1234` |
| Context | Both models loaded at 262144 |

| Model | Arch | Params | Active | On disk | Resident |
|---|---|---|---|---|---|
| `qwen/qwen3.6-35b-a3b` | `qwen3_5_moe` | 35B | ~3B | 20.43 GB | 20.4 GB |
| `qwen/qwen3.6-27b` | `qwen3_5` (dense) | 27B | 27B | 29.53 GB | 27.5 GiB |

Only one model was resident at a time — both together plus KV cache would not
fit in 64 GB.

## Results

| Case | Prompt tok | Gen tok | 27b dense prefill | 27b dense decode | 35b-a3b MoE prefill | 35b-a3b MoE decode |
|---|---|---|---|---|---|---|
| short-ctx | 93 | 255 | 58.3 | 11.2 | 223.4 | 66.0 |
| medium-ctx | 1,221 | 255 | 86.5 | 11.0 | 493.1 | 64.8 |
| long-ctx | 4,821 | 255 | 113.4 | 10.9 | 695.5 | 63.7 |
| long-ctx | 19,221 | 255 | 114.1 | 10.3 | 716.2 | 55.9 |
| gen-heavy | 93 | 923–1023 | 57.0 | 11.1 | 224.8 | 66.3 |

All figures tok/s. Prefill = `prompt_tokens / time-to-first-token`; decode
excludes the first token.

| | 27b dense | 35b-a3b MoE | MoE advantage |
|---|---|---|---|
| Mean decode | 10.9 tok/s | 63.3 tok/s | **5.8×** |
| Peak prefill | 114 tok/s | 716 tok/s | **6.3×** |
| TTFT on 19k prompt | 168 s | 26.8 s | **6.3×** |

## Analysis

**The MoE wins decisively on both axes, despite being the larger model.** It
has 30% more total parameters than the dense 27B and still runs ~6× faster,
because only ~3B parameters are active per token.

**Decode is memory-bandwidth-bound, and the numbers confirm it.** The dense
model must stream all 27.5 GiB of weights per token: 27.5 GiB × 10.9 tok/s ≈
**300 GB/s effective**, roughly 75% of the M1 Max's theoretical peak. That is
close to hardware limit — there is no tuning headroom here, the model is simply
too big to go faster on this machine. Working backwards at the same effective
bandwidth, the MoE reads ~4.7 GB per token, about 23% of its 20.4 GB — the
attention stack and shared weights plus the routed experts actually selected.

**Prefill is compute-bound and scales differently.** Both models climb with
prompt length as the GPU saturates: the dense model plateaus at ~114 tok/s past
~5k tokens, the MoE at ~716 tok/s. Short prompts understate both (58 and 223
tok/s at 93 tokens) because fixed request overhead dominates TTFT, not because
of any real ceiling.

**Context degrades decode gracefully for both.** 66 → 56 tok/s for the MoE and
11.2 → 10.3 tok/s for the dense model going from ~100 to 19k tokens of KV
cache, roughly −15% and −8%.

## Practical takeaways

- **Use the MoE for anything interactive.** 63 tok/s is comfortably faster than
  reading speed; 11 tok/s is not. The dense model's 168-second wait before the
  first token of a 19k-token prompt makes it unusable for document work.
- **The dense 27B is not worth it on this hardware** unless it is measurably
  better at your task — you are paying ~6× latency and 7 GB more memory for a
  model with fewer active parameters doing the work.
- **Both are prefill-bound on long inputs**, so prompt caching / KV reuse is
  where the wins are, not decode tuning. Decode is already at ~75% of the
  memory-bandwidth ceiling.
- **Only one of these fits at a time.** Swapping costs ~30 s to load the 27B.

## Caveats

- Single run per case, no repeats — expect a few percent of run-to-run noise.
- The dense `gen-heavy` case stopped naturally at 923 tokens rather than
  hitting the 1024 cap; the rate is unaffected.
- Prefill rates derived from TTFT include request and sampling overhead, so
  short-prompt figures are pessimistic.

[bench]: bench.py
