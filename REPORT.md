# Qwen3.6 MoE vs dense on M1 Max

Measured 2026-08-02 on a Mac Studio, Apple M1 Max — 32 GPU cores, 64 GB unified
memory (~400 GB/s). LM Studio + MLX. Numbers come from `lmstudio-bench`, which
reads LM Studio's own perf counters and randomizes every prompt to defeat the
KV cache. Raw data in `results/`.

> **Read the caveat before quoting the headline ratio.** The two models are not
> at matched precision — the dense model is 8-bit, the MoE is 4-bit. See
> [Precision mismatch](#precision-mismatch-the-big-caveat).

| Model | Arch | Params | Active | Quant | On disk |
|---|---|---|---|---|---|
| `qwen/qwen3.6-35b-a3b` | `qwen3_5_moe` | 35B | ~3B | 4-bit | 19 GB |
| `qwen/qwen3.6-27b` | `qwen3_5` (dense) | 27B | 27B | 8-bit | 28 GB |

## Results

Median of 3 runs per row, 256 tokens generated.

| Prompt tok | MoE 4-bit prefill | MoE 4-bit decode | Dense 8-bit prefill | Dense 8-bit decode |
|---|---|---|---|---|
| 30 (short) | – | 59.2 | – | 9.6 |
| 837 | 518 | 63.6 | 87.6 | 10.8 |
| 3,218 | 593 | 62.1 | 88.9 | 10.5 |
| 12,853 | 589 | 58.5 | 86.7 | 10.0 |

All figures tok/s. Prefill rate is omitted for the short prompt, where fixed
request overhead dominates TTFT and the ratio means nothing.

| | MoE 4-bit | Dense 8-bit | Ratio |
|---|---|---|---|
| Mean decode | 61.4 tok/s | 10.4 tok/s | **5.9×** |
| Prefill plateau | ~590 tok/s | ~87 tok/s | **6.8×** |
| TTFT on ~13k prompt | 21.9 s | 148.2 s | **6.8×** |

## Precision mismatch: the big caveat

The dense model is 8-bit and the MoE is 4-bit, so it carries roughly twice the
bytes per parameter (~1.09 vs ~0.58). Decode is memory-bandwidth-bound, so
**a large share of the 5.9× decode gap is quantization, not architecture.**

Estimating the split: the dense model streams all 29.5 GB per token at 10.4
tok/s ≈ **307 GB/s effective**, about 77% of the M1 Max's theoretical peak —
essentially at the hardware wall. A hypothetical 4-bit 27B would be ~15 GB and
should decode near 20 tok/s at that same effective bandwidth. Against the MoE's
61.4 tok/s, the *architectural* advantage at matched precision would be closer
to **~3×** than 5.9×.

An 8-bit-vs-8-bit run is possible without downloading anything — the 8-bit MoE
variant is already on disk (35 GB). That is the honest comparison and it has
not been run yet.

## Analysis

**The MoE is dramatically faster, and would remain so at matched precision.**
It has 30% more total parameters than the dense 27B yet activates only ~3B per
token, and that is what the numbers reward.

**Decode is bandwidth-bound for both.** The dense model at ~307 GB/s effective
is at ~77% of peak — there is no tuning headroom, it is simply too big to go
faster here. Working back at the same effective bandwidth, the MoE reads
~5.0 GB per token, about 24% of its 20.4 GB: the attention stack and shared
weights plus whichever experts get routed.

**Prefill is compute-bound and plateaus early.** Both models flatten by ~3k
tokens — ~590 tok/s for the MoE, ~87 tok/s for the dense model — and hold that
rate out to 13k.

**Context costs decode modestly.** 63.6 → 58.5 tok/s for the MoE and 10.8 →
10.0 for the dense model between 837 and ~13k tokens of KV cache, about −8% for
both.

## Practical takeaways

- **Use the MoE for anything interactive.** 60 tok/s outruns reading speed;
  10 tok/s does not. A 148-second wait before the first token of a 13k prompt
  rules the dense 8-bit model out for document work.
- **Both are prefill-bound on long inputs**, so prompt caching / KV reuse is
  where the wins are. Decode is already near the memory-bandwidth ceiling.
- **Only one large model fits at a time.** Attempting to JIT-load the 27B while
  the MoE is resident trips LM Studio's resource guardrail with an HTTP 400.

## Methodology notes

- `lmstudio-bench` talks to LM Studio's native `/api/v0` endpoint, which
  returns real `stats` (tokens/sec, time-to-first-token). The OpenAI-compatible
  `/v1` route returns an empty `stats` object.
- Every prompt body is seeded from a per-run nonce. This matters enormously:
  LM Studio reuses KV cache across requests, and an identical repeated prompt
  prefills **11.6× faster** (6471 vs 557 tok/s measured). Any benchmark whose
  prompts share a prefix will report inflated prefill rates.
- The `ttft_spread` column flags rows whose runs disagree by >25%, the
  signature of an accidental cache hit. All rows here were 1–6%.

## Caveats

- **Precision is not matched** — see above. This is the dominant caveat.
- The ~13k dense row is a **single sample**, not a median of 3: two runs failed
  with HTTP 400 when a concurrent `pi` session JIT-loaded the MoE and exhausted
  memory. Its 86.7 tok/s is consistent with the 837 and 3,218 rows (87.6, 88.9),
  so it is probably sound, but it is unverified.
- An earlier run of these models used prompts built by repeating a fixed filler
  string, so each size shared a prefix with the one before and prefill rates
  were inflated by 15–20%. Those results are kept in `results/superseded/` and
  should not be cited.
