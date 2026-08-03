# Qwen3.6 MoE vs dense on M1 Max: architecture vs quantization

Measured 2026-08-02 on a Mac Studio, Apple M1 Max — 32 GPU cores, 64 GB unified
memory (~400 GB/s theoretical). LM Studio + MLX, 262144 context, median of 3
runs per row, 256 tokens generated. Numbers come from `lmstudio-bench`, which
reads LM Studio's own perf counters and randomizes every prompt to defeat the
KV cache. Raw data in `results/`.

Four builds, all four quantization/architecture combinations:

| Variant | Arch | Params | Active | Quant | Weights |
|---|---|---|---|---|---|
| `qwen3.6-35b-a3b@4bit` | MoE | 35B | ~3B | 4-bit | 20.43 GB |
| `qwen3.6-35b-a3b@8bit` | MoE | 35B | ~3B | 8-bit | 37.75 GB |
| `qwen3.6-27b@4bit` | dense | 27B | 27B | 4-bit | 16.08 GB |
| `qwen3.6-27b@8bit` | dense | 27B | 27B | 8-bit | 29.53 GB |

## Decode (tok/s)

| Prompt tok | MoE 4-bit | MoE 8-bit | Dense 4-bit | Dense 8-bit |
|---|---|---|---|---|
| 30 | 58.3 | 48.5 | 16.3 | 9.7 |
| 837 | 63.9 | 51.7 | 18.5 | 10.8 |
| 3,218 | 63.9 | 51.9 | 18.5 | 10.7 |
| 12,853 | 59.8 | 48.3 | 16.8 | 10.4 |

## Prefill (tok/s)

| Prompt tok | MoE 4-bit | MoE 8-bit | Dense 4-bit | Dense 8-bit |
|---|---|---|---|---|
| 837 | 516 | 510 | 88.1 | 88.4 |
| 3,218 | 591 | 590 | 89.2 | 90.2 |
| 12,853 | 591 | 588 | 88.5 | 88.8 |

## The two effects are separable, and they hit different things

Using the 3,218-token row throughout:

| | Decode | Prefill |
|---|---|---|
| **Architecture** (MoE vs dense, same precision) | 3.5× at 4-bit, **4.9×** at 8-bit | **6.6×** at both |
| **Quantization** (4-bit vs 8-bit, same arch) | 1.23× MoE, **1.76×** dense | **1.00×** — no effect |

**Quantization does nothing for prefill.** Dense prefill is 89.2 vs 90.2 tok/s
at 4-bit vs 8-bit; MoE is 591 vs 590. Prefill is compute-bound — it is doing
large batched matmuls, and halving the weight bytes does not reduce the
arithmetic. If you are prefill-bound on long prompts, quantizing further buys
you nothing.

**Quantization is most of the dense decode story.** Dense decode goes 10.7 →
18.5 tok/s (1.76×) purely from 8-bit → 4-bit, because decode streams every
weight once per token and 4-bit halves the bytes.

**Architecture is worth 3.5–4.9× on decode and a flat 6.6× on prefill.** The
MoE has 30% more total parameters than the dense 27B and still wins everywhere,
because only ~3B activate per token.

## Where the simple bandwidth model works, and where it breaks

Decode should be memory-bandwidth-bound: tok/s ≈ bandwidth ÷ bytes read per
token. For the **dense** models this is almost exact, and it is a real
prediction, not a fit — the 4-bit decode figure was predicted at ~19 tok/s
before the run and measured 18.5.

| Dense variant | Weights | Decode | Implied bandwidth |
|---|---|---|---|
| 27b@8bit | 29.53 GB | 10.7 | 316 GB/s |
| 27b@4bit | 16.08 GB | 18.5 | 297 GB/s |

Two independent points agreeing at ~300 GB/s, about 77% of the M1 Max's
theoretical peak. The dense models are at the hardware wall; there is no tuning
headroom.

**For the MoE the same model fails.** Predicting the 8-bit MoE by assuming it
reads a fixed ~24% of its weights per token gave ~33 tok/s. It measured
**51.9** — off by 57%.

| MoE variant | Weights | Decode | Implied bytes/token | As % of weights |
|---|---|---|---|---|
| 35b-a3b@4bit | 20.43 GB | 63.9 | 4.7 GB | 23% |
| 35b-a3b@8bit | 37.75 GB | 51.9 | 5.8 GB | 15% |

Doubling precision cost the MoE only **1.23×** in decode, where a pure
bandwidth model demands ~1.85× (the ratio of the two file sizes). Fitting
`time_per_token = bytes/bandwidth + overhead` to the two MoE points yields
~1.3 GB read per token and **~11 ms of fixed per-token overhead**; the same fit
on the two dense points gives ~0 ms.

The likely reading is that **MoE decode is substantially latency-bound rather
than bandwidth-bound** — expert routing and gather produce many small
operations whose cost does not shrink when the weights get smaller. Treat this
as a hypothesis fitted to two points, not a confirmed mechanism; distinguishing
it from other explanations (lower achieved bandwidth on gather-heavy access,
different MLX kernel efficiency per precision) would need per-layer profiling.

The practical consequence is concrete: **quantizing an MoE down buys much less
than quantizing a dense model down.** Dense gained 1.76× from 8→4 bit; the MoE
gained 1.23×.

## Practical takeaways

- **`35b-a3b@4bit` is the right default** at 63.9 tok/s decode and 591 tok/s
  prefill — the fastest build on both axes, and the smallest of the four
  at 20.43 GB.
- **`35b-a3b@8bit` costs surprisingly little**: 51.9 tok/s, only 19% slower
  than the 4-bit build, with identical prefill. If 8-bit quality matters, it is
  cheap here in a way it is not for dense models.
- **Neither dense build is competitive on this hardware.** Even at 4-bit the
  dense 27B decodes at 18.5 tok/s and prefills at 89 — a 12,853-token prompt
  costs 145 s before the first token, versus 22 s for the MoE.
- **Long prompts are prefill-bound for everything.** Prompt caching / KV reuse
  is where the wins are; quantization cannot help there.
- **Only one large model fits at a time.** JIT-loading a second trips LM
  Studio's guardrail with an HTTP 400.

## Energy cost of this benchmarking

Active GPU time, derived from the measured per-request latencies (each row is
3 timed runs plus one discarded warm-up):

| Phase | GPU time |
|---|---|
| First pass (flawed, superseded) | 8.0 min |
| Clean re-runs | 16.9 min |
| 4-variant matrix | 42.3 min |
| Probes and one-off requests | 2.5 min |
| **Total at full GPU load** | **1.16 h** |

Plus ~7.5 min of model loading at lower draw. At an assumed 85 W sustained wall
draw under inference and 30 W while loading:

**≈0.10 kWh, ≈$0.033** at $0.32/kWh — about three cents.

The range across plausible draw assumptions (75–100 W) is $0.029–$0.039, so the
answer is "a few cents" regardless. The wall-clock elapsed time was far longer
than 1.16 h, but idle waiting draws roughly baseline and is not attributable.

Caveat: the power figures are **estimated from typical M1 Max sustained-
inference draw, not measured** — `powermetrics` requires root and was not
sampled during the runs. To measure it directly next time:

```sh
sudo powermetrics --samplers cpu_power,gpu_power -n 5 -i 1000
```

## Methodology notes

- `lmstudio-bench` uses LM Studio's native `/api/v0` endpoint, which returns
  real `stats`; the OpenAI-compatible `/v1` route returns an empty `stats`
  object.
- Every prompt body is seeded from a per-run nonce. LM Studio reuses KV cache
  across requests — an identical repeated prompt prefills **11.6×** faster
  (6471 vs 557 tok/s measured). Any benchmark whose prompts share a prefix
  reports inflated prefill rates.
- Variants cannot be selected with `lms load`, which matches only base model
  keys and silently loads whichever variant is "selected". The REST API does
  resolve `model@quant`, so `run-matrix.sh` passes variant ids straight through
  and lets JIT loading pick the build.
- `ttft_spread` was 0.1–9.3% on every row; the two rows above 8% are the
  837-token cases, where a small absolute TTFT makes the ratio noisy.

## Caveats

- Reproducibility is good but not perfect: re-running `35b-a3b@4bit` gave
  516/591/591 tok/s prefill against 518/593/589 in the earlier clean pass, and
  63.9/63.9/59.8 decode against 63.6/62.1/58.5.
- The ~11 ms MoE overhead figure is fitted to two data points. It is
  suggestive, not established.
- Power figures are estimated, not measured — see above.
- An early pass built prompts by repeating a fixed filler string, so each size
  shared a prefix with the one before and prefill came out 15–20% high. Those
  results are in `results/superseded/` and should not be cited.
