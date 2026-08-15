# Qwen3.6 MoE vs dense on M1 Max: speed, power, and energy

Measured 2026-08-03 on a Mac Studio, Apple M1 Max — 32 GPU cores, 64 GB unified
memory (~400 GB/s theoretical). LM Studio + MLX, 262144 context, median of 3
runs per row, 256 tokens generated. Throughput comes from LM Studio's own perf
counters; power from `powermetrics` sampled across the run. Raw data in
`results/2026-08-03/`.

| Variant | Arch | Params | Active | Quant | Weights |
|---|---|---|---|---|---|
| `qwen3.6-35b-a3b@4bit` | MoE | 35B | ~3B | 4-bit | 20.43 GB |
| `qwen3.6-35b-a3b@8bit` | MoE | 35B | ~3B | 8-bit | 37.75 GB |
| `qwen3.6-27b@4bit` | dense | 27B | 27B | 4-bit | 16.08 GB |
| `qwen3.6-27b@8bit` | dense | 27B | 27B | 8-bit | 29.53 GB |

## Full matrix

Each cell: decode tok/s · prefill tok/s · watts · generated tokens per Wh.

| Prompt tok | MoE 4-bit | MoE 8-bit | Dense 4-bit | Dense 8-bit |
|---|---|---|---|---|
| 30 | 57.4 · – · 23.7 · 10707 | 48.5 · – · 22.0 · 10019 | 17.1 · – · 40.7 · 1708 | 9.8 · – · 28.8 · 1512 |
| 837 | 65.0 · 518 · 31.8 · 5462 | 52.4 · 520 · 27.2 · 5838 | 18.4 · 87 · 43.9 · 925 | 10.9 · 89 · 33.7 · 915 |
| 3,218 | 64.4 · 599 · 37.0 · 2920 | 52.2 · 594 · 34.8 · 2723 | 18.6 · 90 · 46.4 · 418 | 10.7 · 91 · 40.2 · 423 |
| 12,853 | 59.2 · 593 · 44.3 · 845 | 48.6 · 592 · 42.8 · 855 | 17.4 · 89 · 49.1 · 122 | 10.3 · 90 · 45.7 · 129 |

`tok/Wh` counts **generated** tokens against energy **above idle**, so it falls
with prompt length: a 12,853-token prompt spends most of its energy on prefill
before producing any of its 256 output tokens. Compare across a row, not down a
column.

## Quantization buys speed, not energy

The clearest result, and it holds independently in both architectures. At the
3,218-token row:

| | 4-bit | 8-bit | speed | power | **energy/token** |
|---|---|---|---|---|---|
| Dense 27B | 18.6 tok/s, 46.4 W | 10.7 tok/s, 40.2 W | 1.74× | 1.15× | **0.99× — level** |
| MoE 35B-A3B | 64.4 tok/s, 37.0 W | 52.2 tok/s, 34.8 W | 1.23× | 1.06× | **1.07× — level** |

Going to 4-bit makes decode substantially faster, but power rises almost in
proportion, so watt-hours per token barely move. The mechanism is consistent
with decode being memory-bound: halving the bytes per token lets the GPU spend
more of its time computing rather than waiting, which is faster *and* hotter.
You finish sooner and draw more while you do.

**The practical consequence: choosing 8-bit costs latency, not electricity.**
For the MoE the electricity difference is nil — 8-bit is actually ahead at 837
and 12,853 tokens — while decode is 19% slower. If 8-bit is better for the work
in question, the running cost of preferring it is zero.

## Architecture is where the efficiency actually is

At the same 3,218-token row, MoE 4-bit vs dense 4-bit:

| | Dense 4-bit | MoE 4-bit | ratio |
|---|---|---|---|
| Decode | 18.6 tok/s | 64.4 tok/s | 3.5× |
| Power | 46.4 W | 37.0 W | 0.80× |
| **Energy per token** | 418 tok/Wh | 2920 tok/Wh | **7.0×** |

The efficiency gain is double the throughput gain, because the MoE is faster
*and* draws less power. Activating ~3B of 35B parameters means less work per
token, not merely less waiting on memory — so the two effects multiply rather
than trade off. This is the opposite of the quantization result above, and it
is why architecture rather than precision is the lever worth pulling here.

Prefill tells the same story more starkly: 599 vs 90 tok/s, a 6.6× gap that is
unaffected by quantization in either direction (518/599/593 for MoE 4-bit
against 520/594/592 for 8-bit; 87/90/89 dense 4-bit against 89/91/90 8-bit).
Prefill is compute-bound, so fewer active parameters help and fewer bits do not.

## Power scales with prompt length

Draw climbs steadily with context in every variant — MoE 4-bit goes 23.7 → 44.3
W from the shortest prompt to the longest. Prefill saturates the GPU harder than
decode does, so a long-prompt workload runs the machine hotter as well as
longer. Idle baseline measured 1.9–3.2 W across runs.

## Practical takeaways

- **`35b-a3b@4bit` remains the default**: fastest on both axes and the most
  energy-efficient at short-to-medium context.
- **`35b-a3b@8bit` is nearly free in energy terms.** 19% slower decode,
  identical prefill, same watt-hours per token. The only cost is waiting.
- **Neither dense build is competitive**, on speed or on energy. Dense 4-bit
  needs 7× the energy per token of MoE 4-bit and still takes 145 s to first
  token on a 12.8k prompt.
- **Quantizing further is not an energy optimization.** If the goal is lower
  power draw rather than lower latency, quantization is the wrong lever.

## What this benchmark cost

Measured, not estimated — energy above idle, for the timed windows only:

| Variant | Above idle |
|---|---|
| MoE 4-bit | 1.38 Wh |
| MoE 8-bit | 1.38 Wh |
| Dense 4-bit | 9.39 Wh |
| Dense 8-bit | 9.09 Wh |
| **Total** | **21.2 Wh** |

At $0.32/kWh that is **$0.007 — about two-thirds of a cent.** The two dense
variants account for 87% of it.

An earlier version of this report estimated the energy from assumed wall draw
(85 W) and put a comparable workload near 60 Wh. The measurement came in about
**2.7× lower**. The assumption was wrong in two ways: sustained draw is far
below the machine's peak, and on a machine left powered on all day the relevant
quantity is the increment over idle, not total draw.

Caveat in the other direction: `powermetrics` reports **SoC package power only**
(CPU + GPU + ANE). It excludes DRAM, PSU losses, and the rest of the machine, so
true incremental draw at the wall is somewhat higher than these figures — likely
by a factor well under two, which leaves the conclusion ("about a cent")
unchanged.

## Addendum 2026-08-14: Qwen3.8-27B (GGUF Q4_K_M)

Measured 2026-08-14, same machine and settings, **throughput only** — no
`powermetrics` sampler was running, so there are no watts or tok/Wh for this
row. Raw data in `results/2026-08-14/`.

This is a different runtime as well as a different model: `qwen/qwen3.8-27b`
is a llama.cpp GGUF build (Q4_K_M, 17.74 GB), not MLX. The dense 27B rows
above are MLX 4-bit/8-bit. Treat the comparison as "the GGUF Q4_K_M build of
3.8 vs the MLX 4-bit build of 3.6", not as a pure model-generation change.

| Prompt tok | decode tok/s | prefill tok/s | TTFT | vs 3.6-27b MLX 4-bit |
|---|---|---|---|---|
| 72 | 11.5 | 50 | 1.4 s | decode 0.67× |
| 879 | 10.1 | 118 | 7.5 s | decode 0.55×, prefill 1.35× |
| 3,260 | 10.5 | 123 | 26.6 s | decode 0.57×, prefill 1.36× |
| 12,895 | 9.7 | 118 | 110 s | decode 0.55×, prefill 1.32× |

- **Decode is ~45% slower** than the MLX 4-bit dense 3.6 (≈10 vs ≈18 tok/s),
  landing right on the MLX *8-bit* dense number (10.7). Q4_K_M weights are
  slightly larger than MLX 4-bit (17.7 vs 16.1 GB), but not enough to explain
  that; the rest is the llama.cpp Metal path vs MLX on this GPU.
- **Prefill is ~35% faster** (≈120 vs ≈90 tok/s), so TTFT on the 12.9k prompt
  drops from 145 s to 110 s. Still far from the MoE's ≈600 tok/s.
- `ttft_spread` ≤ 1.2% on the sized rows — no cache-hit contamination.
- The model stopped early on every row (98–192 tokens generated rather than
  256), so per-row wall time is shorter than the 3.6 dense runs; decode tok/s
  is unaffected.
- No energy figure. From the 3.6 dense rows one would guess a similar draw and
  therefore worse tok/Wh than MLX 4-bit given the slower decode, but that is an
  assumption, not a measurement.

The MoE `qwen3.6-35b-a3b@4bit` remains the default; this build does not change
that on speed.

## Methodology notes

- `lmstudio-bench` uses LM Studio's native `/api/v0` endpoint, which returns
  real `stats`; the OpenAI-compatible `/v1` route returns an empty `stats`
  object and no logprobs.
- Every prompt body is seeded from a per-run nonce. LM Studio reuses KV cache
  across requests — an identical repeated prompt prefills **11.6×** faster
  (6471 vs 557 tok/s measured). Any benchmark whose prompts share a prefix
  reports inflated prefill rates.
- Variants cannot be selected with `lms load`, which matches only base model
  keys and silently loads whichever variant is "selected". The REST API does
  resolve `model@quant`, so variant ids are passed straight through and JIT
  loading picks the build.
- The idle baseline is taken from samples outside the benchmark windows, with a
  10 s guard band and at the 25th percentile rather than the mean. The
  out-of-window samples are not all idle — model loading falls in the gaps, and
  a shared power log contains other runs — so a low quantile finds the floor
  where an average would land in the traffic.

## Caveats

- Reproducibility is good: this run's decode figures land within ~1–4% of the
  2026-08-02 run (MoE 4-bit 64.4 vs 63.9 at 3,218 tokens; dense 4-bit 18.6 vs
  18.5), with power instrumentation adding no measurable perturbation.
- Power is SoC package only — see above.
- **No quality measurement.** Everything here is speed and energy. The 4-bit vs
  8-bit choice cannot be settled on these numbers alone, since they show the two
  builds cost nearly the same energy; which is *better* is a separate question
  this benchmark does not address.
