# Qwen local inference on M1 Max: architecture, quantization, and disk

Mac Studio, Apple M1 Max — 32 GPU cores, 64 GB unified memory (~400 GB/s
theoretical). LM Studio + MLX unless noted. Throughput from LM Studio's own
perf counters; power from `powermetrics` sampled across the run. See the
[Log](#log) below for what was measured when and where the raw data lives.

## Findings

### Quantization buys speed, not energy

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

### Architecture is where the efficiency actually is

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

### Power scales with prompt length

Draw climbs steadily with context in every variant — MoE 4-bit goes 23.7 → 44.3
W from the shortest prompt to the longest. Prefill saturates the GPU harder than
decode does, so a long-prompt workload runs the machine hotter as well as
longer. Idle baseline measured 1.9–3.2 W across runs.

### Qwen3.6 vs 3.8: the decode gap was the runtime, not the model

The 2026-08-14 addendum benchmarked `qwen3.8-27b` as a llama.cpp GGUF Q4_K_M
and found decode ~45% slower than the MLX 4-bit build of 3.6 (≈10 vs ≈18
tok/s), with prefill ~35% faster. That comparison confounded model generation
with runtime (GGUF vs MLX).

The 2026-08-19 disk experiment isolated it by measuring an MLX 4-bit build of
3.8-27b directly: **18.4–19.0 tok/s decode** — 1.8× the GGUF figure, and
matching 3.6-27b MLX 4-bit (≈18.5 tok/s). So:

- **Qwen3.8-27B in MLX matches Qwen3.6-27B in MLX on decode.** The generation
  change is a wash on speed.
- **The GGUF's prefill advantage is real and survives** (88–89 tok/s MLX vs
  ~120 GGUF) — it buys that with roughly half the decode rate. That's a
  runtime/format tradeoff, not a model one.

### Internal vs external disk: inference is unaffected, cold load pays a small, sub-linear cost

Measured 2026-08-19 with `qwen/qwen3.8-27b` MLX 4-bit (14.98 GiB) on internal
NVMe (6.60 GB/s sequential, cache bypassed) vs a USB SSD `ext1` (APFS, 1.8 TB,
3.46 GB/s).

| Location | Cold load | Decode (3,200 prompt) | Prefill | TTFT |
|---|---|---|---|---|
| Internal | **18.4 s** | 18.4 tok/s | 88 | 29.68 s |
| `ext1` | **24 s** (+5.6 s) | 18.5 tok/s | 89 | 29.10 s |

- **Once loaded, the disk doesn't matter.** Decode, prefill, and TTFT are
  identical within run-to-run noise (2–6%); weights live in unified memory
  after load, so the source disk stops mattering the moment loading finishes.
- **Cold load is 1.3× slower on `ext1`, not the 1.9× the raw disk gap
  suggests**, because loading isn't disk-bound: internal cold load (18.4 s)
  and internal *warm* load (18.1 s) differ by only 0.3 s, so the read is only
  ~2% of the 18 s — the rest is MLX weight setup and GPU upload, which storage
  doesn't touch. Streaming 15 GB predicts ~2 s of extra delay from the slower
  disk; the measured 5.6 s delta is the right order, with the remainder likely
  USB per-I/O latency.
- **Practical cost: ~6 s per cold load, once per model per session** (LM
  Studio keeps models resident for an hour), nothing thereafter. This result
  would look different on a slow USB 3.0 enclosure (~0.4 GB/s) — cold load
  would grow to roughly 37 s — but inference would still be unaffected.

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
- **Disk choice for the model library is a non-issue for inference**, and only
  a several-second tax on cold load — safe to keep models on a fast external
  drive if internal storage is tight.

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
- Disk read throughput is measured with `F_NOCACHE` (`fcntl` 48) so the macOS
  unified buffer cache is out of the path; a cached read returns at memory
  speed and measures nothing. Page cache is flushed before cold-load
  measurements by streaming 77 GB through it (more than the machine's RAM),
  since `purge` needs root.

## Caveats

- Reproducibility is good: the 2026-08-03 run's decode figures land within
  ~1–4% of the 2026-08-02 run (MoE 4-bit 64.4 vs 63.9 at 3,218 tokens; dense
  4-bit 18.6 vs 18.5), with power instrumentation adding no measurable
  perturbation.
- Power is SoC package only — see above.
- **No quality measurement.** Everything here is speed and energy. The 4-bit vs
  8-bit choice cannot be settled on these numbers alone, since they show the two
  builds cost nearly the same energy; which is *better* is a separate question
  this benchmark does not address.
- The disk experiment ran with another workload sharing the same LM Studio
  server (chat completions against `qwen/qwen3.6-27b`, plus `nomic-embed`
  requests); timed runs had only the model under test resident, but GPU
  contention cannot be fully excluded. Its cold-load numbers are also a single
  measurement per location, not a median.

## Log

### 2026-08-03 — Qwen3.6 MoE vs dense, full matrix

Compared `qwen3.6-35b-a3b` (MoE, ~3B active) against `qwen3.6-27b` (dense) at
4-bit and 8-bit, across four prompt lengths (30 to 12,853 tokens), measuring
decode/prefill throughput and power via `powermetrics`. Raw data in
`results/2026-08-03/`.

Learned: architecture (MoE vs dense) drives a 7× energy-per-token difference,
while quantization (4-bit vs 8-bit) is roughly energy-neutral and only trades
latency. Power draw scales with prompt length in every variant. MoE 4-bit is
the clear default.

### 2026-08-14 — Addendum: Qwen3.8-27B (GGUF Q4_K_M), throughput only

Benchmarked `qwen/qwen3.8-27b` as a llama.cpp GGUF build, same machine and
settings as above, but no power sampler running. Raw data in
`results/2026-08-14/`.

Learned: decode looked ~45% slower and prefill ~35% faster than the MLX 4-bit
dense 3.6 build. At the time this read as a model-generation regression on
decode. It was later shown (2026-08-19) to be a runtime artifact — GGUF vs
MLX — not a property of Qwen3.8 itself.

### 2026-08-19 — Internal vs external (USB) disk for model storage

Measured cold/warm load time and inference throughput for `qwen/qwen3.8-27b`
MLX 4-bit from internal NVMe vs a USB-attached SSD (`ext1`, 3.46 GB/s
sequential). Raw data and full write-up in `results/2026-08-19/external-disk.md`.

Learned two things: (1) disk location is irrelevant to inference once a model
is loaded, and costs only ~6 s extra on cold load with this disk (sub-linear
vs the 1.9× raw throughput gap, because loading is mostly MLX setup/GPU
upload, not I/O); (2) as a side effect of using an MLX build of 3.8-27b for
this test, resolved the 2026-08-14 addendum's apparent decode regression —
it was the GGUF runtime, not the model.
