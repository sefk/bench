# Storing models on external USB SSD: measured impact

Measured 2026-08-19, Mac Studio M1 Max, 64 GB. Model under test:
`qwen/qwen3.8-27b` MLX 4-bit, 14.98 GiB of weights.

## Disks

| | Device | Sequential read (cache bypassed) |
|---|---|---|
| Internal | Apple SSD AP1024R (NVMe) | **6.60 GB/s** |
| `ext1` | USB SSD, APFS, 1.8 TB | **3.46 GB/s** |

Read throughput measured with `F_NOCACHE` (`fcntl` 48) so the macOS unified
buffer cache is out of the path — a second cached read of a model file returns
at memory speed and measures nothing. Median of 3 × 8 GiB.

## Cold model load

Page cache flushed before each measurement by streaming 77 GB through it
(more than the machine's RAM), since `purge` needs root.

| Location | Cold load | Warm load |
|---|---|---|
| Internal | **18.4 s** | 18.1 s |
| `ext1` | **24 s** | – |

## Inference, once loaded

Three timed runs per size, 128 tokens generated, via `lmstudio-bench`.

| Location | Prompt | decode tok/s | ttft | prefill tok/s |
|---|---|---|---|---|
| Internal | short | 18.5 | 1.26 s | – |
| `ext1` | short | 19.0 | 1.24 s | – |
| Internal | 3,200 | 18.4 | 29.68 s | 88 |
| `ext1` | 3,200 | 18.5 | 29.10 s | 89 |

## Conclusion

**Inference is unaffected.** Decode, prefill, and TTFT are identical between
the two disks — every difference is inside the 2–6% run-to-run spread, and the
external disk is nominally *ahead* on three of four rows, which is noise, not a
speedup. Weights live in unified memory once loaded; the disk they came from
stops mattering the moment the load finishes.

**Only cold load pays, and less than the disk gap suggests.** 18.4 s → 24 s,
+5.6 s, 1.3×. The disk is 1.9× slower but the load is only 1.3× slower, because
loading is not disk-bound: internal cold load (18.4 s) and internal *warm* load
(18.1 s) are within 0.3 s of each other, so on the internal SSD the read is ~2%
of the 18 s. The other ~16 s is MLX weight setup and GPU upload, which no
storage change touches. At 6.6 GB/s the 15 GB streams in 2.3 s; at 3.46 GB/s it
takes 4.3 s. That ~2 s predicted delta is the right order for the 5.6 s
measured — the remainder is USB per-I/O latency the sequential number doesn't
capture.

Moving the model library to `ext1` costs about six seconds per cold load, once
per model per session (LM Studio's TTL keeps models resident for an hour), and
nothing at all thereafter.

## Incidental finding: the MLX build of 3.8-27b is much faster than the GGUF

The 2026-08-14 addendum in REPORT.md benchmarked `qwen3.8-27b` as a llama.cpp
GGUF Q4_K_M and measured ~10 tok/s decode. The MLX 4-bit build of the same
model, measured here, does **18.4–19.0 tok/s** — 1.8×, on the same machine.

That isolates what the addendum could not: the gap it reported against
`qwen3.6-27b` MLX 4-bit (~18.5 tok/s) was the *runtime*, not the model
generation. Qwen3.8-27B in MLX matches Qwen3.6-27B in MLX on decode. Prefill is
88–89 tok/s here against the GGUF's ~120, so the GGUF's prefill advantage is
real and survives; it buys that with roughly half the decode rate.

## Caveats

- Another workload was using the same LM Studio server during these runs
  (chat completions against `qwen/qwen3.6-27b`, plus `nomic-embed` embedding
  requests). Timed runs were taken with only the model under test resident,
  but GPU contention cannot be fully excluded.
- Single cold-load measurement per location, not a median. The 5.6 s delta is
  larger than the ~2 s the throughput numbers predict; more trials would tell
  whether that is USB latency or run-to-run variance.
- `ext1` is USB-attached but fast (3.46 GB/s, so an NVMe enclosure on a
  high-speed port). A cheaper USB 3.0 enclosure at ~0.4 GB/s would read the
  15 GB in ~37 s and the conclusion about cold-load cost would change
  substantially. Inference would still be unaffected.
