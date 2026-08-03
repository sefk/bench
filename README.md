# Local LLM benchmarks

Throughput benchmarks and other investigations of models running locally under
[LM Studio][lms] on Apple Silicon.

## Hardware

Mac Studio, Apple M1 Max — 10 CPU cores (8P/2E), 32 GPU cores, 64 GB unified
memory, ~400 GB/s memory bandwidth.

## Tooling

`lmstudio-bench` measures generation throughput, time-to-first-token, and
prefill rate per model and prompt size:

```sh
./lmstudio-bench                                   # every chat model on disk
./lmstudio-bench qwen/qwen3.6-27b -s 0,1000,4000 -n 3
./lmstudio-bench qwen/qwen3.6-35b-a3b --json > results/moe.json
```

Quantization variants are addressed with `model@quant`. Note that `lms load`
cannot do this — it matches only base model keys and silently loads whichever
variant happens to be selected — but the REST API resolves them, so JIT
loading picks the right build:

```sh
./lmstudio-bench qwen/qwen3.6-27b@4bit qwen/qwen3.6-27b@8bit
```

`run-matrix.sh` sweeps every variant, one invocation each so a late failure
does not cost earlier results.

### Measuring power

Energy is measured only when asked for, because `powermetrics` needs root:

```sh
sudo -v                        # cache credentials first
./lmstudio-bench <model> --power
POWER=1 ./run-matrix.sh        # same, across the whole matrix
```

This adds `watts` and `tok/Wh` columns and a total energy figure. Without
`--power` the tool prints a warning that no energy is being measured — any
energy number derived from a run without it is an assumption, not a
measurement.

Both entry points check for sudo up front and refuse to start rather than
failing partway through a long run. `run-matrix.sh` also refreshes the sudo
timestamp in the background, since it expires after ~5 minutes but the matrix
runs for the better part of an hour.

What is measured is **SoC package power** (CPU + GPU + ANE). It excludes DRAM,
PSU losses, and the rest of the machine, so treat it as a floor on wall draw,
not a substitute for a plug meter.

Two things it does that a naive benchmark gets wrong:

- **Defeats the KV cache.** LM Studio reuses cached prefixes across requests —
  a repeated prompt prefills 11.6× faster. Prompts whose bodies share a prefix
  produce badly inflated prefill numbers, so every run's body is seeded from a
  unique nonce.
- **Reads LM Studio's own perf counters** via the native `/api/v0` endpoint.
  The OpenAI-compatible `/v1` route returns an empty `stats` object.

The `ttft_spread` column flags rows whose runs disagree by more than 25% — the
signature of an accidental cache hit. Treat those rows as suspect.

## Gotchas on 64 GB

Only one large model fits at a time. JIT-loading a second one trips LM Studio's
resource guardrail with an HTTP 400 (`insufficient system resources`), which
surfaces as sporadic failed runs rather than an obvious error. Unload first:

```sh
lms unload --all
```

Watch for other clients — an interactive `pi` session will JIT-load its own
model mid-benchmark and cause exactly this. Check with `lms ps` and
`lsof -nP -iTCP:1234`.

## Results

See [REPORT.md][report]. Raw per-run JSON is in `results/`;
`results/superseded/` holds earlier runs with known methodology flaws, kept
only so the numbers aren't accidentally re-derived.

[lms]: https://lmstudio.ai
[report]: REPORT.md
