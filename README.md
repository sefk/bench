# Local LLM benchmarks

Throughput benchmarks and other investigations of models running locally under
[LM Studio][lms] on Apple Silicon.

## Hardware

Mac Studio, Apple M1 Max — 10 CPU cores (8P/2E), 32 GPU cores, 64 GB unified
memory, ~400 GB/s memory bandwidth.

## Tooling

Benchmarks are run with `lmstudio-bench` (currently living in
`~/src/sef-dotfiles/bin/`). It measures generation throughput, time-to-first-
token, and prefill rate per model and prompt size:

```sh
lmstudio-bench                                   # every chat model on disk
lmstudio-bench qwen/qwen3.6-27b -s 0,1000,4000 -n 3
lmstudio-bench qwen/qwen3.6-35b-a3b --json > results/moe.json
```

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
