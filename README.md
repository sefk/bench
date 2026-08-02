# Local LLM benchmarks

Throughput benchmarks for models served locally by [LM Studio][lms] on Apple
Silicon. Measures prefill (prompt processing) and decode (token generation)
rates across a spread of context and generation lengths.

## Hardware

Mac Studio, Apple M1 Max — 10 CPU cores (8P/2E), 32 GPU cores, 64 GB unified
memory. Roughly 400 GB/s memory bandwidth.

## Usage

Load a model in LM Studio and make sure the local server is running:

```sh
lms server start
lms load qwen/qwen3.6-27b --context-length 262144
```

Then benchmark it and render the comparison:

```sh
./bench.py qwen/qwen3.6-27b --out results/qwen3.6-27b.json
./report.py
```

`bench.py` streams from the OpenAI-compatible endpoint at
`localhost:1234` (override with `LMS_URL`) and reports, per case:

- `ttft_s` — time to first token
- `prefill_tok_s` — `prompt_tokens / ttft`, prompt processing rate
- `decode_tok_s` — generation rate after the first token

Only one large model fits comfortably alongside its KV cache on 64 GB, so
unload the previous one before loading the next:

```sh
lms unload <previous-model>
```

## Results

See [REPORT.md][report] for measurements and analysis. Raw per-run JSON lives
in `results/`.

[lms]: https://lmstudio.ai
[report]: REPORT.md
