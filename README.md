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

`powermetrics` needs root, and these tools never invoke `sudo`. Start the
sampler yourself, run the benchmark against its log, then stop it:

```sh
sudo powermetrics --samplers cpu_power,gpu_power -i 1000 -o /tmp/power.log &

POWER_LOG=/tmp/power.log ./run-matrix.sh          # or:
./lmstudio-bench <model> --power-log /tmp/power.log

sudo pkill -INT powermetrics
```

Start the sampler **before** the benchmark and stop it **after**. The samples
that fall outside the benchmark windows are what establish the idle baseline;
without them there is no increment to report.

Output gains `watts`, `over idle`, and `tok/Wh` columns plus a total. Because
this machine is left powered on regardless, the figure that matters is energy
**above idle**, so `tok/Wh` is computed against incremental energy, not total
draw.

Without `--power-log` the tool says plainly that it is not measuring power. Any
energy number derived from a run without it is an assumption, not a
measurement.

Baseline handling: samples within 10 s of a benchmark window are excluded (the
GPU neither drops to idle instantly nor ramps instantly), and the baseline is
the median rather than the mean, so a leaked busy sample cannot inflate it — an
inflated baseline would silently understate the increment.

What is measured is **SoC package power** (CPU + GPU + ANE). It excludes DRAM,
PSU losses, and the rest of the machine, so treat it as a floor on wall draw,
not a substitute for a plug meter.

## Results layout

Results are kept per run date, since runs are infrequent and worth keeping
apart:

```
results/2026-08-02/qwen3.6-35b-a3b-4bit.json
```

`run-matrix.sh` writes to `results/$(date +%F)/` by default; override with
`OUTDIR=...`.

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

See [REPORT.md][report]. Raw per-run JSON is under `results/<date>/`.

## Dashboard

`dashboard/` is a small localhost-only [Django][django] app for exploring
results interactively — filterable charts and a sortable table, no build
step. It reads every `results/<date>/*.json` file directly; nothing is
written back.

```sh
cd dashboard && uv sync && uv run manage.py runserver
```

Then open <http://127.0.0.1:8000/>. Run `uv run manage.py test` to run its
test suite.

The loader (`dashboard/bench/loader.py`) derives extra dimensions from each
row's `model` field so results can be filtered and grouped:

- `family` / `base` — e.g. `qwen` / `qwen3.6-35b-a3b`, split on the first `/`
  and `@` in `model`.
- `version` — e.g. `3.6`, pulled from the base with a `qwen(\d+\.\d+)`-style
  regex; `unknown` if it doesn't match.
- `arch` — `moe` if the base has an active-params suffix (`-a3b`), else
  `dense`.
- `params` / `active` — total and active parameter counts, e.g. `35b` / `3b`
  (active equals params for dense models).
- `quant` — the part after `@` (`4bit`, `8bit`, `q4_k_m`), `unknown` if none.
- `runtime` — `gguf` for llama.cpp-style quant names (`q4_k_m`), `mlx`
  otherwise.
- `variant` — a short label combining base and quant, e.g.
  `qwen3.6-35b-a3b@4bit`.

Results are cached in memory and re-read automatically when a result file's
mtime changes, so new runs show up on refresh without restarting the server.

[django]: https://www.djangoproject.com
[lms]: https://lmstudio.ai
[report]: REPORT.md
