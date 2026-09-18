# Local LLM benchmarks

Throughput benchmarks and other investigations of models running locally under
[LM Studio][lms] on Apple Silicon.

## Hardware

Mac Studio, Apple M1 Max — 10 CPU cores (8P/2E), 32 GPU cores, 64 GB unified
memory, ~400 GB/s memory bandwidth.

## Tooling

### Speed: `lmstudio-bench`

Measures generation throughput, time-to-first-token, time to a complete answer,
and prefill rate per target and prompt size:

```sh
./lmstudio-bench                                   # every chat model on disk
./lmstudio-bench qwen/qwen3.6-27b -s 0,1000,4000 -n 3
./lmstudio-bench qwen/qwen3.6-35b-a3b --json > results/moe.json
./lmstudio-bench fm:system                         # Apple's on-device model
```

Every request is streamed, so each row carries two clocks:

| | from | use it for |
|---|---|---|
| `ttft`, `gen_tps` | the server's own counters | comparing builds under one server |
| `ttft_client`, `total_s`, `decode_tps` | measured at the socket | comparing *across* servers |

The server figures exclude HTTP and queueing overhead and keep rows comparable
with runs recorded before this tool streamed. The client figures are slightly
pessimistic but mean the same thing for every backend, which is the only way an
Apple-vs-Qwen latency number is worth printing. `total_s` — request in, last
token out — is what a person actually waits.

#### Apple's on-device model

macOS 27 ships `/usr/bin/fm`. `fm serve` exposes the on-device Foundation Model
over a chat-completions endpoint, and `fm:system` benchmarks it alongside the
LM Studio builds. Two things differ from LM Studio: it always streams, and it
reports no token usage at all, so prompt and completion lengths come from
`fm count-tokens` (Apple's own tokenizer) rather than from the response.

Its context window is **4096 tokens**, far smaller than the Qwen builds, and it
**overflows silently**: a prompt that does not fit returns HTTP 200 with a
well-formed but empty stream, no error of any kind. Measured boundary — 3,765
prompt tokens answer normally, 4,376 return nothing. `lmstudio-bench` detects
the empty response and names the cause, so the larger prompt sizes are reported
as skipped rather than silently filled in.

#### Variants

Quantization variants are addressed with `model@quant`. Note that `lms load`
cannot do this — it matches only base model keys and silently loads whichever
variant happens to be selected — but the REST API resolves them, so JIT
loading picks the right build:

```sh
./lmstudio-bench qwen/qwen3.6-27b@4bit qwen/qwen3.6-27b@8bit
```

#### The prompt demands a long answer

Decode rate is only meaningful measured over enough tokens. The earlier
instruction — "reply with a single short sentence" — produced **255 tokens from
Qwen** (which ignored it and ran to the cap) and **15 from Apple's model**
(which obeyed it), so the two were being compared over a 24-second decode
window and a 0.17-second one. The short-prompt case had the same problem.

Both prompts now ask for output long enough to saturate any sane `max_tokens`.
Models that already hit the cap are unaffected by the wording, so this does not
disturb comparability with earlier runs.

Note that `fm serve` **ignores `max_tokens`** and writes until it is done
(1,026–1,492 tokens in practice), so `total_s` is not comparable across
backends. Compare decode *rate*, or normalise to a fixed answer length.

#### Warm-up

Large models ramp. Measured on the 37.75 GB 8-bit MoE, decode ran **9.7 tok/s
over the first 25 requests, 33.2 over the next 60, and 41.9 once warm** — so
discarding a single warm-up run and timing the next banks a number from
partway up that ramp, and understates the largest models most.

`lmstudio-bench` therefore warms up until the rate stops climbing (three
consecutive runs within 8%, up to 12 attempts) before timing anything, once
per model rather than once per prompt size. A model that never settles is
reported with a warning instead of a quiet number.

`run-matrix.sh` sweeps every variant, one invocation each so a late failure
does not cost earlier results.

**LM Studio does not error on a variant it cannot resolve.** Ask for
`qwen/qwen3.8-27b@q4_k_m` after that GGUF build has been deleted and it answers
with whichever model is currently resident, reporting that model's id in the
response and nothing else to mark the substitution. `lmstudio-bench` and
`quality-bench` both check the served `model` against the one requested and
fail loudly, because the alternative is filing one model's numbers under
another model's name.

### Quality: `quality-bench`

Published benchmark scores describe a model at full precision on someone else's
hardware. What runs here is a 4- or 8-bit build under MLX, and the point is to
measure *that* — what you actually get by sending a question to localhost.

```sh
./quality-bench lmstudio:qwen/qwen3.6-35b-a3b@4bit -o results/2026-09-14/quality.jsonl
./quality-bench fm:system -o results/2026-09-14/quality.jsonl
./quality-bench --score results/2026-09-14/quality.jsonl          # table
./quality-bench --score results/2026-09-14/quality.jsonl --json    # for the dashboard
```

Two auto-gradable tasks, 750 items, sampled once with a fixed seed into
`quality/items.jsonl` so every variant faces the same questions and a re-run
months later is comparable:

- **GSM8K** (250 items) — grade-school math, graded on the final number.
  Reasoning.
- **MMLU** (500 items) — 4-way multiple choice, graded on the letter, sampled
  evenly across all 57 subjects. Knowledge. Even sampling matters: MMLU's
  subjects are wildly uneven, and `professional_law` alone is ~1500 of the 14k
  items, so a flat sample is mostly a law exam.

`quality/make-items.py` rebuilds the item file from the upstream sources and
exists for provenance; the benchmark loop does not use it.

#### Reasoning mode is off by default

Qwen3.x thinks by default, and a single multiple-choice item then costs 300+
tokens of reasoning — which on the slowest variant turns this into a ~10 hour
run, and makes the comparison against Apple's non-reasoning model meaningless.
LM Studio honours neither `/no_think` nor `chat_template_kwargs`, but it does
honour an assistant prefill of an empty `<think></think>` block, which is what
the default `--no-think` sends. `--think` leaves reasoning on.

**Scores from the two modes are not comparable**, so the mode is recorded on
every row and the dashboard joins only `no-think` runs.

#### Token budgets

MMLU gets 8 tokens (it wants one letter); GSM8K gets 1536.

GSM8K's budget started at 512, and that was too tight to measure what it
claimed to. At 512 all three Qwen builds scored **98% on the answers they
finished**, and their apparent differences — 94.4%, 94.4%, 91.1% — came
entirely from how often they ran past the cap (4.0%, 3.2%, 9.4% of items).
The metric was ranking verbosity and calling it reasoning.

Re-running *only* the truncated items under a larger ceiling is equivalent to
having used it all along, since an answer that stopped on its own would not
have changed. Resuming does exactly that: rows truncated under a smaller
budget than the current one are retried, rows truncated at the current budget
are not (so repeated resumes do not loop). Truncation rate stays in the
output, because a model that cannot finish inside a sane budget is a real
cost — it just should not masquerade as getting the sums wrong.

#### What the scores do and don't support

Percentages come with a 95% Wilson interval. At 750 items that is roughly ±3
points, so two variants whose intervals overlap have not been shown to differ —
the intervals are drawn on the scatter plot for exactly this reason.

Items that error are dropped, not scored as wrong: an error means the server
went away, not that the model answered badly. Re-running against the same
output file retries precisely those items, and a target that lost some shows up
as a smaller `n` rather than a quietly depressed score.

### Plotting quality against speed

```sh
./plot-quality-speed results/2026-09-14 -o results/2026-09-14/quality-vs-speed.svg
```

Decode tok/s on Y, composite quality on X, 95% intervals as horizontal bars.
Writes a standalone SVG — no plotting library, no build step, and the output
diffs when the numbers change.

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

## Tests

```sh
./tests.py                          # the benchmark scripts
cd dashboard && uv run manage.py test
```

`tests.py` covers the logic that fails quietly rather than loudly: answer
grading, score aggregation, resume-after-failure, and warm-up convergence. It
touches neither the GPU, LM Studio, nor `fm`.

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
cd dashboard && uv run manage.py runserver 0.0.0.0:8000   # reachable from the LAN
cd dashboard && honcho start                            # via Procfile, on port 8101; PORT=8123 honcho start to change it
```

Then open <http://127.0.0.1:8000/> (or <http://127.0.0.1:8101/> via
`honcho start`). Run `uv run manage.py test` to run its
test suite.

The page has three tabs, all driven by the same filter sidebar:

1. **Quality vs speed** — the interactive version of `plot-quality-speed`: one
   point per variant (its most recent run), composite quality with its 95%
   interval on X, a speed measure at one prompt size on Y.
2. **Speed by dimension** — a speed measure across prompt sizes, with series
   by variant, precision, quant, arch, version, runtime or date, plus a bar
   comparison at one size.
3. **Detailed measures** — the sortable table of every row.

The selected tab and controls are kept in the URL, so a view can be
bookmarked.

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
- `precision` — nominal weight width (`4-bit`, `8-bit`), which deliberately
  spans runtimes.
- `runtime` — `gguf` for llama.cpp-style quant names (`q4_k_m`), `mlx`
  otherwise, `afm` for Apple's on-device model.
- `variant` — a short label combining base and quant, e.g.
  `qwen3.6-35b-a3b@4bit`.

`precision` and `runtime` are separate on purpose. `4bit` (MLX) and `q4_k_m`
(GGUF) are both 4-bit and are *not* interchangeable — measured on the same
model at the same precision, the two differ by ~1.8× on decode. Group by
`precision` to ask "what does 8-bit cost?", by `runtime` to ask "what does
llama.cpp cost?", and by `quant` for the raw build id. Collapsing the two into
one series would hide the largest confound in this dataset.

Quality scores are loaded separately from `results/<date>/quality*.json` and
joined onto the throughput rows by model id, which is why `quality-bench`
records a `target` field matching `lmstudio-bench`'s `model`. Only `no-think`
runs are joined.

Results are cached in memory and re-read automatically when a result file's
mtime changes, so new runs show up on refresh without restarting the server.

[django]: https://www.djangoproject.com
[lms]: https://lmstudio.ai
[report]: REPORT.md
