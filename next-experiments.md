# Next experiments on the M5 Max

Candidate runs as of 2026-11-05, after the overnight ds4 pipeline. Each one
says what question it answers, what it costs, and the command shape. The
harness needs no code for any of them unless noted. Pick from here; the
results and verdict go into [REPORT.md][report] as usual.

Current standing, for reference: Flash-Next on ds4 Q2 `--mtp` is the daily
driver (71–79 tok/s, 42 GiB resident, 87.6% composite); ds4 Q4 is 89.1% at
70 GiB; MTPLX 4-bit is 65–86 tok/s at 87 GiB and 86.4%.

## 1. LiveCodeBench at 16k: the fair ds4-vs-MTPLX code score

- **Question.** At 4k, ds4 Q2 scored 54.3% against MTPLX's 45.7%, but 71 and
  93 answers were truncated. On the 75 problems both finished they tied.
  Does the gap survive when the answers can finish?
- **Cost.** No download. The 4k run took two hours per engine on ds4; the
  16k re-run touches only truncated items (71 and 93), each up to four times
  longer, so one to two hours per engine. Needs ds4 Q2 up, then MTPLX.
- **How.** `BUDGET=16384 ./run-code-quality.sh` with the ds4 target, then the
  MTPLX one; it re-asks only the truncated rows. `DS4_PREFILL=0` for ds4.
- **Expected.** Both rise to the mid-60s and the gap closes to noise. If ds4
  still leads by five points or more, the 2-bit weights are not the story and
  the engines' sampling or templates are.

## 2. DeepSeek V4 Flash Q2 on ds4: the model DwarfStar recommends

- **Question.** DwarfStar's [hardware guide][ds4-hw] calls DeepSeek V4 Flash
  Q2 the baseline for a 128 GB Mac (81 GiB resident, 34.4 tok/s at 32K on an
  M5 Max). Its quality at 2-bit is unmeasured here; Unsloth's KL figures are
  the only signal (see [m5-ultra-models.md][m5-models]). Is it worth half the
  speed of Flash-Next Q2?
- **Cost.** About 87 GB download to the internal SSD (the n-gram-style
  Engram table may add more; check `ds4 download` first). One speed sweep
  (ten minutes), the 750-item set (about an hour at 34 tok/s), LiveCodeBench
  at 4k (three hours or so). An overnight.
- **How.** Same shape as `.tmp/overnight2.sh`: serve, `./lmstudio-bench
  ds4:<alias> --label deepseek/deepseek-v4-flash@iq2_xxs`, `quality-bench`
  with `DS4_PREFILL=0`, with and without `--mtp`. ds4's model alias and
  reasoning-off switch need checking against its docs before the run.
- **Expected.** Composite above 90% would make it the "smart" model for
  agentic work with Flash-Next as the fast one; below 88% and it loses on
  both axes and leaves the roster.

## 3. llama.cpp for Flash-Next: stock, then the MTP fork

- **Question.** [dreamingwell's tracker][tracker] picks llama.cpp as the best
  fit for this machine, but it ranks only oMLX, llama.cpp, Ollama, LM Studio
  and vLLM; it knows nothing of ds4 or MTPLX. llama.cpp is untested here:
  the LM Studio Flash-Next run was its MLX engine (safetensors pack), not a
  GGUF. Published figures are 48 tok/s with UD-Q4_K_XL on an M5 Max 128 GB
  ([modelfit][modelfit]) and 36 tok/s with an 85 GB IQ4_XS on a 64 GB M5 Max
  ([AtomicChat][atomic]), both plain decode. [Unsloth][unsloth] claims MTP drafting adds
  1.3–1.7×, which would put it level with ds4 Q2 `--mtp`.
- **Why it matters even if slower.** It is the engine every client speaks,
  it runs the exact Unsloth GGUFs everyone else benchmarks, and it is the
  cleanest control for "is the engine or the quant doing this".
- **Cost.** `brew install llama.cpp` (0.8.0; qwen4exp since v0.4.0).
  UD-Q4_K_XL is 111 GB on disk (the 51B n-gram table pages from SSD under
  mmap, so resident is less); UD-Q2_K_XL is 79.6 GB and is the 2-bit
  comparison to ds4 Q2. Speed sweep ten minutes, quality set about an hour.
  The MTP step needs a source build of Daniel Hanchen's `qwen4exp/mtp` fork
  (upstream has MTP for Qwen3.6 and Qwen3.8-27B; Flash-Next there is
  unconfirmed) plus the 4 GB MTP head GGUF.
- **How.**

  ```bash
  brew install llama.cpp
  hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "*UD-Q4_K_XL*" \
     --local-dir ~/src/ds4/gguf/unsloth-flash-next
  llama-server -m ~/src/ds4/gguf/unsloth-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
     --port 8080 -c 32768 -fa on -fit off --jinja
  ```

  Keep mmap on (default): it is what lets the n-gram table page from SSD.
  AtomicChat reports automatic fitting mis-sizes this architecture, hence
  `-fit off`. Reasoning off is
  `--chat-template-kwargs '{"enable_thinking":false}'`. Then
  `LLAMACPP_URL=http://localhost:8080/v1/chat/completions ./lmstudio-bench
  llamacpp:<served name> --label qwen/qwen3.8-flash-next@q4_k_xl`, and
  `quality-bench` the same way; it should accept the empty-think prefill.

  MTP, if the stock numbers warrant it:

  ```bash
  git clone --branch qwen4exp/mtp https://github.com/danielhanchen/llama.cpp ~/src/llama.cpp-mtp
  cmake ~/src/llama.cpp-mtp -B ~/src/llama.cpp-mtp/build
  cmake --build ~/src/llama.cpp-mtp/build -j --target llama-server
  hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "*mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf*" \
     --local-dir ~/src/ds4/gguf/unsloth-flash-next
  ~/src/llama.cpp-mtp/build/bin/llama-server ... \
     -md ~/src/ds4/gguf/unsloth-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
     --spec-type draft-mtp --spec-draft-n-max 5
  ```

- **Expected.** Stock 45–50 tok/s, quality within a point of ds4 Q4 (same
  bit budget). With MTP, 60–80. It stays behind ds4 Q2 on memory either way.
  Recommended as one evening, stock build first; the fork only if stock
  quality is good and speed is the only gap.

## 4. MTPLX's "exact" claim

- **Question.** MTPLX says its MTP drafting is lossless against a plain run
  of the same weights. On Flash-Next there is no plain run to compare; on
  Qwen3.8-27B LM Studio is healthy and the same 4-bit weights exist on both.
- **Cost.** MTPLX 27B pack download (about 16 GB). Two quality runs of 750
  items, twenty minutes each.
- **How.** `quality-bench` on `mtplx:<27b>` and on the LM Studio 27B at
  temperature 0, then diff the per-item answers, not just the scores.
- **Expected.** Identical or near-identical answers. A measurable drift
  would mean MTPLX's Flash-Next numbers carry an unquantified quality tax.

## 5. `iogpu.wired_limit_mb` at long context

- **Question.** The GPU wired limit is at its default (about 75% of RAM).
  Does raising it to 120 GB change long-context speed or let a 16k run of
  the 87 GiB MTPLX pack succeed where LM Studio's 104 GiB MLX pack failed?
- **Cost.** `sudo sysctl iogpu.wired_limit_mb=122880` (resets on reboot),
  one speed sweep per engine. Every result row already records the value.
- **Expected.** Nothing measurable for ds4 Q2 (51 GiB planned), possibly a
  few percent at 16k for MTPLX. Cheap enough to fold into any other run.

## 6. Splash (Inco AI) on Qwen3.8-27B

- **Question.** The engine dimension was built for it; it claims to be tuned
  for exactly this model on M3+ Macs. Does it beat LM Studio's MLX on 27B?
- **Cost.** Install and a server; no model it does not fetch itself. One
  speed sweep, one quality run.
- **How.** `SPLASH_URL=...` then `./lmstudio-bench splash:<model> --label
  qwen/qwen3.8-27b@4bit`; `SPLASH_PREFILL=0` if it cannot continue a prefill.
- **Expected.** A speed win on 27B is plausible; it does not touch the
  Flash-Next decision.

## Suggested order

1. **Tonight, no download:** 16k code re-run (1), with the wired-limit
   sysctl (5) set beforehand so it is recorded on the rows.
2. **Next overnight:** DeepSeek V4 Flash Q2 on ds4 (2). It is the one run
   that could change the daily-driver recommendation.
3. **An evening:** llama.cpp stock (3). Its MTP fork, 4 and 6 only if there
   is appetite after that.

[report]: REPORT.md
[m5-models]: m5-ultra-models.md
[ds4-hw]: https://dwarfstar.sh/hardware
[tracker]: https://dreamingwell.github.io/apple-llm-performance/
[modelfit]: https://modelfit.io/blog/qwen38-flash-next-open-weights/
[atomic]: https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF
[unsloth]: https://unsloth.ai/docs/models/qwen3.8-flash-next
