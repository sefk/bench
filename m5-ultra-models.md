# Local models for the new Mac: M5 Max 128 GB vs M5 Ultra 96 GB

Research notes, first written 2026-10-17 for a 256 GB M5 Ultra, revised
2026-10-29 when that config was ruled out on cost. The choice is now:

| | Max-128 | Ultra-96 |
|---|---|---|
| Chip | M5 Max, up to 40 GPU cores | M5 Ultra, up to 80 GPU cores |
| Memory | 128 GB | 96 GB |
| Bandwidth | 460-614 GB/s (614 with the 40-core GPU) | 1.2 TB/s |
| Usable for one model | ~100 GB | ~75 GB |

"Usable" assumes ~78% of RAM, the share oMLX allowed one model on the 256 GB
review unit. macOS's GPU limit can be raised (`sysctl iogpu.wired_limit_mb`)
to squeeze out more, at the cost of headroom for everything else. Primary use
is coding and text processing, with image and video generation as a later
experiment.

## Short version

The two machines trade capacity for speed. Decode is bandwidth-bound, so the
Ultra runs anything that fits in both at roughly twice the speed of the Max.
The Max holds models the Ultra cannot: Qwen3.5-122B at Q5, the 2-bit
DeepSeek V4 Flash and GLM-5.3-Flash builds from ds4, and long contexts on
Flash-Next.

For coding, the best-scoring model that fits either machine is Qwen3.8
Flash-Next at 4-bit (62.5% SWE-bench Pro), with Qwen3.8-27B dense (61.7%) as
the reliable fallback. Both fit both machines. The Ultra runs them at about
88 and 48 tok/s, the Max at about 49 and 29 (Max figures are estimates). The
Ultra's weakness is context: Flash-Next at 4-bit leaves only ~7 GB of the
~75 GB for KV cache.

The models that need the Max's extra 32 GB do not score better on coding.
DeepSeek V4 Flash scores 52.6% on SWE-bench Pro at full precision, and the
2-bit build that fits 128 GB is measurably worse than that (see ds4 below). So
the extra capacity buys variety and context length, not a stronger model.

Lean Ultra-96 if speed on Flash-Next and 27B is what matters; lean Max-128 if
you want long contexts or the larger 2-bit MoE models to experiment with.

## Text and coding, in the order to try them

Speeds below are decode tok/s. Ultra figures are measured on the 256 GB
M5 Ultra (same chip and bandwidth, so they stand in for the 96 GB). Max
figures are estimates unless marked measured.

### Qwen3.8 Flash-Next (125B MoE)

The best-measured model on the Ultra chip. MacStories ran the 256 GB M5 Ultra
with oMLX and got about 108 tok/s on prose and 143 tok/s on code at 4-bit,
with prefill around 2,700 to 2,900 tok/s even at 65K context, and 87.8 tok/s
decode on a 16K prompt. LLMCheck estimates 49 tok/s on the Max-128; community
oMLX submissions on that machine range from 5 to 86, with the top ten at
76-86, so the Max figure is soft. License: Qwen 3.8's larger models carry a
revenue-share clause above $50M per year.

Memory is the catch. At 4-bit it is ~68 GB at moderate context but peaks at
~155 GB at 256K.

- Max-128: fits, with ~30 GB of usable memory for context. 256K does not fit.
- Ultra-96: tight. ~7 GB left for context at 4-bit. The ds4 2-bit build
  (42 GiB) would leave ~33 GB, at an unmeasured quality cost.
- 5-bit and up (~179 GB peak) fits neither.

### Qwen3.8-27B dense (Apache 2.0)

Fits both comfortably, at 8-bit or even BF16. On the Ultra chip, about
48 tok/s on a 6K prompt, 39 at 64K, 32 at 128K, 24 at 256K, with prefill
around 1,500 to 1,700 tok/s. LLMCheck estimates 29 tok/s on the Max. Slower
than the MoE, but one developer's writeup says dense 27B models are
noticeably more reliable at tool calling and long edits than MoE
alternatives. At about 15 GB at 4-bit it can stay loaded alongside Flash-Next
on the Max; on the Ultra the two together leave almost nothing for context.

### Qwen3-Coder-Next

Q4_K_M is ~52 GB and fits both. Q8_0 is ~85 GB and fits only the Max (one
Reddit user runs it on a 128 GB machine). No SWE-bench Pro score.

### GPT-OSS 120B

~65 GB. Fits the Max with room; tight on the Ultra (~10 GB left). ModelFit
estimates 43 tok/s on the Ultra chip. No SWE-bench Pro score.

### Max-only: Qwen3.5-122B-A10B at Q5

~92 GB at Q5_K_XL per a Reddit user, which leaves ~8 GB for context on the
Max and does not fit the Ultra. No SWE-bench Pro score.

### Max-only: DeepSeek V4 Flash and GLM-5.3-Flash via ds4 at 2-bit

DeepSeek V4 Flash (284B MoE, MIT) ships with 4-bit (MXFP4) experts at about
156 GB, which fits neither machine. [ds4][ds4] (DwarfStar 4, antirez's beta C
engine) quantizes the routed experts to 2-bit and keeps attention, shared
experts and the output head at Q8, bringing it to 81 GiB (~87 GB). A mixed
build with the last layers at 4-bit is ~98 GB. GLM-5.3-Flash at 2-bit is
90 GiB.

- Max-128: DeepSeek fits. ds4's own published benchmark is this exact
  machine: 39.4 tok/s decode at 2K context, 36.1 at 16K, 34.4 at 32K, 27.6 at
  64K, with prefill 790 falling to 400 tok/s. Reddit users report 27-30.
  GLM-5.3-Flash at 2-bit fits with little context to spare.
- Ultra-96: does not fit at ~75 GB usable. ds4 says "96 GB or more," which
  means raising the GPU limit and leaving a few GB for KV cache. The fallback
  is streaming experts from the SSD, measured at ~11 tok/s on a 64 GB M5 Pro.
- DeepSeek V4.1 Flash at 2-bit streams from SSD on either machine (13.4 tok/s
  measured on an M5 Max).

Quality is the reason not to make this the main model. Unsloth measured the
2-bit build against the official weights on wikitext-2:

| Build | Size | Perplexity | Mean KL divergence | Same top token |
|---|---|---|---|---|
| Official (MXFP4 experts) | 156 GB | 4.53 | 0 | 100% |
| MXFP4 / UD-Q4_K_XL | 155 GB | 4.535 | 0.010 | 96% |
| antirez mixed Q2/Q4 | 98 GB | 5.82 | 0.36 | 80% |
| antirez IQ2XXS | 87 GB | 6.08 | 0.41 | 78% |

On tasks the loss looks smaller: a gist of unclear authorship reports
SWE-bench Verified ~76% vs 79% (100-case subset) and MMLU-Pro 78% vs 83%. Even
before quantization, DeepSeek V4 Flash's 52.6% SWE-bench Pro is ten points
under Flash-Next. ds4 publishes no quality numbers of its own; its `ds4-eval`
is an integration check.

### Also worth benchmarking

- Laguna S 2.1 (Poolside, 118B, OpenMDW license). Coding-specialized, about
  96 GB at 4-bit, so Max-only and tight. No measured numbers found.
- KAT-Coder-V2.5 (35B). Coding-specialized, small and fast. Fits both. No
  measured numbers found.
- Qwen3.6-35B-A3B. Very fast small MoE for cheap subagent work. Fits both.
- Gemma 4 31B. ~22-28 GB, fits both.
- Mistral Medium 3.5 (128B dense, ~70 GB at 4-bit). Fits the Max; the Ultra
  only with almost no context. Dense 128B is slow on either. Modified MIT
  with a revenue cap.

### What does not fit

Qwen3.8 Flash-Next at 8-bit (187 GB peak), Qwen3 235B-A22B (~130 GB at
4-bit), DeepSeek V4 Flash at 4-bit, and the 750B-class models (GLM-5.2 and
GLM-5.3 full, Kimi K3, Qwen3.8 Max, Inkling) fit neither machine. GLM-5.3
full at 2-bit is 197 GiB; even the 256 GB machine was marginal for these.

## Image generation

- Qwen-Image-2.1 ran on the 256 GB review unit at 78 GB peak RAM, about 180
  seconds for a large wallpaper. That fits the Max and is over the Ultra's
  ~75 GB, so on the Ultra expect to use a smaller quant or offload. Best for
  text rendering inside images, Apache 2.0.
- FLUX.2 [dev] (32B) is the strongest open photorealism model and fits both.
  FLUX.2 [klein] 4B is the small Apache 2.0 version.
- Yahoo's review measured Z-Image Turbo in ComfyUI on the M5 Ultra at two to
  three times M3 Ultra speed. Diffusion is compute-bound, so the Ultra's 80 GPU
  cores should roughly double the Max here.
- Apps: Draw Things is the most Mac-native. ComfyUI is the most flexible.
  DiffusionBee is the simplest.

## Video generation

Wan 2.7 and LTX-2.3 are the open models to watch. Memory now matters: the
larger video models want more than 75 GB at full precision, which favours the
Max, while the compute favours the Ultra. The bigger problem is software:
PyTorch's MPS backend lacks torch.compile, several LoRA tools are CUDA-only,
and a Hacker News commenter called the M5 Ultra "way too slow for video." No
M5 video benchmarks found. Treat as experimental on either.

## Hardware notes

- Decode speed tracks bandwidth: 1.2 TB/s on the Ultra vs 614 GB/s on the
  40-core Max, so roughly 2x for the same model. Take the 40-core Max; the
  lower-binned GPU drops to 460 GB/s.
- Prefill tracks GPU cores (80 vs 40), so the Ultra is also roughly 2x there.
  The M5 generation's big win over M3 is prefill, up 3 to 4.6x thanks to the
  per-core Neural Accelerators; decode is only up about 1.2 to 1.3x. Prefill
  matters most for coding agents stuffing large contexts.
- The usable-memory figures are assumptions, not measurements. On today's
  M1 Max 64 GB, ds4 reports a Metal working-set ceiling of ~51 GiB (~80%),
  which is consistent.
- Plan on an external drive for the model library either way.

## Caveats

Most "tokens per second" pages for these machines are bandwidth estimates,
not measurements (LLMCheck and ModelFit label theirs as estimated). No
independent measurements of a 96 GB Ultra were found; its speeds are
borrowed from the 256 GB unit. The Max-128 has more community anecdotes than
controlled benchmarks, apart from ds4's own. Benchmark candidates locally
before settling.

## Sources

- [MacStories M5 Ultra review][macstories]
- [Hacker News discussion of that review][hn]
- [MacRumors M5 Max vs M5 Ultra][macrumors]
- [LLMCheck estimates for the M5 Max 128 GB][llmcheck-max]
- [ModelFit M5 Mac Studio estimates][modelfit]
- [Reddit: DeepSeek V4 Flash 2-bit on a 128 GB MacBook Pro][reddit-ds4]
- [ds4 (DwarfStar 4)][ds4], [its benchmarks][ds4-bench], [issue #810 (SSD streaming on 64 GB)][ds4-810]
- [Unsloth DeepSeek V4 guide (KL divergence table)][unsloth-ds4]
- [Q2 task-benchmark gist][ds4-gist]
- [Neotiler 256 vs 512 GB benchmarks][neotiler]
- [Apple MLX distributed inference post][apple-mlx]
- [Yahoo Tech review][yahoo]
- [Tom's Hardware review][toms]
- [BenchLM agentic leaderboard][benchlm-agentic]
- [BenchLM open-source leaderboard][benchlm-open]
- [Hugging Face local models guide][hf-local]
- [DEV: local LLMs on Apple Silicon][dev-to]
- [Qwen on Wikipedia][qwen-wiki]
- [Medium: local image models for Mac][medium-image]
- [Draw Things guide][draw-things]
- [LTX-Video Mac support issue][ltx-issue]
- [Lekh AI local video tools][lekh]
- [Implicator on prompt processing][implicator]

[macstories]: https://www.macstories.net/stories/m5-ultra-mac-studio-review-the-dream-mac-for-local-ai-agents/
[hn]: https://news.ycombinator.com/item?id=49787313
[macrumors]: https://www.macrumors.com/guide/m5-max-vs-m5-ultra/
[llmcheck-max]: https://llmcheck.net/best-llm/macbook-pro-m5-max-128gb/
[modelfit]: https://modelfit.io/mac-studio/m5/
[reddit-ds4]: https://www.reddit.com/r/LocalLLM/comments/1uvojrw/best_local_opensource_ai_model_for_a_macbook_pro/
[ds4]: https://github.com/antirez/ds4
[ds4-bench]: https://dwarfstar.sh/benchmarks/
[ds4-810]: https://github.com/antirez/ds4/issues/810
[unsloth-ds4]: https://unsloth.ai/docs/models/deepseek-v4
[ds4-gist]: https://gist.github.com/Farenheith/c7c6c75b2918d7074af96ec06672ff07
[neotiler]: https://getneotiler.com/blog/m5-ultra-local-llm-benchmarks/
[apple-mlx]: https://machinelearning.apple.com/research/mlx-distributed-inference
[yahoo]: https://tech.yahoo.com/ai/apple-intelligence/articles/mac-studio-m5-ultra-review-160000594.html
[toms]: https://www.tomshardware.com/desktops/mini-pcs/apple-mac-studio-m5-ultra-review
[benchlm-agentic]: https://benchlm.ai/agentic
[benchlm-open]: https://benchlm.ai/best/open-source
[hf-local]: https://huggingface.co/blog/daya-shankar/open-source-llm-models-to-run-locally
[dev-to]: https://dev.to/brunocerberus/running-local-llms-on-apple-silicon-2ecm
[qwen-wiki]: https://en.wikipedia.org/wiki/Qwen
[medium-image]: https://medium.com/@michael.hannecke/the-best-local-ai-image-models-for-mac-in-2026-flux-vs-qwen-vs-sdxl-c68956c0cde0
[draw-things]: https://jonbrown.org/blog/running-image-generation-locally-on-macos-with-draw-things-2026/
[ltx-issue]: https://github.com/Lightricks/ComfyUI-LTXVideo/issues/473
[lekh]: https://lekhai.app/blog/best-local-ai-video-generation-tools-2026/
[implicator]: https://www.implicator.ai/apple-mac-studio-m5-ultra-prompt-processing/
