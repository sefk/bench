# Local models for a 256 GB M5 Ultra Mac Studio

Research notes, 2026-10-17. Target machine: M5 Ultra, 30-core CPU, 256 GB
unified memory, 2 TB SSD. Primary use is coding and text processing, with
image and video generation as a later experiment.

## Short version

The 256 GB config's sweet spot is mid-size mixture-of-experts models in the
100 to 320B range at 4 to 5-bit, run through MLX. For coding, start with
Qwen3.8 Flash-Next, keep Qwen3.8-27B dense as a fallback for tool-calling
reliability, and try DeepSeek V4 Flash or GLM-5.3-Flash for the strongest
reasoning that fits. Image generation already works well. Local video is
possible but slow and the tooling is rough.

## Text and coding, in the order to try them

### Qwen3.8 Flash-Next (125B MoE)

The best-measured model on this exact machine. MacStories ran the 256 GB
M5 Ultra with oMLX and got about 108 tok/s on prose and 143 tok/s on code at
4-bit, with prefill around 2,700 to 2,900 tok/s even at 65K context. Their
pick is the 5-bit build at roughly 179 GB, which still leaves room for long
context. License: Qwen 3.8's larger models carry a revenue-share clause above
$50M per year.

| Quant | RAM     | Decode prose / code (tok/s) | Prefill at 256K (tok/s) |
|-------|---------|-----------------------------|-------------------------|
| 4-bit | ~155 GB | 108 / 143                   | 2,544                   |
| 5-bit | ~179 GB | 100 / 133                   | 2,207                   |
| 6-bit | ~156 GB*| 95 / 123                    | 2,394                   |
| 8-bit | ~187 GB*| 87 / 119                    | 2,252                   |

\* embedding tables on SSD.

Decode at 4-bit by context: 90.7 at 4K, 83.8 at 64K, 74.7 at 256K.

### Qwen3.8-27B dense (Apache 2.0)

About 48 tok/s on a 6K prompt, 39 at 64K, 32 at 128K, 24 at 256K. Prefill
around 1,500 to 1,700 tok/s. Slower than the MoE, but one developer's
writeup says dense 27B models are noticeably more reliable at tool calling
and long edits than MoE alternatives. At about 15 GB at 4-bit it can stay
loaded alongside a bigger model. An RTX 5090 beats it on this model at every
context size (59/51/44 vs 48/39/32 at 8K/64K/128K).

### DeepSeek V4 Flash (284B MoE, MIT)

79% on SWE-bench per Vellum, the best coding score of anything that fits.
Around 167 GB as shipped. Community reports put decode at 40 to 53 tok/s
depending on runtime (llama.cpp vs MLX). Expect it to feel slower than
Flash-Next.

### GLM-5.3-Flash (320B MoE, MIT)

Tops BenchLM's agentic board among open models. Measured at about 41 tok/s
on the M5 Ultra at mixed 4/8-bit (4-bit experts, 8-bit attention), prefill
around 1,000 to 1,100 tok/s. Ran out of memory at 256K context, so this one
is tight. A Hacker News commenter reports about 30 tok/s.

### Also worth benchmarking

- Laguna S 2.1 (Poolside, 118B, OpenMDW license). Coding-specialized, about
  96 GB at 4-bit. No measured M5 Ultra numbers found.
- KAT-Coder-V2.5 (35B). Coding-specialized, small and fast. No measured
  numbers found.
- Qwen3.6-35B-A3B. Very fast small MoE for cheap subagent work.
- Mistral Medium 3.5 (128B dense, ~70 GB at 4-bit). Modified MIT with a
  revenue cap.

### What does not fit well

GLM-5.2 and GLM-5.3 full (753B), Kimi K3, Qwen3.8 Max, and Inkling (975B)
only fit at 1 to 2-bit quants, and quality suffers badly. GLM-5.2 at 3-bit is
343 GB. Those are the models the 512 GB config exists for. Hacker News
consensus is that 256 GB is the sweet spot unless you specifically want the
750B-class MoE models.

## Image generation

- Qwen-Image-2.1 ran on the review unit with 78 GB peak RAM, about 180
  seconds for a large wallpaper. Best for text rendering inside images,
  Apache 2.0.
- FLUX.2 [dev] (32B) is the strongest open photorealism model and fits
  easily. FLUX.2 [klein] 4B is the small Apache 2.0 version.
- Yahoo's review measured Z-Image Turbo in ComfyUI at two to three times
  M3 Ultra speed.
- Apps: Draw Things is the most Mac-native. ComfyUI is the most flexible.
  DiffusionBee is the simplest.

## Video generation

Wan 2.7 and LTX-2.3 are the open models to watch, and 256 GB is more than
enough memory. The problem is software: PyTorch's MPS backend lacks
torch.compile, several LoRA tools are CUDA-only, and a Hacker News commenter
called the M5 Ultra "way too slow for video." No M5 Ultra video benchmarks
found. Treat as experimental.

## Hardware notes

- Memory bandwidth is 1.2 TB/s on both GPU options (up from 819 GB/s on
  M3 Ultra), so the cheaper GPU gives up prefill compute but not decode
  speed.
- The M5 generation's big win over M3 Ultra is prefill, up 3 to 4.6x thanks
  to the per-core Neural Accelerators. Decode is only up about 1.2 to 1.3x.
  Prefill matters most for coding agents stuffing large contexts.
- 2 TB of disk fills fast. Flash-Next at 5-bit plus DeepSeek V4 Flash plus
  GLM-5.3-Flash plus a couple of image models is already over 600 GB. Plan
  on an external drive for the model library.

## Caveats

Most "tokens per second" pages for this machine are bandwidth estimates, not
measurements (LLMCheck and Modelfit label theirs as estimated). The only
independent measured numbers found are from MacStories and the Yahoo and
Tom's Hardware reviews. Benchmark candidates locally before settling.

## Sources

- [MacStories M5 Ultra review][macstories]
- [Hacker News discussion of that review][hn]
- [Neotiler 256 vs 512 GB benchmarks][neotiler]
- [LLMCheck ranking for 256 GB][llmcheck]
- [Apple MLX distributed inference post][apple-mlx]
- [Yahoo Tech review][yahoo]
- [Tom's Hardware review][toms]
- [Vellum open LLM leaderboard][vellum]
- [BenchLM agentic leaderboard][benchlm-agentic]
- [BenchLM open-source leaderboard][benchlm-open]
- [Hugging Face local models guide][hf-local]
- [DEV: local LLMs on Apple Silicon][dev-to]
- [Qwen on Wikipedia][qwen-wiki]
- [Medium: local image models for Mac][medium-image]
- [Draw Things guide][draw-things]
- [LTX-Video Mac support issue][ltx-issue]
- [Lekh AI local video tools][lekh]
- [Pinggy on the 512 GB M5 Ultra][pinggy]
- [Implicator on prompt processing][implicator]

[macstories]: https://www.macstories.net/stories/m5-ultra-mac-studio-review-the-dream-mac-for-local-ai-agents/
[hn]: https://news.ycombinator.com/item?id=49787313
[neotiler]: https://getneotiler.com/blog/m5-ultra-local-llm-benchmarks/
[llmcheck]: https://llmcheck.net/best-llm/mac-studio-m5-ultra-256gb/
[apple-mlx]: https://machinelearning.apple.com/research/mlx-distributed-inference
[yahoo]: https://tech.yahoo.com/ai/apple-intelligence/articles/mac-studio-m5-ultra-review-160000594.html
[toms]: https://www.tomshardware.com/desktops/mini-pcs/apple-mac-studio-m5-ultra-review
[vellum]: https://www.vellum.ai/open-llm-leaderboard
[benchlm-agentic]: https://benchlm.ai/agentic
[benchlm-open]: https://benchlm.ai/best/open-source
[hf-local]: https://huggingface.co/blog/daya-shankar/open-source-llm-models-to-run-locally
[dev-to]: https://dev.to/brunocerberus/running-local-llms-on-apple-silicon-2ecm
[qwen-wiki]: https://en.wikipedia.org/wiki/Qwen
[medium-image]: https://medium.com/@michael.hannecke/the-best-local-ai-image-models-for-mac-in-2026-flux-vs-qwen-vs-sdxl-c68956c0cde0
[draw-things]: https://jonbrown.org/blog/running-image-generation-locally-on-macos-with-draw-things-2026/
[ltx-issue]: https://github.com/Lightricks/ComfyUI-LTXVideo/issues/473
[lekh]: https://lekhai.app/blog/best-local-ai-video-generation-tools-2026/
[pinggy]: https://pinggy.io/blog/self_hosting_llms_on_512gb_m5_ultra_mac_studio/
[implicator]: https://www.implicator.ai/apple-mac-studio-m5-ultra-prompt-processing/
