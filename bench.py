#!/usr/bin/env python3
"""Benchmark a model served by LM Studio: prefill + decode throughput.

Usage:
    ./bench.py <model-id> [--out results/<name>.json]

Talks to the LM Studio OpenAI-compatible endpoint on localhost:1234 and
measures time-to-first-token and decode rate across a spread of prompt
lengths and generation lengths.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

URL = os.environ.get("LMS_URL", "http://localhost:1234/v1/chat/completions")

FILLER = ("The quick brown fox jumps over the lazy dog near the riverbank while "
          "seventeen curious observers record every detail in their notebooks. ")


def make_prompt(approx_tokens):
    """Build a prompt of roughly approx_tokens (FILLER is ~20 tokens)."""
    reps = max(1, approx_tokens // 20)
    return (FILLER * reps) + "\n\nSummarize the above in one short sentence."


def run(model, prompt, max_tokens, label):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n_chunks += 1
    total = time.perf_counter() - t0
    pt = usage.get("prompt_tokens") if usage else None
    ct = usage.get("completion_tokens") if usage else n_chunks
    decode_time = total - (ttft or 0)
    return {
        "label": label,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "ttft_s": ttft,
        "total_s": total,
        # first token is produced by prefill, so decode rate excludes it
        "decode_tok_s": (ct - 1) / decode_time if decode_time > 0 and ct else None,
        "prefill_tok_s": pt / ttft if pt and ttft else None,
    }


# (label, approx prompt tokens, max output tokens)
CASES = [
    ("warmup", 20, 64),
    ("short-ctx (~64)", 64, 256),
    ("medium-ctx (~1k)", 1000, 256),
    ("long-ctx (~4k)", 4000, 256),
    ("long-ctx (~16k)", 16000, 256),
    ("gen-heavy (~64 in / 1024 out)", 64, 1024),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help='model id, e.g. "qwen/qwen3.6-27b"')
    ap.add_argument("--out", help="write JSON results here")
    args = ap.parse_args()

    results = []
    for label, approx_in, max_out in CASES:
        sys.stderr.write(f"running {label} ...\n")
        sys.stderr.flush()
        r = run(args.model, make_prompt(approx_in), max_out, label)
        r["model"] = args.model
        results.append(r)
        sys.stderr.write(f"  -> {json.dumps(r)}\n")
        sys.stderr.flush()

    blob = json.dumps(results, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(blob + "\n")
        sys.stderr.write(f"wrote {args.out}\n")
    else:
        print(blob)


if __name__ == "__main__":
    main()
