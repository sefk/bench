#!/usr/bin/env python3
"""Render benchmark JSON files in results/ into a comparison table.

Usage:
    ./report.py results/*.json
"""
import argparse
import glob
import json


def short(model):
    return model.split("/")[-1]


def fmt(v, spec=".1f"):
    return "n/a" if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", default=sorted(glob.glob("results/*.json")))
    args = ap.parse_args()
    files = args.files or sorted(glob.glob("results/*.json"))

    runs = {}
    for path in files:
        for r in json.load(open(path)):
            runs.setdefault(short(r["model"]), {})[r["label"]] = r

    models = list(runs)
    labels = []
    for m in models:
        for lbl in runs[m]:
            if lbl not in labels and lbl != "warmup":
                labels.append(lbl)

    hdr = ["Case", "Prompt tok", "Gen tok"]
    for m in models:
        hdr += [f"{m} prefill tok/s", f"{m} decode tok/s"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")

    for lbl in labels:
        any_r = next((runs[m][lbl] for m in models if lbl in runs[m]), None)
        row = [lbl, str(any_r["prompt_tokens"]), str(any_r["completion_tokens"])]
        for m in models:
            r = runs[m].get(lbl)
            row += [fmt(r["prefill_tok_s"]) if r else "-",
                    fmt(r["decode_tok_s"]) if r else "-"]
        print("| " + " | ".join(row) + " |")

    print()
    for m in models:
        rs = [r for lbl, r in runs[m].items() if lbl != "warmup" and r["decode_tok_s"]]
        if rs:
            avg = sum(r["decode_tok_s"] for r in rs) / len(rs)
            peak = max(r["prefill_tok_s"] for r in rs if r["prefill_tok_s"])
            print(f"{m}: mean decode {avg:.1f} tok/s, peak prefill {peak:.0f} tok/s")


if __name__ == "__main__":
    main()
