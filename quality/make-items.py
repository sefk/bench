#!/usr/bin/env python3
"""Build the fixed quality-eval item set from GSM8K and MMLU.

Both sources are sampled once with a fixed seed and the result is committed as
`items.jsonl`, so every variant is graded on exactly the same questions and a
re-run months later is comparable. Re-running this script reproduces the same
file; it is here for provenance, not as part of the benchmark loop.

Sources (downloaded, not vendored -- they are large and externally hosted):
  GSM8K test split  https://github.com/openai/grade-school-math
  MMLU  test split  https://people.eecs.berkeley.edu/~hendrycks/data.tar

Usage:  ./make-items.py --gsm8k test.jsonl --mmlu data/test -o items.jsonl
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

SEED = 20260914
N_GSM8K = 250
N_MMLU = 500


def load_gsm8k(path):
    """GSM8K answers end with a '#### <number>' line; that number is the key."""
    items = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        answer = row["answer"].rsplit("####", 1)[1].strip().replace(",", "")
        items.append({"task": "gsm8k", "question": row["question"], "answer": answer})
    return items


def load_mmlu(path):
    """One CSV per subject: question, four choices, then the correct letter."""
    items = []
    for csv_path in sorted(Path(path).glob("*_test.csv")):
        subject = csv_path.name.removesuffix("_test.csv")
        with csv_path.open(newline="") as fh:
            for row in csv.reader(fh):
                if len(row) != 6:
                    continue
                items.append(
                    {
                        "task": "mmlu",
                        "subject": subject,
                        "question": row[0],
                        "choices": row[1:5],
                        "answer": row[5].strip().upper(),
                    }
                )
    return items


def sample_mmlu(items, n, rng):
    """Stratify across subjects so no single subject dominates the score.

    MMLU's subjects are wildly uneven in size (professional_law alone is ~1500
    of the 14k items). A flat sample would make the score largely a law exam.
    """
    by_subject = {}
    for item in items:
        by_subject.setdefault(item["subject"], []).append(item)

    subjects = sorted(by_subject)
    per_subject, remainder = divmod(n, len(subjects))
    # The remainder goes to a seeded random subset of subjects, not the first
    # few alphabetically, so the extra items are not all humanities.
    extra = set(rng.sample(subjects, remainder))

    picked = []
    for subject in subjects:
        take = per_subject + (1 if subject in extra else 0)
        picked.extend(rng.sample(by_subject[subject], min(take, len(by_subject[subject]))))
    rng.shuffle(picked)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsm8k", required=True, help="GSM8K test.jsonl")
    ap.add_argument("--mmlu", required=True, help="MMLU data/test directory")
    ap.add_argument("-o", "--out", default="items.jsonl")
    args = ap.parse_args()

    rng = random.Random(SEED)

    gsm = load_gsm8k(args.gsm8k)
    gsm_pick = rng.sample(gsm, N_GSM8K)

    mmlu = load_mmlu(args.mmlu)
    mmlu_pick = sample_mmlu(mmlu, N_MMLU, rng)

    with open(args.out, "w") as fh:
        for i, item in enumerate(gsm_pick + mmlu_pick):
            item["id"] = f"{item['task']}-{i:04d}"
            fh.write(json.dumps(item, sort_keys=True) + "\n")

    print(f"{len(gsm_pick)} gsm8k + {len(mmlu_pick)} mmlu -> {args.out}")
    print(f"gsm8k pool {len(gsm)}, mmlu pool {len(mmlu)} across "
          f"{len({i['subject'] for i in mmlu})} subjects")


if __name__ == "__main__":
    main()
