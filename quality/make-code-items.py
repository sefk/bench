#!/usr/bin/env python3
"""Build the fixed coding item set from LiveCodeBench.

Uses LiveCodeBench's `code_generation_lite` release v6 -- every problem in it,
175 of them, published on AtCoder and LeetCode between 2025-01-04 and
2025-04-06. That is the newest slice LiveCodeBench has released; the project
stopped publishing after it.

Contamination: LiveCodeBench's selling point is that problems are dated, so a
model can be scored only on problems published after its training cutoff.
Every model benchmarked here was released in 2026, after all of these
problems, so that protection does not apply. What the set still offers is
difficulty -- it does not saturate the way GSM8K and MMLU do -- and every
variant faces the same problems, so comparisons *between* builds of one model
remain fair.

Outputs:
  code-items.jsonl        problem statements, starter code, public examples.
                          Small; committed.
  code-tests/<id>.json.gz public + hidden tests per problem. ~280 MB
                          uncompressed, so gitignored; re-run this script to
                          rebuild them.

Hidden tests arrive as base64(zlib(pickle(json string))). They are unpickled
with an unpickler that refuses every global, so the only thing it can build is
the plain string it is supposed to contain.

Source (revision pinned for provenance):
  https://huggingface.co/datasets/livecodebench/code_generation_lite
  revision 0fe84c3912ea0c4d4a78037083943e8f0c4dd505, file test6.jsonl

Usage:  ./make-code-items.py test6.jsonl
"""

import argparse
import base64
import gzip
import io
import json
import pickle
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"


class _StringsOnly(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load global {module}.{name}")


def decode_hidden(blob):
    raw = zlib.decompress(base64.b64decode(blob))
    return json.loads(_StringsOnly(io.BytesIO(raw)).load())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="test6.jsonl from code_generation_lite")
    ap.add_argument("-o", "--out", default=str(HERE / "code-items.jsonl"))
    ap.add_argument("--tests-dir", default=str(HERE / "code-tests"))
    args = ap.parse_args()

    tests_dir = Path(args.tests_dir)
    tests_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for line in Path(args.source).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        public = json.loads(row["public_test_cases"])
        hidden = decode_hidden(row["private_test_cases"])
        tests = public + hidden
        testtypes = {t["testtype"] for t in tests}
        if len(testtypes) != 1:
            raise SystemExit(f"{row['question_id']}: mixed test types {testtypes}")

        item_id = f"lcb-{row['question_id']}"
        items.append(
            {
                "id": item_id,
                "task": "code",
                "title": row["question_title"],
                "platform": row["platform"],
                "difficulty": row["difficulty"],
                "contest_date": row["contest_date"][:10],
                "question": row["question_content"],
                "starter_code": row["starter_code"],
                "func_name": json.loads(row["metadata"] or "{}").get("func_name"),
                "testtype": testtypes.pop(),
                "n_tests": len(tests),
                # Graded by passing every test; there is no single answer string.
                "answer": "pass",
            }
        )
        with gzip.open(tests_dir / f"{item_id}.json.gz", "wt") as fh:
            json.dump(tests, fh)

    items.sort(key=lambda i: (i["contest_date"], i["id"]))
    with open(args.out, "w") as fh:
        for item in items:
            fh.write(json.dumps(item, sort_keys=True) + "\n")
    print(f"{len(items)} items -> {args.out}; tests -> {tests_dir} (revision {REVISION[:7]})")


if __name__ == "__main__":
    main()
