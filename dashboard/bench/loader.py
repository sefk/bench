"""Loading and deriving dimensions from benchmark result JSON files.

Results live at ``<RESULTS_DIR>/<date>/<slug>.json``, each file a JSON array
of row dicts produced by ``lmstudio-bench`` (see the repo README). This
module adds a handful of derived dimensions to each row -- pulled out of the
``model`` field -- that make the rows easy to filter and group in the
dashboard.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

# module-level cache: {"mtimes": {path: mtime}, "rows": [...]}
_CACHE: dict[str, Any] = {"mtimes": {}, "rows": []}
# runserver is threaded and the page fetches /api/meta/ and /api/rows/ at once.
# Without the lock, the second request sees the mtimes the first one just
# recorded, concludes nothing changed, and returns the still-empty cache.
_CACHE_LOCK = threading.Lock()


# Nominal weight precision, separated from the runtime that implements it.
# ``4bit`` (MLX) and ``q4_k_m`` (llama.cpp GGUF) are both 4-bit and are *not*
# interchangeable: measured on the same model, the two differ by ~1.8x on
# decode. Collapsing them into one series would hide the single largest
# confound in this dataset. So precision and runtime are separate dimensions --
# group by ``precision`` to ask "what does 8-bit cost?", by ``runtime`` to ask
# "what does llama.cpp cost?", and by ``quant`` for the raw build id.
_PRECISION = {
    "4bit": "4-bit",
    "8bit": "8-bit",
    "6bit": "6-bit",
    "3bit": "3-bit",
    "q4_k_m": "4-bit",
    "q4_k_s": "4-bit",
    "q5_k_m": "5-bit",
    "q6_k": "6-bit",
    "q8_0": "8-bit",
    # gpt-oss ships its MoE expert weights in MXFP4 (4-bit floats with a shared
    # scale per block); attention and embeddings stay at bf16. The experts are
    # nearly all the weight, so 4-bit is the honest nominal width.
    "mxfp4": "4-bit",
}

# What a model id cannot say -- gpt-oss-20b is MoE with no `-a3b`-style suffix,
# and its card gives 21B total, 3.6B active. Lives at the repo root so
# `plot-quality-speed` reads the same facts.
MODEL_FACTS_PATH = Path(__file__).resolve().parents[2] / "model-facts.json"


def _load_model_facts(path: Path = MODEL_FACTS_PATH) -> dict[str, dict]:
    try:
        with path.open() as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_")}


MODEL_FACTS = _load_model_facts()

# Scores other people measured, on hardware we do not own. Kept out of the
# result rows on purpose: they have no throughput, no sample we can vouch for,
# and must never be joined onto a measured variant.
PUBLISHED_SCORES_PATH = Path(__file__).resolve().parents[2] / "published-scores.json"


def load_published(path: Path = PUBLISHED_SCORES_PATH) -> dict:
    try:
        with path.open() as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_")}
_FACT_FIELDS = ("arch", "params", "active", "version")


def derive_dimensions(model: str) -> dict[str, str]:
    """Pull family/base/version/arch/params/active/quant/precision/runtime/
    variant out of a model id like ``qwen/qwen3.6-35b-a3b@4bit``.

    Apple's on-device model arrives as ``fm:system`` rather than a path, and is
    given its own family so it does not land in ``unknown`` next to parse
    failures.
    """
    if model.startswith("fm:"):
        return {
            "family": "apple",
            "base": "foundation-model",
            "version": "26",
            "arch": "dense",
            # Apple publishes neither a parameter count nor a weight format for
            # the on-device model. Guessing one would put a number in the
            # dashboard that nothing measured.
            "params": "undisclosed",
            "active": "undisclosed",
            "quant": "undisclosed",
            "precision": "undisclosed",
            "runtime": "afm",
            "variant": "apple-foundation-model",
        }

    if "/" in model:
        family, rest = model.split("/", 1)
    else:
        family, rest = "unknown", model

    if "@" in rest:
        base, quant = rest.split("@", 1)
    else:
        base, quant = rest, ""

    quant = quant or "unknown"

    version_match = re.search(re.escape(family) + r"(\d+\.\d+)", base)
    version = version_match.group(1) if version_match else "unknown"

    arch = "moe" if re.search(r"-a\d+b", base) else "dense"

    params_match = re.search(r"(\d+b)(?:-a(\d+b))?$", base)
    if params_match:
        params = params_match.group(1)
        active = params_match.group(2) if params_match.group(2) else params
    else:
        params = "unknown"
        active = "unknown"

    if quant == "unknown":
        runtime = "unknown"
    elif re.match(r"q\d", quant):
        runtime = "gguf"
    else:
        runtime = "mlx"

    precision = _PRECISION.get(quant.lower(), "unknown")

    variant = f"{base}@{quant}" if quant != "unknown" else base

    dims = {
        "family": family,
        "base": base,
        "version": version,
        "arch": arch,
        "params": params,
        "active": active,
        "quant": quant,
        "precision": precision,
        "runtime": runtime,
        "variant": variant,
    }
    facts = MODEL_FACTS.get(f"{family}/{base}", {})
    dims.update({k: facts[k] for k in _FACT_FIELDS if k in facts})
    return dims


# Quality summaries are written alongside the throughput results but have a
# different shape (one entry per model, no prompt size), so they are picked out
# by name and joined onto the throughput rows rather than loaded as rows.
QUALITY_PREFIX = "quality"


def _is_quality_file(path: Path) -> bool:
    return path.name.startswith(QUALITY_PREFIX)


def _iter_result_files(results_dir: Path):
    if not results_dir.exists():
        return
    for path in sorted(results_dir.glob("*/*.json")):
        if not _is_quality_file(path):
            yield path


def _iter_quality_files(results_dir: Path):
    if not results_dir.exists():
        return
    for path in sorted(results_dir.glob("*/*.json")):
        if _is_quality_file(path):
            yield path


# Code scores depend heavily on the token budget: at 4k, half the hard
# problems are cut off before any code is written. A run under a larger budget
# is therefore its own measure (`quality_code_16k`), not a replacement for the
# 4k one -- both are worth seeing side by side.
CODE_BASE_BUDGET = 4096


def _quality_field(task: str, max_tokens: int | None) -> str:
    if task == "code" and max_tokens and max_tokens > CODE_BASE_BUDGET:
        return f"quality_code_{max_tokens // 1024}k"
    return f"quality_{task}"


# Modes that are joined, best first. `no-think` is the baseline. `effort-low`
# is the least reasoning a model that cannot switch it off will do (gpt-oss),
# so it is that model's nearest equivalent -- shown, but marked with
# `quality_mode` so it is never read as the same measurement. A full `think`
# run is never joined.
JOINED_MODES = ("no-think", "effort-low")


def load_quality(results_dir: Path) -> dict[str, dict]:
    """Map target id -> quality scores, from `quality*.json` summary files.

    Only the modes in JOINED_MODES are joined, and a target's no-think scores
    win over any other mode it was also run in. Scores from a reasoning-mode
    run are not comparable with no-think ones -- the model is doing a
    different amount of work per item -- so each row records the mode its
    scores came from. A later date wins, so a re-run supersedes an older score.

    Scores are merged per field, not per entry: the code task is summarised in
    its own file (`quality-code.json`), and its entry for a model must not
    blank out that model's GSM8K/MMLU scores from `quality.json`.
    """
    by_mode: dict[str, dict[str, dict]] = {}
    for path in sorted(_iter_quality_files(results_dir)):
        try:
            with path.open() as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            mode = entry.get("mode", "no-think")
            if mode not in JOINED_MODES:
                continue
            target = entry.get("target") or entry.get("model")
            if not target:
                continue
            fields = {"quality_mode": mode}
            composite = entry.get("composite")
            if composite:
                ci = composite.get("ci95") or [None, None]
                fields.update(
                    quality_pct=composite.get("pct"),
                    quality_ci_low=ci[0],
                    quality_ci_high=ci[1],
                    quality_n=composite.get("n"),
                    quality_date=path.parent.name,
                )
            for task in ("gsm8k", "mmlu", "code"):
                scored = entry.get(task)
                if not scored:
                    continue
                name = _quality_field(task, scored.get("max_tokens"))
                ci = scored.get("ci95") or [None, None]
                fields[name] = scored.get("pct")
                fields[f"{name}_ci_low"] = ci[0]
                fields[f"{name}_ci_high"] = ci[1]
                fields[f"{name}_n"] = scored.get("n")
                # >0 while a raised-budget run is still working through the
                # answers truncated under the smaller one: the score is a
                # lower bound, and the dashboard says so.
                fields[f"{name}_pending"] = scored.get("pending") or 0
            by_mode.setdefault(target, {}).setdefault(mode, {}).update(fields)

    # One mode per target, never a mix: a no-think composite beside an
    # effort-low code score would be two measurements under one label.
    scores: dict[str, dict] = {}
    for target, modes in by_mode.items():
        best = next(m for m in JOINED_MODES if m in modes)
        scores[target] = modes[best]
    return scores


def _current_mtimes(results_dir: Path) -> dict[str, float]:
    current = {}
    for path in list(_iter_result_files(results_dir)) + list(
        _iter_quality_files(results_dir)
    ):
        try:
            current[str(path)] = path.stat().st_mtime
        except OSError:
            continue
    return current


def load_rows(results_dir: Path, force: bool = False) -> list[dict]:
    """Load and flatten all result rows under ``results_dir``, adding
    derived dimensions. Cached in memory; re-read automatically when any
    result file is added, removed, or its mtime changes.
    """
    results_dir = Path(results_dir)
    with _CACHE_LOCK:
        current = _current_mtimes(results_dir)
        if force or current != _CACHE["mtimes"]:
            _CACHE["rows"] = _read_rows(results_dir)
            _CACHE["mtimes"] = current
        return _CACHE["rows"]


def _read_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in _iter_result_files(results_dir):
        date = path.parent.name
        try:
            with path.open() as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, list):
            continue
        for raw_row in data:
            row = dict(raw_row)
            row["date"] = date
            row["source_file"] = path.name
            model = row.get("model", "")
            row.update(derive_dimensions(model))
            rows.append(row)

    quality = load_quality(results_dir)
    for row in rows:
        row.update(quality.get(row.get("model", ""), {}))
    return rows


MEASURES = [
    ("gen_tps", "Decode tok/s"),
    ("prefill_tps", "Prefill tok/s"),
    ("ttft", "TTFT (s)"),
    ("total_s", "Time to full answer (s)"),
    ("quality_pct", "Quality (% correct)"),
    ("quality_gsm8k", "Quality: GSM8K (%)"),
    ("quality_mmlu", "Quality: MMLU (%)"),
    ("quality_code", "Quality: LiveCodeBench, 4k budget (%)"),
    ("quality_code_16k", "Quality: LiveCodeBench, 16k budget (%)"),
    ("watts", "Watts"),
    ("tokens_per_wh", "Tokens/Wh"),
]

DIMENSIONS = [
    "version", "arch", "quant", "precision", "runtime", "date", "variant",
]

SIZE_ORDER = ["short", "1000tok", "4000tok", "16000tok"]


def build_meta(rows: list[dict]) -> dict:
    meta: dict[str, Any] = {}
    for dim in DIMENSIONS:
        values = sorted({row.get(dim, "unknown") for row in rows})
        meta[dim] = values
    sizes_present = {row.get("size") for row in rows if row.get("size")}
    meta["size"] = [s for s in SIZE_ORDER if s in sizes_present] + sorted(
        sizes_present - set(SIZE_ORDER)
    )
    meta["measures"] = [{"key": key, "label": label} for key, label in MEASURES]
    return meta
