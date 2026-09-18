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
}


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

    return {
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


def load_quality(results_dir: Path) -> dict[str, dict]:
    """Map target id -> quality scores, from `quality*.json` summary files.

    Only `no-think` runs are joined. Scores from a reasoning-mode run are not
    comparable with them -- the model is doing a different amount of work per
    item -- so mixing the two into one lookup would silently compare across
    modes. A later date wins, so a re-run supersedes an older score.
    """
    scores: dict[str, dict] = {}
    for path in sorted(_iter_quality_files(results_dir)):
        try:
            with path.open() as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            if entry.get("mode", "no-think") != "no-think":
                continue
            target = entry.get("target") or entry.get("model")
            if not target:
                continue
            composite = entry.get("composite") or {}
            scores[target] = {
                "quality_pct": composite.get("pct"),
                "quality_ci_low": (composite.get("ci95") or [None, None])[0],
                "quality_ci_high": (composite.get("ci95") or [None, None])[1],
                "quality_n": composite.get("n"),
                "quality_gsm8k": (entry.get("gsm8k") or {}).get("pct"),
                "quality_mmlu": (entry.get("mmlu") or {}).get("pct"),
                "quality_date": path.parent.name,
            }
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
