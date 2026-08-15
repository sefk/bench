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
from pathlib import Path
from typing import Any

# module-level cache: {"mtimes": {path: mtime}, "rows": [...]}
_CACHE: dict[str, Any] = {"mtimes": {}, "rows": []}


def derive_dimensions(model: str) -> dict[str, str]:
    """Pull family/base/version/arch/params/active/quant/runtime/variant out
    of a model id like ``qwen/qwen3.6-35b-a3b@4bit``.
    """
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

    variant = f"{base}@{quant}" if quant != "unknown" else base

    return {
        "family": family,
        "base": base,
        "version": version,
        "arch": arch,
        "params": params,
        "active": active,
        "quant": quant,
        "runtime": runtime,
        "variant": variant,
    }


def _iter_result_files(results_dir: Path):
    if not results_dir.exists():
        return
    for path in sorted(results_dir.glob("*/*.json")):
        yield path


def _needs_reload(results_dir: Path) -> bool:
    current = {}
    for path in _iter_result_files(results_dir):
        try:
            current[str(path)] = path.stat().st_mtime
        except OSError:
            continue
    if current != _CACHE["mtimes"]:
        _CACHE["mtimes"] = current
        return True
    return False


def load_rows(results_dir: Path, force: bool = False) -> list[dict]:
    """Load and flatten all result rows under ``results_dir``, adding
    derived dimensions. Cached in memory; re-read automatically when any
    result file is added, removed, or its mtime changes.
    """
    results_dir = Path(results_dir)
    if force or _needs_reload(results_dir):
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
        _CACHE["rows"] = rows
    return _CACHE["rows"]


MEASURES = [
    ("gen_tps", "Decode tok/s"),
    ("prefill_tps", "Prefill tok/s"),
    ("ttft", "TTFT (s)"),
    ("watts", "Watts"),
    ("tokens_per_wh", "Tokens/Wh"),
]

DIMENSIONS = ["version", "arch", "quant", "runtime", "date", "variant"]

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
