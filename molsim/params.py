"""Loading parameter files.

A value entry looks like:
    {value: 1008, unit: "GB/s", label: published, source: "...", note: "...", range: [lo, hi]}
and becomes a Value. Model files are plain shapes plus a few labelled values.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .labeled import Value

PARAMS_DIR = Path(__file__).resolve().parent.parent / "params"


def _is_value(entry) -> bool:
    return isinstance(entry, dict) and "value" in entry and "label" in entry


def _to_value(entry: dict) -> Value:
    return Value.of(
        entry["value"],
        entry.get("unit", ""),
        label=entry["label"],
        source=entry.get("source", ""),
        note=entry.get("note", ""),
        range=entry.get("range"),
    )


def _convert(node):
    if _is_value(node):
        return _to_value(node)
    if isinstance(node, dict):
        return {k: _convert(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_convert(v) for v in node]
    return node


_CACHE: dict[str, tuple[float, dict]] = {}


def _copy(node):
    """Fresh dicts and lists around the shared, immutable Values (callers may edit the structure)."""
    if isinstance(node, dict):
        return {k: _copy(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_copy(v) for v in node]
    return node


def load(name: str) -> dict:
    """Load params/<name>.yaml, turning every labelled entry into a Value (parsed once per file change)."""
    path = PARAMS_DIR / f"{name}.yaml"
    mtime = path.stat().st_mtime
    hit = _CACHE.get(name)
    if hit is None or hit[0] != mtime:
        with path.open(encoding="utf-8") as f:
            hit = (mtime, _convert(yaml.safe_load(f) or {}))
        _CACHE[name] = hit
    return _copy(hit[1])


def all_values(tree, prefix: str = "") -> list[tuple[str, Value]]:
    """Every Value in a loaded file, with its dotted path (for checks and reports)."""
    out: list[tuple[str, Value]] = []
    if isinstance(tree, Value):
        return [(prefix, tree)]
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.extend(all_values(v, f"{prefix}.{k}" if prefix else str(k)))
    if isinstance(tree, list):
        for i, v in enumerate(tree):
            out.extend(all_values(v, f"{prefix}[{i}]"))
    return out
