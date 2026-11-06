"""Shared helpers for measurements on this PC: system info, saving results, statistics."""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def gpu_info() -> dict:
    q = "name,driver_version,memory.total,clocks.max.memory,clocks.max.graphics"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout
    name, driver, mem, mclk, gclk = [x.strip() for x in out.splitlines()[0].split(",")]
    return {"name": name, "driver": driver, "memory_total": mem, "max_memory_clock": mclk, "max_graphics_clock": gclk}


def on_mains_power() -> bool | None:
    """True when a laptop is plugged in (Windows battery status 2), None if unknown."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Battery | Select-Object -First 1).BatteryStatus"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        return out == "2" if out else None
    except Exception:
        return None


def context() -> dict:
    return {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": gpu_info(),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "on_mains_power": on_mains_power(),
    }


def summary(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {
        "median": statistics.median(xs),
        "min": xs[0],
        "max": xs[-1],
        "spread_pct": 100 * (xs[-1] - xs[0]) / statistics.median(xs) if statistics.median(xs) else None,
        "n": len(xs),
    }


def _plain(o):
    """numpy and pandas scalars (from sweep tables) as plain Python numbers."""
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(type(o).__name__)


def save(name: str, payload: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_plain), encoding="utf-8")
    return path
