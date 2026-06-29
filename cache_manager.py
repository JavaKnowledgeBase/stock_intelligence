"""
Shared file-based cache for cross-process data sharing.

All Streamlit worker processes share this cache, so expensive market scans
(yfinance calls, options chains) are computed once and served to all 200+
concurrent users instantly from disk.

Keys map to JSON files under data/shared_cache/.
Atomic rename-on-write prevents torn reads.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

_CACHE_DIR = Path("data/shared_cache")


def _path(key: str) -> Path:
    safe = key.replace("/", "_").replace("\\", "_").replace(":", "_")
    return _CACHE_DIR / f"{safe}.json"


def save(key: str, data: Any, ttl_seconds: int = 900) -> None:
    """Persist data to disk. Uses atomic rename so readers never see partial writes."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"ts": time.time(), "ttl": ttl_seconds, "data": data}
    p = _path(key)
    tmp = p.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, default=str)
        tmp.replace(p)
    except Exception:
        pass


def load(key: str) -> Optional[Any]:
    """Return data if present and not expired, else None."""
    p = _path(key)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            payload = json.load(f)
        if time.time() - payload["ts"] > payload["ttl"]:
            return None
        return payload["data"]
    except Exception:
        return None


def load_stale(key: str) -> Optional[tuple[Any, float]]:
    """Return (data, age_seconds) even if TTL expired. None if file missing."""
    p = _path(key)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            payload = json.load(f)
        age = time.time() - payload["ts"]
        return payload["data"], age
    except Exception:
        return None


def age_seconds(key: str) -> Optional[float]:
    """How many seconds old is the cached value? None if not found."""
    result = load_stale(key)
    return result[1] if result else None


def invalidate(key: str) -> None:
    """Delete a cache entry so the next read forces a fresh fetch."""
    try:
        _path(key).unlink(missing_ok=True)
    except Exception:
        pass


def age_label(key: str) -> str:
    """Human-readable age string for display in the dashboard."""
    secs = age_seconds(key)
    if secs is None:
        return "never fetched"
    if secs < 60:
        return f"{int(secs)}s ago"
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    return f"{secs / 3600:.1f}h ago"
