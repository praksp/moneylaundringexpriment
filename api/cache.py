"""
api/cache.py — Lightweight TTL in-memory cache
================================================
No external dependencies — uses stdlib `time` only.

Usage:
    from api.cache import ttl_cache, invalidate, get_cached, set_cached

    @ttl_cache(ttl=300)                  # 5-minute cache
    def expensive_query():
        ...

    # Manual get/set (for pre-warming)
    set_cached("world_map", data, ttl=300)
    data = get_cached("world_map")        # None if expired/missing
    invalidate("world_map")
"""
from __future__ import annotations

import functools
import threading
import time
from typing import Any, Callable, Optional

_store: dict[str, tuple[float, Any]] = {}   # key → (expires_at, value)
_lock  = threading.Lock()


def get_cached(key: str) -> Optional[Any]:
    """Return cached value or None if missing/expired."""
    with _lock:
        entry = _store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() > expires_at:
            del _store[key]
            return None
        return value


def set_cached(key: str, value: Any, ttl: int = 60) -> None:
    """Store a value with a TTL (seconds)."""
    with _lock:
        _store[key] = (time.monotonic() + ttl, value)


def invalidate(key: str) -> None:
    with _lock:
        _store.pop(key, None)


def invalidate_prefix(prefix: str) -> None:
    """Remove all keys that start with prefix."""
    with _lock:
        for k in list(_store):
            if k.startswith(prefix):
                del _store[k]


def ttl_cache(ttl: int = 60, key_fn: Optional[Callable] = None):
    """
    Decorator that caches the result of a function for `ttl` seconds.

    key_fn(args, kwargs) → str can be provided to build the cache key;
    defaults to using the function name + repr of all args/kwargs.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if key_fn is not None:
                cache_key = key_fn(args, kwargs)
            else:
                cache_key = f"{fn.__module__}.{fn.__qualname__}:{repr(args)}:{repr(sorted(kwargs.items()))}"

            cached = get_cached(cache_key)
            if cached is not None:
                return cached

            result = fn(*args, **kwargs)
            set_cached(cache_key, result, ttl=ttl)
            return result

        wrapper.invalidate = lambda *a, **kw: invalidate(
            key_fn(a, kw) if key_fn else
            f"{fn.__module__}.{fn.__qualname__}:{repr(a)}:{repr(sorted(kw.items()))}"
        )
        return wrapper
    return decorator
