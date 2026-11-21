"""Is a game running? Used to pause background analysis while you play (Windows only)."""
from __future__ import annotations

import os
import subprocess
import time

from .media import CREATE_NO_WINDOW

_cache: dict = {"t": -1e9, "names": (), "running": False}


def game_running(names: list[str], ttl: float = 10.0) -> bool:
    if os.name != "nt" or not names:
        return False
    now = time.monotonic()
    key = tuple(n.lower() for n in names)
    if now - _cache["t"] < ttl and _cache["names"] == key:
        return _cache["running"]
    try:
        out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=10,
                             creationflags=CREATE_NO_WINDOW).stdout.lower()
        running = any(f'"{n}"' in out for n in key)
    except (OSError, subprocess.TimeoutExpired):
        running = False
    _cache.update(t=now, names=key, running=running)
    return running
