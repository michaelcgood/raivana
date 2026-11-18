"""User configuration: API keys and settings from environment variables or a `.env` file.

Raivana never stores keys itself. It reads them, in this order (first match wins):
  1. real environment variables
  2. `.env` in the current directory
  3. `~/.raivana/.env`
Existing environment variables are never overwritten. See `.env.example`.
"""

from __future__ import annotations

import os
from pathlib import Path

_loaded = False


def env_paths() -> list[Path]:
    home = Path(os.environ.get("RAIVANA_HOME", Path.home() / ".raivana"))
    return [Path.cwd() / ".env", home / ".env"]


def load_env(force: bool = False) -> list[Path]:
    """Load KEY=VALUE lines from .env files into os.environ (never overriding). Returns files loaded."""
    global _loaded
    if _loaded and not force:
        return []
    _loaded = True
    loaded = []
    for path in env_paths():
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:]
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            if key and value and key not in os.environ:
                os.environ[key] = value
        loaded.append(path)
    return loaded


def mask(secret: str | None) -> str:
    if not secret:
        return "not set"
    return f"{secret[:7]}...{secret[-4:]}" if len(secret) > 14 else "set"
