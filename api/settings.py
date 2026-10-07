"""
Runtime settings, read from environment variables (never from code, never from the browser).

  FRONTEND_ORIGIN   allowed browser origin(s) for CORS, comma separated.  Default: http://localhost:3000
  ENVIRONMENT       development (default) or production.  In production a wildcard origin is refused.
  CONFIG_PATH       optional path to config.yaml.  Default: the project's config.yaml

The Angel One credentials listed in .env.example are deliberately NOT read anywhere yet (Phase 8E).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FRONTEND_ORIGIN = "http://localhost:3000"
_ORIGIN = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d{1,5})?$")


@dataclass(frozen=True)
class Settings:
    environment: str
    frontend_origins: tuple
    config_path: Path


def parse_origins(raw: str) -> tuple:
    """'http://a.com, http://b.com/' -> ('http://a.com', 'http://b.com'). Anything that is not scheme://host[:port] is refused."""
    origins = []
    for item in raw.split(","):
        item = item.strip().rstrip("/")
        if not item:
            continue
        if item != "*" and not _ORIGIN.match(item):
            raise RuntimeError(f"FRONTEND_ORIGIN contains {item!r}, which is not an origin like https://example.com")
        if item not in origins:
            origins.append(item)
    return tuple(origins)


def get_settings() -> Settings:
    environment = os.environ.get("ENVIRONMENT", "development").strip().lower() or "development"
    if environment not in {"development", "production"}:
        raise RuntimeError(f"ENVIRONMENT must be 'development' or 'production', got {environment!r}.")
    if environment == "production" and not os.environ.get("FRONTEND_ORIGIN", "").strip():
        raise RuntimeError("In production FRONTEND_ORIGIN must be set to the exact frontend origin (there is no default).")
    origins = parse_origins(os.environ.get("FRONTEND_ORIGIN", DEFAULT_FRONTEND_ORIGIN))
    if environment == "production" and ("*" in origins or not origins):
        raise RuntimeError("In production FRONTEND_ORIGIN must list the exact frontend origin(s); '*' is not allowed.")
    config_path = Path(os.environ.get("CONFIG_PATH") or PROJECT_ROOT / "config.yaml").resolve()
    return Settings(environment=environment, frontend_origins=origins, config_path=config_path)
