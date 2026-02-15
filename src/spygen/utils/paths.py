from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_path(*parts: str) -> Path:
    return Path("data", *parts)


def output_path(*parts: str) -> Path:
    return Path("outputs", *parts)
