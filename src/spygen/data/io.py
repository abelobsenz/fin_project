from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from spygen.utils.paths import ensure_dir


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p, index=False)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def append_dedup_underlying(df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    p = Path(path)
    ensure_dir(p.parent)
    if p.exists():
        existing = pd.read_parquet(p)
        merged = pd.concat([existing, df], ignore_index=True)
        merged["date"] = pd.to_datetime(merged["date"])  # type: ignore[index]
        merged = merged.drop_duplicates(subset=["date"]).sort_values("date")
    else:
        merged = df.copy()
    merged.to_parquet(p, index=False)
    return merged


def write_metadata(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(payload, indent=2, default=str))


def list_raw_chain_files(raw_dir: str | Path, start: date, end: date) -> list[Path]:
    root = Path(raw_dir)
    files = []
    for file in sorted(root.glob("*.parquet")):
        day = date.fromisoformat(file.stem)
        if start <= day <= end:
            files.append(file)
    return files


def load_raw_chains(raw_dir: str | Path, start: date, end: date) -> dict[date, pd.DataFrame]:
    out: dict[date, pd.DataFrame] = {}
    for file in list_raw_chain_files(raw_dir, start, end):
        out[date.fromisoformat(file.stem)] = pd.read_parquet(file)
    return out
