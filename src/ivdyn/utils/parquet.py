"""Parquet write helpers with pragmatic fallbacks for mixed object columns."""

from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

_COLUMN_ERROR_RE = re.compile(r"conversion failed for column ([^ ]+) with type object", flags=re.IGNORECASE)
_BYTES_LIKE = (bytes, bytearray, memoryview)


def _is_missing_engine_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("unable to find a usable engine" in msg) or (
        "missing optional dependency" in msg and ("pyarrow" in msg or "fastparquet" in msg)
    )


def _is_arrow_object_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "conversion failed for column" in msg
        or "expected bytes, got a" in msg
        or "did not recognize python value type" in msg
        or "could not convert" in msg
    )


def _extract_problem_column(exc: Exception, columns: pd.Index) -> str | None:
    match = _COLUMN_ERROR_RE.search(str(exc))
    if not match:
        return None
    name = match.group(1)
    return name if name in columns else None


def _is_missing_value(v: object) -> bool:
    if v is None:
        return True
    try:
        out = pd.isna(v)
    except Exception:
        return False
    if isinstance(out, (bool, np.bool_)):
        return bool(out)
    return False


def _stringify_value(v: object) -> object:
    if _is_missing_value(v):
        return pd.NA
    if isinstance(v, _BYTES_LIKE):
        raw = bytes(v)
        try:
            return raw.decode("utf-8")
        except Exception:
            return raw.hex()
    return str(v)


def _needs_object_string_coercion(series: pd.Series) -> bool:
    if series.dtype != object:
        return False
    non_null = [v for v in series.to_numpy(copy=False) if not _is_missing_value(v)]
    if not non_null:
        return False

    has_bytes = any(isinstance(v, _BYTES_LIKE) for v in non_null)
    has_non_bytes = any(not isinstance(v, _BYTES_LIKE) for v in non_null)
    if has_bytes and has_non_bytes:
        return True

    value_types = {type(v) for v in non_null}
    if len(value_types) > 1:
        return True

    only_type = next(iter(value_types))
    if issubclass(only_type, (str, *_BYTES_LIKE)):
        return False
    if issubclass(only_type, (int, float, bool, np.integer, np.floating, np.bool_)):
        return False
    return True


def _coerce_object_columns_for_parquet(df: pd.DataFrame, exc: Exception) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    changed: list[str] = []

    explicit_col = _extract_problem_column(exc, out.columns)
    if explicit_col is not None:
        cols = [explicit_col]
    else:
        cols = [c for c in out.columns if out[c].dtype == object]

    for col in cols:
        series = out[col]
        if series.dtype != object:
            continue
        if explicit_col is None and not _needs_object_string_coercion(series):
            continue
        out[col] = series.map(_stringify_value).astype("string")
        changed.append(col)

    return out, changed


def write_parquet_with_fallback(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    """Write parquet with robust handling for common pyarrow object-dtype failures.

    Behavior:
    - Normal path: write parquet directly.
    - Missing parquet engine: write CSV fallback with warning.
    - Arrow object conversion errors: coerce problematic object columns to string and retry.
    """
    try:
        df.to_parquet(path, index=index)
        return
    except Exception as exc:
        if _is_missing_engine_error(exc):
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=index)
            warnings.warn(
                (
                    f"Parquet engine unavailable; wrote {csv_path.name} instead of {path.name}. "
                    "Install pyarrow or fastparquet to enable parquet output."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            return

        if not _is_arrow_object_error(exc):
            raise

        coerced, changed_cols = _coerce_object_columns_for_parquet(df, exc)
        if not changed_cols:
            raise

        coerced.to_parquet(path, index=index)
        warnings.warn(
            (
                "Coerced object columns to string for parquet compatibility: "
                + ", ".join(changed_cols)
            ),
            RuntimeWarning,
            stacklevel=2,
        )
