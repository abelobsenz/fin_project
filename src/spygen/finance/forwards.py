from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def infer_forward_from_parity(
    chain: pd.DataFrame, rate: float = 0.01, spot: float | None = None
) -> float:
    cols = {"strike", "mid", "call_put"}
    missing = cols.difference(chain.columns)
    if missing:
        raise ValueError(f"Chain missing required columns: {missing}")

    calls = chain.loc[chain["call_put"] == "C", ["strike", "mid", "dte"]].copy()
    puts = chain.loc[chain["call_put"] == "P", ["strike", "mid", "dte"]].copy()
    merged = calls.merge(puts, on=["strike", "dte"], suffixes=("_c", "_p"))
    if merged.empty:
        raise ValueError("Need both calls and puts at matching strikes to infer forward")

    if spot is not None:
        merged = merged.assign(dist=(merged["strike"] - spot).abs()).sort_values("dist").head(8)

    dte = float(merged["dte"].median())
    t = max(dte / 365.0, 1.0 / 365.0)
    disc = np.exp(-rate * t)
    forwards = (merged["mid_c"] - merged["mid_p"]) / disc + merged["strike"]
    return float(np.median(forwards))


def infer_forwards_by_expiry(chain: pd.DataFrame, rate: float = 0.01) -> dict[pd.Timestamp, float]:
    out: dict[pd.Timestamp, float] = {}
    for expiry, g in chain.groupby("expiry"):
        spot = float(g["underlying_close"].iloc[0])
        out[pd.Timestamp(expiry)] = infer_forward_from_parity(g, rate=rate, spot=spot)
    return out


def robust_median(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=float)
    return float(np.median(arr))
