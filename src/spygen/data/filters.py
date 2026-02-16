from __future__ import annotations

import pandas as pd


def apply_liquidity_filters(
    chain: pd.DataFrame,
    min_volume: int = 1,
    min_open_interest: int = 1,
    max_relative_spread: float = 0.5,
) -> pd.DataFrame:
    df = chain.copy()
    spread = df["ask"] - df["bid"]
    mid = df["mid"].replace(0.0, 1e-8)
    rel_spread = spread / mid
    keep = (
        (df["volume"] >= min_volume)
        & (df["open_interest"] >= min_open_interest)
        & (rel_spread <= max_relative_spread)
        & (df["ask"] > df["bid"])
    )
    return df.loc[keep].reset_index(drop=True)
