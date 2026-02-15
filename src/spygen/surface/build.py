from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from spygen.finance.conventions import discount_factor, log_moneyness, normalize_call
from spygen.finance.forwards import infer_forward_from_parity
from spygen.surface.grid import SurfaceGrid


@dataclass(slots=True)
class BuildSurfaceResult:
    date: pd.Timestamp
    surface_raw: np.ndarray
    forwards: dict[pd.Timestamp, float]
    discounts: dict[pd.Timestamp, float]
    expiries: list[pd.Timestamp]


def _calls_for_expiry(expiry_df: pd.DataFrame, forward: float, rate: float = 0.01) -> pd.DataFrame:
    dte = float(expiry_df["dte"].iloc[0])
    t = max(dte / 365.0, 1.0 / 365.0)
    disc = discount_factor(rate, t)

    calls = expiry_df.loc[expiry_df["call_put"] == "C", ["strike", "mid"]].rename(
        columns={"mid": "call_mid"}
    )
    puts = expiry_df.loc[expiry_df["call_put"] == "P", ["strike", "mid"]].rename(
        columns={"mid": "put_mid"}
    )
    merged = calls.merge(puts, on="strike", how="outer")

    merged["call_price"] = merged["call_mid"]
    missing_call = merged["call_price"].isna() & merged["put_mid"].notna()
    if missing_call.any():
        merged.loc[missing_call, "call_price"] = (
            merged.loc[missing_call, "put_mid"]
            + disc * (forward - merged.loc[missing_call, "strike"])
        )
    merged = merged.dropna(subset=["call_price"]).sort_values("strike")
    return merged[["strike", "call_price"]]


def build_surface_from_chain(
    chain: pd.DataFrame,
    grid: SurfaceGrid,
    rate: float = 0.01,
) -> BuildSurfaceResult:
    if chain.empty:
        raise ValueError("Empty chain passed to build_surface_from_chain")

    asof = pd.Timestamp(chain["date"].iloc[0])
    x_grid = grid.x
    tenors = np.array(grid.tenors_days, dtype=float)

    forwards: dict[pd.Timestamp, float] = {}
    discounts: dict[pd.Timestamp, float] = {}
    expiry_surfaces: dict[pd.Timestamp, np.ndarray] = {}
    expiry_dtes: dict[pd.Timestamp, int] = {}

    for expiry, g in chain.groupby("expiry"):
        g = g.copy()
        exp_ts = pd.Timestamp(expiry)
        spot = float(g["underlying_close"].iloc[0])
        forward = infer_forward_from_parity(g, rate=rate, spot=spot)
        calls = _calls_for_expiry(g, forward=forward, rate=rate)
        if calls.empty:
            continue

        dte = int(g["dte"].iloc[0])
        t = max(dte / 365.0, 1.0 / 365.0)
        disc = discount_factor(rate, t)

        x_obs = log_moneyness(calls["strike"].to_numpy(dtype=float), forward)
        y_obs = normalize_call(calls["call_price"].to_numpy(dtype=float), disc, forward)
        order = np.argsort(x_obs)
        x_obs = x_obs[order]
        y_obs = y_obs[order]
        interp = np.interp(x_grid, x_obs, y_obs, left=y_obs[0], right=y_obs[-1])

        forwards[exp_ts] = float(forward)
        discounts[exp_ts] = float(disc)
        expiry_surfaces[exp_ts] = interp
        expiry_dtes[exp_ts] = dte

    if not expiry_surfaces:
        raise ValueError("No valid expiries found while building surface")

    expiry_list = sorted(expiry_surfaces.keys())
    surf = np.zeros((grid.nx, len(tenors)), dtype=float)
    for j, tenor in enumerate(tenors):
        nearest = min(expiry_list, key=lambda e: abs(expiry_dtes[e] - tenor))
        surf[:, j] = expiry_surfaces[nearest]

    return BuildSurfaceResult(
        date=asof,
        surface_raw=surf,
        forwards=forwards,
        discounts=discounts,
        expiries=expiry_list,
    )
