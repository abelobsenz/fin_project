"""Massive-compatible data plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
import gzip
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import requests

from ivdyn.data.schemas import normalize_chain_df

OPRA_TICKER_RE = re.compile(
    r"^O:(?P<underlying>[A-Z]+)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<cp>[CP])(?P<strike>\d{8})$"
)


def parse_opra_ticker(ticker: str) -> dict[str, Any] | None:
    m = OPRA_TICKER_RE.match(str(ticker))
    if not m:
        return None
    yy = int(m.group("yy"))
    year = 2000 + yy
    month = int(m.group("mm"))
    day = int(m.group("dd"))
    strike = int(m.group("strike")) / 1000.0
    return {
        "underlying": m.group("underlying"),
        "expiry": date(year, month, day),
        "call_put": m.group("cp"),
        "strike": strike,
    }


class OptionsDataPlugin(ABC):
    """Plugin contract for options data backends."""

    name: str

    @abstractmethod
    def list_dates(self, symbol: str) -> list[date]:
        raise NotImplementedError

    @abstractmethod
    def load_day(self, symbol: str, asof: date) -> pd.DataFrame:
        raise NotImplementedError


@dataclass(slots=True)
class MassiveRawParquetPlugin(OptionsDataPlugin):
    """Reads canonical Massive-derived snapshots from data/symbols/<SYMBOL>/options/raw/*.parquet."""

    raw_dir: Path
    name: str = "massive_raw_parquet"

    def list_dates(self, symbol: str) -> list[date]:
        dates: list[date] = []
        for p in sorted(self.raw_dir.glob("*.parquet")):
            try:
                dates.append(date.fromisoformat(p.stem))
            except ValueError:
                continue
        return dates

    def load_day(self, symbol: str, asof: date) -> pd.DataFrame:
        p = self.raw_dir / f"{asof.isoformat()}.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_parquet(p)
        return normalize_chain_df(df, asof=asof, symbol=symbol)


@dataclass(slots=True)
class MassiveFlatfileAggsPlugin(OptionsDataPlugin):
    """Reads local Massive OPRA day aggs csv.gz caches and converts to canonical schema."""

    flatfile_root: Path
    underlying_path: Path | None = None
    name: str = "massive_flatfile_aggs"

    def _underlying_lookup(self) -> dict[date, float]:
        if self.underlying_path is None or not self.underlying_path.exists():
            return {}
        udf = pd.read_parquet(self.underlying_path)
        udf["date"] = pd.to_datetime(udf["date"]).dt.date
        return dict(zip(udf["date"], udf["close"], strict=False))

    def _path_for_date(self, asof: date) -> Path:
        return (
            self.flatfile_root
            / f"{asof.year:04d}"
            / f"{asof.month:02d}"
            / f"{asof.isoformat()}.csv.gz"
        )

    def list_dates(self, symbol: str) -> list[date]:
        dates: list[date] = []
        for p in sorted(self.flatfile_root.glob("*/*/*.csv.gz")):
            try:
                dates.append(date.fromisoformat(p.stem.replace(".csv", "")))
            except ValueError:
                continue
        return dates

    def load_day(self, symbol: str, asof: date) -> pd.DataFrame:
        p = self._path_for_date(asof)
        if not p.exists():
            raise FileNotFoundError(p)

        with gzip.open(p, "rt") as fh:
            raw = pd.read_csv(fh)

        parsed = raw["ticker"].apply(parse_opra_ticker)
        mask = parsed.notna()
        raw = raw.loc[mask].copy()
        parsed = parsed.loc[mask]

        raw["underlying"] = parsed.apply(lambda x: x["underlying"])  # type: ignore[index]
        raw = raw[raw["underlying"] == symbol.upper()]
        parsed = parsed.loc[raw.index]

        if raw.empty:
            return normalize_chain_df(pd.DataFrame(columns=["date"]), asof=asof, symbol=symbol)

        expiry = parsed.apply(lambda x: x["expiry"])
        dte = parsed.apply(lambda x: (x["expiry"] - asof).days)
        call_put = parsed.apply(lambda x: x["call_put"])
        strike = parsed.apply(lambda x: x["strike"])
        mid = pd.to_numeric(raw.get("close"), errors="coerce")
        volume = pd.to_numeric(raw.get("volume"), errors="coerce")

        under_map = self._underlying_lookup()
        spot = float(under_map.get(asof, np.nan))

        out = pd.DataFrame(
            {
                "date": asof,
                "expiry": expiry,  # type: ignore[index]
                "dte": dte,  # type: ignore[index]
                "call_put": call_put,  # type: ignore[index]
                "symbol": raw["ticker"].astype(str),
                "strike": strike,  # type: ignore[index]
                "bid": np.nan,
                "ask": np.nan,
                "mid": mid,
                "last": mid,
                "volume": volume,
                "open_interest": np.nan,
                "underlying_close": spot,
                "delta": np.nan,
                "gamma": np.nan,
                "theta": np.nan,
                "vega": np.nan,
                "iv": np.nan,
            }
        )
        return normalize_chain_df(out, asof=asof, symbol=symbol)


@dataclass(slots=True)
class MassiveRESTPlugin(OptionsDataPlugin):
    """REST plugin for Massive (best-effort historical compatibility)."""

    api_key: str
    base_url: str = "https://api.massive.com"
    timeout_seconds: int = 30
    name: str = "massive_rest"

    def _get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        params = {**params, "apiKey": self.api_key}
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        resp = requests.get(url, params=params, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def list_dates(self, symbol: str) -> list[date]:
        # Massive REST APIs do not provide a direct historical options-date index in a
        # single endpoint. This method intentionally returns an empty list and relies on
        # caller-provided date ranges.
        return []

    def load_day(self, symbol: str, asof: date) -> pd.DataFrame:
        # Best-effort snapshot route. Historical availability depends on account/endpoint.
        payload = self._get_json(
            f"v3/snapshot/options/{symbol.upper()}",
            {"limit": 5000, "as_of": asof.isoformat()},
        )
        rows = payload.get("results", [])
        chain_rows: list[dict[str, Any]] = []
        for row in rows:
            details = row.get("details", {})
            quote = row.get("last_quote", {})
            trade = row.get("last_trade", {})
            greeks = row.get("greeks", {})
            iv = row.get("implied_volatility")
            chain_rows.append(
                {
                    "date": asof,
                    "expiry": details.get("expiration_date"),
                    "dte": None,
                    "call_put": details.get("contract_type", "").upper()[:1],
                    "symbol": details.get("ticker"),
                    "strike": details.get("strike_price"),
                    "bid": quote.get("bid_price"),
                    "ask": quote.get("ask_price"),
                    "mid": None,
                    "last": trade.get("price"),
                    "volume": row.get("day", {}).get("volume"),
                    "open_interest": row.get("open_interest"),
                    "underlying_close": row.get("underlying_asset", {}).get("price"),
                    "delta": greeks.get("delta"),
                    "gamma": greeks.get("gamma"),
                    "theta": greeks.get("theta"),
                    "vega": greeks.get("vega"),
                    "iv": iv,
                }
            )
        df = pd.DataFrame(chain_rows)
        return normalize_chain_df(df, asof=asof, symbol=symbol)


@dataclass(slots=True)
class PluginFactory:
    data_root: Path

    def build(
        self,
        plugin_name: str,
        symbol: str = "SPY",
        api_key: str | None = None,
    ) -> OptionsDataPlugin:
        plugin_name = plugin_name.lower().strip()
        sym_u = symbol.upper()
        sym_l = symbol.lower()
        sym_root = self.data_root / "symbols" / sym_u
        if plugin_name == "massive_raw_parquet":
            return MassiveRawParquetPlugin(raw_dir=sym_root / "options" / "raw")
        if plugin_name == "massive_flatfile_aggs":
            return MassiveFlatfileAggsPlugin(
                flatfile_root=self.data_root / "options_source" / "us_options_opra" / "day_aggs_v1",
                underlying_path=sym_root / "underlying" / f"{sym_l}_eod.parquet",
            )
        if plugin_name == "massive_rest":
            if not api_key:
                raise ValueError("Massive REST plugin requires api_key.")
            return MassiveRESTPlugin(api_key=api_key)
        raise ValueError(f"Unknown plugin '{plugin_name}'.")


def read_metadata(raw_dir: Path, asof: date) -> dict[str, Any] | None:
    p = raw_dir / f"{asof.isoformat()}.metadata.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)
