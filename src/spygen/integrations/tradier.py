from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TradierConfig:
    base_url: str = os.getenv("TRADIER_BASE_URL", "https://api.tradier.com")
    token_env_var: str = "TRADIER_TOKEN"
    cache_enabled: bool = False
    cache_dir: Path = Path("data/tradier_cache")
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    max_retries: int = 4
    backoff_base: float = 0.5


class TradierClient:
    def __init__(self, config: TradierConfig | None = None) -> None:
        self.config = config or TradierConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _token(self) -> str:
        token = os.getenv(self.config.token_env_var)
        if not token:
            raise RuntimeError(
                f"Missing Tradier token. Set environment variable {self.config.token_env_var}."
            )
        return token

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._token()}",
        }

    @staticmethod
    def _cache_key(method: str, url: str, params: dict[str, Any] | None) -> str:
        normalized = "&".join(
            f"{k}={v}" for k, v in sorted((params or {}).items(), key=lambda kv: kv[0])
        )
        payload = f"{method.upper()}|{url}|{normalized}".encode()
        return hashlib.sha256(payload).hexdigest()

    def _cache_path(self, endpoint: str, key: str) -> Path:
        endpoint_name = endpoint.strip("/").replace("/", "_")
        return self.config.cache_dir / endpoint_name / f"{key}.json"

    def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}{endpoint}"
        cache_key = self._cache_key("GET", url, params)
        cache_path = self._cache_path(endpoint, cache_key)

        if self.config.cache_enabled and cache_path.exists():
            return json.loads(cache_path.read_text())

        timeout = httpx.Timeout(
            timeout=self.config.read_timeout,
            connect=self.config.connect_timeout,
        )
        for attempt in range(self.config.max_retries + 1):
            try:
                response = httpx.get(url, params=params, headers=self._headers(), timeout=timeout)
            except httpx.HTTPError as exc:
                if attempt >= self.config.max_retries:
                    raise RuntimeError(f"Tradier request failed: {exc}") from exc
                sleep_s = self.config.backoff_base * (2**attempt)
                time.sleep(sleep_s)
                continue

            remaining = response.headers.get("X-Ratelimit-Available") or response.headers.get(
                "X-Ratelimit-Remaining"
            )
            expiry = response.headers.get("X-Ratelimit-Expiry")
            if remaining is not None or expiry is not None:
                logger.info("Tradier rate-limit remaining=%s expiry=%s", remaining, expiry)

            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= self.config.max_retries:
                    response.raise_for_status()
                sleep_s = self.config.backoff_base * (2**attempt)
                if response.status_code == 429 and expiry:
                    try:
                        reset_ts = int(expiry)
                        now = int(time.time())
                        sleep_s = max(sleep_s, float(reset_ts - now))
                    except ValueError:
                        pass
                time.sleep(max(0.0, sleep_s))
                continue

            response.raise_for_status()
            payload = response.json()
            if self.config.cache_enabled:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(payload, indent=2))
            return payload

        raise RuntimeError("Unreachable request failure path")

    def get_market_history(
        self, symbol: str, start: date, end: date, interval: str = "daily"
    ) -> pd.DataFrame:
        payload = self._request(
            "/v1/markets/history",
            params={
                "symbol": symbol,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "interval": interval,
            },
        )
        return parse_market_history(payload)

    def get_quotes(self, symbols: list[str], greeks: bool = False) -> dict[str, dict[str, Any]]:
        payload = self._request(
            "/v1/markets/quotes",
            params={"symbols": ",".join(symbols), "greeks": str(greeks).lower()},
        )
        return parse_quotes(payload)

    def get_option_expirations(
        self,
        symbol: str = "SPY",
        include_all_roots: bool = False,
        strikes: bool = False,
        contract_size: bool = False,
        expiration_type: bool = False,
    ) -> list[date]:
        payload = self._request(
            "/v1/markets/options/expirations",
            params={
                "symbol": symbol,
                "includeAllRoots": str(include_all_roots).lower(),
                "strikes": str(strikes).lower(),
                "contractSize": str(contract_size).lower(),
                "expirationType": str(expiration_type).lower(),
            },
        )
        return parse_expirations(payload)

    def lookup_option_symbols(
        self,
        underlying: str = "SPY",
        strike: float | None = None,
        expiration: date | None = None,
        opt_type: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"underlying": underlying}
        if strike is not None:
            params["strike"] = strike
        if expiration is not None:
            params["expiration"] = expiration.isoformat()
        if opt_type is not None:
            params["type"] = opt_type
        payload = self._request("/v1/markets/options/lookup", params=params)
        return parse_lookup(payload)

    def get_option_chain(
        self,
        symbol: str = "SPY",
        expiration: date | None = None,
        greeks: bool = False,
    ) -> pd.DataFrame:
        if expiration is None:
            raise ValueError("expiration is required for option chain")
        payload = self._request(
            "/v1/markets/options/chains",
            params={
                "symbol": symbol,
                "expiration": expiration.isoformat(),
                "greeks": str(greeks).lower(),
            },
        )
        return parse_option_chain(payload)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _to_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    return float(value)


def _to_int(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    return int(value)


def parse_market_history(payload: dict[str, Any]) -> pd.DataFrame:
    days = _as_list(payload.get("history", {}).get("day"))
    if not days:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    rows = []
    for day in days:
        rows.append(
            {
                "date": pd.to_datetime(day["date"]).date(),
                "open": float(day["open"]),
                "high": float(day["high"]),
                "low": float(day["low"]),
                "close": float(day["close"]),
                "volume": int(day.get("volume", 0)),
            }
        )
    df = pd.DataFrame(rows).sort_values("date").set_index("date")
    return df


def parse_quotes(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    quotes = _as_list(payload.get("quotes", {}).get("quote"))
    out: dict[str, dict[str, Any]] = {}
    for q in quotes:
        symbol = q.get("symbol")
        if symbol:
            out[symbol] = q
    return out


def parse_expirations(payload: dict[str, Any]) -> list[date]:
    expiries_raw = payload.get("expirations", {}).get("date", [])
    expiries = _as_list(expiries_raw)
    parsed: list[date] = []
    for exp in expiries:
        if isinstance(exp, dict):
            exp = exp.get("date")
        parsed.append(datetime.strptime(str(exp), "%Y-%m-%d").date())
    return parsed


def parse_lookup(payload: dict[str, Any]) -> list[dict[str, Any]]:
    options = _as_list(payload.get("options", {}).get("option"))
    rows: list[dict[str, Any]] = []
    for opt in options:
        rows.append(
            {
                "symbol": opt.get("symbol"),
                "rootsymbol": opt.get("rootsymbol"),
                "strike": _to_float(opt.get("strike")),
                "expiration": datetime.strptime(str(opt.get("expiration")), "%Y-%m-%d").date(),
                "type": opt.get("type"),
                "description": opt.get("description", ""),
            }
        )
    return rows


def parse_option_chain(payload: dict[str, Any]) -> pd.DataFrame:
    options = _as_list(payload.get("options", {}).get("option"))
    rows: list[dict[str, Any]] = []
    for opt in options:
        greeks = opt.get("greeks") or {}
        rows.append(
            {
                "symbol": opt.get("symbol"),
                "option_type": opt.get("option_type") or opt.get("type"),
                "strike": _to_float(opt.get("strike")),
                "expiration_date": pd.to_datetime(opt.get("expiration_date")).date(),
                "bid": _to_float(opt.get("bid")),
                "ask": _to_float(opt.get("ask")),
                "last": _to_float(opt.get("last")),
                "volume": _to_int(opt.get("volume")) or 0,
                "open_interest": _to_int(opt.get("open_interest")) or 0,
                "delta": _to_float(greeks.get("delta")),
                "gamma": _to_float(greeks.get("gamma")),
                "theta": _to_float(greeks.get("theta")),
                "vega": _to_float(greeks.get("vega")),
                "iv": _to_float(greeks.get("mid_iv") or greeks.get("iv")),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "option_type",
                "strike",
                "expiration_date",
                "bid",
                "ask",
                "last",
                "volume",
                "open_interest",
                "delta",
                "gamma",
                "theta",
                "vega",
                "iv",
            ]
        )
    return pd.DataFrame(rows)
