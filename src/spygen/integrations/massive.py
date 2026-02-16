from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urljoin, urlparse, urlunparse

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MassiveConfig:
    base_url: str = os.getenv("MASSIVE_BASE_URL", "https://api.massive.com")
    api_key_env_var: str = "MASSIVE_API_KEY"
    cache_enabled: bool = False
    cache_dir: Path = Path("data/massive_cache")
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    max_retries: int = 4
    backoff_base: float = 0.5


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


def parse_stock_aggs(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in _as_list(payload.get("results")):
        ts = item.get("t")
        if ts is None:
            continue
        rows.append(
            {
                "date": pd.to_datetime(int(ts), unit="ms").date(),
                "open": float(item.get("o", 0.0)),
                "high": float(item.get("h", 0.0)),
                "low": float(item.get("l", 0.0)),
                "close": float(item.get("c", 0.0)),
                "volume": float(item.get("v", 0.0)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.DataFrame(rows).sort_values("date").set_index("date")


def parse_options_contracts(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in _as_list(payload.get("results")):
        exp = item.get("expiration_date")
        if exp is None:
            continue
        rows.append(
            {
                "ticker": item.get("ticker"),
                "underlying_ticker": item.get("underlying_ticker"),
                "expiration_date": pd.to_datetime(exp).date(),
                "strike_price": _to_float(item.get("strike_price")),
                "contract_type": item.get("contract_type"),
                "exercise_style": item.get("exercise_style"),
                "shares_per_contract": _to_float(item.get("shares_per_contract")),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "underlying_ticker",
                "expiration_date",
                "strike_price",
                "contract_type",
                "exercise_style",
                "shares_per_contract",
            ]
        )
    return pd.DataFrame(rows)


def parse_options_chain_snapshot(
    payload: dict[str, Any],
    *,
    asof: date,
    fallback_underlying_close: float | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in _as_list(payload.get("results")):
        details = item.get("details") or {}
        quote = item.get("last_quote") or item.get("quote") or {}
        trade = item.get("last_trade") or item.get("trade") or {}
        day = item.get("day") or {}
        greeks = item.get("greeks") or {}
        under = item.get("underlying_asset") or {}

        symbol = details.get("ticker") or item.get("ticker")
        exp = details.get("expiration_date")
        strike = _to_float(details.get("strike_price"))
        ctype = (details.get("contract_type") or "").lower()
        if not symbol or exp is None or strike is None:
            continue
        expiry = pd.to_datetime(exp).date()
        dte = int((expiry - asof).days)
        if dte <= 0:
            continue

        bid = _to_float(quote.get("bid") or quote.get("bp"))
        ask = _to_float(quote.get("ask") or quote.get("ap"))
        last = _to_float(trade.get("price") or trade.get("p") or day.get("close") or day.get("c"))
        if bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid:
            mid = 0.5 * (bid + ask)
        else:
            mid = last
        if mid is None:
            continue

        underlying_close = _to_float(under.get("price"))
        if underlying_close is None:
            underlying_close = fallback_underlying_close

        rows.append(
            {
                "date": asof,
                "expiry": expiry,
                "dte": dte,
                "call_put": "C" if ctype.startswith("c") else "P",
                "symbol": symbol,
                "strike": float(strike),
                "bid": float(bid) if bid is not None else float("nan"),
                "ask": float(ask) if ask is not None else float("nan"),
                "mid": float(mid),
                "last": float(last) if last is not None else float("nan"),
                "volume": _to_int(day.get("volume") or day.get("v")) or 0,
                "open_interest": _to_int(item.get("open_interest")) or 0,
                "underlying_close": float(underlying_close)
                if underlying_close is not None
                else float("nan"),
                "delta": _to_float(greeks.get("delta")),
                "gamma": _to_float(greeks.get("gamma")),
                "theta": _to_float(greeks.get("theta")),
                "vega": _to_float(greeks.get("vega")),
                "iv": _to_float(item.get("implied_volatility") or item.get("iv")),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "expiry",
                "dte",
                "call_put",
                "symbol",
                "strike",
                "bid",
                "ask",
                "mid",
                "last",
                "volume",
                "open_interest",
                "underlying_close",
                "delta",
                "gamma",
                "theta",
                "vega",
                "iv",
            ]
        )
    return pd.DataFrame(rows)


class MassiveClient:
    def __init__(self, config: MassiveConfig | None = None) -> None:
        self.config = config or MassiveConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _api_key(self) -> str:
        api_key = os.getenv(self.config.api_key_env_var)
        if not api_key:
            raise RuntimeError(
                f"Missing Massive API key. Set environment variable {self.config.api_key_env_var}."
            )
        return api_key

    @staticmethod
    def _cache_key(method: str, url: str, params: dict[str, Any] | None) -> str:
        normalized = "&".join(
            f"{k}={v}" for k, v in sorted((params or {}).items(), key=lambda kv: kv[0])
        )
        payload = f"{method.upper()}|{url}|{normalized}".encode()
        return hashlib.sha256(payload).hexdigest()

    def _cache_path(self, endpoint: str, key: str) -> Path:
        endpoint_name = endpoint.strip("/").replace("/", "_") or "root"
        return self.config.cache_dir / endpoint_name / f"{key}.json"

    def _merge_api_key(
        self,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        merged = dict(params or {})
        if "apiKey" not in merged:
            merged["apiKey"] = self._api_key()
        parsed = urlparse(url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        if "apiKey" not in query:
            query["apiKey"] = self._api_key()
        clean_url = urlunparse(
            (parsed.scheme, parsed.netloc, parsed.path, parsed.params, "", parsed.fragment)
        )
        merged.update({k: v for k, v in query.items() if k not in merged})
        return clean_url, merged

    def _request_url(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        clean_url, merged_params = self._merge_api_key(url, params=params)
        parsed_path = urlparse(clean_url).path
        cache_key = self._cache_key("GET", clean_url, merged_params)
        cache_path = self._cache_path(parsed_path, cache_key)

        if self.config.cache_enabled and cache_path.exists():
            return json.loads(cache_path.read_text())

        timeout = httpx.Timeout(
            timeout=self.config.read_timeout,
            connect=self.config.connect_timeout,
        )
        for attempt in range(self.config.max_retries + 1):
            try:
                response = httpx.get(clean_url, params=merged_params, timeout=timeout)
            except httpx.HTTPError as exc:
                if attempt >= self.config.max_retries:
                    raise RuntimeError(f"Massive request failed: {exc}") from exc
                time.sleep(self.config.backoff_base * (2**attempt))
                continue

            remaining = response.headers.get("X-RateLimit-Remaining")
            reset = response.headers.get("X-RateLimit-Reset")
            if remaining is not None or reset is not None:
                logger.info("Massive rate-limit remaining=%s reset=%s", remaining, reset)

            if response.status_code in {429, 500, 502, 503, 504}:
                if attempt >= self.config.max_retries:
                    response.raise_for_status()
                sleep_s = self.config.backoff_base * (2**attempt)
                if response.status_code == 429 and reset:
                    try:
                        sleep_s = max(sleep_s, float(reset) - time.time())
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

        raise RuntimeError("Unreachable Massive request failure path")

    def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        base = self.config.base_url.rstrip("/")
        return self._request_url(urljoin(f"{base}/", endpoint.lstrip("/")), params=params)

    def _paginate(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        payload = self._request(endpoint, params=params)
        results = _as_list(payload.get("results"))
        next_url = payload.get("next_url")
        while next_url:
            payload = self._request_url(next_url)
            results.extend(_as_list(payload.get("results")))
            next_url = payload.get("next_url")
        return results

    def get_stock_aggs(
        self,
        symbol: str,
        start: date,
        end: date,
        multiplier: int = 1,
        timespan: str = "day",
        adjusted: bool = True,
        sort: str = "asc",
        limit: int = 50_000,
    ) -> pd.DataFrame:
        endpoint = (
            f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/"
            f"{start.isoformat()}/{end.isoformat()}"
        )
        payload = self._request(
            endpoint,
            params={
                "adjusted": str(adjusted).lower(),
                "sort": sort,
                "limit": int(limit),
            },
        )
        return parse_stock_aggs(payload)

    def list_options_contracts(
        self,
        *,
        underlying_ticker: str,
        as_of: date | None = None,
        expiration_gte: date | None = None,
        expiration_lte: date | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        params: dict[str, Any] = {"underlying_ticker": underlying_ticker, "limit": int(limit)}
        if as_of is not None:
            params["as_of"] = as_of.isoformat()
        if expiration_gte is not None:
            params["expiration_date.gte"] = expiration_gte.isoformat()
        if expiration_lte is not None:
            params["expiration_date.lte"] = expiration_lte.isoformat()
        rows = self._paginate("/v3/reference/options/contracts", params=params)
        return parse_options_contracts({"results": rows})

    def get_options_chain_snapshot(
        self,
        *,
        underlying_ticker: str,
        as_of: date,
        fallback_underlying_close: float | None = None,
        expiration_gte: date | None = None,
        expiration_lte: date | None = None,
        contract_type: str | None = None,
        strike_gte: float | None = None,
        strike_lte: float | None = None,
        limit: int = 250,
    ) -> pd.DataFrame:
        params: dict[str, Any] = {"limit": int(limit)}
        if expiration_gte is not None:
            params["expiration_date.gte"] = expiration_gte.isoformat()
        if expiration_lte is not None:
            params["expiration_date.lte"] = expiration_lte.isoformat()
        if contract_type is not None:
            params["contract_type"] = contract_type
        if strike_gte is not None:
            params["strike_price.gte"] = float(strike_gte)
        if strike_lte is not None:
            params["strike_price.lte"] = float(strike_lte)
        rows = self._paginate(f"/v3/snapshot/options/{underlying_ticker}", params=params)
        return parse_options_chain_snapshot(
            {"results": rows},
            asof=as_of,
            fallback_underlying_close=fallback_underlying_close,
        )

    def get_option_open_close(
        self,
        *,
        options_ticker: str,
        as_of: date,
        adjusted: bool = True,
    ) -> dict[str, Any] | None:
        endpoint = f"/v1/open-close/{options_ticker}/{as_of.isoformat()}"
        try:
            return self._request(
                endpoint,
                params={"adjusted": str(adjusted).lower()},
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
