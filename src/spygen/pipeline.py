from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from spygen.data.filters import apply_liquidity_filters
from spygen.data.io import append_dedup_underlying, list_raw_chain_files, write_metadata
from spygen.data.synth import SynthConfig, write_synthetic_dataset
from spygen.integrations.massive import MassiveClient, MassiveConfig
from spygen.integrations.massive_flatfiles import (
    MassiveFlatFileClient,
    MassiveFlatFilesConfig,
    parse_option_symbol_osi,
)
from spygen.integrations.tradier import TradierClient, TradierConfig
from spygen.models.context import build_context_features
from spygen.models.sampling import (
    conditional_mean_surface,
    load_checkpoint,
    log_likelihood,
    sample_surfaces,
)
from spygen.models.training import TrainConfig, train_flow_model
from spygen.strategy.backtester import BacktestConfig, run_backtest
from spygen.strategy.providers import available_signal_providers
from spygen.surface.arb_checks import arb_violation_counts, is_arb_free
from spygen.surface.build import build_surface_from_chain
from spygen.surface.grid import SurfaceGrid
from spygen.surface.repair_qp import repair_surface_qp
from spygen.surface.representation import softplus_inverse, surface_to_theta
from spygen.utils.dates import parse_date
from spygen.utils.paths import ensure_dir

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    validate_config(cfg)
    return cfg


def save_config(path: str | Path, config: dict[str, Any]) -> None:
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False))


def validate_config(config: dict[str, Any]) -> None:
    required_top = {"paths", "surface", "train", "strategy"}
    missing_top = required_top.difference(config.keys())
    if missing_top:
        raise ValueError(f"Config missing required top-level keys: {sorted(missing_top)}")

    required_paths = {"raw_dir", "processed_dir", "underlying_path", "outputs_dir"}
    missing_paths = required_paths.difference(config["paths"].keys())
    if missing_paths:
        raise ValueError(f"Config missing required path keys: {sorted(missing_paths)}")

    surface = config["surface"]
    for key in ("x_min", "x_max", "nx", "tenors_days"):
        if key not in surface:
            raise ValueError(f"Config surface missing key: {key}")
    if float(surface["x_min"]) >= float(surface["x_max"]):
        raise ValueError("surface.x_min must be < surface.x_max")
    if int(surface["nx"]) < 5:
        raise ValueError("surface.nx must be >= 5")
    if not surface["tenors_days"]:
        raise ValueError("surface.tenors_days must not be empty")

    strategy = config["strategy"]
    max_spread_abs = float(strategy.get("max_spread_abs", strategy.get("max_spread", 1.0)))
    if max_spread_abs <= 0:
        raise ValueError("strategy.max_spread_abs must be > 0")
    if float(strategy.get("max_spread_rel", 0.0)) <= 0:
        raise ValueError("strategy.max_spread_rel must be > 0")
    if float(strategy.get("max_notional", 0.0)) <= 0:
        raise ValueError("strategy.max_notional must be > 0")
    if int(strategy.get("max_contracts", 0)) <= 0:
        raise ValueError("strategy.max_contracts must be > 0")


def _merge_provider_strategy(base: dict[str, Any], provider_name: str) -> dict[str, Any]:
    merged = dict(base)
    overrides = (
        base.get("provider_overrides", {}).get(provider_name, {})
        if isinstance(base.get("provider_overrides"), dict)
        else {}
    )
    if not isinstance(overrides, dict):
        return merged
    for key, value in overrides.items():
        merged[key] = value
    return merged


def _make_backtest_config(
    bt_cfg: dict[str, Any],
    provider_name: str,
    *,
    fallback_seed: int,
) -> BacktestConfig:
    return BacktestConfig(
        threshold=float(bt_cfg.get("threshold", 4.0)),
        zscore_quantile=float(bt_cfg.get("zscore_quantile", 0.9)),
        min_history_for_quantile=int(bt_cfg.get("min_history_for_quantile", 20)),
        min_signal_abs=float(bt_cfg.get("min_signal_abs", 0.001)),
        edge_cost_multiplier=float(bt_cfg.get("edge_cost_multiplier", 1.25)),
        residual_clip=float(bt_cfg.get("residual_clip", 1.0)),
        reject_if_residual_clipped=bool(bt_cfg.get("reject_if_residual_clipped", False)),
        n_samples=int(bt_cfg.get("n_samples", 32)),
        max_trades_per_day=int(bt_cfg.get("max_trades_per_day", 3)),
        max_contracts=int(bt_cfg.get("max_contracts", 10)),
        max_notional=float(bt_cfg.get("max_notional", 50_000.0)),
        direction_mode=str(bt_cfg.get("direction_mode", "mean_revert")),
        signal_provider=str(provider_name),
        seed=int(bt_cfg.get("seed", fallback_seed)),
        execution_mode=str(bt_cfg.get("execution_mode", "worse_than_touch")),
        execution_impact_bps=float(bt_cfg.get("execution_impact_bps", 0.0)),
        execution_fee_per_contract=float(bt_cfg.get("execution_fee_per_contract", 0.0)),
        execution_worse_touch_extra_half_spread=float(
            bt_cfg.get("execution_worse_touch_extra_half_spread", 0.5)
        ),
        spread_gate_mode=str(bt_cfg.get("spread_gate_mode", "abs_or_rel")),
        max_spread_abs=float(bt_cfg.get("max_spread_abs", bt_cfg.get("max_spread", 1.5))),
        max_spread_rel=float(bt_cfg.get("max_spread_rel", 0.35)),
        edge_signal_to_usd_scale=float(bt_cfg.get("edge_signal_to_usd_scale", 1.0)),
        unit_sanity_check=bool(bt_cfg.get("unit_sanity_check", True)),
        unit_sanity_fail_fast=bool(bt_cfg.get("unit_sanity_fail_fast", False)),
    )


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_call_price(x: float, t: float, sigma: float) -> float:
    if t <= 0.0:
        return max(0.0, 1.0 - math.exp(x))
    vol_sqrt_t = max(sigma, 1e-8) * math.sqrt(t)
    d1 = (-x + 0.5 * vol_sqrt_t * vol_sqrt_t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return _normal_cdf(d1) - math.exp(x) * _normal_cdf(d2)


def _iv_from_norm_call(c_norm: float, x: float, t: float) -> float:
    intrinsic = max(0.0, 1.0 - math.exp(x))
    target = min(max(float(c_norm), intrinsic + 1e-8), 1.0 - 1e-8)
    lo, hi = 1e-4, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        price = _norm_call_price(x=x, t=t, sigma=mid)
        if abs(price - target) < 1e-6:
            return mid
        if price > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _surface_to_iv(surface: np.ndarray, x_grid: np.ndarray, tenors_days: np.ndarray) -> np.ndarray:
    iv = np.zeros_like(surface, dtype=float)
    for i, x in enumerate(x_grid):
        for j, tenor in enumerate(tenors_days):
            t = max(float(tenor) / 365.0, 1.0 / 365.0)
            iv[i, j] = _iv_from_norm_call(float(surface[i, j]), float(x), t)
    return iv


def _massive_cutoff_date(config: dict[str, Any], today: date | None = None) -> date:
    massive_cfg = config.get("massive", {})
    max_history_days = int(massive_cfg.get("max_history_days", 730))
    base = today or datetime.now(UTC).date()
    return base - timedelta(days=max(1, max_history_days))


def _clip_massive_date_range(
    start_d: date,
    end_d: date,
    config: dict[str, Any],
    today: date | None = None,
) -> tuple[date, date, date]:
    cutoff = _massive_cutoff_date(config=config, today=today)
    if end_d < cutoff:
        raise ValueError(
            f"Massive entitlement window exceeded. End date {end_d.isoformat()} is older than "
            f"cutoff {cutoff.isoformat()} (max_history_days)."
        )
    adjusted_start = max(start_d, cutoff)
    if adjusted_start != start_d:
        logger.warning(
            "Massive start date clipped from %s to entitlement cutoff %s",
            start_d.isoformat(),
            cutoff.isoformat(),
        )
    return adjusted_start, end_d, cutoff


def _validate_massive_asof(
    asof_d: date,
    config: dict[str, Any],
    today: date | None = None,
) -> date:
    cutoff = _massive_cutoff_date(config=config, today=today)
    if asof_d < cutoff:
        raise ValueError(
            f"Massive entitlement window exceeded. asof={asof_d.isoformat()} is older than "
            f"cutoff {cutoff.isoformat()} (max_history_days)."
        )
    return cutoff


def fetch_underlying_range(
    start: str,
    end: str,
    symbol: str,
    config: dict[str, Any],
) -> Path:
    start_d = parse_date(start)
    end_d = parse_date(end)
    tradier_cfg = config.get("tradier", {})
    client = TradierClient(
        TradierConfig(
            base_url=tradier_cfg.get("base_url", "https://api.tradier.com"),
            token_env_var=tradier_cfg.get("token_env_var", "TRADIER_TOKEN"),
            cache_enabled=bool(tradier_cfg.get("cache_enabled", False)),
            cache_dir=Path(tradier_cfg.get("cache_dir", "data/tradier_cache")),
            connect_timeout=float(tradier_cfg.get("connect_timeout", 10.0)),
            read_timeout=float(tradier_cfg.get("read_timeout", 30.0)),
            max_retries=int(tradier_cfg.get("max_retries", 4)),
            backoff_base=float(tradier_cfg.get("backoff_base", 0.5)),
        )
    )
    history = client.get_market_history(symbol=symbol, start=start_d, end=end_d)
    if history.empty:
        raise ValueError("Tradier returned empty market history")

    out = history.reset_index().rename(columns={"index": "date"})
    path = Path(config["paths"]["underlying_path"])
    append_dedup_underlying(out, path)

    meta = {
        "source": "Tradier /v1/markets/history",
        "symbol": symbol,
        "start": start,
        "end": end,
        "fetched_at": datetime.now(UTC).isoformat(),
        "rows": int(len(out)),
    }
    write_metadata(path.with_suffix(".metadata.json"), meta)
    return path


def fetch_underlying_range_massive(
    start: str,
    end: str,
    symbol: str,
    config: dict[str, Any],
) -> Path:
    start_d = parse_date(start)
    end_d = parse_date(end)
    start_d, end_d, cutoff = _clip_massive_date_range(start_d, end_d, config=config)
    massive_cfg = config.get("massive", {})
    client = MassiveClient(
        MassiveConfig(
            base_url=massive_cfg.get("base_url", "https://api.massive.com"),
            api_key_env_var=massive_cfg.get("api_key_env_var", "MASSIVE_API_KEY"),
            cache_enabled=bool(massive_cfg.get("cache_enabled", False)),
            cache_dir=Path(massive_cfg.get("cache_dir", "data/massive_cache")),
            connect_timeout=float(massive_cfg.get("connect_timeout", 10.0)),
            read_timeout=float(massive_cfg.get("read_timeout", 30.0)),
            max_retries=int(massive_cfg.get("max_retries", 4)),
            backoff_base=float(massive_cfg.get("backoff_base", 0.5)),
        )
    )
    history = client.get_stock_aggs(symbol=symbol, start=start_d, end=end_d)
    if history.empty:
        raise ValueError("Massive returned empty market history")

    out = history.reset_index().rename(columns={"index": "date"})
    path = Path(config["paths"]["underlying_path"])
    append_dedup_underlying(out, path)

    meta = {
        "source": "Massive /v2/aggs",
        "symbol": symbol,
        "start": start_d.isoformat(),
        "end": end_d.isoformat(),
        "requested_start": start,
        "requested_end": end,
        "entitlement_cutoff": cutoff.isoformat(),
        "fetched_at": datetime.now(UTC).isoformat(),
        "rows": int(len(out)),
    }
    write_metadata(path.with_suffix(".metadata.json"), meta)
    return path


def _to_call_put(option_type: str | None) -> str:
    t = (option_type or "").lower()
    return "C" if t.startswith("c") else "P"


def _quote_mid(bid: float | None, ask: float | None, last: float | None) -> float:
    if bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid:
        return float(0.5 * (bid + ask))
    if last is not None:
        return float(last)
    return float("nan")


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    if value in (None, "", "null"):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _massive_open_close_row(
    *,
    payload: dict[str, Any],
    asof_d: date,
    expiry: date,
    call_put: str,
    symbol: str,
    strike: float,
    underlying_close: float,
    fallback_rel_spread: float,
    min_mid: float,
) -> dict[str, Any] | None:
    close_px = _safe_float(payload.get("close"))
    if close_px is None:
        close_px = _safe_float(payload.get("c"))
    if close_px is None:
        return None

    mid = max(float(close_px), min_mid)
    bid = _safe_float(payload.get("bid"))
    if bid is None:
        bid = _safe_float(payload.get("b"))
    ask = _safe_float(payload.get("ask"))
    if ask is None:
        ask = _safe_float(payload.get("a"))

    if bid is None or ask is None or ask <= bid:
        half = max(0.01, 0.5 * fallback_rel_spread * max(mid, 0.01))
        bid = max(0.0, mid - half)
        ask = mid + half
    else:
        mid = 0.5 * (bid + ask)

    return {
        "date": asof_d,
        "expiry": expiry,
        "dte": int((expiry - asof_d).days),
        "call_put": call_put,
        "symbol": symbol,
        "strike": float(strike),
        "bid": float(bid),
        "ask": float(ask),
        "mid": float(mid),
        "last": float(mid),
        "volume": _safe_int(payload.get("volume") or payload.get("v")),
        "open_interest": _safe_int(payload.get("open_interest")),
        "underlying_close": float(underlying_close),
        "delta": float("nan"),
        "gamma": float("nan"),
        "theta": float("nan"),
        "vega": float("nan"),
        "iv": _safe_float(payload.get("implied_volatility") or payload.get("iv")) or float("nan"),
    }


def collect_eod_chains_asof(
    asof: str,
    symbol: str,
    tenors_days: list[int],
    greeks: bool,
    config: dict[str, Any],
) -> Path:
    asof_d = parse_date(asof)
    tradier_cfg = config.get("tradier", {})
    client = TradierClient(
        TradierConfig(
            base_url=tradier_cfg.get("base_url", "https://api.tradier.com"),
            token_env_var=tradier_cfg.get("token_env_var", "TRADIER_TOKEN"),
            cache_enabled=bool(tradier_cfg.get("cache_enabled", True)),
            cache_dir=Path(tradier_cfg.get("cache_dir", "data/tradier_cache")),
            connect_timeout=float(tradier_cfg.get("connect_timeout", 10.0)),
            read_timeout=float(tradier_cfg.get("read_timeout", 30.0)),
            max_retries=int(tradier_cfg.get("max_retries", 4)),
            backoff_base=float(tradier_cfg.get("backoff_base", 0.5)),
        )
    )

    history = client.get_market_history(symbol=symbol, start=asof_d, end=asof_d)
    if not history.empty:
        underlying_close = float(history.iloc[-1]["close"])
    else:
        quotes = client.get_quotes([symbol]).get(symbol, {})
        px = quotes.get("last") or quotes.get("bid") or quotes.get("ask")
        if px is None:
            raise ValueError("Could not infer underlying close from Tradier history or quotes")
        underlying_close = float(px)

    expiries = client.get_option_expirations(symbol=symbol)
    if not expiries:
        raise ValueError("No option expirations returned by Tradier")

    exp_dtes = [(exp, (exp - asof_d).days) for exp in expiries if (exp - asof_d).days > 0]
    if not exp_dtes:
        raise ValueError("No future expirations available for requested as-of date")

    chosen: list[Any] = []
    for t in tenors_days:
        exp, _ = min(exp_dtes, key=lambda ed: abs(ed[1] - int(t)))
        if exp not in chosen:
            chosen.append(exp)
    chosen = sorted(chosen)

    frames: list[pd.DataFrame] = []
    for exp in chosen:
        chain = client.get_option_chain(symbol=symbol, expiration=exp, greeks=greeks)
        if chain.empty:
            continue
        c = chain.copy()
        c["date"] = pd.Timestamp(asof_d)
        c["expiry"] = pd.to_datetime(c["expiration_date"])
        c["dte"] = (c["expiry"].dt.date - asof_d).apply(lambda x: int(x.days))
        c["call_put"] = c["option_type"].apply(_to_call_put)
        c["underlying_close"] = underlying_close
        c["mid"] = [
            _quote_mid(bid, ask, last)
            for bid, ask, last in zip(c["bid"], c["ask"], c["last"], strict=True)
        ]
        keep = [
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
        c = c[[col for col in keep if col in c.columns]]
        c = c.dropna(subset=["strike", "mid", "expiry", "dte", "underlying_close"])
        frames.append(c)

    if not frames:
        raise ValueError("No option chain rows produced from selected expirations")

    out = pd.concat(frames, ignore_index=True)
    raw_dir = ensure_dir(config["paths"]["raw_dir"])
    out_path = raw_dir / f"{asof_d.isoformat()}.parquet"
    out.to_parquet(out_path, index=False)
    meta = {
        "source": "tradier_live_snapshot",
        "symbol": symbol,
        "asof": asof_d.isoformat(),
        "tenors_days": [int(t) for t in tenors_days],
        "selected_expiries": [e.isoformat() for e in chosen],
        "rows": int(len(out)),
        "greeks": bool(greeks),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    write_metadata(raw_dir / f"{asof_d.isoformat()}.metadata.json", meta)
    return out_path


def collect_eod_chains_asof_massive(
    asof: str,
    symbol: str,
    tenors_days: list[int],
    config: dict[str, Any],
) -> Path:
    asof_d = parse_date(asof)
    cutoff = _validate_massive_asof(asof_d=asof_d, config=config)
    massive_cfg = config.get("massive", {})
    client = MassiveClient(
        MassiveConfig(
            base_url=massive_cfg.get("base_url", "https://api.massive.com"),
            api_key_env_var=massive_cfg.get("api_key_env_var", "MASSIVE_API_KEY"),
            cache_enabled=bool(massive_cfg.get("cache_enabled", True)),
            cache_dir=Path(massive_cfg.get("cache_dir", "data/massive_cache")),
            connect_timeout=float(massive_cfg.get("connect_timeout", 10.0)),
            read_timeout=float(massive_cfg.get("read_timeout", 30.0)),
            max_retries=int(massive_cfg.get("max_retries", 4)),
            backoff_base=float(massive_cfg.get("backoff_base", 0.5)),
        )
    )

    history = client.get_stock_aggs(symbol=symbol, start=asof_d, end=asof_d)
    if history.empty:
        raise ValueError("Massive returned empty underlying bars for as-of date")
    underlying_close = float(history.iloc[-1]["close"])

    max_tenor = max(int(t) for t in tenors_days) if tenors_days else 180
    contracts = client.list_options_contracts(
        underlying_ticker=symbol,
        as_of=asof_d,
        expiration_gte=asof_d,
        expiration_lte=asof_d + timedelta(days=max_tenor + 45),
    )
    if contracts.empty:
        raise ValueError("Massive returned no options contracts")

    expiries = sorted({pd.Timestamp(d).date() for d in contracts["expiration_date"] if pd.notna(d)})
    exp_dtes = [(exp, (exp - asof_d).days) for exp in expiries if (exp - asof_d).days > 0]
    if not exp_dtes:
        raise ValueError("Massive returned no future expirations")

    chosen: list[Any] = []
    for t in tenors_days:
        exp, _ = min(exp_dtes, key=lambda ed: abs(ed[1] - int(t)))
        if exp not in chosen:
            chosen.append(exp)
    chosen = sorted(chosen)

    today = datetime.now(UTC).date()
    use_snapshot = asof_d >= today
    frames: list[pd.DataFrame] = []
    if use_snapshot:
        for exp in chosen:
            frame = client.get_options_chain_snapshot(
                underlying_ticker=symbol,
                as_of=asof_d,
                fallback_underlying_close=underlying_close,
                expiration_gte=exp,
                expiration_lte=exp,
            )
            if not frame.empty:
                frames.append(frame)

    mode = "massive_snapshot_chain"
    if not frames:
        fallback_rel_spread = float(massive_cfg.get("fallback_rel_spread", 0.12))
        strike_band_pct = float(massive_cfg.get("fallback_strike_band_pct", 0.25))
        max_contracts_per_expiry = int(massive_cfg.get("fallback_max_contracts_per_expiry", 80))
        min_mid = float(massive_cfg.get("fallback_min_mid", 0.01))

        fallback_rows: list[dict[str, Any]] = []
        low_strike = underlying_close * (1.0 - strike_band_pct)
        high_strike = underlying_close * (1.0 + strike_band_pct)
        oc_not_found = 0
        oc_other_errors = 0
        oc_success = 0

        for exp in chosen:
            by_expiry = contracts.loc[contracts["expiration_date"] == exp].copy()
            if by_expiry.empty:
                continue

            by_expiry["strike_price"] = by_expiry["strike_price"].astype(float)
            strike_filtered = by_expiry.loc[
                (by_expiry["strike_price"] >= low_strike)
                & (by_expiry["strike_price"] <= high_strike)
            ].copy()
            if strike_filtered.empty:
                strike_filtered = by_expiry.copy()
            strike_filtered["distance"] = (strike_filtered["strike_price"] - underlying_close).abs()
            strike_filtered = strike_filtered.sort_values("distance").head(max_contracts_per_expiry)

            for _, row in strike_filtered.iterrows():
                ticker = row.get("ticker")
                strike = row.get("strike_price")
                contract_type = row.get("contract_type")
                if not isinstance(ticker, str) or strike is None:
                    continue
                try:
                    payload = client.get_option_open_close(options_ticker=ticker, as_of=asof_d)
                except Exception:
                    oc_other_errors += 1
                    continue
                if payload is None:
                    oc_not_found += 1
                    continue
                parsed = _massive_open_close_row(
                    payload=payload,
                    asof_d=asof_d,
                    expiry=exp,
                    call_put=_to_call_put(contract_type),
                    symbol=ticker,
                    strike=float(strike),
                    underlying_close=underlying_close,
                    fallback_rel_spread=fallback_rel_spread,
                    min_mid=min_mid,
                )
                if parsed is not None:
                    fallback_rows.append(parsed)
                    oc_success += 1

        if fallback_rows:
            frames = [pd.DataFrame(fallback_rows)]
            mode = "massive_open_close_fallback"
            logger.info(
                "Massive open-close fallback for %s: success=%d not_found=%d other_errors=%d",
                asof_d.isoformat(),
                oc_success,
                oc_not_found,
                oc_other_errors,
            )

    if not frames:
        raise ValueError(
            "Massive returned no option rows for selected expiries. "
            "For historical dates, use the open-close fallback settings or a narrower strike band."
        )

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["expiry"] = pd.to_datetime(out["expiry"]).dt.date
    out = out.sort_values(["expiry", "strike", "call_put"]).drop_duplicates(
        subset=["date", "symbol"],
        keep="last",
    )
    raw_dir = ensure_dir(config["paths"]["raw_dir"])
    out_path = raw_dir / f"{asof_d.isoformat()}.parquet"
    out.to_parquet(out_path, index=False)

    meta = {
        "source": mode,
        "symbol": symbol,
        "asof": asof_d.isoformat(),
        "entitlement_cutoff": cutoff.isoformat(),
        "tenors_days": [int(t) for t in tenors_days],
        "selected_expiries": [e.isoformat() for e in chosen],
        "rows": int(len(out)),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    write_metadata(raw_dir / f"{asof_d.isoformat()}.metadata.json", meta)
    return out_path


def collect_eod_chains_asof_massive_flatfile(
    asof: str,
    symbol: str,
    tenors_days: list[int],
    config: dict[str, Any],
    *,
    underlying_close: float | None = None,
) -> Path:
    asof_d = parse_date(asof)
    cutoff = _validate_massive_asof(asof_d=asof_d, config=config)
    massive_cfg = config.get("massive", {})
    flat_cfg = config.get("massive_flatfiles", {})
    client = MassiveFlatFileClient(
        MassiveFlatFilesConfig(
            endpoint_url=flat_cfg.get("endpoint_url", "https://files.massive.com"),
            bucket=flat_cfg.get("bucket", "flatfiles"),
            prefix=flat_cfg.get("prefix", "us_options_opra/day_aggs_v1"),
            aws_access_key_id_env_var=flat_cfg.get(
                "aws_access_key_id_env_var",
                "MASSIVE_FILES_ACCESS_KEY_ID",
            ),
            aws_secret_access_key_env_var=flat_cfg.get(
                "aws_secret_access_key_env_var",
                "MASSIVE_FILES_SECRET_ACCESS_KEY",
            ),
            cache_enabled=bool(flat_cfg.get("cache_enabled", True)),
            cache_dir=Path(flat_cfg.get("cache_dir", "data/massive_cache/flatfiles")),
            read_chunksize=int(flat_cfg.get("read_chunksize", 250_000)),
        )
    )

    if underlying_close is None:
        history = MassiveClient(
            MassiveConfig(
                base_url=massive_cfg.get("base_url", "https://api.massive.com"),
                api_key_env_var=massive_cfg.get("api_key_env_var", "MASSIVE_API_KEY"),
                cache_enabled=bool(massive_cfg.get("cache_enabled", True)),
                cache_dir=Path(massive_cfg.get("cache_dir", "data/massive_cache")),
                connect_timeout=float(massive_cfg.get("connect_timeout", 10.0)),
                read_timeout=float(massive_cfg.get("read_timeout", 30.0)),
                max_retries=int(massive_cfg.get("max_retries", 4)),
                backoff_base=float(massive_cfg.get("backoff_base", 0.5)),
            )
        ).get_stock_aggs(symbol=symbol, start=asof_d, end=asof_d)
        if history.empty:
            raise ValueError("Massive returned empty underlying bars for as-of date")
        underlying_close = float(history.iloc[-1]["close"])
    else:
        underlying_close = float(underlying_close)

    raw = client.read_day_aggs_for_underlying(asof_d, symbol)
    if raw.empty:
        raise ValueError(f"No rows found in Massive flat file for {symbol} on {asof_d.isoformat()}")

    close_col = next((c for c in raw.columns if c.lower() in {"close", "c"}), None)
    vol_col = next((c for c in raw.columns if c.lower() in {"volume", "v"}), None)
    oi_col = next((c for c in raw.columns if c.lower() in {"open_interest", "oi"}), None)
    if close_col is None:
        raise ValueError("Massive flat file missing close column")

    rows: list[dict[str, Any]] = []
    rel_spread = float(flat_cfg.get("fallback_rel_spread", 0.08))
    min_mid = float(flat_cfg.get("fallback_min_mid", 0.01))
    for _, row in raw.iterrows():
        ticker = str(row.get("ticker", "")).upper()
        parsed = parse_option_symbol_osi(ticker)
        if parsed is None:
            continue
        if parsed["underlying"] != symbol.upper():
            continue
        expiry = parsed["expiry"]
        dte = int((expiry - asof_d).days)
        if dte <= 0:
            continue

        close_px = _safe_float(row.get(close_col))
        if close_px is None:
            continue
        mid = max(min_mid, float(close_px))
        spread_abs = max(0.01, rel_spread * max(mid, 0.01))
        bid = max(0.0, mid - 0.5 * spread_abs)
        ask = mid + 0.5 * spread_abs
        rows.append(
            {
                "date": asof_d,
                "expiry": expiry,
                "dte": dte,
                "call_put": parsed["call_put"],
                "symbol": ticker,
                "strike": float(parsed["strike"]),
                "bid": float(bid),
                "ask": float(ask),
                "mid": float(mid),
                "last": float(mid),
                "volume": _safe_int(row.get(vol_col)) if vol_col else 0,
                "open_interest": _safe_int(row.get(oi_col)) if oi_col else 0,
                "underlying_close": float(underlying_close),
                "delta": float("nan"),
                "gamma": float("nan"),
                "theta": float("nan"),
                "vega": float("nan"),
                "iv": float("nan"),
            }
        )
    if not rows:
        raise ValueError(
            f"Massive flat file had no valid {symbol} option rows for {asof_d.isoformat()}"
        )

    out = pd.DataFrame(rows)
    expiries = sorted(out["expiry"].dropna().unique())
    exp_dtes = []
    for exp in expiries:
        exp_date = pd.Timestamp(exp).date()
        exp_dtes.append((exp_date, (exp_date - asof_d).days))
    chosen: list[date] = []
    for t in tenors_days:
        exp, _ = min(exp_dtes, key=lambda ed: abs(ed[1] - int(t)))
        if exp not in chosen:
            chosen.append(exp)
    chosen = sorted(chosen)

    out["expiry"] = pd.to_datetime(out["expiry"]).dt.date
    out = out.loc[out["expiry"].isin(chosen)].copy()
    if out.empty:
        raise ValueError(
            f"No rows after tenor selection for {symbol} on {asof_d.isoformat()}"
        )
    out = out.sort_values(["expiry", "strike", "call_put"]).drop_duplicates(
        subset=["date", "symbol"],
        keep="last",
    )

    raw_dir = ensure_dir(config["paths"]["raw_dir"])
    out_path = raw_dir / f"{asof_d.isoformat()}.parquet"
    out.to_parquet(out_path, index=False)

    meta = {
        "source": "massive_flatfiles_day_aggs_v1",
        "symbol": symbol,
        "asof": asof_d.isoformat(),
        "entitlement_cutoff": cutoff.isoformat(),
        "tenors_days": [int(t) for t in tenors_days],
        "selected_expiries": [e.isoformat() for e in chosen],
        "rows": int(len(out)),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    write_metadata(raw_dir / f"{asof_d.isoformat()}.metadata.json", meta)
    return out_path


def _clear_tree(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    count = 0
    for child in p.rglob("*"):
        if child.is_file():
            count += 1
    shutil.rmtree(p)
    return count


def clear_market_data(config: dict[str, Any]) -> dict[str, int]:
    paths_cfg = config.get("paths", {})
    raw_dir = ensure_dir(paths_cfg["raw_dir"])
    processed_dir = ensure_dir(paths_cfg["processed_dir"])
    underlying_path = Path(paths_cfg["underlying_path"])

    removed = {"raw": 0, "underlying": 0, "processed": 0, "cache": 0}

    for p in raw_dir.glob("*.parquet"):
        p.unlink()
        removed["raw"] += 1
    for p in raw_dir.glob("*.metadata.json"):
        p.unlink()
        removed["raw"] += 1

    for p in (underlying_path, underlying_path.with_suffix(".metadata.json")):
        if p.exists():
            p.unlink()
            removed["underlying"] += 1

    for p in processed_dir.glob("*"):
        if p.is_file() and p.suffix in {".npz", ".json", ".parquet"}:
            p.unlink()
            removed["processed"] += 1

    for cache_key in (
        config.get("massive", {}).get("cache_dir"),
        config.get("tradier", {}).get("cache_dir"),
        config.get("massive_flatfiles", {}).get("cache_dir"),
    ):
        if cache_key:
            removed["cache"] += _clear_tree(cache_key)

    return removed


def fetch_market_data_range_massive(
    start: str,
    end: str,
    symbol: str,
    tenors_days: list[int],
    config: dict[str, Any],
    *,
    clean: bool = False,
    stop_on_error: bool = False,
    options_source: str = "flatfiles",
) -> dict[str, Any]:
    start_d = parse_date(start)
    end_d = parse_date(end)

    removed = {"raw": 0, "underlying": 0, "processed": 0}
    if clean:
        removed = clear_market_data(config=config)

    underlying_path = fetch_underlying_range_massive(
        start=start,
        end=end,
        symbol=symbol,
        config=config,
    )
    underlying = pd.read_parquet(underlying_path)
    underlying["date"] = pd.to_datetime(underlying["date"]).dt.date
    date_candidates = sorted(
        {d for d in underlying["date"].tolist() if start_d <= d <= end_d}
    )
    if not date_candidates:
        raise ValueError("No underlying dates available in requested range")

    successes: list[str] = []
    failures: list[dict[str, str]] = []
    skipped_existing = 0
    raw_dir = Path(config["paths"]["raw_dir"])
    for day in date_candidates:
        day_str = day.isoformat()
        existing = raw_dir / f"{day_str}.parquet"
        if existing.exists() and existing.stat().st_size > 0:
            successes.append(day_str)
            skipped_existing += 1
            continue
        try:
            if options_source == "api":
                collect_eod_chains_asof_massive(
                    asof=day_str,
                    symbol=symbol,
                    tenors_days=tenors_days,
                    config=config,
                )
            elif options_source == "flatfiles":
                day_underlying = underlying.loc[underlying["date"] == day, "close"]
                if day_underlying.empty:
                    raise ValueError(f"Missing underlying close for {day_str}")
                collect_eod_chains_asof_massive_flatfile(
                    asof=day_str,
                    symbol=symbol,
                    tenors_days=tenors_days,
                    config=config,
                    underlying_close=float(day_underlying.iloc[-1]),
                )
            else:
                raise ValueError("options_source must be one of: flatfiles, api")
        except Exception as exc:
            failures.append({"date": day_str, "error": str(exc)})
            logger.warning("Massive chain collection failed for %s: %s", day_str, exc)
            if stop_on_error:
                raise
            continue
        successes.append(day_str)

    if not successes:
        raise ValueError("No option chains collected for requested range")

    aligned_days = {parse_date(d) for d in successes}
    aligned_underlying = underlying.loc[underlying["date"].isin(aligned_days)].copy()
    aligned_underlying = aligned_underlying.sort_values("date").drop_duplicates(
        subset=["date"],
        keep="last",
    )
    aligned_underlying.to_parquet(underlying_path, index=False)

    summary = {
        "symbol": symbol,
        "requested_start": start_d.isoformat(),
        "requested_end": end_d.isoformat(),
        "clean_requested": bool(clean),
        "removed_counts": removed,
        "underlying_path": str(underlying_path),
        "underlying_rows_raw": int(len(underlying)),
        "underlying_rows_aligned": int(len(aligned_underlying)),
        "days_attempted": int(len(date_candidates)),
        "days_succeeded": int(len(successes)),
        "days_failed": int(len(failures)),
        "days_skipped_existing": int(skipped_existing),
        "options_source": options_source,
        "success_range": {
            "start": min(successes),
            "end": max(successes),
        },
        "failed_examples": failures[:20],
        "generated_at": datetime.now(UTC).isoformat(),
    }
    raw_dir = ensure_dir(config["paths"]["raw_dir"])
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    summary_path = raw_dir / f"_massive_pull_summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path)
    return summary


def synth_data_range(start: str, end: str, config: dict[str, Any]) -> None:
    start_d = parse_date(start)
    end_d = parse_date(end)
    synth_cfg = config.get("synth", {})
    paths_cfg = config["paths"]

    write_synthetic_dataset(
        start=start_d,
        end=end_d,
        raw_dir=paths_cfg["raw_dir"],
        underlying_path=paths_cfg["underlying_path"],
        config=SynthConfig(
            seed=int(synth_cfg.get("seed", 123)),
            strike_points=int(synth_cfg.get("strike_points", 21)),
            bad_quote_prob=float(synth_cfg.get("bad_quote_prob", 0.02)),
        ),
    )


def build_dataset_range(start: str, end: str, config: dict[str, Any]) -> Path:
    start_d = parse_date(start)
    end_d = parse_date(end)

    grid = SurfaceGrid.from_config(config["surface"])
    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = ensure_dir(config["paths"]["processed_dir"])

    underlying = pd.read_parquet(config["paths"]["underlying_path"])
    context_df = build_context_features(underlying)

    dates: list[str] = []
    contexts: list[np.ndarray] = []
    surfaces: list[np.ndarray] = []
    raw_surfaces: list[np.ndarray] = []
    theta_raw_list: list[np.ndarray] = []
    theta_list: list[np.ndarray] = []
    skip_counts: dict[str, int] = {}

    def bump(reason: str) -> None:
        skip_counts[reason] = skip_counts.get(reason, 0) + 1

    files = list_raw_chain_files(raw_dir, start=start_d, end=end_d)
    for file in files:
        chain = pd.read_parquet(file)
        if chain.empty:
            bump("empty_chain")
            continue
        day = file.stem
        if pd.Timestamp(day) not in context_df.index:
            bump("missing_context")
            continue

        filtered = apply_liquidity_filters(
            chain, min_volume=0, min_open_interest=0, max_relative_spread=5.0
        )
        if filtered.empty:
            filtered = chain

        try:
            built = build_surface_from_chain(filtered, grid=grid)
            repaired = repair_surface_qp(
                built.surface_raw,
                x_grid=grid.x,
                lambda_smooth=float(config.get("repair", {}).get("lambda_smooth", 1e-3)),
                data_weight=float(config.get("repair", {}).get("data_weight", 1.0)),
            )
        except Exception as exc:
            bump("surface_build_or_repair_error")
            logger.warning("Skipping day %s due to build/repair error: %s", day, exc)
            continue

        theta = surface_to_theta(repaired.repaired)
        theta_raw = softplus_inverse(theta)

        dates.append(day)
        contexts.append(context_df.loc[pd.Timestamp(day)].to_numpy(dtype=float))
        surfaces.append(repaired.repaired)
        raw_surfaces.append(built.surface_raw)
        theta_list.append(theta)
        theta_raw_list.append(theta_raw)

    if not dates:
        raise ValueError("No dataset rows built. Check raw data and date range.")

    context_arr = np.vstack(contexts).astype(np.float32)
    surface_arr = np.stack(surfaces).astype(np.float32)
    surface_raw_arr = np.stack(raw_surfaces).astype(np.float32)
    theta_arr = np.stack(theta_list).astype(np.float32)
    theta_raw_arr = np.stack(theta_raw_list).astype(np.float32)

    ds_path = processed_dir / "dataset.npz"
    np.savez_compressed(
        ds_path,
        dates=np.array(dates),
        context=context_arr,
        surface=surface_arr,
        surface_raw=surface_raw_arr,
        theta=theta_arr,
        theta_raw=theta_raw_arr,
        x_grid=grid.x.astype(np.float32),
        tenors_days=np.array(grid.tenors_days, dtype=np.int32),
    )

    meta = {
        "rows": len(dates),
        "attempted_days": len(files),
        "context_features": list(context_df.columns),
        "nx": grid.nx,
        "nt": len(grid.tenors_days),
        "built_at": datetime.now(UTC).isoformat(),
        "skip_counts": skip_counts,
    }
    write_metadata(processed_dir / "dataset_meta.json", meta)

    preview = pd.DataFrame({"date": dates, "z_theta_norm": np.linalg.norm(theta_arr, axis=1)})
    preview.to_parquet(processed_dir / "dataset_preview.parquet", index=False)
    return ds_path


def train_from_config(config: dict[str, Any], dataset_path: str | Path | None = None) -> Path:
    train_cfg = config.get("train", {})
    processed_dir = Path(config["paths"]["processed_dir"])
    ds_path = Path(dataset_path) if dataset_path else processed_dir / "dataset.npz"

    out_dir = ensure_dir(Path(config["paths"]["outputs_dir"]) / "checkpoints")
    grid = SurfaceGrid.from_config(config["surface"])
    ckpt = train_flow_model(
        dataset_path=ds_path,
        output_dir=out_dir,
        config=TrainConfig(
            seed=int(train_cfg.get("seed", 42)),
            batch_size=int(train_cfg.get("batch_size", 64)),
            epochs=int(train_cfg.get("epochs", 20)),
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-6)),
            hidden_size=int(train_cfg.get("hidden_size", 128)),
            flow_layers=int(train_cfg.get("flow_layers", 4)),
            early_stopping_patience=int(train_cfg.get("early_stopping_patience", 5)),
        ),
        nx=grid.nx,
        nt=len(grid.tenors_days),
    )
    return ckpt


def eval_checkpoint(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    dataset_path: str | Path | None = None,
    n_samples: int = 32,
) -> Path:
    processed_dir = Path(config["paths"]["processed_dir"])
    ds_path = Path(dataset_path) if dataset_path else processed_dir / "dataset.npz"
    ds = np.load(ds_path, allow_pickle=True)

    context = ds["context"].astype(np.float32)
    theta_raw = ds["theta_raw"].astype(np.float32)
    model = load_checkpoint(checkpoint_path)

    ll = log_likelihood(model, theta_raw=theta_raw, context=context)
    samples = sample_surfaces(model, context=context[:1], n_samples=n_samples)[0]
    mean_surfaces = conditional_mean_surface(model, context=context, n_samples=n_samples)

    arb_ok = [is_arb_free(samples[i]) for i in range(samples.shape[0])]
    arb_rate = float(np.mean(arb_ok))
    counts = [arb_violation_counts(samples[i]) for i in range(samples.shape[0])]

    obs_surface = ds["surface"].astype(np.float32)
    x_grid = ds["x_grid"].astype(np.float32)
    tenors_days = ds["tenors_days"].astype(np.int32)
    iv_obs = np.stack(
        [_surface_to_iv(s, x_grid=x_grid, tenors_days=tenors_days) for s in obs_surface]
    )
    iv_pred = np.stack(
        [_surface_to_iv(s, x_grid=x_grid, tenors_days=tenors_days) for s in mean_surfaces]
    )
    iv_diff = iv_obs - iv_pred

    run_name = f"eval_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    out_dir = ensure_dir(Path(config["paths"]["outputs_dir"]) / run_name)
    summary = {
        "n_obs": int(context.shape[0]),
        "mean_log_likelihood": float(np.mean(ll)),
        "median_log_likelihood": float(np.median(ll)),
        "sample_arb_pass_rate": arb_rate,
        "sample_violations_avg": {
            "strike_monotonic": float(np.mean([c["strike_monotonic"] for c in counts])),
            "strike_convex": float(np.mean([c["strike_convex"] for c in counts])),
            "calendar": float(np.mean([c["calendar"] for c in counts])),
        },
        "iv_rmse": float(np.sqrt(np.mean(iv_diff**2))),
        "iv_mae": float(np.mean(np.abs(iv_diff))),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    mean_surface = mean_surfaces[0]
    np.save(out_dir / "mean_surface.npy", mean_surface)
    np.save(out_dir / "iv_obs.npy", iv_obs)
    np.save(out_dir / "iv_pred.npy", iv_pred)
    return out_dir


def backtest_from_config(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    dataset_path: str | Path | None = None,
) -> Path:
    processed_dir = Path(config["paths"]["processed_dir"])
    ds_path = Path(dataset_path) if dataset_path else processed_dir / "dataset.npz"

    bt_cfg = dict(config.get("strategy", {}))
    grid = SurfaceGrid.from_config(config["surface"])
    fallback_seed = int(config.get("train", {}).get("seed", 42))
    default_provider = str(bt_cfg.get("signal_provider", "deep_flow"))
    merged_cfg = _merge_provider_strategy(bt_cfg, default_provider)
    bt_config = _make_backtest_config(
        merged_cfg,
        default_provider,
        fallback_seed=fallback_seed,
    )

    run_all = bool(bt_cfg.get("run_all_providers", False))
    if not run_all:
        return run_backtest(
            dataset_path=ds_path,
            checkpoint_path=checkpoint_path,
            raw_dir=config["paths"]["raw_dir"],
            output_dir=config["paths"]["outputs_dir"],
            tenor_days=grid.tenors_days,
            x_grid=grid.x,
            config=bt_config,
            signal_provider_name=bt_config.signal_provider,
        )

    providers = bt_cfg.get("providers", available_signal_providers())
    compare_dir = ensure_dir(
        Path(config["paths"]["outputs_dir"])
        / "backtests_compare"
        / f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    )
    compare: dict[str, Any] = {}
    for provider_name in providers:
        provider_name_str = str(provider_name)
        merged_cfg = _merge_provider_strategy(bt_cfg, provider_name_str)
        provider_bt_config = _make_backtest_config(
            merged_cfg,
            provider_name_str,
            fallback_seed=fallback_seed,
        )
        run_dir = run_backtest(
            dataset_path=ds_path,
            checkpoint_path=checkpoint_path,
            raw_dir=config["paths"]["raw_dir"],
            output_dir=config["paths"]["outputs_dir"],
            tenor_days=grid.tenors_days,
            x_grid=grid.x,
            config=provider_bt_config,
            signal_provider_name=provider_name_str,
        )
        summary = json.loads((run_dir / "summary.json").read_text())
        compare[provider_name_str] = {"run_dir": str(run_dir), "summary": summary}

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "compare": compare,
        "providers": [str(p) for p in providers],
    }
    (compare_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    return compare_dir


def _subset_dataset(ds: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for key, value in ds.items():
        if key in {"x_grid", "tenors_days"}:
            payload[key] = value
        else:
            payload[key] = value[idx]
    return payload


def _write_dataset_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(path, **payload)


def walkforward_from_config(config: dict[str, Any], dataset_path: str | Path | None = None) -> Path:
    wf_cfg = config.get("walkforward", {})
    train_days = int(wf_cfg.get("train_days", 40))
    test_days = int(wf_cfg.get("test_days", 10))
    step_days = int(wf_cfg.get("step_days", 10))

    processed_dir = Path(config["paths"]["processed_dir"])
    ds_path = Path(dataset_path) if dataset_path else processed_dir / "dataset.npz"
    ds_raw = np.load(ds_path, allow_pickle=True)
    ds = {k: ds_raw[k] for k in ds_raw.files}
    n = int(ds["dates"].shape[0])
    if n < train_days + test_days:
        raise ValueError("Not enough rows for walk-forward split")

    train_cfg = config.get("train", {})
    grid = SurfaceGrid.from_config(config["surface"])
    bt_cfg = dict(config.get("strategy", {}))
    fallback_seed = int(config.get("train", {}).get("seed", 42))
    default_provider = str(bt_cfg.get("signal_provider", "deep_flow"))
    merged_cfg = _merge_provider_strategy(bt_cfg, default_provider)
    bt_config = _make_backtest_config(
        merged_cfg,
        default_provider,
        fallback_seed=fallback_seed,
    )

    root = ensure_dir(
        Path(config["paths"]["outputs_dir"])
        / "walkforward"
        / f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    )
    windows: list[dict[str, Any]] = []

    window_id = 0
    for start in range(0, n - (train_days + test_days) + 1, step_days):
        train_idx = np.arange(start, start + train_days)
        test_idx = np.arange(start + train_days, start + train_days + test_days)
        window_dir = ensure_dir(root / f"window_{window_id:02d}")
        train_ds = _subset_dataset(ds, train_idx)
        test_ds = _subset_dataset(ds, test_idx)
        train_ds_path = window_dir / "train_dataset.npz"
        test_ds_path = window_dir / "test_dataset.npz"
        _write_dataset_npz(train_ds_path, train_ds)
        _write_dataset_npz(test_ds_path, test_ds)

        ckpt = train_flow_model(
            dataset_path=train_ds_path,
            output_dir=window_dir / "checkpoints",
            config=TrainConfig(
                seed=int(train_cfg.get("seed", 42)),
                batch_size=int(train_cfg.get("batch_size", 64)),
                epochs=int(train_cfg.get("epochs", 20)),
                lr=float(train_cfg.get("lr", 1e-3)),
                weight_decay=float(train_cfg.get("weight_decay", 1e-6)),
                hidden_size=int(train_cfg.get("hidden_size", 128)),
                flow_layers=int(train_cfg.get("flow_layers", 4)),
                early_stopping_patience=int(train_cfg.get("early_stopping_patience", 5)),
            ),
            nx=grid.nx,
            nt=len(grid.tenors_days),
        )

        bt_dir = run_backtest(
            dataset_path=test_ds_path,
            checkpoint_path=ckpt,
            raw_dir=config["paths"]["raw_dir"],
            output_dir=window_dir,
            tenor_days=grid.tenors_days,
            x_grid=grid.x,
            config=bt_config,
            signal_provider_name=bt_config.signal_provider,
        )
        summary = json.loads((bt_dir / "summary.json").read_text())
        windows.append(
            {
                "window_id": window_id,
                "train_start": str(train_ds["dates"][0]),
                "train_end": str(train_ds["dates"][-1]),
                "test_start": str(test_ds["dates"][0]),
                "test_end": str(test_ds["dates"][-1]),
                "run_dir": str(bt_dir),
                "summary": summary,
            }
        )
        window_id += 1

    agg = {
        "windows": len(windows),
        "mean_total_pnl": float(np.mean([w["summary"]["total_pnl"] for w in windows])),
        "mean_sharpe": float(np.mean([w["summary"]["sharpe"] for w in windows])),
        "mean_turnover_ratio": float(np.mean([w["summary"]["turnover_ratio"] for w in windows])),
    }
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "config": config,
        "walkforward": {
            "train_days": train_days,
            "test_days": test_days,
            "step_days": step_days,
        },
        "aggregate": agg,
        "windows": windows,
    }
    (root / "walkforward_summary.json").write_text(json.dumps(payload, indent=2))
    return root


def run_sanity(config: dict[str, Any]) -> dict[str, Path]:
    cfg = dict(config)
    cfg["train"] = dict(config.get("train", {}))
    cfg["train"]["epochs"] = min(4, int(cfg["train"].get("epochs", 20)))
    cfg["train"]["flow_layers"] = min(2, int(cfg["train"].get("flow_layers", 4)))
    cfg["train"]["hidden_size"] = min(64, int(cfg["train"].get("hidden_size", 128)))
    cfg["strategy"] = dict(config.get("strategy", {}))
    cfg["strategy"]["n_samples"] = min(16, int(cfg["strategy"].get("n_samples", 32)))

    start = "2024-01-02"
    end = "2024-03-29"
    synth_data_range(start=start, end=end, config=cfg)
    ds_path = build_dataset_range(start=start, end=end, config=cfg)
    ckpt = train_from_config(cfg, dataset_path=ds_path)
    eval_dir = eval_checkpoint(ckpt, cfg, dataset_path=ds_path, n_samples=16)
    bt_dir = backtest_from_config(ckpt, cfg, dataset_path=ds_path)

    return {
        "dataset": ds_path,
        "checkpoint": ckpt,
        "eval": eval_dir,
        "backtest": bt_dir,
    }
