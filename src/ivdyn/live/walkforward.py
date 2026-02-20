"""Live walk-forward sync and trading pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from ivdyn.backtest import BacktestConfig, run_backtest
from ivdyn.data import DatasetBuildConfig, build_dataset
from ivdyn.data.schemas import normalize_chain_df
from ivdyn.utils.paths import resolve_latest, utc_timestamp

_DEFAULT_X_GRID: tuple[float, ...] = (
    -0.35,
    -0.30,
    -0.25,
    -0.20,
    -0.16,
    -0.12,
    -0.09,
    -0.06,
    -0.04,
    -0.02,
    0.0,
    0.02,
    0.04,
    0.06,
    0.09,
    0.12,
    0.16,
    0.20,
    0.25,
    0.30,
    0.35,
)
_DEFAULT_TENOR_DAYS: tuple[int, ...] = (7, 14, 30, 60, 90, 180)


@dataclass(slots=True)
class LiveWalkforwardConfig:
    data_root: Path = Path("data")
    output_root: Path = Path("outputs/live_walkforward")
    lookback_days: int = 120
    trade_window_days: int = 14
    week_days: int = 5
    timeout_seconds: int = 30
    options_limit: int = 250
    max_pages: int = 0
    base_url: str = "https://api.massive.com"
    force_run: bool = False


@dataclass(slots=True)
class LiveWalkforwardResult:
    symbol: str
    asof_date: str
    status: str
    message: str
    session_dir: str | None = None
    last_week_pnl: float | None = None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off", "n"}


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except Exception:
        return default
    return max(minimum, val)


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_parquet_or_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def _run_recency(run_dir: Path) -> float:
    recency = run_dir.stat().st_mtime
    for rel in (
        "backtest/summary.json",
        "backtest/daily.parquet",
        "backtest/daily.csv",
        "evaluation/metrics.json",
        "train_summary.json",
    ):
        p = run_dir / rel
        if p.exists():
            try:
                recency = max(recency, p.stat().st_mtime)
            except Exception:
                continue
    return float(recency)


def _latest_run_for_symbol(symbol: str, *, outputs_root: Path) -> Path | None:
    runs_root = outputs_root / symbol.upper() / "runs"
    if not runs_root.is_dir():
        return None

    candidates: list[Path] = [p.resolve() for p in runs_root.glob("**/run_*") if p.is_dir()]
    for latest_path in runs_root.glob("**/latest.txt"):
        try:
            target = Path(latest_path.read_text(encoding="utf-8").strip()).expanduser().resolve()
        except Exception:
            continue
        if target.is_dir():
            candidates.append(target)

    uniq: dict[str, Path] = {str(p): p for p in candidates}
    if not uniq:
        latest = resolve_latest(runs_root)
        return latest.resolve() if latest is not None else None
    return max(uniq.values(), key=_run_recency)


def _latest_symbol_from_outputs(outputs_root: Path) -> str | None:
    best_symbol: str | None = None
    best_recency = -1.0
    if not outputs_root.exists():
        return None
    for symbol_dir in outputs_root.iterdir():
        if not symbol_dir.is_dir():
            continue
        symbol = symbol_dir.name.upper()
        if symbol in {"RUNS", "LIVE_WALKFORWARD"}:
            continue
        run_dir = _latest_run_for_symbol(symbol, outputs_root=outputs_root)
        if run_dir is None:
            continue
        recency = _run_recency(run_dir)
        if recency > best_recency:
            best_recency = recency
            best_symbol = symbol
    return best_symbol


def _symbol_from_path(path: Path) -> str | None:
    parts = path.resolve().parts
    for i in range(len(parts) - 2):
        if parts[i] == "outputs" and parts[i + 2] in {"runs", "datasets"}:
            symbol = parts[i + 1].strip().upper()
            if symbol and symbol not in {"RUNS", "DATASETS", "LIVE_WALKFORWARD"}:
                return symbol
    for i in range(len(parts) - 1):
        if parts[i] == "symbols":
            symbol = parts[i + 1].strip().upper()
            if symbol:
                return symbol
    return None


def _symbol_from_dataset_path(dataset_path: Path) -> str | None:
    sym = _symbol_from_path(dataset_path)
    if sym:
        return sym
    meta_path = dataset_path.parent / "dataset_meta.json"
    payload = _safe_json(meta_path)
    raw = str(payload.get("symbol", "")).strip().upper()
    return raw or None


def _dedupe_ordered(symbols: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        sym = str(raw).upper().strip()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _discover_symbols(ns: Any, *, outputs_root: Path) -> list[str]:
    env_symbols = os.environ.get("IVDYN_LIVE_SYMBOLS", "").strip()
    if env_symbols:
        return _dedupe_ordered([s for s in env_symbols.replace(";", ",").split(",") if s.strip()])

    inferred: list[str] = []

    raw_symbol = getattr(ns, "symbol", None)
    if raw_symbol:
        inferred.append(str(raw_symbol).upper())

    raw_run_dir = getattr(ns, "run_dir", None)
    if raw_run_dir:
        sym = _symbol_from_path(Path(str(raw_run_dir)).expanduser())
        if sym:
            inferred.append(sym)

    raw_dataset = getattr(ns, "dataset", None)
    if raw_dataset:
        sym = _symbol_from_dataset_path(Path(str(raw_dataset)).expanduser())
        if sym:
            inferred.append(sym)

    if not inferred:
        latest = _latest_symbol_from_outputs(outputs_root)
        if latest:
            inferred.append(latest)

    return _dedupe_ordered(inferred)


def _market_today_ny() -> date:
    try:
        return datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:
        return datetime.utcnow().date()


def _options_day_path(data_root: Path, symbol: str, asof: date) -> Path:
    return data_root / "symbols" / symbol / "options" / "raw" / f"{asof.isoformat()}.parquet"


def _underlying_path(data_root: Path, symbol: str) -> Path:
    return data_root / "symbols" / symbol / "underlying" / f"{symbol.lower()}_eod.parquet"


def _has_underlying_date(path: Path, asof: date) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path)
    except Exception:
        return False
    if "date" not in df.columns:
        return False
    d = pd.to_datetime(df["date"], errors="coerce").dt.date
    return bool((d == asof).any())


def _resolve_api_key() -> str | None:
    key = os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY")
    key = str(key).strip() if key else ""
    return key or None


def _pull_underlying_day(
    *,
    symbol: str,
    asof: date,
    data_root: Path,
    api_key: str,
    base_url: str,
    timeout_seconds: int,
    max_pages: int,
) -> bool:
    out_dir = data_root / "symbols" / symbol / "underlying"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _underlying_path(data_root, symbol)
    meta_path = out_dir / f"{symbol.lower()}_eod.metadata.json"

    url = f"{base_url.rstrip('/')}/v2/aggs/ticker/{symbol}/range/1/day/{asof.isoformat()}/{asof.isoformat()}"
    params: dict[str, Any] | None = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }
    rows: list[dict[str, Any]] = []
    pages = 0
    while url:
        resp = requests.get(url, params=params, timeout=timeout_seconds)
        params = None
        resp.raise_for_status()
        payload = resp.json()
        batch = payload.get("results", [])
        if isinstance(batch, list):
            rows.extend(batch)
        pages += 1
        if max_pages > 0 and pages >= max_pages:
            break
        next_url = payload.get("next_url")
        if next_url and "apiKey=" not in str(next_url):
            sep = "&" if "?" in str(next_url) else "?"
            next_url = f"{next_url}{sep}apiKey={api_key}"
        url = next_url

    if not rows:
        return False

    raw = pd.DataFrame(rows)
    if "t" not in raw.columns or "c" not in raw.columns:
        return False

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["t"], unit="ms", utc=True).dt.date,
            "open": pd.to_numeric(raw.get("o"), errors="coerce"),
            "high": pd.to_numeric(raw.get("h"), errors="coerce"),
            "low": pd.to_numeric(raw.get("l"), errors="coerce"),
            "close": pd.to_numeric(raw.get("c"), errors="coerce"),
            "volume": pd.to_numeric(raw.get("v"), errors="coerce"),
            "vwap": pd.to_numeric(raw.get("vw"), errors="coerce"),
            "trades": pd.to_numeric(raw.get("n"), errors="coerce"),
            "timestamp_ms": pd.to_numeric(raw.get("t"), errors="coerce"),
        }
    ).dropna(subset=["date", "close"])
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    if out_path.exists():
        prev = pd.read_parquet(out_path)
        if not prev.empty and "date" in prev.columns:
            prev = prev.copy()
            prev["date"] = pd.to_datetime(prev["date"], errors="coerce").dt.date
            out = pd.concat([prev, out], ignore_index=True)
            out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    out.to_parquet(out_path, index=False)

    meta = {
        "symbol": symbol,
        "source": "massive_v2_aggs",
        "base_url": base_url,
        "start_date": asof.isoformat(),
        "end_date": asof.isoformat(),
        "rows": int(len(out)),
        "path": str(out_path.resolve()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return _has_underlying_date(out_path, asof)


def _pull_options_day(
    *,
    symbol: str,
    asof: date,
    data_root: Path,
    api_key: str,
    base_url: str,
    timeout_seconds: int,
    options_limit: int,
    max_pages: int,
) -> bool:
    out_dir = data_root / "symbols" / symbol / "options" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    day_path = _options_day_path(data_root, symbol, asof)
    meta_path = out_dir / f"{asof.isoformat()}.metadata.json"

    url = f"{base_url.rstrip('/')}/v3/snapshot/options/{symbol}"
    params: dict[str, Any] | None = {
        "as_of": asof.isoformat(),
        "limit": int(options_limit),
        "sort": "ticker",
        "order": "asc",
        "expired": "true",
        "apiKey": api_key,
    }

    page_count = 0
    chain_rows: list[dict[str, Any]] = []
    while url:
        resp = requests.get(url, params=params, timeout=timeout_seconds)
        params = None
        resp.raise_for_status()

        payload = resp.json()
        results = payload.get("results", [])
        if isinstance(results, list):
            for row in results:
                details = row.get("details", {}) if isinstance(row, dict) else {}
                quote = row.get("last_quote", {}) if isinstance(row, dict) else {}
                trade = row.get("last_trade", {}) if isinstance(row, dict) else {}
                greeks = row.get("greeks", {}) if isinstance(row, dict) else {}
                day = row.get("day", {}) if isinstance(row, dict) else {}
                under = row.get("underlying_asset", {}) if isinstance(row, dict) else {}

                bid = quote.get("bid_price")
                ask = quote.get("ask_price")
                mid = None
                if bid is not None and ask is not None:
                    try:
                        mid = (float(bid) + float(ask)) / 2.0
                    except Exception:
                        mid = None

                chain_rows.append(
                    {
                        "date": asof.isoformat(),
                        "expiry": details.get("expiration_date"),
                        "call_put": str(details.get("contract_type", "")).upper()[:1],
                        "symbol": details.get("ticker"),
                        "strike": details.get("strike_price"),
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "last": trade.get("price"),
                        "volume": day.get("volume"),
                        "open_interest": row.get("open_interest"),
                        "underlying_close": under.get("price"),
                        "delta": greeks.get("delta"),
                        "gamma": greeks.get("gamma"),
                        "theta": greeks.get("theta"),
                        "vega": greeks.get("vega"),
                        "iv": row.get("implied_volatility"),
                    }
                )

        page_count += 1
        if max_pages > 0 and page_count >= max_pages:
            break
        next_url = payload.get("next_url")
        if next_url and "apiKey=" not in str(next_url):
            sep = "&" if "?" in str(next_url) else "?"
            next_url = f"{next_url}{sep}apiKey={api_key}"
        url = next_url

    norm = normalize_chain_df(pd.DataFrame(chain_rows), asof=asof, symbol=symbol) if chain_rows else pd.DataFrame()
    rows_saved = int(len(norm))
    if rows_saved > 0:
        norm.to_parquet(day_path, index=False)

    meta = {
        "symbol": symbol,
        "as_of": asof.isoformat(),
        "source": "massive_v3_snapshot_options",
        "base_url": base_url,
        "rows_raw": int(len(chain_rows)),
        "rows_saved": rows_saved,
        "pages": page_count,
        "path": str(day_path.resolve()) if rows_saved > 0 else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return bool(rows_saved > 0)


def _ensure_market_data(
    *,
    symbol: str,
    asof: date,
    cfg: LiveWalkforwardConfig,
    api_key: str | None,
) -> dict[str, Any]:
    options_path = _options_day_path(cfg.data_root, symbol, asof)
    under_path = _underlying_path(cfg.data_root, symbol)

    had_options = options_path.exists()
    had_underlying = _has_underlying_date(under_path, asof)
    pulled_options = False
    pulled_underlying = False
    errors: list[str] = []

    if api_key and not had_options:
        try:
            pulled_options = _pull_options_day(
                symbol=symbol,
                asof=asof,
                data_root=cfg.data_root,
                api_key=api_key,
                base_url=cfg.base_url,
                timeout_seconds=cfg.timeout_seconds,
                options_limit=cfg.options_limit,
                max_pages=cfg.max_pages,
            )
        except Exception as exc:
            errors.append(f"options pull failed: {exc}")

    if api_key and not had_underlying:
        try:
            pulled_underlying = _pull_underlying_day(
                symbol=symbol,
                asof=asof,
                data_root=cfg.data_root,
                api_key=api_key,
                base_url=cfg.base_url,
                timeout_seconds=cfg.timeout_seconds,
                max_pages=cfg.max_pages,
            )
        except Exception as exc:
            errors.append(f"underlying pull failed: {exc}")

    if not api_key and (not had_options or not had_underlying):
        errors.append("market data missing and MASSIVE_API_KEY/POLYGON_API_KEY not set")

    options_present = options_path.exists()
    underlying_present = _has_underlying_date(under_path, asof)
    return {
        "options_present": bool(options_present),
        "underlying_present": bool(underlying_present),
        "pulled_options": bool(pulled_options),
        "pulled_underlying": bool(pulled_underlying),
        "errors": errors,
        "options_path": str(options_path.resolve()),
        "underlying_path": str(under_path.resolve()),
    }


def _coerce_x_grid(v: Any) -> tuple[float, ...]:
    if not isinstance(v, list) or len(v) < 2:
        return _DEFAULT_X_GRID
    out: list[float] = []
    for x in v:
        try:
            out.append(float(x))
        except Exception:
            continue
    return tuple(out) if len(out) >= 2 else _DEFAULT_X_GRID


def _coerce_tenors(v: Any) -> tuple[int, ...]:
    if not isinstance(v, list) or len(v) < 2:
        return _DEFAULT_TENOR_DAYS
    out: list[int] = []
    for x in v:
        try:
            out.append(int(x))
        except Exception:
            continue
    return tuple(out) if len(out) >= 2 else _DEFAULT_TENOR_DAYS


def _load_training_grid(run_dir: Path) -> tuple[tuple[float, ...], tuple[int, ...]]:
    train_summary = _safe_json(run_dir / "train_summary.json")
    dataset_path_raw = train_summary.get("dataset_path")
    if not dataset_path_raw:
        return _DEFAULT_X_GRID, _DEFAULT_TENOR_DAYS
    dataset_path = Path(str(dataset_path_raw)).expanduser()
    meta = _safe_json(dataset_path.parent / "dataset_meta.json")
    return _coerce_x_grid(meta.get("x_grid")), _coerce_tenors(meta.get("tenor_days"))


def _ensure_live_dataset(
    *,
    symbol: str,
    run_dir: Path,
    asof: date,
    cfg: LiveWalkforwardConfig,
    api_key: str | None,
) -> Path:
    dataset_dir = cfg.output_root / symbol / "dataset"
    dataset_path = dataset_dir / "dataset.npz"
    meta_path = dataset_dir / "dataset_meta.json"
    meta = _safe_json(meta_path)
    if (
        dataset_path.exists()
        and not cfg.force_run
        and str(meta.get("date_max", "")) == asof.isoformat()
        and int(meta.get("rows_dates", 0)) >= 2
    ):
        return dataset_path

    lookback = max(int(cfg.lookback_days), 35)
    start_date = (asof - timedelta(days=lookback)).isoformat()
    x_grid, tenor_days = _load_training_grid(run_dir)
    out = build_dataset(
        DatasetBuildConfig(
            data_root=cfg.data_root,
            out_dir=dataset_dir,
            symbol=symbol,
            plugin="massive_raw_parquet",
            api_key=api_key,
            start_date=start_date,
            end_date=asof.isoformat(),
            x_grid=x_grid,
            tenor_days=tenor_days,
            max_contracts_per_day=900,
            random_seed=7,
            num_workers=0,
        )
    )
    path = Path(out["dataset"]).resolve()
    built_meta = _safe_json(path.parent / "dataset_meta.json")
    if int(built_meta.get("rows_dates", 0)) < 2:
        raise RuntimeError(
            f"Live dataset has too few dates for walk-forward (rows_dates={built_meta.get('rows_dates', 0)})."
        )
    return path


def _append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        hist = pd.read_csv(path)
    else:
        hist = pd.DataFrame()
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    if "session_dir" in hist.columns:
        hist = hist.drop_duplicates(subset=["session_dir"], keep="last")
    if "session_utc" in hist.columns:
        hist = hist.sort_values("session_utc").reset_index(drop=True)
    hist.to_csv(path, index=False)


def _load_state(path: Path) -> dict[str, Any]:
    return _safe_json(path)


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_session_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = root / f"wf_{utc_timestamp()}"
    if not base.exists():
        return base
    i = 2
    while True:
        cand = root / f"{base.name}_{i}"
        if not cand.exists():
            return cand
        i += 1


def _result(
    *,
    symbol: str,
    asof: date,
    status: str,
    message: str,
    session_dir: Path | None = None,
    last_week_pnl: float | None = None,
) -> LiveWalkforwardResult:
    return LiveWalkforwardResult(
        symbol=symbol,
        asof_date=asof.isoformat(),
        status=status,
        message=message,
        session_dir=str(session_dir.resolve()) if session_dir is not None else None,
        last_week_pnl=last_week_pnl,
    )


def run_live_walkforward_for_symbol(
    *,
    symbol: str,
    asof: date,
    cfg: LiveWalkforwardConfig,
) -> LiveWalkforwardResult:
    symbol = symbol.upper().strip()
    if not symbol:
        return _result(symbol="UNKNOWN", asof=asof, status="skipped", message="empty symbol")
    if asof.weekday() >= 5:
        return _result(symbol=symbol, asof=asof, status="skipped", message="weekend (no market session)")

    run_dir = _latest_run_for_symbol(symbol, outputs_root=Path("outputs"))
    if run_dir is None:
        return _result(symbol=symbol, asof=asof, status="skipped", message="no trained run found")
    if not (run_dir / "model.pt").exists():
        return _result(symbol=symbol, asof=asof, status="skipped", message=f"model.pt missing in {run_dir}")

    symbol_root = cfg.output_root / symbol
    state_path = symbol_root / "state.json"
    state = _load_state(state_path)
    if str(state.get("last_asof_date", "")) == asof.isoformat() and not cfg.force_run:
        return _result(symbol=symbol, asof=asof, status="skipped", message="already processed for this date")

    api_key = _resolve_api_key()
    market = _ensure_market_data(symbol=symbol, asof=asof, cfg=cfg, api_key=api_key)
    if not bool(market.get("options_present", False)):
        err = "; ".join(market.get("errors", [])) or "options data unavailable"
        return _result(symbol=symbol, asof=asof, status="skipped", message=err)

    dataset_path = _ensure_live_dataset(symbol=symbol, run_dir=run_dir, asof=asof, cfg=cfg, api_key=api_key)
    session_dir = _make_session_dir(symbol_root / "sessions")
    bt_dir = session_dir / "backtest"
    bt_dir.mkdir(parents=True, exist_ok=True)

    trade_window = max(int(cfg.trade_window_days), int(cfg.week_days) + 2, 7)
    start_date = (asof - timedelta(days=trade_window)).isoformat()
    end_date = asof.isoformat()
    run_backtest(
        BacktestConfig(
            run_dir=run_dir,
            dataset_path=dataset_path,
            output_dir=bt_dir,
            start_date=start_date,
            end_date=end_date,
            num_workers=0,
        )
    )

    summary = _safe_json(bt_dir / "summary.json")
    daily = _read_parquet_or_csv(bt_dir / "daily.parquet")
    if "date" in daily.columns:
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        daily = daily.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    pnl_series = pd.to_numeric(daily.get("pnl"), errors="coerce").fillna(0.0)
    trade_series = pd.to_numeric(daily.get("trades"), errors="coerce").fillna(0.0)
    last_week = daily.tail(max(int(cfg.week_days), 1))
    last_week_pnl = float(pd.to_numeric(last_week.get("pnl"), errors="coerce").fillna(0.0).sum())
    last_week_trades = int(pd.to_numeric(last_week.get("trades"), errors="coerce").fillna(0.0).sum())
    last_trade_date = (
        pd.to_datetime(last_week["date"], errors="coerce").dt.date.max().isoformat()
        if not last_week.empty and "date" in last_week.columns
        else None
    )

    session_payload = {
        "symbol": symbol,
        "asof_date": asof.isoformat(),
        "session_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_run_dir": str(run_dir.resolve()),
        "live_dataset_path": str(dataset_path.resolve()),
        "backtest_dir": str(bt_dir.resolve()),
        "last_week_pnl": last_week_pnl,
        "last_week_trades": last_week_trades,
        "last_trade_date": last_trade_date,
        "total_pnl_window": float(pnl_series.sum()),
        "total_trades_window": int(trade_series.sum()),
        "summary_total_pnl": summary.get("total_pnl"),
        "summary_trades": summary.get("trades"),
        "market_data": market,
    }
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "metadata.json").write_text(json.dumps(session_payload, indent=2), encoding="utf-8")
    (symbol_root / "latest.txt").write_text(str(session_dir.resolve()), encoding="utf-8")

    history_row = {
        "session_utc": session_payload["session_utc"],
        "asof_date": session_payload["asof_date"],
        "symbol": symbol,
        "session_dir": str(session_dir.resolve()),
        "model_run_dir": str(run_dir.resolve()),
        "live_dataset_path": str(dataset_path.resolve()),
        "last_trade_date": last_trade_date,
        "last_week_pnl": last_week_pnl,
        "last_week_trades": last_week_trades,
        "total_pnl_window": session_payload["total_pnl_window"],
        "total_trades_window": session_payload["total_trades_window"],
        "options_present": bool(market.get("options_present", False)),
        "underlying_present": bool(market.get("underlying_present", False)),
    }
    _append_history(symbol_root / "history.csv", history_row)
    _append_history(cfg.output_root / "history.csv", history_row)
    _write_state(
        state_path,
        {
            "last_asof_date": asof.isoformat(),
            "last_session_dir": str(session_dir.resolve()),
            "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    )

    return _result(
        symbol=symbol,
        asof=asof,
        status="ran",
        message=f"live walk-forward updated (last_week_pnl={last_week_pnl:.2f})",
        session_dir=session_dir,
        last_week_pnl=last_week_pnl,
    )


def load_live_walkforward_config_from_env() -> LiveWalkforwardConfig:
    return LiveWalkforwardConfig(
        data_root=Path(os.environ.get("IVDYN_LIVE_DATA_ROOT", "data")).expanduser(),
        output_root=Path(os.environ.get("IVDYN_LIVE_OUTPUT_ROOT", "outputs/live_walkforward")).expanduser(),
        lookback_days=_env_int("IVDYN_LIVE_WF_LOOKBACK_DAYS", 120, minimum=21),
        trade_window_days=_env_int("IVDYN_LIVE_WF_TRADE_WINDOW_DAYS", 14, minimum=7),
        week_days=_env_int("IVDYN_LIVE_WF_WEEK_DAYS", 5, minimum=1),
        timeout_seconds=_env_int("IVDYN_LIVE_TIMEOUT_SECONDS", 30, minimum=5),
        options_limit=_env_int("IVDYN_LIVE_OPTIONS_LIMIT", 250, minimum=1),
        max_pages=_env_int("IVDYN_LIVE_MAX_PAGES", 0, minimum=0),
        base_url=str(os.environ.get("IVDYN_LIVE_BASE_URL", "https://api.massive.com")).strip(),
        force_run=_env_flag("IVDYN_LIVE_FORCE_RUN", False),
    )


def maybe_run_live_walkforward(
    ns: Any,
    *,
    logger: Callable[[str], None] | None = None,
) -> list[LiveWalkforwardResult]:
    if not _env_flag("IVDYN_ENABLE_LIVE_WALKFORWARD", True):
        return []

    outputs_root = Path("outputs")
    symbols = _discover_symbols(ns, outputs_root=outputs_root)
    if not symbols:
        return []

    cfg = load_live_walkforward_config_from_env()
    asof = _market_today_ny()
    results: list[LiveWalkforwardResult] = []
    for symbol in symbols:
        try:
            result = run_live_walkforward_for_symbol(symbol=symbol, asof=asof, cfg=cfg)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            result = LiveWalkforwardResult(
                symbol=symbol,
                asof_date=asof.isoformat(),
                status="failed",
                message=str(exc),
            )
        results.append(result)
        if logger is not None:
            logger(f"[live-wf] {result.symbol} [{result.status}] {result.message}")
    return results
