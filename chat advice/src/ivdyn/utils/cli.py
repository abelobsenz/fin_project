"""Command-line interface for ivdyn."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import date, timedelta
from pathlib import Path
import os
import subprocess
import sys
from typing import Any, Sequence

from ivdyn.utils.paths import resolve_latest, utc_timestamp


def _to_path(v: str) -> Path:
    return Path(v).expanduser()


def _autoload_dotenv() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [Path.cwd() / ".env", repo_root / ".env"]
    seen: set[Path] = set()
    for candidate in candidates:
        p = candidate.resolve()
        if p in seen or not p.exists() or not p.is_file():
            continue
        seen.add(p)

        for raw_line in p.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue

            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                q = value[0]
                value = value[1:-1]
                if q == '"':
                    value = value.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
            else:
                comment_pos = value.find(" #")
                if comment_pos >= 0:
                    value = value[:comment_pos].rstrip()

            os.environ.setdefault(key, value)


def _resolve_run_dir(raw: str | None) -> Path:
    if raw:
        return _to_path(raw).resolve()
    latest = resolve_latest(Path("outputs/runs"))
    if latest is None:
        raise RuntimeError("No run directory found. Provide --run-dir explicitly.")
    return latest


def _resolve_dataset(dataset_arg: str | None, run_dir: Path) -> Path:
    if dataset_arg:
        return _to_path(dataset_arg).resolve()
    train_summary = run_dir / "train_summary.json"
    if not train_summary.exists():
        raise RuntimeError("No dataset argument and no train_summary.json found in run dir.")
    import json

    payload = json.loads(train_summary.read_text(encoding="utf-8"))
    if not (dataset := payload.get("dataset_path")):
        raise RuntimeError("train_summary.json exists but missing dataset_path.")
    return _to_path(str(dataset)).resolve()


def _train_command(ns: Any) -> None:
    from ivdyn.training import TrainingConfig, train

    run_dir = train(
        _to_path(ns.dataset).resolve(),
        TrainingConfig(
            out_dir=_to_path(ns.out_dir),
            seed=ns.seed,
            train_frac=ns.train_frac,
            val_frac=ns.val_frac,
            latent_dim=ns.latent_dim,
            vae_epochs=ns.vae_epochs,
            vae_batch_size=ns.vae_batch_size,
            vae_lr=ns.vae_lr,
            vae_kl_beta=ns.vae_kl_beta,
            noarb_lambda=ns.noarb_lambda,
            head_epochs=ns.head_epochs,
            dyn_batch_size=ns.dyn_batch_size,
            contract_batch_size=ns.contract_batch_size,
            head_lr=ns.head_lr,
            joint_epochs=ns.joint_epochs,
            joint_lr=ns.joint_lr,
            joint_contract_batch_size=ns.joint_contract_batch_size,
            joint_dyn_lambda=ns.joint_dyn_lambda,
            joint_price_lambda=ns.joint_price_lambda,
            joint_exec_lambda=ns.joint_exec_lambda,
            weight_decay=ns.weight_decay,
            price_risk_weight=ns.price_risk_weight,
            exec_risk_weight=ns.exec_risk_weight,
            risk_focus_abs_x=ns.risk_focus_abs_x,
            risk_focus_tau_days=ns.risk_focus_tau_days,
            exec_label_smoothing=ns.exec_label_smoothing,
            exec_logit_l2=ns.exec_logit_l2,
        ),
    )
    print(run_dir)


def _build_dataset_command(ns: Any) -> None:
    from ivdyn.data import DatasetBuildConfig, build_dataset

    out = build_dataset(
        DatasetBuildConfig(
            data_root=_to_path(ns.data_root),
            out_dir=_to_path(ns.out_dir),
            symbol=ns.symbol,
            plugin=ns.plugin,
            api_key=ns.api_key,
            start_date=ns.start_date,
            end_date=ns.end_date,
            x_grid=tuple(ns.x_grid),
            tenor_days=tuple(ns.tenor_days),
            max_contracts_per_day=ns.max_contracts_per_day,
            random_seed=ns.random_seed,
            num_workers=ns.num_workers,
        )
    )
    print(out["dataset"])


def _evaluate_command(ns: Any) -> None:
    from ivdyn.eval import evaluate

    run_dir = _resolve_run_dir(ns.run_dir)
    dataset = _resolve_dataset(ns.dataset, run_dir)
    out_dir = evaluate(
        run_dir=run_dir,
        dataset_path=dataset,
        device=ns.device,
        num_workers=ns.num_workers,
    )
    print(out_dir)


def _backtest_command(ns: Any) -> None:
    try:
        from ivdyn.backtest import BacktestConfig, run_backtest
    except Exception as exc:
        raise RuntimeError(
            "Backtest support is not available in this checkout. Reinstall from a build "
            "that includes ivdyn.backtest."
        ) from exc

    run_dir = _resolve_run_dir(ns.run_dir)
    dataset = _resolve_dataset(ns.dataset, run_dir)
    out_dir = run_backtest(
        BacktestConfig(
            run_dir=run_dir,
            dataset_path=dataset,
            start_date=getattr(ns, "start_date", None),
            end_date=getattr(ns, "end_date", None),
            device=ns.device,
            num_workers=ns.num_workers,
            inference_batch_size=ns.inference_batch_size,
            initial_capital=ns.initial_capital,
            fill_gate=ns.fill_gate,
            fill_model=getattr(ns, "fill_model", "expected"),
            slippage_bps=ns.slippage_bps,
            spread_cross_fraction=getattr(ns, "spread_cross_fraction", 0.75),
            option_commission_per_contract=getattr(ns, "option_commission_per_contract", 0.65),
            option_fee_per_contract=getattr(ns, "option_fee_per_contract", 0.05),
            min_edge_to_cost_ratio=getattr(ns, "min_edge_to_cost_ratio", 1.2),
            max_trades_per_day=ns.max_trades_per_day,
            max_contracts_per_trade=getattr(ns, "max_contracts_per_trade", 4),
            volume_participation_rate=getattr(ns, "volume_participation_rate", 0.01),
            open_interest_participation_rate=getattr(ns, "open_interest_participation_rate", 0.01),
            selector_long_score_scale=getattr(ns, "long_score_scale", 0.0),
            selector_allow_long_puts=bool(getattr(ns, "allow_long_puts", True)),
            signal_abs_gate=ns.signal_abs_gate,
            min_dte=ns.min_dte,
            max_dte=ns.max_dte,
            min_moneyness=ns.min_moneyness,
            max_moneyness=ns.max_moneyness,
            max_rel_spread=ns.max_rel_spread,
            strategy_mode=getattr(ns, "strategy_mode", "vertical"),
            vertical_wing_width_pct_target=getattr(ns, "vertical_wing_width_pct_target", 0.03),
            vertical_wing_width_pct_min=getattr(ns, "vertical_wing_width_pct_min", 0.01),
            vertical_wing_width_pct_max=getattr(ns, "vertical_wing_width_pct_max", 0.08),
            vertical_wing_max_premium_ratio=getattr(ns, "vertical_wing_max_premium_ratio", 0.35),
            vertical_wing_fill_gate=getattr(ns, "vertical_wing_fill_gate", 0.50),
            vertical_wing_max_rel_spread=getattr(ns, "vertical_wing_max_rel_spread", 0.15),
            vertical_wing_min_moneyness=getattr(ns, "vertical_wing_min_moneyness", 0.75),
            vertical_wing_max_moneyness=getattr(ns, "vertical_wing_max_moneyness", 1.30),
            vertical_wing_rich_signal_penalty=getattr(ns, "vertical_wing_rich_signal_penalty", 0.75),
            vertical_skip_if_no_wing=bool(getattr(ns, "vertical_skip_if_no_wing", True)),
            hedge_underlying_delta=bool(getattr(ns, "hedge_underlying_delta", False)),
            hedge_underlying_ratio=getattr(ns, "hedge_underlying_ratio", 1.0),
            hedge_underlying_min_abs_shares=getattr(ns, "hedge_underlying_min_abs_shares", 25.0),
            hedge_underlying_max_shares=getattr(ns, "hedge_underlying_max_shares", 5000),
            hedge_underlying_slippage_bps=getattr(ns, "hedge_underlying_slippage_bps", 1.0),
            hedge_policy=getattr(ns, "hedge_policy", "fixed"),
            hedge_policy_path=getattr(ns, "hedge_policy_path", None),
            enforce_portfolio_constraints=bool(getattr(ns, "enforce_portfolio_constraints", True)),
            buying_power_leverage=getattr(ns, "buying_power_leverage", 1.0),
            option_short_margin_rate=getattr(ns, "option_short_margin_rate", 0.20),
            underlying_margin_rate=getattr(ns, "underlying_margin_rate", 0.50),
        )
    )
    print(out_dir)


def _train_hedge_policy_command(ns: Any) -> None:
    """Train a state-dependent underlying hedge ratio policy.

    This uses an existing backtest run under <run_dir>/backtest to build daily
    hedge episodes (options PnL and net option delta) and then learns an MLP
    mapping model state -> hedge ratio.
    """
    from ivdyn.hedge_policy import HedgePolicyTrainConfig, train_hedge_policy

    run_dir = _resolve_run_dir(ns.run_dir)
    dataset = _resolve_dataset(ns.dataset, run_dir)
    out_dir = train_hedge_policy(
        HedgePolicyTrainConfig(
            run_dir=run_dir,
            dataset_path=dataset,
            out_dir=_to_path(ns.out_dir).resolve() if ns.out_dir else None,
            device=ns.device,
            hidden_dim=ns.hidden_dim,
            depth=ns.depth,
            max_ratio=ns.max_ratio,
            epochs=ns.epochs,
            lr=ns.lr,
            weight_decay=ns.weight_decay,
            train_frac=ns.train_frac,
            seed=ns.seed,
            risk_aversion=ns.risk_aversion,
            underlying_slippage_bps=ns.underlying_slippage_bps,
            min_abs_shares=ns.min_abs_shares,
            max_shares=ns.max_shares,
        )
    )
    print(out_dir)


def _ui_command(ns: Any) -> None:
    env = os.environ.copy()
    if ns.run_dir:
        env["IVDYN_DEFAULT_RUN_DIR"] = _to_path(ns.run_dir).resolve().as_posix()
    app_path = Path(__file__).resolve().parent.parent / "ui" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    rc = subprocess.call(cmd, env=env)
    raise SystemExit(rc)


def _ui2_command(ns: Any) -> None:
    env = os.environ.copy()
    if getattr(ns, "symbol", None):
        env["IVDYN_LIVE_DEFAULT_SYMBOL"] = str(ns.symbol).upper().strip()
    app_path = Path(__file__).resolve().parent.parent / "ui" / "live_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    rc = subprocess.call(cmd, env=env)
    raise SystemExit(rc)


def _wf_command(ns: Any) -> None:
    from ivdyn.live import maybe_run_live_walkforward

    def _log(msg: str) -> None:
        print(msg, file=sys.stderr)

    prev_symbols = os.environ.get("IVDYN_LIVE_SYMBOLS")
    prev_force = os.environ.get("IVDYN_LIVE_FORCE_RUN")
    try:
        if getattr(ns, "symbol", None):
            os.environ["IVDYN_LIVE_SYMBOLS"] = str(ns.symbol).upper().strip()
        if bool(getattr(ns, "force", False)):
            os.environ["IVDYN_LIVE_FORCE_RUN"] = "1"

        results = maybe_run_live_walkforward(ns, logger=_log)
    finally:
        if prev_symbols is None:
            os.environ.pop("IVDYN_LIVE_SYMBOLS", None)
        else:
            os.environ["IVDYN_LIVE_SYMBOLS"] = prev_symbols
        if prev_force is None:
            os.environ.pop("IVDYN_LIVE_FORCE_RUN", None)
        else:
            os.environ["IVDYN_LIVE_FORCE_RUN"] = prev_force

    if not results:
        print("No symbols resolved for walk-forward run.")
        return

    failed = 0
    for row in results:
        print(f"{row.symbol}\t{row.status}\t{row.message}")
        if row.status == "failed":
            failed += 1
    if failed > 0:
        raise SystemExit(1)


def _resolve_massive_api_key(explicit: str | None) -> str:
    api_key = explicit or os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set MASSIVE_API_KEY (or POLYGON_API_KEY) or pass --api-key.")
    return api_key


def _normalize_flatfiles_prefix(bucket: str, prefix: str) -> str:
    p = str(prefix).strip().lstrip("/")
    b = str(bucket).strip().strip("/")
    if p.startswith(f"{b}/"):
        p = p[len(b) + 1 :]
    return p.strip("/")


def _resolve_flatfiles_credentials(ns: Any) -> dict[str, str | None]:
    access_key = ns.access_key or os.environ.get("MASSIVE_FLATFILES_ACCESS_KEY") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = ns.secret_key or os.environ.get("MASSIVE_FLATFILES_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    session_token = ns.session_token or os.environ.get("AWS_SESSION_TOKEN")

    if not access_key or not secret_key:
        raise RuntimeError(
            "Missing flatfiles credentials. Set MASSIVE_FLATFILES_ACCESS_KEY and "
            "MASSIVE_FLATFILES_SECRET_ACCESS_KEY (or pass --access-key / --secret-key)."
        )

    endpoint_url = (
        ns.endpoint_url
        or os.environ.get("MASSIVE_FLATFILES_ENDPOINT_URL")
        or "https://files.massive.com"
    )
    bucket = ns.bucket or os.environ.get("MASSIVE_FLATFILES_BUCKET") or "flatfiles"
    prefix = ns.prefix or os.environ.get("MASSIVE_FLATFILES_PREFIX") or "us_options_opra/day_aggs_v1"
    prefix = _normalize_flatfiles_prefix(bucket=bucket, prefix=prefix)

    return {
        "access_key": access_key,
        "secret_key": secret_key,
        "session_token": session_token,
        "endpoint_url": endpoint_url,
        "bucket": bucket,
        "prefix": prefix,
    }


def _flatfile_key(prefix: str, asof: date) -> str:
    return f"{prefix}/{asof.year:04d}/{asof.month:02d}/{asof.isoformat()}.csv.gz"


def _pull_flatfiles_command(ns: Any) -> None:
    import json

    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError

    cfg = _resolve_flatfiles_credentials(ns)
    data_root = _to_path(ns.data_root)
    out_root = data_root / "options_source"
    out_root.mkdir(parents=True, exist_ok=True)

    session = boto3.Session(
        aws_access_key_id=str(cfg["access_key"]),
        aws_secret_access_key=str(cfg["secret_key"]),
        aws_session_token=str(cfg["session_token"]) if cfg["session_token"] else None,
    )
    s3 = session.client(
        "s3",
        endpoint_url=str(cfg["endpoint_url"]),
        config=Config(signature_version="s3v4"),
    )

    all_days = _iter_weekdays(ns.start_date, ns.end_date)
    if ns.max_days > 0:
        all_days = all_days[: ns.max_days]

    downloaded = 0
    skipped_existing = 0
    missing_or_error = 0
    failures: list[dict[str, str]] = []

    prefix = str(cfg["prefix"])
    bucket = str(cfg["bucket"])
    for asof in all_days:
        rel_key = _flatfile_key(prefix=prefix, asof=asof)
        local_path = out_root / rel_key
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists() and not ns.overwrite:
            skipped_existing += 1
            continue

        try:
            s3.download_file(bucket, rel_key, str(local_path))
            downloaded += 1
        except (ClientError, BotoCoreError, OSError) as exc:
            missing_or_error += 1
            failures.append(
                {
                    "date": asof.isoformat(),
                    "key": rel_key,
                    "error": str(exc),
                }
            )
            if ns.fail_fast:
                raise RuntimeError(f"Failed to download {rel_key}: {exc}") from exc

    summary = {
        "source": "massive_flatfiles_s3",
        "endpoint_url": cfg["endpoint_url"],
        "bucket": bucket,
        "prefix": prefix,
        "start_date": ns.start_date,
        "end_date": ns.end_date,
        "trading_days_considered": len(all_days),
        "downloaded": downloaded,
        "skipped_existing": skipped_existing,
        "missing_or_error": missing_or_error,
        "output_root": str(out_root.resolve()),
        "failures_sample": failures[:25],
    }
    summary_path = out_root / f"_flatfile_pull_summary_{utc_timestamp()}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


def _pull_options_symbol_command(ns: Any) -> None:
    import json

    import pandas as pd

    from ivdyn.data.massive import parse_opra_ticker
    from ivdyn.data.schemas import normalize_chain_df

    symbol = str(ns.symbol).upper().strip()
    if not symbol:
        raise RuntimeError("Symbol is required.")

    data_root = _to_path(ns.data_root)
    if ns.source_root:
        source_root = _to_path(ns.source_root)
    else:
        source_prefix = ns.source_prefix or os.environ.get("MASSIVE_FLATFILES_PREFIX") or "us_options_opra/day_aggs_v1"
        source_prefix = _normalize_flatfiles_prefix(bucket="flatfiles", prefix=source_prefix)
        source_root = data_root / "options_source" / source_prefix

    out_raw_root = data_root / "symbols" / symbol / "options" / "raw"
    out_raw_root.mkdir(parents=True, exist_ok=True)

    underlying_path = data_root / "symbols" / symbol / "underlying" / f"{symbol.lower()}_eod.parquet"
    spot_lookup: dict[date, float] = {}
    if underlying_path.exists():
        udf = pd.read_parquet(underlying_path)
        if "date" in udf.columns and "close" in udf.columns:
            udf = udf.copy()
            udf["date"] = pd.to_datetime(udf["date"], errors="coerce").dt.date
            close = pd.to_numeric(udf["close"], errors="coerce")
            mask = udf["date"].notna() & close.notna()
            spot_lookup = dict(zip(udf.loc[mask, "date"], close.loc[mask], strict=False))
    if not spot_lookup and not ns.allow_missing_underlying:
        raise RuntimeError(
            f"No usable underlying closes found at {underlying_path}. "
            "Run pull-underlying-massive first or pass --allow-missing-underlying."
        )

    all_days = _iter_weekdays(ns.start_date, ns.end_date)
    if ns.max_days > 0:
        all_days = all_days[: ns.max_days]

    saved_days = 0
    skipped_existing = 0
    missing_source = 0
    zero_rows = 0
    total_rows_saved = 0

    for asof in all_days:
        day_str = asof.isoformat()
        source_path = source_root / f"{asof.year:04d}" / f"{asof.month:02d}" / f"{day_str}.csv.gz"
        out_path = out_raw_root / f"{day_str}.parquet"
        meta_path = out_raw_root / f"{day_str}.metadata.json"

        if out_path.exists() and not ns.overwrite:
            skipped_existing += 1
            continue

        if not source_path.exists():
            missing_source += 1
            meta = {
                "symbol": symbol,
                "as_of": day_str,
                "source_path": str(source_path),
                "rows_source": 0,
                "rows_symbol_raw": 0,
                "rows_saved": 0,
                "missing_source": True,
                "path": None,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            continue

        raw = pd.read_csv(source_path, compression="gzip")
        if "ticker" not in raw.columns:
            meta = {
                "symbol": symbol,
                "as_of": day_str,
                "source_path": str(source_path),
                "rows_source": int(len(raw)),
                "rows_symbol_raw": 0,
                "rows_saved": 0,
                "missing_source": False,
                "error": "missing ticker column",
                "path": None,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            zero_rows += 1
            continue

        ticker = raw["ticker"].astype(str)
        scoped = raw.loc[ticker.str.startswith(f"O:{symbol}", na=False)].copy()
        if scoped.empty:
            rows_saved = 0
            norm = pd.DataFrame()
            rows_symbol_raw = 0
        else:
            parsed = scoped["ticker"].astype(str).apply(parse_opra_ticker)
            mask = parsed.notna()
            scoped = scoped.loc[mask].copy()
            parsed = parsed.loc[mask]

            if not scoped.empty:
                under = parsed.apply(lambda x: x["underlying"])  # type: ignore[index]
                scoped = scoped.loc[under == symbol].copy()
                parsed = parsed.loc[scoped.index]

            if scoped.empty:
                rows_saved = 0
                norm = pd.DataFrame()
                rows_symbol_raw = 0
            else:
                expiry = parsed.apply(lambda x: x["expiry"])  # type: ignore[index]
                dte = parsed.apply(lambda x: (x["expiry"] - asof).days)  # type: ignore[index]
                call_put = parsed.apply(lambda x: x["call_put"])  # type: ignore[index]
                strike = parsed.apply(lambda x: x["strike"])  # type: ignore[index]

                mid = pd.to_numeric(scoped.get("close"), errors="coerce")
                volume = pd.to_numeric(scoped.get("volume"), errors="coerce")
                open_interest = pd.to_numeric(scoped.get("open_interest"), errors="coerce")
                spot = float(spot_lookup.get(asof, float("nan")))

                day_df = pd.DataFrame(
                    {
                        "date": asof,
                        "expiry": expiry,
                        "dte": dte,
                        "call_put": call_put,
                        "symbol": scoped["ticker"].astype(str),
                        "strike": strike,
                        "bid": pd.NA,
                        "ask": pd.NA,
                        "mid": mid,
                        "last": mid,
                        "volume": volume,
                        "open_interest": open_interest,
                        "underlying_close": spot,
                        "delta": pd.NA,
                        "gamma": pd.NA,
                        "theta": pd.NA,
                        "vega": pd.NA,
                        "iv": pd.NA,
                    }
                )
                rows_symbol_raw = int(len(day_df))
                norm = normalize_chain_df(day_df, asof=asof, symbol=symbol)
                rows_saved = int(len(norm))

        if rows_saved > 0:
            norm.to_parquet(out_path, index=False)
            saved_days += 1
            total_rows_saved += rows_saved
        else:
            if out_path.exists() and ns.overwrite:
                out_path.unlink()
            zero_rows += 1

        meta = {
            "symbol": symbol,
            "as_of": day_str,
            "source_path": str(source_path),
            "rows_source": int(len(raw)),
            "rows_symbol_raw": int(rows_symbol_raw),
            "rows_saved": int(rows_saved),
            "missing_source": False,
            "path": str(out_path.resolve()) if rows_saved > 0 else None,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    summary = {
        "symbol": symbol,
        "start_date": ns.start_date,
        "end_date": ns.end_date,
        "source_root": str(source_root),
        "output_root": str(out_raw_root),
        "trading_days_considered": len(all_days),
        "saved_days": saved_days,
        "skipped_existing_days": skipped_existing,
        "missing_source_days": missing_source,
        "zero_row_days": zero_rows,
        "rows_saved_total": total_rows_saved,
    }
    summary_path = out_raw_root / f"_symbol_pull_summary_{utc_timestamp()}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


def _pull_underlying_massive_command(ns: Any) -> None:
    import json

    import pandas as pd
    import requests

    symbol = str(ns.symbol).upper().strip()
    if not symbol:
        raise RuntimeError("Symbol is required.")

    api_key = _resolve_massive_api_key(ns.api_key)
    base_url = str(ns.base_url).rstrip("/")

    out_dir = _to_path(ns.data_root) / "symbols" / symbol / "underlying"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol.lower()}_eod.parquet"
    meta_path = out_dir / f"{symbol.lower()}_eod.metadata.json"

    url = f"{base_url}/v2/aggs/ticker/{symbol}/range/1/day/{ns.start_date}/{ns.end_date}"
    params: dict[str, Any] | None = {
        "adjusted": "true" if ns.adjusted else "false",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }

    rows: list[dict[str, Any]] = []
    pages = 0
    while url:
        resp = requests.get(url, params=params, timeout=ns.timeout_seconds)
        params = None
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            snippet = resp.text[:500]
            raise RuntimeError(f"Massive request failed ({resp.status_code}): {snippet}") from exc

        payload = resp.json()
        batch = payload.get("results", [])
        if isinstance(batch, list):
            rows.extend(batch)

        pages += 1
        if ns.max_pages > 0 and pages >= ns.max_pages:
            break

        next_url = payload.get("next_url")
        if next_url and "apiKey=" not in str(next_url):
            sep = "&" if "?" in str(next_url) else "?"
            next_url = f"{next_url}{sep}apiKey={api_key}"
        url = next_url

    if not rows:
        raise RuntimeError(
            "No underlying bars returned for the requested range. "
            "Check symbol/date range and API permissions."
        )

    raw = pd.DataFrame(rows)
    if "t" not in raw.columns or "c" not in raw.columns:
        raise RuntimeError("Unexpected Massive response schema: expected fields 't' and 'c'.")

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
    )
    out = out.dropna(subset=["date", "close"])
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    if out_path.exists() and not ns.overwrite:
        prev = pd.read_parquet(out_path)
        if not prev.empty and "date" in prev.columns:
            prev = prev.copy()
            prev["date"] = pd.to_datetime(prev["date"]).dt.date
            out = pd.concat([prev, out], ignore_index=True)
            out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    out.to_parquet(out_path, index=False)

    meta = {
        "symbol": symbol,
        "source": "massive_v2_aggs",
        "base_url": base_url,
        "start_date": ns.start_date,
        "end_date": ns.end_date,
        "adjusted": bool(ns.adjusted),
        "rows": int(len(out)),
        "path": str(out_path.resolve()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(out_path)


def _iter_weekdays(start_date: str, end_date: str) -> list[date]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        raise RuntimeError(f"Invalid date range: start-date {start_date} is after end-date {end_date}.")

    out: list[date] = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def _pull_massive_command(ns: Any) -> None:
    import json

    import pandas as pd
    import requests

    from ivdyn.data.schemas import normalize_chain_df

    symbol = str(ns.symbol).upper().strip()
    if not symbol:
        raise RuntimeError("Symbol is required.")

    api_key = _resolve_massive_api_key(ns.api_key)
    base_url = str(ns.base_url).rstrip("/")

    out_dir = _to_path(ns.data_root) / "symbols" / symbol / "options" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    weekdays = _iter_weekdays(ns.start_date, ns.end_date)
    saved_days = 0
    skipped_existing = 0
    empty_days = 0
    total_rows = 0

    for asof in weekdays:
        day_str = asof.isoformat()
        day_path = out_dir / f"{day_str}.parquet"
        day_meta_path = out_dir / f"{day_str}.metadata.json"

        if day_path.exists() and not ns.overwrite:
            skipped_existing += 1
            continue

        url = f"{base_url}/v3/snapshot/options/{symbol}"
        params: dict[str, Any] | None = {
            "as_of": day_str,
            "limit": int(ns.limit),
            "sort": "ticker",
            "order": "asc",
            "expired": "true" if ns.include_expired else "false",
            "apiKey": api_key,
        }

        page_count = 0
        chain_rows: list[dict[str, Any]] = []
        while url:
            resp = requests.get(url, params=params, timeout=ns.timeout_seconds)
            params = None
            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                snippet = resp.text[:500]
                raise RuntimeError(f"Massive options request failed on {day_str} ({resp.status_code}): {snippet}") from exc

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
                            "date": day_str,
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
            if ns.max_pages > 0 and page_count >= ns.max_pages:
                break

            next_url = payload.get("next_url")
            if next_url and "apiKey=" not in str(next_url):
                sep = "&" if "?" in str(next_url) else "?"
                next_url = f"{next_url}{sep}apiKey={api_key}"
            url = next_url

        if chain_rows:
            norm = normalize_chain_df(pd.DataFrame(chain_rows), asof=asof, symbol=symbol)
        else:
            norm = pd.DataFrame()

        rows_saved = int(len(norm))
        if rows_saved > 0:
            norm.to_parquet(day_path, index=False)
            saved_days += 1
            total_rows += rows_saved
        else:
            if day_path.exists() and ns.overwrite:
                day_path.unlink()
            empty_days += 1

        meta = {
            "symbol": symbol,
            "as_of": day_str,
            "source": "massive_v3_snapshot_options",
            "base_url": base_url,
            "include_expired": bool(ns.include_expired),
            "rows_raw": int(len(chain_rows)),
            "rows_saved": rows_saved,
            "pages": page_count,
            "path": str(day_path.resolve()) if rows_saved > 0 else None,
        }
        day_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    summary = {
        "symbol": symbol,
        "start_date": ns.start_date,
        "end_date": ns.end_date,
        "trading_days_considered": len(weekdays),
        "saved_days": saved_days,
        "skipped_existing_days": skipped_existing,
        "empty_days": empty_days,
        "rows_saved_total": total_rows,
    }
    summary_path = out_dir / f"_massive_pull_summary_{utc_timestamp()}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


def _not_implemented(_: Any) -> None:
    raise RuntimeError(
        "This command is not implemented in the local checkout. "
        "Use external data-prep tooling and keep files in the expected locations."
    )


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", default="outputs/runs")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--vae-epochs", type=int, default=120)
    p.add_argument("--vae-batch-size", type=int, default=32)
    p.add_argument("--vae-lr", type=float, default=2e-3)
    p.add_argument("--vae-kl-beta", type=float, default=0.02)
    p.add_argument("--noarb-lambda", type=float, default=0.2)
    p.add_argument("--head-epochs", type=int, default=130)
    p.add_argument("--dyn-batch-size", type=int, default=64)
    p.add_argument("--contract-batch-size", type=int, default=2048)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--joint-epochs", type=int, default=120)
    p.add_argument("--joint-lr", type=float, default=5e-4)
    p.add_argument("--joint-contract-batch-size", type=int, default=4096)
    p.add_argument("--joint-dyn-lambda", type=float, default=1.0)
    p.add_argument("--joint-price-lambda", type=float, default=1.0)
    p.add_argument("--joint-exec-lambda", type=float, default=0.25)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--price-risk-weight", type=float, default=1.0)
    p.add_argument("--exec-risk-weight", type=float, default=0.5)
    p.add_argument("--risk-focus-abs-x", type=float, default=0.06)
    p.add_argument("--risk-focus-tau-days", type=float, default=20.0)
    p.add_argument("--exec-label-smoothing", type=float, default=0.03)
    p.add_argument("--exec-logit-l2", type=float, default=2e-4)
    p.set_defaults(func=_train_command)

    p = sub.add_parser("build-dataset")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--plugin", default="massive_raw_parquet")
    p.add_argument("--api-key")
    p.add_argument("--start-date")
    p.add_argument("--end-date")
    p.add_argument(
        "--x-grid",
        nargs="+",
        type=float,
        default=[-0.35, -0.30, -0.25, -0.20, -0.16, -0.12, -0.09, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.20, 0.25, 0.30, 0.35],
    )
    p.add_argument("--tenor-days", nargs="+", type=int, default=[7, 14, 30, 60, 90, 180])
    p.add_argument("--max-contracts-per-day", type=int, default=900)
    p.add_argument("--random-seed", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=0)
    p.set_defaults(func=_build_dataset_command)

    p = sub.add_parser("evaluate")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--device")
    p.add_argument("--num-workers", type=int, default=0)
    p.set_defaults(func=_evaluate_command)

    p = sub.add_parser("backtest")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--start-date", default=None, help="Optional inclusive start date filter (YYYY-MM-DD).")
    p.add_argument("--end-date", default=None, help="Optional inclusive end date filter (YYYY-MM-DD).")
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--inference-batch-size", type=int, default=65536)
    p.add_argument("--initial-capital", type=float, default=1000000.0)
    p.add_argument("--fill-gate", type=float, default=0.45)
    p.add_argument(
        "--fill-model",
        choices=["assume", "expected"],
        default="expected",
        help="How to apply fill probability. 'assume' is optimistic (trade always fills). 'expected' scales PnL/fees by fill_prob.",
    )
    p.add_argument("--slippage-bps", type=float, default=10.0)
    p.add_argument(
        "--spread-cross-fraction",
        type=float,
        default=0.75,
        help="Fraction of half-spread paid per fill (0=mid, 1=at touch). Applied on entry and exit.",
    )
    p.add_argument("--option-commission-per-contract", type=float, default=0.65)
    p.add_argument("--option-fee-per-contract", type=float, default=0.05)
    p.add_argument(
        "--min-edge-to-cost-ratio",
        type=float,
        default=1.75,
        help="Require estimated edge to exceed estimated round-trip costs by this ratio before selecting trades.",
    )
    p.add_argument(
        "--volume-participation-rate",
        type=float,
        default=0.02,
        help="Target fraction of per-contract daily volume used as a participation cap.",
    )
    p.add_argument(
        "--open-interest-participation-rate",
        type=float,
        default=0.01,
        help="Target fraction of open interest used as an additional cap for contract sizing.",
    )
    p.add_argument(
        "--long-score-scale",
        type=float,
        default=1.0,
        help="Scale factor for selecting LONG option trades based on negative signal. 0 disables longs (default). Typical range: 0.1–0.5.",
    )
    p.add_argument(
        "--allow-long-puts",
        dest="allow_long_puts",
        action="store_true",
        help="Allow LONG put candidates in the selector (enabled by default).",
    )
    p.add_argument(
        "--no-allow-long-puts",
        dest="allow_long_puts",
        action="store_false",
        help="Disable LONG put candidates in the selector.",
    )
    p.set_defaults(allow_long_puts=True)
    p.add_argument("--max-trades-per-day", type=int, default=100)
    p.add_argument(
        "--max-contracts-per-trade",
        type=int,
        default=4,
        help="Maximum contracts for a single trade idea. Additional size is only used for very high-quality/liquid setups.",
    )
    p.add_argument(
        "--signal-abs-gate",
        type=float,
        default=0.04,
        help="Require |signal| >= threshold (signal is relative edge: (mid_now - pred_next) / mid_now) before a contract is eligible for trading.",
    )
    p.add_argument("--min-dte", type=int, default=7)
    p.add_argument("--max-dte", type=int, default=75)
    p.add_argument("--min-moneyness", type=float, default=0.88)
    p.add_argument("--max-moneyness", type=float, default=1.12)
    p.add_argument("--max-rel-spread", type=float, default=0.10)

    # Multi-leg / hedged extensions
    p.add_argument(
        "--strategy-mode",
        choices=["single", "vertical"],
        default="vertical",
        help="Trade construction mode. 'single' = 1-leg option trades (current behavior). 'vertical' = defined-risk vertical spreads (adds a further-OTM wing leg).",
    )
    p.add_argument("--vertical-wing-width-pct-target", type=float, default=0.03, help="Vertical wing target distance as a fraction of spot (e.g., 0.02 ~= 2%% of spot).")
    p.add_argument("--vertical-wing-width-pct-min", type=float, default=0.01, help="Minimum wing distance as a fraction of spot.")
    p.add_argument("--vertical-wing-width-pct-max", type=float, default=0.08, help="Maximum wing distance as a fraction of spot.")
    p.add_argument("--vertical-wing-max-premium-ratio", type=float, default=0.35, help="Require wing mid_now <= ratio * anchor mid_now.")
    p.add_argument("--vertical-wing-fill-gate", type=float, default=0.6)
    p.add_argument("--vertical-wing-max-rel-spread", type=float, default=0.15)
    p.add_argument("--vertical-wing-min-moneyness", type=float, default=0.3)
    p.add_argument("--vertical-wing-max-moneyness", type=float, default=1.9)
    p.add_argument("--vertical-wing-rich-signal-penalty", type=float, default=0.75, help="Penalize long wings that are also 'rich' per the model (higher = avoid paying away alpha).")
    p.add_argument(
        "--vertical-skip-if-no-wing",
        dest="vertical_skip_if_no_wing",
        action="store_true",
        help="Skip anchor trades when a suitable wing cannot be found (enabled by default).",
    )
    p.add_argument(
        "--no-vertical-skip-if-no-wing",
        dest="vertical_skip_if_no_wing",
        action="store_false",
        help="Allow fallback to single-leg anchor trade when no suitable wing is found.",
    )
    p.set_defaults(vertical_skip_if_no_wing=True)

    p.add_argument(
        "--hedge-underlying-delta",
        dest="hedge_underlying_delta",
        action="store_true",
        help="Enable daily underlying delta hedge.",
    )
    p.add_argument(
        "--no-hedge-underlying-delta",
        dest="hedge_underlying_delta",
        action="store_false",
        help="Disable daily underlying delta hedge.",
    )
    p.set_defaults(hedge_underlying_delta=True)
    p.add_argument("--hedge-underlying-ratio", type=float, default=1.0, help="0=off. 1.0=full delta neutralization; 0.5=half hedge.")
    p.add_argument("--hedge-underlying-min-abs-shares", type=float, default=15.0, help="Do not place a hedge trade unless |shares| exceeds this threshold.")
    p.add_argument("--hedge-underlying-max-shares", type=int, default=50)
    p.add_argument("--hedge-underlying-slippage-bps", type=float, default=1.0)
    p.add_argument(
        "--hedge-policy",
        choices=["fixed", "learned"],
        default="fixed",
        help="Underlying hedge policy. 'fixed' uses --hedge-underlying-ratio. 'learned' loads --hedge-policy-path.",
    )
    p.add_argument(
        "--hedge-policy-path",
        default=None,
        help="Path to a trained hedge_policy.pt (required when --hedge-policy learned).",
    )
    p.add_argument(
        "--enforce-portfolio-constraints",
        dest="enforce_portfolio_constraints",
        action="store_true",
        help="Reject trades/hedges that exceed available buying power.",
    )
    p.add_argument(
        "--no-enforce-portfolio-constraints",
        dest="enforce_portfolio_constraints",
        action="store_false",
        help="Disable buying power checks (legacy behavior).",
    )
    p.set_defaults(enforce_portfolio_constraints=True)
    p.add_argument(
        "--buying-power-leverage",
        type=float,
        default=1.0,
        help="Buying power multiplier on equity for funding checks.",
    )
    p.add_argument(
        "--option-short-margin-rate",
        type=float,
        default=0.20,
        help="Short option margin proxy as fraction of spot notional.",
    )
    p.add_argument(
        "--underlying-margin-rate",
        type=float,
        default=0.50,
        help="Underlying hedge margin proxy as fraction of stock notional.",
    )
    p.set_defaults(func=_backtest_command)

    p = sub.add_parser("train-hedge-policy")
    p.add_argument("--run-dir", default=None, help="Run directory containing model.pt and backtest artifacts.")
    p.add_argument("--dataset", default=None, help="Dataset used for the backtest (defaults to train_summary.json).")
    p.add_argument("--out-dir", default=None, help="Output directory for the trained policy (default: <run_dir>/hedge_policy/<timestamp>).")
    p.add_argument("--device", default=None)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--max-ratio", type=float, default=1.25)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--train-frac", type=float, default=0.7, help="Fraction of days (by time) used for training; remaining are validation.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--risk-aversion",
        type=float,
        default=0.50,
        help="Objective is mean(daily_pnl) - risk_aversion*std(daily_pnl). Higher => more hedging / less variance.",
    )
    p.add_argument("--underlying-slippage-bps", type=float, default=1.0)
    p.add_argument("--min-abs-shares", type=float, default=20.0)
    p.add_argument("--max-shares", type=float, default=200.0)
    p.set_defaults(func=_train_hedge_policy_command)

    p = sub.add_parser("ui")
    p.add_argument("--run-dir", default=None)
    p.set_defaults(func=_ui_command)

    p = sub.add_parser("ui2")
    p.add_argument("--symbol", default=None, help="Optional symbol focus for live walk-forward dashboard.")
    p.set_defaults(func=_ui2_command)

    p = sub.add_parser("wf")
    p.add_argument("--symbol", default=None, help="Optional symbol override for live walk-forward run.")
    p.add_argument("--force", action="store_true", default=False, help="Force a rerun even if already processed today.")
    p.set_defaults(func=_wf_command)

    p = sub.add_parser("pull-underlying-massive")
    p.add_argument("--data-root", default="data")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default="https://api.massive.com")
    p.add_argument("--timeout-seconds", type=int, default=30)
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--max-pages", type=int, default=0, help="Optional page cap for debugging (0 = no cap).")
    p.add_argument("--adjusted", dest="adjusted", action="store_true")
    p.add_argument("--unadjusted", dest="adjusted", action="store_false")
    p.set_defaults(adjusted=True, func=_pull_underlying_massive_command)

    p = sub.add_parser("pull-massive")
    p.add_argument("--data-root", default="data")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default="https://api.massive.com")
    p.add_argument("--timeout-seconds", type=int, default=30)
    p.add_argument("--limit", type=int, default=250, help="API page size (max depends on Massive plan).")
    p.add_argument("--max-pages", type=int, default=0, help="Optional page cap per day (0 = no cap).")
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--include-expired", dest="include_expired", action="store_true")
    p.add_argument("--exclude-expired", dest="include_expired", action="store_false")
    p.set_defaults(include_expired=True, func=_pull_massive_command)

    p = sub.add_parser("pull-flatfiles")
    p.add_argument("--data-root", default="data")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--access-key", default=None)
    p.add_argument("--secret-key", default=None)
    p.add_argument("--session-token", default=None)
    p.add_argument("--endpoint-url", default=None)
    p.add_argument("--bucket", default=None)
    p.add_argument("--prefix", default=None)
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--fail-fast", action="store_true", default=False)
    p.add_argument("--max-days", type=int, default=0, help="Optional cap for debugging (0 = no cap).")
    p.set_defaults(func=_pull_flatfiles_command)

    p = sub.add_parser("pull-options-symbol")
    p.add_argument("--data-root", default="data")
    p.add_argument("--symbol", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument(
        "--source-root",
        default=None,
        help="Root containing full OPRA day files. Default: <data-root>/options_source/<source-prefix>.",
    )
    p.add_argument(
        "--source-prefix",
        default=None,
        help="Used when --source-root is omitted. Default env MASSIVE_FLATFILES_PREFIX or us_options_opra/day_aggs_v1.",
    )
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--allow-missing-underlying", action="store_true", default=False)
    p.add_argument("--max-days", type=int, default=0, help="Optional cap for debugging (0 = no cap).")
    p.set_defaults(func=_pull_options_symbol_command)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    _autoload_dotenv()
    parser = _build_parser()
    ns = parser.parse_args(argv)
    ns.func(ns)


if __name__ == "__main__":
    main()
