from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from spygen.data.filters import apply_liquidity_filters
from spygen.data.io import append_dedup_underlying, list_raw_chain_files, write_metadata
from spygen.data.synth import SynthConfig, write_synthetic_dataset
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

    files = list_raw_chain_files(raw_dir, start=start_d, end=end_d)
    for file in files:
        chain = pd.read_parquet(file)
        if chain.empty:
            continue
        day = file.stem
        if pd.Timestamp(day) not in context_df.index:
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
                lambda_smooth=float(config.get("repair", {}).get("lambda_smooth", 1e-3)),
                data_weight=float(config.get("repair", {}).get("data_weight", 1.0)),
            )
        except Exception:
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
        "context_features": list(context_df.columns),
        "nx": grid.nx,
        "nt": len(grid.tenors_days),
        "built_at": datetime.now(UTC).isoformat(),
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

    arb_ok = [is_arb_free(samples[i]) for i in range(samples.shape[0])]
    arb_rate = float(np.mean(arb_ok))
    counts = [arb_violation_counts(samples[i]) for i in range(samples.shape[0])]

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
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    mean_surface = conditional_mean_surface(model, context=context[:1], n_samples=n_samples)[0]
    np.save(out_dir / "mean_surface.npy", mean_surface)
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
