from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from spygen.models.sampling import load_checkpoint
from spygen.strategy.basis import basis_vectors, default_trade_structures
from spygen.strategy.execution import (
    ExecutionModel,
    ExecutionResult,
    estimate_structure_roundtrip_cost,
    execute_structure_one_day,
)
from spygen.strategy.metrics import max_drawdown, sharpe_ratio, turnover_ratio
from spygen.strategy.providers import make_signal_provider
from spygen.utils.paths import ensure_dir


@dataclass(slots=True)
class BacktestConfig:
    threshold: float = 4.0
    zscore_quantile: float = 0.9
    min_history_for_quantile: int = 20
    min_signal_abs: float = 0.001
    edge_cost_multiplier: float = 1.25
    residual_clip: float = 1.0
    reject_if_residual_clipped: bool = False
    n_samples: int = 32
    max_trades_per_day: int = 3
    max_contracts: int = 10
    max_notional: float = 50_000.0
    direction_mode: str = "mean_revert"
    signal_provider: str = "deep_flow"
    seed: int = 42
    execution_mode: str = "worse_than_touch"
    execution_impact_bps: float = 0.0
    execution_fee_per_contract: float = 0.0
    execution_worse_touch_extra_half_spread: float = 0.5
    spread_gate_mode: str = "abs_or_rel"
    max_spread_abs: float = 3.0
    max_spread_rel: float = 0.35
    edge_signal_to_usd_scale: float = 1.0
    unit_sanity_check: bool = True
    unit_sanity_fail_fast: bool = False


def _load_chain(raw_dir: str | Path, day: str) -> pd.DataFrame | None:
    p = Path(raw_dir) / f"{day}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def _bump(counter: dict[str, int], reason: str) -> None:
    counter[reason] = counter.get(reason, 0) + 1


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


def _trade_direction(signal: float, mode: str) -> int:
    if mode == "mean_revert":
        return -1 if signal > 0 else 1
    if mode == "trend_follow":
        return 1 if signal > 0 else -1
    raise ValueError(f"Unsupported direction_mode: {mode}")


def _pnl_attribution(blotter: pd.DataFrame) -> dict[str, Any]:
    edge_gross_col = "edge_gross_usd" if "edge_gross_usd" in blotter.columns else "edge_gross"
    edge_net_col = "edge_net_usd" if "edge_net_usd" in blotter.columns else "edge_net"

    if blotter.empty:
        return {
            "n_trades": 0,
            "cost_breakdown": {
                "spread_paid": 0.0,
                "fill_slippage": 0.0,
                "fees": 0.0,
            },
            "edge": {
                "mean_edge_gross_usd": 0.0,
                "median_edge_gross_usd": 0.0,
                "mean_edge_net_usd": 0.0,
                "median_edge_net_usd": 0.0,
            },
            "hit_rate": 0.0,
            "avg_hold_days": 0.0,
            "tail_losses": {"p1": 0.0, "p5": 0.0, "p10": 0.0},
            "per_structure": {},
            "exposure_proxy_totals": {
                "delta_proxy": 0.0,
                "vega_proxy": 0.0,
                "gamma_proxy": 0.0,
            },
        }

    losses = blotter["realized_pnl"].to_numpy(dtype=float)
    by_structure: dict[str, Any] = {}
    for name, g in blotter.groupby("structure"):
        by_structure[name] = {
            "trades": int(len(g)),
            "total_pnl": float(g["realized_pnl"].sum()),
            "mean_pnl": float(g["realized_pnl"].mean()),
            "median_pnl": float(g["realized_pnl"].median()),
            "hit_rate": float((g["realized_pnl"] > 0).mean()),
            "spread_paid": float(g["spread_paid"].sum()),
            "fill_slippage": float(g["fill_slippage"].sum()),
            "fees": float(g["fees"].sum()),
            "edge_gross_usd": float(g[edge_gross_col].sum()),
            "edge_net_usd": float(g[edge_net_col].sum()),
            "delta_proxy": float(g["delta_proxy"].sum()),
            "vega_proxy": float(g["vega_proxy"].sum()),
            "gamma_proxy": float(g["gamma_proxy"].sum()),
        }

    return {
        "n_trades": int(len(blotter)),
        "cost_breakdown": {
            "spread_paid": float(blotter["spread_paid"].sum()),
            "fill_slippage": float(blotter["fill_slippage"].sum()),
            "fees": float(blotter["fees"].sum()),
        },
        "edge": {
            "mean_edge_gross_usd": float(blotter[edge_gross_col].mean()),
            "median_edge_gross_usd": float(blotter[edge_gross_col].median()),
            "mean_edge_net_usd": float(blotter[edge_net_col].mean()),
            "median_edge_net_usd": float(blotter[edge_net_col].median()),
        },
        "hit_rate": float((blotter["realized_pnl"] > 0).mean()),
        "avg_hold_days": 1.0,
        "tail_losses": {
            "p1": float(np.percentile(losses, 1)),
            "p5": float(np.percentile(losses, 5)),
            "p10": float(np.percentile(losses, 10)),
        },
        "per_structure": by_structure,
        "exposure_proxy_totals": {
            "delta_proxy": float(blotter["delta_proxy"].sum()),
            "vega_proxy": float(blotter["vega_proxy"].sum()),
            "gamma_proxy": float(blotter["gamma_proxy"].sum()),
        },
    }


def _execution_summary(
    blotter: pd.DataFrame,
    gate_reasons: dict[str, Any],
    exec_model: ExecutionModel,
) -> dict[str, Any]:
    attempts = float(gate_reasons.get("meta", {}).get("attempted_structure_checks", 0))
    spread_rejects = float(
        gate_reasons.get("global", {}).get(
            "rejected_execution_spread_too_wide",
            0,
        )
    )
    pct_spread = spread_rejects / attempts if attempts > 0 else 0.0

    if blotter.empty:
        return {
            "n_trades": 0,
            "execution_model": asdict(exec_model),
            "avg_spread_paid": 0.0,
            "avg_fill_slippage": 0.0,
            "avg_fees": 0.0,
            "spread_paid_quantiles": {"p50": 0.0, "p90": 0.0, "p99": 0.0},
            "fill_slippage_quantiles": {"p50": 0.0, "p90": 0.0, "p99": 0.0},
            "pct_skipped_by_spread": float(pct_spread),
        }

    spread = blotter["spread_paid"].to_numpy(dtype=float)
    slip = blotter["fill_slippage"].to_numpy(dtype=float)

    return {
        "n_trades": int(len(blotter)),
        "execution_model": asdict(exec_model),
        "avg_spread_paid": float(np.mean(spread)),
        "avg_fill_slippage": float(np.mean(slip)),
        "avg_fees": float(blotter["fees"].mean()),
        "spread_paid_quantiles": {
            "p50": float(np.percentile(spread, 50)),
            "p90": float(np.percentile(spread, 90)),
            "p99": float(np.percentile(spread, 99)),
        },
        "fill_slippage_quantiles": {
            "p50": float(np.percentile(slip, 50)),
            "p90": float(np.percentile(slip, 90)),
            "p99": float(np.percentile(slip, 99)),
        },
        "pct_skipped_by_spread": float(pct_spread),
    }


def run_backtest(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    raw_dir: str | Path,
    output_dir: str | Path,
    tenor_days: list[int],
    x_grid: np.ndarray,
    config: BacktestConfig,
    signal_provider_name: str | None = None,
) -> Path:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    ds = np.load(dataset_path, allow_pickle=True)
    dates = ds["dates"].astype(str)
    context = ds["context"].astype(np.float32)
    theta_raw = ds["theta_raw"].astype(np.float32)
    surfaces = ds["surface"].astype(np.float32)

    provider_name = signal_provider_name or config.signal_provider
    basis = basis_vectors(x_grid=x_grid, tenor_days=tenor_days)
    structures = default_trade_structures()

    model = load_checkpoint(checkpoint_path) if provider_name == "deep_flow" else None
    provider = make_signal_provider(
        provider_name,
        basis=basis,
        n_samples=config.n_samples,
        residual_clip=config.residual_clip,
    )
    provider.prepare(
        surfaces=surfaces,
        context=context,
        theta_raw=theta_raw,
        model=model,
    )

    exec_model = ExecutionModel(
        mode=config.execution_mode,
        impact_bps=config.execution_impact_bps,
        fee_per_contract=config.execution_fee_per_contract,
        worse_touch_extra_half_spread=config.execution_worse_touch_extra_half_spread,
        max_spread_abs=config.max_spread_abs,
        max_spread_rel=config.max_spread_rel,
        spread_gate_mode=config.spread_gate_mode,
    )

    daily_rows: list[dict[str, float | str]] = []
    blotter_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    z_history: list[float] = []

    gate_reasons: dict[str, Any] = {
        "global": {},
        "by_structure": {k: {} for k in structures},
        "meta": {
            "attempted_structure_checks": 0,
            "provider": provider_name,
            "units": {
                "edge": "USD (proxy)",
                "cost": "USD",
                "signal": "normalized_call_surface_residual",
            },
        },
    }

    for i in range(len(dates) - 1):
        day = dates[i]
        next_day = dates[i + 1]
        entry_chain = _load_chain(raw_dir, day)
        exit_chain = _load_chain(raw_dir, next_day)
        if entry_chain is None or exit_chain is None:
            _bump(gate_reasons["global"], "rejected_missing_chain")
            continue

        sig = provider.signal_for_day(i)
        z = float(sig.z)

        threshold = config.threshold
        if len(z_history) >= config.min_history_for_quantile:
            quantile_cut = float(np.quantile(z_history, config.zscore_quantile))
            threshold = max(threshold, quantile_cut)
        z_history.append(z)

        if z < threshold:
            _bump(gate_reasons["global"], "rejected_quantile_gate")
            for name in structures:
                _bump(gate_reasons["by_structure"][name], "rejected_quantile_gate")
            daily_rows.append(
                {
                    "date": day,
                    "provider": provider_name,
                    "pnl": 0.0,
                    "turnover": 0.0,
                    "z": z,
                    "trades": 0.0,
                }
            )
            continue

        raw_residual = surfaces[i] - sig.reference_surface
        clipped_any = bool(np.any(np.abs(raw_residual) > config.residual_clip))
        if clipped_any:
            _bump(gate_reasons["global"], "residual_clipped")
        if clipped_any and config.reject_if_residual_clipped:
            _bump(gate_reasons["global"], "rejected_residual_clip")
            for name in structures:
                _bump(gate_reasons["by_structure"][name], "rejected_residual_clip")
            daily_rows.append(
                {
                    "date": day,
                    "provider": provider_name,
                    "pnl": 0.0,
                    "turnover": 0.0,
                    "z": z,
                    "trades": 0.0,
                }
            )
            continue

        pnl = 0.0
        turnover = 0.0
        contracts = 0
        trades = 0

        for name, struct in structures.items():
            gate_reasons["meta"]["attempted_structure_checks"] += 1
            signal = float(sig.projections.get(name, 0.0))
            if abs(signal) < config.min_signal_abs:
                _bump(gate_reasons["global"], "rejected_signal_below_min")
                _bump(gate_reasons["by_structure"][name], "rejected_signal_below_min")
                event_rows.append(
                    {
                        "date": day,
                        "structure": name,
                        "reason": "rejected_signal_below_min",
                        "signal": signal,
                    }
                )
                continue

            est = estimate_structure_roundtrip_cost(entry_chain, struct, exec_model)
            if not est.passed:
                reason = f"rejected_execution_{est.reason}"
                _bump(gate_reasons["global"], reason)
                _bump(gate_reasons["by_structure"][name], reason)
                event_rows.append(
                    {
                        "date": day,
                        "structure": name,
                        "reason": reason,
                        **est.metrics,
                    }
                )
                continue

            notional_usd = float(est.metrics.get("notional_usd", est.metrics.get("notional", 0.0)))
            expected_cost_usd = float(
                est.metrics.get("expected_cost_usd", est.metrics.get("expected_cost", 0.0))
            )
            edge_gross_usd = abs(signal) * notional_usd * config.edge_signal_to_usd_scale
            edge_over_cost = edge_gross_usd / max(expected_cost_usd, 1e-8)

            if edge_gross_usd <= config.edge_cost_multiplier * expected_cost_usd:
                _bump(gate_reasons["global"], "rejected_net_edge")
                _bump(gate_reasons["by_structure"][name], "rejected_net_edge")
                event_rows.append(
                    {
                        "date": day,
                        "structure": name,
                        "reason": "rejected_net_edge",
                        "signal": signal,
                        "edge_gross_usd": edge_gross_usd,
                        "expected_cost_usd": expected_cost_usd,
                        "edge_over_cost": edge_over_cost,
                        "notional_usd": notional_usd,
                    }
                )
                continue

            direction = _trade_direction(signal=signal, mode=config.direction_mode)
            result: ExecutionResult = execute_structure_one_day(
                entry_chain=entry_chain,
                exit_chain=exit_chain,
                structure=struct,
                direction=direction,
                model=exec_model,
                signal=signal,
                edge_gross=edge_gross_usd,
            )
            if result.skipped:
                reason = f"rejected_execution_{result.skip_reason}"
                _bump(gate_reasons["global"], reason)
                _bump(gate_reasons["by_structure"][name], reason)
                event_rows.append(
                    {
                        "date": day,
                        "structure": name,
                        "reason": reason,
                        **result.gate_metrics,
                    }
                )
                continue

            if trades >= config.max_trades_per_day:
                _bump(gate_reasons["global"], "rejected_position_limit")
                _bump(gate_reasons["by_structure"][name], "rejected_position_limit")
                continue

            if contracts + result.contracts > config.max_contracts:
                _bump(gate_reasons["global"], "rejected_position_limit")
                _bump(gate_reasons["by_structure"][name], "rejected_position_limit")
                continue

            if turnover + result.turnover > config.max_notional:
                _bump(gate_reasons["global"], "rejected_notional_limit")
                _bump(gate_reasons["by_structure"][name], "rejected_notional_limit")
                continue

            pnl += result.realized_pnl
            turnover += result.turnover
            contracts += result.contracts
            trades += 1
            _bump(gate_reasons["global"], "accepted")
            _bump(gate_reasons["by_structure"][name], "accepted")

            blotter_rows.append(
                {
                    "date": day,
                    "next_date": next_day,
                    "provider": provider_name,
                    "structure": name,
                    "signal": signal,
                    "z": z,
                    "edge_gross_usd": edge_gross_usd,
                    "edge_net_usd": result.edge_net,
                    "edge_cost_estimate_usd": expected_cost_usd,
                    "edge_over_cost": edge_over_cost,
                    "notional_usd": notional_usd,
                    "fill_slippage": result.fill_slippage,
                    "spread_paid": result.spread_paid,
                    "fees": result.fees,
                    "holding_return": result.holding_return,
                    "realized_pnl": result.realized_pnl,
                    "pnl_before_fees": result.pnl,
                    "turnover": result.turnover,
                    "contracts": result.contracts,
                    "direction": direction,
                    "delta_proxy": result.delta_proxy,
                    "vega_proxy": result.vega_proxy,
                    "gamma_proxy": result.gamma_proxy,
                }
            )

            event_rows.append(
                {
                    "date": day,
                    "structure": name,
                    "reason": "accepted",
                    "signal": signal,
                    "z": z,
                    "edge_gross_usd": edge_gross_usd,
                    "expected_cost_usd": expected_cost_usd,
                    "edge_over_cost": edge_over_cost,
                    "realized_pnl": result.realized_pnl,
                }
            )

        daily_rows.append(
            {
                "date": day,
                "provider": provider_name,
                "pnl": pnl,
                "turnover": turnover,
                "z": z,
                "trades": float(trades),
            }
        )

    if not daily_rows:
        raise ValueError("No backtest rows produced")

    edge_samples: list[float] = []
    cost_samples: list[float] = []
    for row in event_rows:
        edge_val = row.get("edge_gross_usd")
        cost_val = row.get("expected_cost_usd")
        if edge_val is None or cost_val is None:
            continue
        edge_samples.append(float(edge_val))
        cost_samples.append(float(cost_val))

    unit_sanity: dict[str, Any] = {"enabled": bool(config.unit_sanity_check), "checked": False}
    if config.unit_sanity_check and edge_samples and cost_samples:
        edge_arr = np.asarray(edge_samples, dtype=float)
        cost_arr = np.asarray(cost_samples, dtype=float)
        median_edge = float(np.median(edge_arr))
        median_cost = float(np.median(cost_arr))
        median_ratio = median_edge / max(median_cost, 1e-8)
        unit_sanity = {
            "enabled": True,
            "checked": True,
            "median_edge_gross_usd": median_edge,
            "median_expected_cost_usd": median_cost,
            "median_edge_over_cost": median_ratio,
            "edge_signal_to_usd_scale": config.edge_signal_to_usd_scale,
        }
        if config.unit_sanity_fail_fast and median_ratio < 0.01:
            raise ValueError(
                "Unit-sanity check failed: median edge/cost ratio below 0.01. "
                "Check edge units or edge_signal_to_usd_scale."
            )

    df = pd.DataFrame(daily_rows)
    df["equity"] = df["pnl"].cumsum()
    returns = df["pnl"].to_numpy(dtype=float)
    equity = df["equity"].to_numpy(dtype=float)

    summary = {
        "provider": provider_name,
        "n_days": int(len(df)),
        "trade_days": int((df["trades"] > 0).sum()),
        "sum_trades": int(df["trades"].sum()),
        "total_pnl": float(df["pnl"].sum()),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "turnover_ratio": turnover_ratio(df["turnover"].to_numpy(dtype=float), equity),
    }

    run_name = f"run_{provider_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = ensure_dir(Path(output_dir) / "backtests" / run_name)

    df.to_parquet(run_dir / "daily.parquet", index=False)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    blotter_df = pd.DataFrame(blotter_rows)
    blotter_df.to_parquet(run_dir / "trade_blotter.parquet", index=False)

    attribution = _pnl_attribution(blotter_df)
    (run_dir / "pnl_attribution.json").write_text(json.dumps(attribution, indent=2))

    execution_summary = _execution_summary(blotter_df, gate_reasons, exec_model)
    (run_dir / "execution_summary.json").write_text(json.dumps(execution_summary, indent=2))

    (run_dir / "gate_reasons.json").write_text(json.dumps(gate_reasons, indent=2))
    (run_dir / "unit_sanity.json").write_text(json.dumps(unit_sanity, indent=2))

    events_path = run_dir / "events.jsonl"
    with events_path.open("w") as fp:
        for row in event_rows:
            fp.write(json.dumps(row) + "\n")

    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "seed": config.seed,
        "config": asdict(config),
        "unit_sanity": unit_sanity,
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    return run_dir
