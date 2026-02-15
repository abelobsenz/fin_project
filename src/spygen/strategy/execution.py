from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from spygen.strategy.basis import TradeStructure


@dataclass(slots=True)
class ExecutionModel:
    mode: str = "worse_than_touch"
    impact_bps: float = 0.0
    fee_per_contract: float = 0.0
    worse_touch_extra_half_spread: float = 0.5
    max_spread_abs: float = 3.0
    max_spread_rel: float = 0.35
    spread_gate_mode: str = "abs_or_rel"


@dataclass(slots=True)
class GateCheck:
    passed: bool
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionResult:
    pnl: float
    turnover: float
    contracts: int
    skipped: bool
    skip_reason: str
    spread_paid: float
    fill_slippage: float
    fees: float
    holding_return: float
    realized_pnl: float
    edge_gross: float
    edge_net: float
    delta_proxy: float
    vega_proxy: float
    gamma_proxy: float
    blotter_rows: list[dict[str, Any]] = field(default_factory=list)
    gate_metrics: dict[str, float] = field(default_factory=dict)


def select_option_quote(
    chain: pd.DataFrame, target_dte: int, target_x: float, call_put: str
) -> pd.Series | None:
    df = chain.loc[chain["call_put"] == call_put].copy()
    if df.empty:
        return None
    spot = float(df["underlying_close"].iloc[0])
    df["x"] = np.log(df["strike"] / spot)
    df["score"] = (df["dte"] - target_dte).abs() + 20.0 * (df["x"] - target_x).abs()
    return df.sort_values("score").iloc[0]


def _fill_components(
    row: pd.Series,
    side: int,
    model: ExecutionModel,
) -> tuple[float, float, float]:
    bid = float(row["bid"])
    ask = float(row["ask"])
    mid = float(row["mid"])
    half_spread = max(0.0, 0.5 * (ask - bid))

    if side not in (-1, 1):
        raise ValueError("side must be +/-1")

    spread_paid = 0.0
    slippage = 0.0

    if model.mode == "mid":
        fill = mid
    elif model.mode in {"touch", "mid_plus_half_spread"}:
        fill = ask if side > 0 else bid
        spread_paid = half_spread
    elif model.mode == "worse_than_touch":
        extra = model.worse_touch_extra_half_spread * half_spread
        fill = (ask + extra) if side > 0 else (bid - extra)
        spread_paid = half_spread
        slippage += max(0.0, extra)
    else:
        raise ValueError(f"Unsupported execution mode: {model.mode}")

    impact = model.impact_bps * 1e-4 * max(mid, 0.01)
    fill = fill + impact if side > 0 else fill - impact
    slippage += max(0.0, impact)

    return float(fill), float(spread_paid), float(slippage)


def _spread_gate(spread_abs: float, mid: float, model: ExecutionModel) -> GateCheck:
    spread_rel = spread_abs / max(mid, 1e-8)
    abs_ok = spread_abs <= model.max_spread_abs
    rel_ok = spread_rel <= model.max_spread_rel

    mode = model.spread_gate_mode
    if mode == "abs_only":
        passed = abs_ok
    elif mode == "rel_only":
        passed = rel_ok
    elif mode == "abs_and_rel":
        passed = abs_ok and rel_ok
    elif mode == "abs_or_rel":
        passed = abs_ok or rel_ok
    else:
        raise ValueError(f"Unsupported spread_gate_mode: {mode}")

    return GateCheck(
        passed=passed,
        reason="ok" if passed else "spread_too_wide",
        metrics={
            "spread_abs": float(spread_abs),
            "spread_rel": float(spread_rel),
            "mid": float(mid),
            "spread_gate_abs_ok": float(abs_ok),
            "spread_gate_rel_ok": float(rel_ok),
        },
    )


def estimate_structure_roundtrip_cost(
    entry_chain: pd.DataFrame,
    structure: TradeStructure,
    model: ExecutionModel,
) -> GateCheck:
    expected_cost_usd = 0.0
    expected_spread_usd = 0.0
    expected_slippage_usd = 0.0
    expected_fees_usd = 0.0
    notional_usd = 0.0

    for leg in structure.legs:
        row = select_option_quote(entry_chain, leg.target_dte, leg.target_x, leg.call_put)
        if row is None:
            return GateCheck(False, "missing_entry_quote")

        spread_abs = float(row["ask"] - row["bid"])
        mid = float(row["mid"])
        spread_check = _spread_gate(spread_abs=spread_abs, mid=mid, model=model)
        if not spread_check.passed:
            return spread_check

        qty = int(abs(leg.weight))
        if qty == 0:
            continue

        half_spread = max(0.0, 0.5 * spread_abs)

        per_fill_spread = 0.0 if model.mode == "mid" else half_spread
        extra_slippage = 0.0
        if model.mode == "worse_than_touch":
            extra_slippage += model.worse_touch_extra_half_spread * half_spread
        extra_slippage += model.impact_bps * 1e-4 * max(mid, 0.01)

        leg_spread_cost = 2.0 * per_fill_spread * qty * 100.0
        leg_slippage_cost = 2.0 * extra_slippage * qty * 100.0
        leg_fees = 2.0 * model.fee_per_contract * qty

        expected_spread_usd += leg_spread_cost
        expected_slippage_usd += leg_slippage_cost
        expected_fees_usd += leg_fees
        expected_cost_usd += leg_spread_cost + leg_slippage_cost + leg_fees
        notional_usd += abs(mid) * qty * 100.0

    return GateCheck(
        True,
        "ok",
        {
            "expected_cost": expected_cost_usd,
            "expected_spread": expected_spread_usd,
            "expected_slippage": expected_slippage_usd,
            "expected_fees": expected_fees_usd,
            "notional": notional_usd,
            "expected_cost_usd": expected_cost_usd,
            "expected_spread_usd": expected_spread_usd,
            "expected_slippage_usd": expected_slippage_usd,
            "expected_fees_usd": expected_fees_usd,
            "notional_usd": notional_usd,
        },
    )


def execute_structure_one_day(
    entry_chain: pd.DataFrame,
    exit_chain: pd.DataFrame,
    structure: TradeStructure,
    direction: int,
    model: ExecutionModel,
    signal: float,
    edge_gross: float,
) -> ExecutionResult:
    pnl = 0.0
    turnover = 0.0
    contracts = 0
    spread_paid = 0.0
    fill_slippage = 0.0
    fees = 0.0
    delta_proxy = 0.0
    vega_proxy = 0.0
    gamma_proxy = 0.0
    blotter_rows: list[dict[str, Any]] = []

    for leg in structure.legs:
        entry_row = select_option_quote(entry_chain, leg.target_dte, leg.target_x, leg.call_put)
        if entry_row is None:
            return ExecutionResult(
                pnl=0.0,
                turnover=0.0,
                contracts=0,
                skipped=True,
                skip_reason="missing_entry_quote",
                spread_paid=0.0,
                fill_slippage=0.0,
                fees=0.0,
                holding_return=0.0,
                realized_pnl=0.0,
                edge_gross=edge_gross,
                edge_net=edge_gross,
                delta_proxy=0.0,
                vega_proxy=0.0,
                gamma_proxy=0.0,
            )

        spread_abs = float(entry_row["ask"] - entry_row["bid"])
        mid = float(entry_row["mid"])
        spread_check = _spread_gate(spread_abs=spread_abs, mid=mid, model=model)
        if not spread_check.passed:
            return ExecutionResult(
                pnl=0.0,
                turnover=0.0,
                contracts=0,
                skipped=True,
                skip_reason="spread_too_wide",
                spread_paid=0.0,
                fill_slippage=0.0,
                fees=0.0,
                holding_return=0.0,
                realized_pnl=0.0,
                edge_gross=edge_gross,
                edge_net=edge_gross,
                delta_proxy=0.0,
                vega_proxy=0.0,
                gamma_proxy=0.0,
                gate_metrics=spread_check.metrics,
            )

        symbol = str(entry_row["symbol"])
        exit_match = exit_chain.loc[exit_chain["symbol"] == symbol]
        if exit_match.empty:
            exit_row = select_option_quote(
                exit_chain,
                leg.target_dte - 1,
                leg.target_x,
                leg.call_put,
            )
            if exit_row is None:
                return ExecutionResult(
                    pnl=0.0,
                    turnover=0.0,
                    contracts=0,
                    skipped=True,
                    skip_reason="missing_exit_quote",
                    spread_paid=0.0,
                    fill_slippage=0.0,
                    fees=0.0,
                    holding_return=0.0,
                    realized_pnl=0.0,
                    edge_gross=edge_gross,
                    edge_net=edge_gross,
                    delta_proxy=0.0,
                    vega_proxy=0.0,
                    gamma_proxy=0.0,
                )
        else:
            exit_row = exit_match.iloc[0]

        signed_qty = direction * int(np.sign(leg.weight))
        qty = int(abs(leg.weight))
        if qty == 0:
            continue

        entry_fill, entry_spread, entry_slip = _fill_components(entry_row, signed_qty, model)
        exit_fill, exit_spread, exit_slip = _fill_components(exit_row, -signed_qty, model)

        leg_pnl = signed_qty * (exit_fill - entry_fill) * qty * 100.0
        leg_turnover = (abs(entry_fill) + abs(exit_fill)) * qty * 100.0
        leg_spread = (entry_spread + exit_spread) * qty * 100.0
        leg_slippage = (entry_slip + exit_slip) * qty * 100.0
        leg_fees = 2.0 * model.fee_per_contract * qty
        x = float(np.log(float(entry_row["strike"]) / float(entry_row["underlying_close"])))
        tau = max(float(entry_row["dte"]) / 365.0, 1.0 / 365.0)
        signed_contracts = float(signed_qty * qty)
        leg_delta = signed_contracts * float(np.exp(-3.0 * abs(x)))
        leg_gamma = signed_contracts * float(np.exp(-12.0 * (x**2)) / np.sqrt(tau))
        leg_vega = signed_contracts * float(entry_row["mid"]) * float(np.sqrt(tau))

        pnl += leg_pnl
        turnover += leg_turnover
        contracts += qty
        spread_paid += leg_spread
        fill_slippage += leg_slippage
        fees += leg_fees
        delta_proxy += leg_delta
        gamma_proxy += leg_gamma
        vega_proxy += leg_vega

        blotter_rows.append(
            {
                "structure": structure.name,
                "leg": leg.name,
                "symbol": symbol,
                "signal": signal,
                "qty": qty,
                "direction": signed_qty,
                "entry_mid": float(entry_row["mid"]),
                "exit_mid": float(exit_row["mid"]),
                "entry_fill": entry_fill,
                "exit_fill": exit_fill,
                "spread_paid": leg_spread,
                "fill_slippage": leg_slippage,
                "fees": leg_fees,
                "holding_return": leg_pnl / max(leg_turnover, 1e-8),
                "realized_pnl": leg_pnl - leg_fees,
                "delta_proxy": leg_delta,
                "vega_proxy": leg_vega,
                "gamma_proxy": leg_gamma,
            }
        )

    realized_pnl = pnl - fees
    holding_return = realized_pnl / max(turnover, 1e-8)
    edge_net = edge_gross - (spread_paid + fill_slippage + fees)

    return ExecutionResult(
        pnl=float(pnl),
        turnover=float(turnover),
        contracts=contracts,
        skipped=False,
        skip_reason="ok",
        spread_paid=float(spread_paid),
        fill_slippage=float(fill_slippage),
        fees=float(fees),
        holding_return=float(holding_return),
        realized_pnl=float(realized_pnl),
        edge_gross=float(edge_gross),
        edge_net=float(edge_net),
        delta_proxy=float(delta_proxy),
        vega_proxy=float(vega_proxy),
        gamma_proxy=float(gamma_proxy),
        blotter_rows=blotter_rows,
    )
