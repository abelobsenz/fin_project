"""Backtest engine for model-driven options trading simulation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for backtest.") from exc

from ivdyn.model import ModelBundle, device_auto, to_numpy
from ivdyn.finance import bs_delta
from ivdyn.eval.metrics import probabilistic_sharpe_ratio, sample_skew_kurtosis

_OPRA_TICKER_RE = re.compile(r"^O:(?P<underlying>[A-Z]+)(?P<exp>\d{6})[CP]\d{8}$")
_BACKTEST_VERSION = "2026-02-19-rf46-sizing-v1"


@dataclass(slots=True)
class BacktestConfig:
    run_dir: Path
    dataset_path: Path
    # Optional explicit output directory. If provided, artifacts are written here
    # instead of <run_dir>/backtest so live workflows can stay isolated.
    output_dir: Path | None = None
    # Optional inclusive date filter (YYYY-MM-DD). Useful for debugging and walk-forward.
    start_date: str | None = None
    end_date: str | None = None
    device: str | None = None
    num_workers: int = 0
    inference_batch_size: int = 65536
    initial_capital: float = 10_000.0
    risk_free_rate_annual: float = 0.045

    fill_gate: float = 0.65
    slippage_bps: float = 10.0
    signal_abs_gate: float = 0.04
    # Execution realism.
    spread_cross_fraction: float = 0.75
    # Explicit costs (round-trip; applied per-leg per-contract).
    option_commission_per_contract: float = 0.65
    option_fee_per_contract: float = 0.05
    # Filtering / selection on net edge.
    min_edge_to_cost_ratio: float = 1.25
    # Fill modeling: "assume" (legacy optimistic) or "expected" (EV scaled by fill_prob).
    fill_model: str = "expected"
    max_trades_per_day: int = 5
    max_contracts_per_trade: int = 4
    volume_participation_rate: float = 0.02
    open_interest_participation_rate: float = 0.01
    selector_edge_clip_quantile: float = 0.95
    selector_mid_norm_floor: float = 0.0025
    selector_signal_soft_cap: float = 250.0
    selector_long_score_scale: float = 0.0
    selector_long_abs_signal_cap: float = 6.0
    selector_allow_long_puts: bool = False

    min_dte: int = 7
    max_dte: int = 75
    min_moneyness: float = 0.88
    max_moneyness: float = 1.12
    max_rel_spread: float = 0.10
    hedge_max_net_delta_ratio: float = 0.20
    hedge_relaxed_net_delta_ratio: float = 0.30
    hedge_max_net_delta_abs: float = 0.75
    hedge_max_side_imbalance_ratio: float = 0.25

    # --- Multi-leg support ---
    #
    # strategy_mode:
    #   - "single": current behavior (one option leg per trade).
    #   - "vertical": attach a same-expiry, further-OTM wing to every option trade
    #                 to create defined-risk vertical spreads.
    strategy_mode: str = "single"

    # Vertical spread construction (wing selection)
    vertical_wing_width_pct_target: float = 0.02
    vertical_wing_width_pct_min: float = 0.01
    vertical_wing_width_pct_max: float = 0.08
    vertical_wing_max_premium_ratio: float = 0.35
    vertical_wing_fill_gate: float = 0.50
    vertical_wing_max_rel_spread: float = 0.15
    vertical_wing_min_moneyness: float = 0.75
    vertical_wing_max_moneyness: float = 1.30
    vertical_wing_rich_signal_penalty: float = 0.75
    vertical_skip_if_no_wing: bool = False
    vertical_fallback_to_single: bool = True

    # Portfolio hedging (daily close-to-close) using the underlying.
    # The hedge is sized from BS delta at entry using the day's observed IV surface.
    hedge_underlying_delta: bool = False
    hedge_underlying_ratio: float = 1.0
    hedge_underlying_min_abs_shares: float = 25.0
    hedge_underlying_max_shares: int = 5000
    hedge_underlying_slippage_bps: float = 1.0

    # Underlying hedge policy.
    # - "fixed": use hedge_underlying_ratio.
    # - "learned": load a HedgePolicyBundle and predict a state-dependent ratio.
    hedge_policy: str = "fixed"
    hedge_policy_path: str | None = None

    # Portfolio accounting constraints.
    # These gates reject trades that exceed available buying power.
    enforce_portfolio_constraints: bool = True
    buying_power_leverage: float = 1.0
    option_short_margin_rate: float = 0.20
    underlying_margin_rate: float = 0.50


def _clamp01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _annual_to_daily_rate(rate_annual: float, *, periods_per_year: int = 252) -> float:
    r = float(rate_annual)
    if not np.isfinite(r):
        return 0.0
    if r <= -1.0:
        r = -0.999999999
    return float((1.0 + r) ** (1.0 / float(max(int(periods_per_year), 1))) - 1.0)


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=True)
    out: dict[str, np.ndarray] = {}
    for k in npz.files:
        arr = npz[k]
        if arr.dtype == object:
            arr = arr.astype(str)
        out[k] = arr
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _opra_expiry(symbol: str) -> str:
    m = _OPRA_TICKER_RE.match(str(symbol))
    if not m:
        return ""
    return m.group("exp")


def _opra_underlying(symbol: str) -> str:
    m = _OPRA_TICKER_RE.match(str(symbol))
    if not m:
        return ""
    return m.group("underlying")


def _leg_pnl(
    *,
    mid_now: float,
    mid_next: float,
    rel_spread_now: float,
    rel_spread_next: float,
    slippage: float,
    spread_cross_fraction: float,
    side: int,
) -> float:
    # Realistic fill model:
    # - mid +/- (half-spread * cross_fraction) for each fill
    # - plus slippage on both entry and exit
    rel_now = float(np.clip(rel_spread_now, 0.0, 3.0))
    rel_nxt = float(np.clip(rel_spread_next, 0.0, 3.0))
    cross = float(np.clip(spread_cross_fraction, 0.0, 1.0))
    entry_cost = float(slippage + 0.5 * rel_now * cross)
    exit_cost = float(slippage + 0.5 * rel_nxt * cross)
    entry = float(mid_now * (1.0 + side * entry_cost))
    exit_ = float(mid_next * (1.0 - side * exit_cost))
    return float(side * (exit_ - entry))


def _execution_cost_norm(mid_now: float, rel_spread: float, slippage: float, spread_cross_fraction: float) -> float:
    rel_sp = float(np.clip(rel_spread, 0.0, 3.0))
    cross = float(np.clip(spread_cross_fraction, 0.0, 1.0))
    return float(mid_now * (slippage + 0.5 * rel_sp * cross))


def _option_roundtrip_fees(
    *,
    contracts: float,
    legs: int,
    commission_per_contract: float,
    fee_per_contract: float,
) -> float:
    # Round-trip: entry + exit, per leg.
    c = float(max(contracts, 0.0))
    per = float(max(commission_per_contract, 0.0) + max(fee_per_contract, 0.0))
    return float(2.0 * c * float(max(legs, 1)) * per)


def _underlying_pnl(
    *,
    spot_now: float,
    spot_next: float,
    shares: float,
    slippage: float,
) -> float:
    """Close-to-close PnL for an underlying position.

    shares > 0 means long, shares < 0 means short.
    slippage is a *relative* cost (e.g., 1 bps = 0.0001).
    """
    if not np.isfinite(shares) or shares == 0.0:
        return 0.0
    if not np.isfinite(spot_now) or not np.isfinite(spot_next):
        return 0.0
    if spot_now <= 0.0 or spot_next <= 0.0:
        return 0.0

    side = 1.0 if shares > 0 else -1.0
    sh = float(abs(shares))
    cost = float(max(slippage, 0.0))
    entry = float(spot_now * (1.0 + side * cost))
    exit_ = float(spot_next * (1.0 - side * cost))
    return float(side * sh * (exit_ - entry))


def _option_entry_price(
    *,
    mid_now: float,
    rel_spread_now: float,
    slippage: float,
    spread_cross_fraction: float,
    side: int,
) -> float:
    """Estimated option entry fill price per share used for capital checks."""
    rel_now = float(np.clip(rel_spread_now, 0.0, 3.0))
    cross = float(np.clip(spread_cross_fraction, 0.0, 1.0))
    entry_cost = float(slippage + 0.5 * rel_now * cross)
    return float(mid_now * (1.0 + side * entry_cost))


def _interp_surface_bilinear(
    surface: np.ndarray,
    *,
    x_grid: np.ndarray,
    tenor_days: np.ndarray,
    x: np.ndarray,
    dte_days: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation into a surface defined on (x_grid, tenor_days).

    surface has shape (len(x_grid), len(tenor_days)).
    x and dte_days are broadcastable to the same shape.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    tenor_days = np.asarray(tenor_days, dtype=float)
    x = np.asarray(x, dtype=float)
    t = np.asarray(dte_days, dtype=float)

    if surface.ndim != 2:
        raise ValueError("surface must be 2D")
    if surface.shape[0] != len(x_grid) or surface.shape[1] != len(tenor_days):
        raise ValueError("surface shape does not match x_grid/tenor_days")

    nx = len(x_grid)
    nt = len(tenor_days)
    if nx < 2 or nt < 2:
        # Degenerate fallback.
        return np.full_like(x, float(np.nanmedian(surface)))

    # Clamp to the grid boundaries.
    x_clamped = np.clip(x, x_grid[0], x_grid[-1])
    t_clamped = np.clip(t, tenor_days[0], tenor_days[-1])

    ix1 = np.searchsorted(x_grid, x_clamped, side="right")
    ix1 = np.clip(ix1, 1, nx - 1)
    ix0 = ix1 - 1

    it1 = np.searchsorted(tenor_days, t_clamped, side="right")
    it1 = np.clip(it1, 1, nt - 1)
    it0 = it1 - 1

    x0 = x_grid[ix0]
    x1 = x_grid[ix1]
    t0 = tenor_days[it0]
    t1 = tenor_days[it1]

    wx = (x_clamped - x0) / np.clip(x1 - x0, 1e-12, None)
    wt = (t_clamped - t0) / np.clip(t1 - t0, 1e-12, None)

    v00 = surface[ix0, it0]
    v10 = surface[ix1, it0]
    v01 = surface[ix0, it1]
    v11 = surface[ix1, it1]

    v0 = v00 + wx * (v10 - v00)
    v1 = v01 + wx * (v11 - v01)
    return v0 + wt * (v1 - v0)


def _parse_strategy_mode(raw: str) -> str:
    mode = str(raw or "").strip().lower()
    if mode in {"single", "1", "one", "one-leg"}:
        return "single"
    if mode in {"vertical", "vertical_spread", "spread", "2", "two", "two-leg"}:
        return "vertical"
    raise ValueError(f"Unknown strategy_mode={raw!r}; expected 'single' or 'vertical'.")


def _write_parquet_with_fallback(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # pragma: no cover
        msg = str(exc).lower()
        missing_engine = (
            ("unable to find a usable engine" in msg)
            or ("missing optional dependency" in msg and ("pyarrow" in msg or "fastparquet" in msg))
        )
        if not missing_engine:
            raise
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        warnings.warn(
            (
                f"Parquet engine unavailable; wrote {csv_path.name} instead of {path.name}. "
                "Install pyarrow or fastparquet to enable parquet output."
            ),
            RuntimeWarning,
            stacklevel=2,
        )


def _tenor_bucket(dte: int) -> int:
    if dte <= 14:
        return 14
    if dte <= 30:
        return 30
    if dte <= 60:
        return 60
    return 90


def _quality_scaled_contracts(
    *,
    max_contracts_per_trade: int,
    edge_to_cost_ratio: float,
    fill_prob: float,
    rel_spread: float,
    selection_score: float,
    top_selection_score: float,
    min_edge_to_cost_ratio: float,
    fill_gate: float,
    max_rel_spread: float,
    est_volume: float | None,
    est_open_interest: float | None,
    volume_participation_rate: float,
    open_interest_participation_rate: float,
) -> tuple[int, float, int]:
    """Conservative contract sizing for high-conviction, liquid ideas.

    Returns:
        contracts_target: integer contracts to attempt.
        quality_score: [0, 1] blended quality metric.
        contracts_cap: final liquidity/config cap used for sizing.
    """
    cfg_cap = max(1, int(max_contracts_per_trade))
    if cfg_cap <= 1:
        return 1, 0.0, 1

    vol_cap = cfg_cap
    if est_volume is not None and np.isfinite(est_volume) and est_volume > 0.0:
        vol_cap = max(1, int(np.floor(float(est_volume) * max(float(volume_participation_rate), 0.0))))

    oi_cap = cfg_cap
    if est_open_interest is not None and np.isfinite(est_open_interest) and est_open_interest > 0.0:
        oi_cap = max(1, int(np.floor(float(est_open_interest) * max(float(open_interest_participation_rate), 0.0))))

    contracts_cap = max(1, min(cfg_cap, vol_cap, oi_cap))
    if contracts_cap <= 1:
        return 1, 0.0, 1

    edge_floor = max(float(min_edge_to_cost_ratio) + 0.75, float(min_edge_to_cost_ratio) * 1.5)
    edge_ceiling = max(edge_floor + 1e-6, edge_floor * 2.0)
    fill_floor = float(np.clip(max(float(fill_gate) + 0.20, 0.80), 0.0, 0.99))
    spread_cap = max(min(float(max_rel_spread), 0.05), 1e-6)

    edge_quality = float(np.clip((float(edge_to_cost_ratio) - edge_floor) / (edge_ceiling - edge_floor), 0.0, 1.0))
    fill_quality = float(np.clip((float(fill_prob) - fill_floor) / max(1.0 - fill_floor, 1e-6), 0.0, 1.0))
    spread_quality = float(np.clip((spread_cap - float(rel_spread)) / spread_cap, 0.0, 1.0))
    top_score = max(float(top_selection_score), 1e-12)
    score_quality = float(np.clip(float(selection_score) / top_score, 0.0, 1.0))

    quality = edge_quality * fill_quality * spread_quality * score_quality
    if quality < 0.55:
        return 1, quality, contracts_cap

    scaled = int(np.floor((quality**2.0) * float(contracts_cap - 1) + 1e-9))
    contracts_target = int(np.clip(1 + scaled, 1, contracts_cap))
    return contracts_target, quality, contracts_cap


def _predict_contracts(
    *,
    model_bundle: ModelBundle,
    ds: dict[str, np.ndarray],
    dev: torch.device,
    batch_size: int,
    contract_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = model_bundle.model.to(dev).eval()
    iv_surface = ds["iv_surface"].astype(np.float32)
    context = ds["context"].astype(np.float32)
    contract_features = ds["contract_features"].astype(np.float32)
    date_idx = ds["contract_date_index"].astype(np.int32)

    n_dates = iv_surface.shape[0]
    surface_flat = iv_surface.reshape(n_dates, -1)
    surface_scaled = model_bundle.surface_scaler.transform(surface_flat)
    context_scaled = model_bundle.context_scaler.transform(context)
    contract_scaled = model_bundle.contract_scaler.transform(contract_features)

    with torch.no_grad():
        sf = torch.as_tensor(surface_scaled, dtype=torch.float32, device=dev)
        mu, _ = model.encode(sf)
        z_now = to_numpy(mu)

        z_prev_t = torch.as_tensor(z_now, dtype=torch.float32, device=dev)
        ctx_t = torch.as_tensor(context_scaled, dtype=torch.float32, device=dev)
        z_next = to_numpy(model.forward_dynamics(z_prev_t, ctx_t))

    n_contracts = len(contract_scaled)
    pred_now = np.full(n_contracts, np.nan, dtype=np.float32)
    pred_next = np.full(n_contracts, np.nan, dtype=np.float32)
    fill_prob = np.zeros(n_contracts, dtype=np.float32)

    if contract_indices is None:
        sel = np.arange(n_contracts, dtype=np.int32)
    else:
        sel = np.asarray(contract_indices, dtype=np.int32).reshape(-1)
        if sel.size == 0:
            return pred_now, pred_next, fill_prob

    with torch.no_grad():
        for i in range(0, int(sel.size), batch_size):
            j = min(i + batch_size, int(sel.size))
            idx_sel = sel[i:j]
            d = date_idx[idx_sel]
            cf = torch.as_tensor(contract_scaled[idx_sel], dtype=torch.float32, device=dev)
            zc_now = torch.as_tensor(z_now[d], dtype=torch.float32, device=dev)
            zc_next = torch.as_tensor(z_next[d], dtype=torch.float32, device=dev)

            p_now_scaled = to_numpy(model.forward_pricer(zc_now, cf)).reshape(-1, 1)
            p_next_scaled = to_numpy(model.forward_pricer(zc_next, cf)).reshape(-1, 1)
            pred_now[idx_sel] = model_bundle.price_scaler.inverse_transform(p_now_scaled).reshape(-1)
            pred_next[idx_sel] = model_bundle.price_scaler.inverse_transform(p_next_scaled).reshape(-1)
            logits = to_numpy(model.forward_execution_logit(zc_now, cf)).reshape(-1)
            fill_prob[idx_sel] = _sigmoid(logits).astype(np.float32)

    return pred_now, pred_next, fill_prob


def _compute_surface_latents(*, model_bundle: ModelBundle, ds: dict[str, np.ndarray], dev: torch.device) -> np.ndarray:
    """Compute per-date latent surface states z_t.

    This is used by learned hedge policies to condition the hedge ratio on the
    same smooth IV surface representation the model trades on.
    """
    model = model_bundle.model.to(dev).eval()
    iv_surface = ds["iv_surface"].astype(np.float32)
    n_dates = iv_surface.shape[0]
    surface_flat = iv_surface.reshape(n_dates, -1)
    surface_scaled = model_bundle.surface_scaler.transform(surface_flat)

    with torch.no_grad():
        sf = torch.as_tensor(surface_scaled, dtype=torch.float32, device=dev)
        mu, _ = model.encode(sf)
        z_now = to_numpy(mu)
    return np.asarray(z_now, dtype=np.float32)


def run_backtest(cfg: BacktestConfig) -> Path:
    run_dir = cfg.run_dir.resolve()
    bt_dir = cfg.output_dir.resolve() if cfg.output_dir is not None else (run_dir / "backtest")
    bt_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = cfg.dataset_path.resolve()
    ds = _load_dataset(dataset_path)

    strategy_mode = _parse_strategy_mode(cfg.strategy_mode)

    dev = torch.device(cfg.device) if cfg.device else device_auto()
    model_path = run_dir / "model.pt"

    dates = ds["dates"].astype(str)
    n_dates = len(dates)
    spot_by_date = ds.get("spot")
    if spot_by_date is None:
        spot_by_date = np.full(n_dates, np.nan, dtype=np.float32)
    spot_by_date = np.asarray(spot_by_date, dtype=np.float32).reshape(-1)

    x_grid = np.asarray(ds.get("x_grid", np.array([], dtype=np.float32)), dtype=np.float32)
    tenor_days = np.asarray(ds.get("tenor_days", np.array([], dtype=np.int32)), dtype=np.int32)
    iv_surface_all = ds.get("iv_surface")

    date_idx = ds["contract_date_index"].astype(np.int32)
    symbol = ds["contract_symbol"].astype(str)
    expiry_all = np.array([_opra_expiry(s) for s in symbol], dtype=object)
    underlying_all = np.array([_opra_underlying(s) for s in symbol], dtype=object)
    valid_underlying = underlying_all != ""
    if valid_underlying.any():
        vals, counts = np.unique(underlying_all[valid_underlying], return_counts=True)
        underlying_symbol = str(vals[int(np.argmax(counts))])
        underlying_mask = underlying_all == underlying_symbol
    else:
        # Fallback for non-standard symbols: keep options-like rows.
        underlying_symbol = "UNDERLYING"
        underlying_mask = np.char.startswith(symbol.astype(str), "O:")
    if not underlying_mask.any():
        underlying_mask = np.ones(len(symbol), dtype=bool)
    call_put = ds["contract_call_put"].astype(str)
    dte = ds["contract_dte"].astype(np.int32)
    strike = ds["contract_strike"].astype(np.float32)
    spot = ds["contract_spot"].astype(np.float32)
    mid_now = ds["contract_mid"].astype(np.float32)
    mid_now_norm = mid_now / np.clip(spot, 1e-6, None)
    date_arr = ds["contract_date"].astype(str)

    features = ds["contract_features"].astype(np.float32)
    feature_names = ds.get("contract_feature_names", np.array([], dtype=str)).astype(str).tolist()
    fidx = {name: i for i, name in enumerate(feature_names)}
    rel_spread = features[:, fidx["rel_spread"]] if "rel_spread" in fidx else np.full(len(features), np.nan, dtype=np.float32)
    log_volume = (
        features[:, fidx["log_volume"]] if "log_volume" in fidx else np.full(len(features), np.nan, dtype=np.float32)
    )
    log_open_interest = (
        features[:, fidx["log_open_interest"]]
        if "log_open_interest" in fidx
        else np.full(len(features), np.nan, dtype=np.float32)
    )
    cp_sign = features[:, fidx["cp_sign"]] if "cp_sign" in fidx else np.where(call_put == "C", 1.0, -1.0).astype(np.float32)
    cp_sign = np.where(np.isfinite(cp_sign), np.sign(cp_sign), np.where(call_put == "C", 1.0, -1.0)).astype(np.float32)
    cp_sign = np.where(cp_sign == 0.0, np.where(call_put == "C", 1.0, -1.0), cp_sign).astype(np.float32)

    moneyness = strike / np.clip(spot, 1e-6, None)
    date_next_idx = np.clip(date_idx + 1, 0, n_dates - 1)
    date_next_arr = dates[date_next_idx]

    next_key_mid: dict[tuple[int, str], float] = {}
    next_key_mid_norm: dict[tuple[int, str], float] = {}
    next_key_rel_spread: dict[tuple[int, str], float] = {}
    for i in range(len(symbol)):
        key = (int(date_idx[i]), str(symbol[i]))
        next_key_mid[key] = float(mid_now[i])
        next_key_mid_norm[key] = float(mid_now_norm[i])
        next_key_rel_spread[key] = float(rel_spread[i]) if np.isfinite(rel_spread[i]) else float("nan")
    mid_next = np.full(len(symbol), np.nan, dtype=np.float32)
    mid_next_norm = np.full(len(symbol), np.nan, dtype=np.float32)
    rel_spread_next = np.full(len(symbol), np.nan, dtype=np.float32)
    for i in range(len(symbol)):
        k = (int(date_idx[i] + 1), str(symbol[i]))
        if k in next_key_mid:
            mid_next[i] = np.float32(next_key_mid[k])
            mid_next_norm[i] = np.float32(next_key_mid_norm[k])
            rel_spread_next[i] = np.float32(next_key_rel_spread.get(k, float("nan")))

    # --- Inference (subset) ---
    # Predict only for contracts in the reasonable trading universe to keep the workflow fast.
    # This prevents inference over the full (often huge) contract universe.
    universe_mask = (
        (date_idx < (n_dates - 1))
        & underlying_mask
        & np.isfinite(mid_next)
        & (dte >= int(cfg.min_dte))
        & (dte <= int(cfg.max_dte))
        & (moneyness >= float(min(cfg.min_moneyness, cfg.vertical_wing_min_moneyness)))
        & (moneyness <= float(max(cfg.max_moneyness, cfg.vertical_wing_max_moneyness)))
        & (rel_spread <= float(max(cfg.max_rel_spread, cfg.vertical_wing_max_rel_spread)))
        & (mid_now_norm >= float(cfg.selector_mid_norm_floor) * 0.50)
    )
    if cfg.start_date or cfg.end_date:
        start = str(cfg.start_date) if cfg.start_date else ""
        end = str(cfg.end_date) if cfg.end_date else "9999-12-31"
        universe_mask &= (date_arr >= start) & (date_arr <= end)

    contract_indices = np.flatnonzero(universe_mask).astype(np.int32)

    bundle = ModelBundle.load(model_path, device=dev)

    # Optional learned hedge policy (state-dependent hedge ratio).
    hedge_policy_bundle = None
    hedge_policy_latents = None
    hedge_policy_context = ds.get("context")
    hedge_policy_kind = str(getattr(cfg, "hedge_policy", "fixed") or "fixed").strip().lower()
    if bool(cfg.hedge_underlying_delta) and hedge_policy_kind not in {"fixed", "learned"}:
        raise ValueError(f"Unknown hedge_policy={cfg.hedge_policy!r}; expected 'fixed' or 'learned'.")

    if bool(cfg.hedge_underlying_delta) and hedge_policy_kind == "learned":
        if not cfg.hedge_policy_path:
            raise RuntimeError("hedge_policy='learned' requires --hedge-policy-path")
        from ivdyn.hedge_policy import HedgePolicyBundle

        hedge_policy_bundle = HedgePolicyBundle.load(Path(cfg.hedge_policy_path), device=dev)
        if hedge_policy_bundle.feature_spec.use_latent:
            hedge_policy_latents = _compute_surface_latents(model_bundle=bundle, ds=ds, dev=dev)

    cache_path = bt_dir / "pred_cache.npz"
    cache_meta_path = bt_dir / "pred_cache_meta.json"
    use_cache = False
    sig = {
        "dataset_name": dataset_path.name,
        "model_name": model_path.name,
        "dataset_mtime": int(dataset_path.stat().st_mtime),
        "model_mtime": int(model_path.stat().st_mtime),
        "dataset_size": int(dataset_path.stat().st_size),
        "model_size": int(model_path.stat().st_size),
        "universe_n": int(contract_indices.size),
        "universe_cfg": {
            "min_dte": int(cfg.min_dte),
            "max_dte": int(cfg.max_dte),
            "min_moneyness": float(cfg.min_moneyness),
            "max_moneyness": float(cfg.max_moneyness),
            "max_rel_spread": float(cfg.max_rel_spread),
            "vertical_wing_min_moneyness": float(cfg.vertical_wing_min_moneyness),
            "vertical_wing_max_moneyness": float(cfg.vertical_wing_max_moneyness),
            "vertical_wing_max_rel_spread": float(cfg.vertical_wing_max_rel_spread),
            "start_date": str(cfg.start_date) if cfg.start_date else None,
            "end_date": str(cfg.end_date) if cfg.end_date else None,
        },
    }
    if cache_path.exists() and cache_meta_path.exists():
        try:
            meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
            if meta == sig:
                use_cache = True
        except Exception:
            use_cache = False

    if use_cache:
        cached = np.load(cache_path, allow_pickle=False)
        pred_now_norm = cached["pred_now_norm"].astype(np.float32)
        pred_next_norm = cached["pred_next_norm"].astype(np.float32)
        fill_prob = cached["fill_prob"].astype(np.float32)
    else:
        pred_now_norm, pred_next_norm, fill_prob = _predict_contracts(
            model_bundle=bundle,
            ds=ds,
            dev=dev,
            batch_size=max(1, int(cfg.inference_batch_size)),
            contract_indices=contract_indices,
        )
        np.savez_compressed(
            cache_path,
            pred_now_norm=pred_now_norm.astype(np.float32),
            pred_next_norm=pred_next_norm.astype(np.float32),
            fill_prob=fill_prob.astype(np.float32),
        )
        cache_meta_path.write_text(json.dumps(sig, indent=2), encoding="utf-8")

    edge_raw_norm = mid_now_norm - pred_next_norm
    edge = np.divide(
        edge_raw_norm,
        mid_now_norm,
        out=np.zeros_like(edge_raw_norm),
        where=mid_now_norm > 0,
    )
    edge_usd_per_share = edge_raw_norm * spot

    tradable = (
        (date_idx < (n_dates - 1))
        & underlying_mask
        & (dte >= int(cfg.min_dte))
        & (dte <= int(cfg.max_dte))
        & (moneyness >= float(cfg.min_moneyness))
        & (moneyness <= float(cfg.max_moneyness))
        & (rel_spread <= float(cfg.max_rel_spread))
        & (fill_prob >= float(cfg.fill_gate))
        & np.isfinite(mid_next)
        & (mid_now_norm >= float(cfg.selector_mid_norm_floor))
        & (np.abs(edge) >= float(cfg.signal_abs_gate))
    )

    if cfg.start_date or cfg.end_date:
        start = str(cfg.start_date) if cfg.start_date else ""
        end = str(cfg.end_date) if cfg.end_date else "9999-12-31"
        tradable &= (date_arr >= start) & (date_arr <= end)

    side = np.full(len(symbol), "", dtype=object)
    long_mask = tradable & (edge < 0.0)
    short_mask = tradable & (edge > 0.0)

    side[short_mask] = "SHORT"
    side[long_mask] = "LONG"
    active = side != ""

    active_idx = np.flatnonzero(active).astype(np.int32)

    candidates = pd.DataFrame(
        {
            "contract_idx": active_idx,
            "date_idx": date_idx[active],
            "date": date_arr[active],
            "date_next": date_next_arr[active],
            "symbol": symbol[active],
            "expiry": expiry_all[active],
            "call_put": call_put[active],
            "strike": strike[active].astype(float),
            "dte": dte[active].astype(int),
            "mid_now": mid_now[active].astype(float),
            "mid_next": mid_next[active].astype(float),
            "mid_now_norm": mid_now_norm[active].astype(float),
            "mid_next_norm": mid_next_norm[active].astype(float),
            "pred_now_norm": pred_now_norm[active].astype(float),
            "pred_next_norm": pred_next_norm[active].astype(float),
            "signal": edge[active].astype(float),
            "edge_usd_per_share": edge_usd_per_share[active].astype(float),
            "fill_prob": fill_prob[active].astype(float),
            "moneyness": moneyness[active].astype(float),
            "rel_spread": rel_spread[active].astype(float),
            "log_volume": log_volume[active].astype(float),
            "log_open_interest": log_open_interest[active].astype(float),
            "cp_sign": cp_sign[active].astype(float),
            "spot": spot[active].astype(float),
            "side": side[active].astype(str),
        }
    )

    keep_cols = [
        "date",
        "date_next",
        "strategy_mode",
        "contract_idx",
        "symbol",
        "side",
        "call_put",
        "strike",
        "dte",
        "mid_now_norm",
        "mid_next_norm",
        "pred_now_norm",
        "pred_next_norm",
        "signal",
        "edge_usd_per_share",
        "selection_score",
        "delta_proxy",
        "fill_prob",
        "tenor_bucket",
        "expiry",
        "wing_symbol",
        "wing_strike",
        "wing_side",
        "wing_signal",
        "wing_edge_usd_per_share",
        "wing_pnl_per_contract",
        "legs",
        "pnl_per_contract",
        "ev_per_contract",
        "risk_score",
        "execution_cost_per_contract",
        "execution_cost_ratio",
        "max_fill_distance",
        "contracts_target",
        "contracts",
        "contracts_cap",
        "contract_quality",
        "notional",
        "capital_required",
        "buying_power_limit",
        "buying_power_used_before_trade",
        "buying_power_used_after_trade",
        "fees",
        "pnl_before_fees",
        "pnl_gross",
        "pnl",
    ]

    slippage = float(cfg.slippage_bps) / 1e4
    selector_edge_q = float(np.clip(cfg.selector_edge_clip_quantile, 0.50, 1.00))
    selector_signal_soft_cap = max(float(cfg.selector_signal_soft_cap), 1.0)
    selector_long_score_scale = float(np.clip(cfg.selector_long_score_scale, 0.0, 1.0))
    selector_long_abs_signal_cap = max(float(cfg.selector_long_abs_signal_cap), 0.0)
    hedge_ratio = max(float(cfg.hedge_max_net_delta_ratio), 0.0)
    hedge_ratio_relaxed = max(float(cfg.hedge_relaxed_net_delta_ratio), hedge_ratio)
    hedge_abs = max(float(cfg.hedge_max_net_delta_abs), 0.0)
    hedge_cp_ratio = float(np.clip(cfg.hedge_max_side_imbalance_ratio, 0.0, 1.0))
    enforce_portfolio_constraints = bool(getattr(cfg, "enforce_portfolio_constraints", True))
    buying_power_leverage = max(float(getattr(cfg, "buying_power_leverage", 1.0)), 0.0)
    option_short_margin_rate = max(float(getattr(cfg, "option_short_margin_rate", 0.20)), 0.0)
    underlying_margin_rate = max(float(getattr(cfg, "underlying_margin_rate", 0.50)), 0.0)
    contract_multiplier = 100.0
    trade_rows: list[dict[str, object]] = []
    leg_rows: list[dict[str, object]] = []
    hedge_rows: list[dict[str, object]] = []
    capital_rows: list[dict[str, object]] = []
    capital_rejected_trades_total = 0
    capital_rejected_hedges_total = 0
    running_equity = float(cfg.initial_capital)
    if not np.isfinite(running_equity):
        raise ValueError(f"initial_capital must be finite, got {cfg.initial_capital!r}")
    trade_id = 0
    for d in sorted(candidates["date_idx"].unique().tolist()):
        day = candidates[candidates["date_idx"] == d].reset_index(drop=True)
        if day.empty:
            continue

        day_date = str(day.at[0, "date"])
        day_date_next = str(day.at[0, "date_next"])
        equity_start_day = float(running_equity)
        buying_power_limit_day = float(max(equity_start_day, 0.0) * buying_power_leverage)
        buying_power_used_options_day = 0.0
        buying_power_used_hedge_day = 0.0
        capital_rejected_trades_day = 0
        capital_rejected_hedges_day = 0
        day_options_pnl_total = 0.0

        day_cap = max(0, int(cfg.max_trades_per_day))
        if day_cap == 0:
            capital_rows.append(
                {
                    "date": day_date,
                    "date_next": day_date_next,
                    "equity_start": equity_start_day,
                    "equity_end": equity_start_day,
                    "buying_power_limit": buying_power_limit_day,
                    "buying_power_used_options": 0.0,
                    "buying_power_used_hedge": 0.0,
                    "buying_power_used_total": 0.0,
                    "capital_rejected_trades": 0,
                    "capital_rejected_hedges": 0,
                }
            )
            continue

        # Optional multi-leg universe (broader than the alpha-selection universe).
        wing_universe_idx: np.ndarray | None = None
        if strategy_mode == "vertical":
            wing_mask = (
                (date_idx == int(d))
                & (date_idx < (n_dates - 1))
                & underlying_mask
                & (dte >= int(cfg.min_dte))
                & (dte <= int(cfg.max_dte))
                & (moneyness >= float(cfg.vertical_wing_min_moneyness))
                & (moneyness <= float(cfg.vertical_wing_max_moneyness))
                & (rel_spread <= float(cfg.vertical_wing_max_rel_spread))
                & (fill_prob >= float(cfg.vertical_wing_fill_gate))
                & np.isfinite(mid_next)
                & (mid_now_norm >= float(cfg.selector_mid_norm_floor) * 0.50)
            )
            wing_universe_idx = np.flatnonzero(wing_mask).astype(np.int32)

        abs_edge_day = np.abs(day["edge_usd_per_share"].to_numpy(dtype=float))
        edge_cap = float(np.quantile(abs_edge_day, selector_edge_q)) if len(abs_edge_day) else 0.0
        if not np.isfinite(edge_cap) or edge_cap <= 0.0:
            edge_cap = float(np.nanmax(abs_edge_day)) if len(abs_edge_day) else 0.0
        edge_rank = np.minimum(abs_edge_day, edge_cap)

        fill_day = np.clip(day["fill_prob"].to_numpy(dtype=float), 0.0, 1.0)
        mid_now_day = np.clip(day["mid_now"].to_numpy(dtype=float), 0.0, None)
        rel_sp_day = np.clip(day["rel_spread"].to_numpy(dtype=float), 0.0, 3.0)
        abs_signal_day = np.abs(day["signal"].to_numpy(dtype=float))

        spread_cross = float(np.clip(cfg.spread_cross_fraction, 0.0, 1.0))
        exec_cost_est = np.array(
            [
                _execution_cost_norm(float(m), float(rs), slippage, spread_cross)
                for m, rs in zip(mid_now_day, rel_sp_day, strict=False)
            ],
            dtype=float,
        )
        # Conservative round-trip cost estimate (entry+exit) using today's spread.
        roundtrip_exec_cost = 2.0 * exec_cost_est
        fees_rt = 2.0 * (float(cfg.option_commission_per_contract) + float(cfg.option_fee_per_contract))
        # Convert fees to "per share" space (contract_multiplier shares).
        fees_rt_per_share = fees_rt / contract_multiplier

        # Net edge after expected execution & fees (still in $/share).
        edge_net = np.clip(edge_rank - roundtrip_exec_cost - fees_rt_per_share, 0.0, None)

        # Require the edge to clear costs by a ratio (prevents micro-edge cost bleed).
        denom = np.clip(roundtrip_exec_cost + fees_rt_per_share, 1e-12, None)
        edge_to_cost = np.divide(edge_rank, denom, out=np.zeros_like(edge_rank), where=denom > 0)
        edge_ok = edge_to_cost >= float(cfg.min_edge_to_cost_ratio)

        spread_penalty = 1.0 + rel_sp_day
        signal_penalty = 1.0 / (1.0 + (abs_signal_day / selector_signal_soft_cap))
        fill_factor = fill_day if str(cfg.fill_model).lower().startswith("exp") else 1.0
        selection_score = np.where(edge_ok, edge_net * fill_factor * signal_penalty / spread_penalty, 0.0)

        cp_day = np.clip(day["cp_sign"].to_numpy(dtype=float), -1.0, 1.0)
        side_day = day["side"].to_numpy(dtype=str)
        is_long_day = side_day == "LONG"
        long_allowed = ~is_long_day | (abs_signal_day <= selector_long_abs_signal_cap)
        if not bool(cfg.selector_allow_long_puts):
            long_allowed &= ~(is_long_day & (cp_day < 0.0))
        selection_score = np.where(is_long_day, selection_score * selector_long_score_scale, selection_score)
        selection_score = np.where(long_allowed, selection_score, 0.0)
        if np.isfinite(selection_score).any():
            top_selection_score = float(np.nanmax(selection_score))
        else:
            top_selection_score = 0.0
        side_sign_day = np.where(side_day == "LONG", 1.0, -1.0)
        moneyness_day = np.clip(day["moneyness"].to_numpy(dtype=float), 0.1, 10.0)
        delta_mag = np.clip(np.exp(-25.0 * np.abs(moneyness_day - 1.0)), 0.05, 1.0)
        delta_proxy_day = side_sign_day * cp_day * delta_mag

        order = np.argsort(-selection_score).astype(np.int64)
        cp_imbalance_limit = max(1, int(np.ceil(day_cap * hedge_cp_ratio)))
        selected_idx: list[int] = []
        selected_set: set[int] = set()
        net_delta_proxy = 0.0
        gross_delta_proxy = 0.0
        call_count = 0
        put_count = 0

        def _try_pick(i: int, ratio: float, side_limit: int) -> bool:
            nonlocal net_delta_proxy, gross_delta_proxy, call_count, put_count
            if i in selected_set:
                return False
            score_i = float(selection_score[i])
            if not np.isfinite(score_i) or score_i <= 0.0:
                return False

            cp_lbl_i = str(day.at[i, "call_put"])
            call_new = call_count + (1 if cp_lbl_i == "C" else 0)
            put_new = put_count + (1 if cp_lbl_i == "P" else 0)
            if abs(call_new - put_new) > side_limit:
                return False

            delta_i = float(delta_proxy_day[i])
            net_new = net_delta_proxy + delta_i
            gross_new = gross_delta_proxy + abs(delta_i)
            allowed = hedge_abs + ratio * max(gross_new, 1.0)
            if abs(net_new) > allowed:
                return False

            selected_idx.append(i)
            selected_set.add(i)
            net_delta_proxy = net_new
            gross_delta_proxy = gross_new
            call_count = call_new
            put_count = put_new
            return True

        for i_raw in order:
            if len(selected_idx) >= day_cap:
                break
            _try_pick(int(i_raw), hedge_ratio, cp_imbalance_limit)

        if len(selected_idx) < day_cap:
            for i_raw in order:
                if len(selected_idx) >= day_cap:
                    break
                _try_pick(int(i_raw), hedge_ratio_relaxed, cp_imbalance_limit)

        # Build trade structures (1- or 2-leg option positions) and optional daily underlying hedge.
        day_legs: list[dict[str, object]] = []
        option_leg_idx: list[int] = []
        option_leg_side_sign: list[float] = []
        option_leg_contracts: list[float] = []

        def _pick_vertical_wing(
            *,
            anchor_contract_idx: int,
            anchor_side_lbl: str,
            anchor_expiry: str,
            anchor_cp: str,
            anchor_strike: float,
            anchor_spot: float,
            anchor_mid_now: float,
        ) -> int | None:
            if wing_universe_idx is None or wing_universe_idx.size == 0:
                return None

            cp = "C" if str(anchor_cp).upper().startswith("C") else "P"
            exp = str(anchor_expiry)
            spot_i = float(anchor_spot)
            if not np.isfinite(spot_i) or spot_i <= 0.0:
                return None

            width_tgt = float(cfg.vertical_wing_width_pct_target) * spot_i
            width_min = float(cfg.vertical_wing_width_pct_min) * spot_i
            width_max = float(cfg.vertical_wing_width_pct_max) * spot_i
            width_min = float(max(width_min, 0.0))
            width_max = float(max(width_max, width_min))
            target_strike = float(anchor_strike + (width_tgt if cp == "C" else -width_tgt))

            cand = wing_universe_idx
            same_cp = call_put[cand] == cp
            same_exp = expiry_all[cand] == exp
            cand = cand[same_cp & same_exp]
            if cand.size == 0:
                return None

            strike_c = strike[cand].astype(float)
            if cp == "C":
                keep = (strike_c >= float(anchor_strike + width_min)) & (strike_c <= float(anchor_strike + width_max))
            else:
                keep = (strike_c <= float(anchor_strike - width_min)) & (strike_c >= float(anchor_strike - width_max))
            cand = cand[keep]
            if cand.size == 0:
                return None

            strike_c = strike[cand].astype(float)
            mid_c = mid_now[cand].astype(float)
            prem_ratio = mid_c / max(float(anchor_mid_now), 1e-12)
            ok_prem = prem_ratio <= float(cfg.vertical_wing_max_premium_ratio)
            if not ok_prem.any():
                return None
            cand = cand[ok_prem]
            strike_c = strike_c[ok_prem]

            dist = np.abs(strike_c - target_strike)

            # Prefer wings that do not look "rich" against the model in the direction we are trading.
            wing_side_lbl = "LONG" if str(anchor_side_lbl) == "SHORT" else "SHORT"
            wing_is_long = wing_side_lbl == "LONG"
            wing_signal = edge[cand].astype(float)
            rich_penalty = np.clip(wing_signal, 0.0, None) if wing_is_long else np.clip(-wing_signal, 0.0, None)
            score = dist + float(cfg.vertical_wing_rich_signal_penalty) * rich_penalty
            j = int(np.argmin(score))
            return int(cand[j])

        for i in selected_idx:
            i = int(i)
            side_lbl = str(day.at[i, "side"])
            cp = str(day.at[i, "call_put"])
            exp = str(day.at[i, "expiry"])
            dte_i = int(day.at[i, "dte"])
            spot_i = float(day.at[i, "spot"])
            if not np.isfinite(spot_i) or spot_i <= 0.0:
                continue
            notional = float(spot_i * contract_multiplier)

            anchor_contract_idx = int(day.at[i, "contract_idx"])
            anchor_symbol = str(day.at[i, "symbol"])

            main_side = 1 if side_lbl == "LONG" else -1
            mid_now_main = float(day.at[i, "mid_now"])
            mid_next_main = float(day.at[i, "mid_next"])
            mid_now_main_norm = float(day.at[i, "mid_now_norm"])
            mid_next_main_norm = float(day.at[i, "mid_next_norm"])
            rel_sp_main = float(day.at[i, "rel_spread"])
            if not np.isfinite(mid_now_main) or not np.isfinite(mid_next_main):
                continue

            rel_sp_next_main = float(rel_spread_next[anchor_contract_idx])
            pnl_per_share_main = _leg_pnl(
                mid_now=mid_now_main,
                mid_next=mid_next_main,
                rel_spread_now=rel_sp_main,
                rel_spread_next=rel_sp_next_main,
                slippage=slippage,
                spread_cross_fraction=float(cfg.spread_cross_fraction),
                side=main_side,
            )
            exec_cost_per_share_main = _execution_cost_norm(mid_now_main, rel_sp_main, slippage, float(cfg.spread_cross_fraction))

            pnl_per_contract_main = float(pnl_per_share_main * contract_multiplier)
            signal_i = float(day.at[i, "signal"])
            edge_usd_per_share_i = float(day.at[i, "edge_usd_per_share"])
            ev_per_contract_main = float((-main_side * edge_usd_per_share_i) * contract_multiplier)
            execution_cost_per_contract_main = float(exec_cost_per_share_main * contract_multiplier)
            execution_cost_ratio_main = float(exec_cost_per_share_main / max(mid_now_main, 1e-12))
            max_fill_distance = float(np.clip(rel_sp_main * 0.25, 0.0, None))

            wing_symbol = ""
            wing_strike = float("nan")
            wing_side_lbl = ""
            wing_side = 0
            wing_mid_now = float("nan")
            wing_rel_sp = float("nan")
            wing_signal = float("nan")
            wing_edge_usd_per_share = float("nan")
            wing_pnl_per_contract = 0.0
            legs = 1
            pnl_per_contract = pnl_per_contract_main
            ev_per_contract = ev_per_contract_main
            execution_cost_per_contract = execution_cost_per_contract_main
            execution_cost_ratio = execution_cost_ratio_main

            fill_p = float(day.at[i, "fill_prob"])
            fill_factor = _clamp01(fill_p) if str(cfg.fill_model).lower().startswith("exp") else 1.0
            log_volume_i = float(day.at[i, "log_volume"])
            log_open_interest_i = float(day.at[i, "log_open_interest"])
            est_volume_i = (
                float(np.expm1(np.clip(log_volume_i, 0.0, 20.0))) if np.isfinite(log_volume_i) else None
            )
            est_oi_i = (
                float(np.expm1(np.clip(log_open_interest_i, 0.0, 20.0)))
                if np.isfinite(log_open_interest_i)
                else None
            )
            contracts_target, contract_quality, contracts_cap = _quality_scaled_contracts(
                max_contracts_per_trade=int(cfg.max_contracts_per_trade),
                edge_to_cost_ratio=float(edge_to_cost[i]),
                fill_prob=fill_p,
                rel_spread=rel_sp_main,
                selection_score=float(selection_score[i]),
                top_selection_score=top_selection_score,
                min_edge_to_cost_ratio=float(cfg.min_edge_to_cost_ratio),
                fill_gate=float(cfg.fill_gate),
                max_rel_spread=float(cfg.max_rel_spread),
                est_volume=est_volume_i,
                est_open_interest=est_oi_i,
                volume_participation_rate=float(cfg.volume_participation_rate),
                open_interest_participation_rate=float(cfg.open_interest_participation_rate),
            )
            contracts_filled = float(contracts_target) * float(fill_factor)
            fees = _option_roundtrip_fees(
                contracts=contracts_filled,
                legs=legs,
                commission_per_contract=float(cfg.option_commission_per_contract),
                fee_per_contract=float(cfg.option_fee_per_contract),
            )
            pnl_before_fees = float(pnl_per_contract * contracts_filled)
            pnl_gross = pnl_before_fees
            pnl_net = float(pnl_before_fees - fees)

            wing_contract_idx: int | None = None
            if strategy_mode == "vertical":
                wing_contract_idx = _pick_vertical_wing(
                    anchor_contract_idx=anchor_contract_idx,
                    anchor_side_lbl=side_lbl,
                    anchor_expiry=exp,
                    anchor_cp=cp,
                    anchor_strike=float(day.at[i, "strike"]),
                    anchor_spot=spot_i,
                    anchor_mid_now=mid_now_main,
                )
                if wing_contract_idx is None and bool(cfg.vertical_skip_if_no_wing):
                    continue

            if wing_contract_idx is not None:
                wing_side_lbl = "LONG" if side_lbl == "SHORT" else "SHORT"
                wing_side = 1 if wing_side_lbl == "LONG" else -1
                wing_symbol = str(symbol[wing_contract_idx])
                wing_strike = float(strike[wing_contract_idx])

                mid_now_w = float(mid_now[wing_contract_idx])
                mid_next_w = float(mid_next[wing_contract_idx])
                rel_sp_w = float(rel_spread[wing_contract_idx])
                wing_mid_now = float(mid_now_w)
                wing_rel_sp = float(rel_sp_w)
                if np.isfinite(mid_now_w) and np.isfinite(mid_next_w):
                    pnl_per_share_w = _leg_pnl(
                        mid_now=mid_now_w,
                        mid_next=mid_next_w,
                        rel_spread_now=rel_sp_w,
                        rel_spread_next=float(rel_spread_next[int(wing_contract_idx)]),
                        slippage=slippage,
                        spread_cross_fraction=float(cfg.spread_cross_fraction),
                        side=wing_side,
                    )
                    exec_cost_per_share_w = _execution_cost_norm(mid_now_w, rel_sp_w, slippage, float(cfg.spread_cross_fraction))

                    wing_pnl_per_contract = float(pnl_per_share_w * contract_multiplier)
                    wing_signal = float(edge[wing_contract_idx])
                    wing_edge_usd_per_share = float(edge_usd_per_share[wing_contract_idx])
                    ev_per_contract_w = float((-wing_side * wing_edge_usd_per_share) * contract_multiplier)
                    execution_cost_per_contract_w = float(exec_cost_per_share_w * contract_multiplier)
                    execution_cost_ratio_w = float(exec_cost_per_share_w / max(mid_now_w, 1e-12))

                    legs = 2
                    pnl_per_contract = pnl_per_contract_main + wing_pnl_per_contract
                    ev_per_contract = ev_per_contract_main + ev_per_contract_w
                    execution_cost_per_contract = execution_cost_per_contract_main + execution_cost_per_contract_w
                    execution_cost_ratio = float(0.5 * (execution_cost_ratio_main + execution_cost_ratio_w))

                    # Update fees and net PnL for 2-leg.
                    fees = _option_roundtrip_fees(
                        contracts=contracts_filled,
                        legs=legs,
                        commission_per_contract=float(cfg.option_commission_per_contract),
                        fee_per_contract=float(cfg.option_fee_per_contract),
                    )
                    pnl_before_fees = float(pnl_per_contract * contracts_filled)
                    pnl_gross = pnl_before_fees
                    pnl_net = float(pnl_before_fees - fees)
                else:
                    if bool(cfg.vertical_skip_if_no_wing):
                        continue
                    wing_contract_idx = None
                    wing_symbol = ""
                    wing_strike = float("nan")
                    wing_side_lbl = ""

            def _option_leg_capital_req(side_sign: int, entry_price: float, spot_ref: float) -> float:
                px = float(max(entry_price, 0.0))
                entry_value = px * contract_multiplier
                if side_sign > 0:
                    return float(max(entry_value, 0.0))
                short_req = float(max(spot_ref, 0.0) * contract_multiplier * option_short_margin_rate)
                return float(max(short_req - entry_value, 0.0))

            entry_main = _option_entry_price(
                mid_now=mid_now_main,
                rel_spread_now=rel_sp_main,
                slippage=slippage,
                spread_cross_fraction=float(cfg.spread_cross_fraction),
                side=main_side,
            )
            capital_per_contract = _option_leg_capital_req(main_side, entry_main, spot_i)
            if wing_side != 0 and np.isfinite(wing_mid_now):
                entry_wing = _option_entry_price(
                    mid_now=float(wing_mid_now),
                    rel_spread_now=float(wing_rel_sp),
                    slippage=slippage,
                    spread_cross_fraction=float(cfg.spread_cross_fraction),
                    side=int(wing_side),
                )
                capital_per_contract += _option_leg_capital_req(int(wing_side), entry_wing, spot_i)

            capital_required = float(max(contracts_filled, 0.0) * max(capital_per_contract, 0.0))
            buying_power_before_trade = float(buying_power_used_options_day)
            if enforce_portfolio_constraints and (capital_required > 0.0):
                if (buying_power_before_trade + capital_required) > (buying_power_limit_day + 1e-9):
                    capital_rejected_trades_day += 1
                    capital_rejected_trades_total += 1
                    continue
            buying_power_used_options_day = float(buying_power_before_trade + capital_required)
            buying_power_after_trade = float(buying_power_used_options_day)

            # Apply fill model scaling to EV and costs too (expected fills).
            ev_per_contract = float(ev_per_contract * fill_factor)
            execution_cost_per_contract = float(execution_cost_per_contract * fill_factor)

            # Structures (trades.parquet) remain one-row per anchor idea for compatibility.
            trade_rows.append(
                {
                    "date": str(day.at[i, "date"]),
                    "date_next": str(day.at[i, "date_next"]),
                    "strategy_mode": strategy_mode,
                    "contract_idx": anchor_contract_idx,
                    "symbol": anchor_symbol,
                    "side": side_lbl,
                    "call_put": cp,
                    "strike": float(day.at[i, "strike"]),
                    "dte": dte_i,
                    "mid_now_norm": mid_now_main_norm,
                    "mid_next_norm": mid_next_main_norm,
                    "pred_now_norm": float(day.at[i, "pred_now_norm"]),
                    "pred_next_norm": float(day.at[i, "pred_next_norm"]),
                    "signal": signal_i,
                    "edge_usd_per_share": edge_usd_per_share_i,
                    "selection_score": float(selection_score[i]),
                    "delta_proxy": float(delta_proxy_day[i]),
                    "fill_prob": float(day.at[i, "fill_prob"]),
                    "tenor_bucket": int(_tenor_bucket(dte_i)),
                    "expiry": exp,
                    "wing_symbol": wing_symbol,
                    "wing_strike": wing_strike,
                    "wing_side": wing_side_lbl,
                    "wing_signal": wing_signal,
                    "wing_edge_usd_per_share": wing_edge_usd_per_share,
                    "wing_pnl_per_contract": wing_pnl_per_contract,
                    "legs": legs,
                    "pnl_per_contract": pnl_per_contract,
                    "ev_per_contract": ev_per_contract,
                    "risk_score": float(abs(edge_usd_per_share_i)),
                    "execution_cost_per_contract": execution_cost_per_contract,
                    "execution_cost_ratio": execution_cost_ratio,
                    "max_fill_distance": max_fill_distance,
                    "contracts_target": int(contracts_target),
                    "contracts": float(contracts_filled),
                    "contracts_cap": int(contracts_cap),
                    "contract_quality": float(contract_quality),
                    "notional": float(notional * contracts_filled),
                    "capital_required": float(capital_required),
                    "buying_power_limit": float(buying_power_limit_day),
                    "buying_power_used_before_trade": float(buying_power_before_trade),
                    "buying_power_used_after_trade": float(buying_power_after_trade),
                    "fees": fees,
                    "pnl_before_fees": pnl_before_fees,
                    "pnl_gross": pnl_gross,
                    "pnl": pnl_net,
                }
            )
            day_options_pnl_total += float(pnl_net)

            # Leg-level log (legs.parquet) for diagnostics and hedging.
            trade_key = f"{str(day.at[i, 'date'])}_{trade_id:06d}"
            trade_id += 1

            day_legs.append(
                {
                    "trade_key": trade_key,
                    "date": str(day.at[i, "date"]),
                    "date_next": str(day.at[i, "date_next"]),
                    "instrument": "OPTION",
                    "leg_role": "ANCHOR",
                    "contract_idx": anchor_contract_idx,
                    "symbol": anchor_symbol,
                    "side": side_lbl,
                    "contracts": float(contracts_filled),
                    "call_put": cp,
                    "strike": float(day.at[i, "strike"]),
                    "dte": dte_i,
                    "spot": spot_i,
                    "mid_now": mid_now_main,
                    "mid_next": mid_next_main,
                    "rel_spread": rel_sp_main,
                    "pred_next_norm": float(day.at[i, "pred_next_norm"]),
                    "signal": signal_i,
                    "pnl": float(pnl_per_contract_main * contracts_filled),
                }
            )
            option_leg_idx.append(anchor_contract_idx)
            option_leg_side_sign.append(float(main_side))
            option_leg_contracts.append(float(contracts_filled))

            if wing_contract_idx is not None:
                wing_side_sign = 1 if wing_side_lbl == "LONG" else -1
                day_legs.append(
                    {
                        "trade_key": trade_key,
                        "date": str(day.at[i, "date"]),
                        "date_next": str(day.at[i, "date_next"]),
                        "instrument": "OPTION",
                        "leg_role": "WING",
                        "contract_idx": int(wing_contract_idx),
                        "symbol": wing_symbol,
                        "side": wing_side_lbl,
                        "contracts": float(contracts_filled),
                        "call_put": str(call_put[wing_contract_idx]),
                        "strike": float(strike[wing_contract_idx]),
                        "dte": int(dte[wing_contract_idx]),
                        "spot": float(spot[wing_contract_idx]),
                        "mid_now": float(mid_now[wing_contract_idx]),
                        "mid_next": float(mid_next[wing_contract_idx]),
                        "rel_spread": float(rel_spread[wing_contract_idx]),
                        "pred_next_norm": float(pred_next_norm[wing_contract_idx]),
                        "signal": float(edge[wing_contract_idx]),
                        "pnl": float(wing_pnl_per_contract * contracts_filled),
                    }
                )
                option_leg_idx.append(int(wing_contract_idx))
                option_leg_side_sign.append(float(wing_side_sign))
                option_leg_contracts.append(float(contracts_filled))

        # Compute per-leg deltas / IVs (needed for the underlying delta hedge and diagnostics).
        can_interp = (
            (iv_surface_all is not None)
            and isinstance(iv_surface_all, np.ndarray)
            and (iv_surface_all.ndim == 3)
            and (x_grid.size >= 2)
            and (tenor_days.size >= 2)
            and (0 <= int(d) < iv_surface_all.shape[0])
        )

        net_option_delta_shares = 0.0
        if can_interp and option_leg_idx:
            surf_day = np.asarray(iv_surface_all[int(d)], dtype=float)
            idx_arr = np.asarray(option_leg_idx, dtype=np.int32)
            spot_arr = np.asarray(spot[idx_arr], dtype=float)
            strike_arr = np.asarray(strike[idx_arr], dtype=float)
            dte_arr = np.asarray(dte[idx_arr], dtype=float)
            cp_arr = np.asarray(cp_sign[idx_arr], dtype=float)

            x_arr = np.log(np.clip(strike_arr / np.clip(spot_arr, 1e-6, None), 1e-12, None))
            iv_arr = _interp_surface_bilinear(
                surf_day,
                x_grid=x_grid,
                tenor_days=tenor_days,
                x=x_arr,
                dte_days=dte_arr,
            )
            iv_fallback = float(np.nanmedian(surf_day)) if np.isfinite(surf_day).any() else 0.20
            iv_arr = np.where(np.isfinite(iv_arr), iv_arr, iv_fallback)
            iv_arr = np.clip(iv_arr, 1e-4, 4.0)

            tau_arr = np.clip(dte_arr, 1.0, None) / 365.0
            delta_arr = bs_delta(
                spot_arr,
                strike_arr,
                tau_arr,
                iv_arr,
                cp_arr,
            )

            side_arr = np.asarray(option_leg_side_sign, dtype=float)
            contracts_arr = np.asarray(option_leg_contracts, dtype=float)
            delta_shares = delta_arr * contract_multiplier * side_arr * contracts_arr
            net_option_delta_shares = float(np.nansum(delta_shares))

            # Write delta and IV back to the corresponding leg dicts.
            k = 0
            for leg in day_legs:
                if str(leg.get("instrument")) != "OPTION":
                    continue
                leg["iv"] = float(iv_arr[k])
                leg["delta"] = float(delta_arr[k])
                leg["delta_shares"] = float(delta_shares[k])
                k += 1

        # Optional: delta hedge with the underlying close-to-close.
        hedge_shares = 0.0
        hedge_pnl = 0.0
        hedge_capital_required = 0.0
        if bool(cfg.hedge_underlying_delta) and option_leg_idx:
            if not can_interp:
                warnings.warn(
                    "Underlying delta hedge requested but IV surface grid is unavailable; skipping hedge.",
                    stacklevel=2,
                )
            else:
                hedge_ratio = float(cfg.hedge_underlying_ratio)
                if hedge_policy_bundle is not None:
                    from ivdyn.hedge_policy.policy import build_hedge_features

                    d_i = int(d)
                    z_d = hedge_policy_latents[d_i] if hedge_policy_latents is not None else None
                    ctx_d = hedge_policy_context[d_i] if hedge_policy_context is not None else None
                    spot_now_d = float(spot_by_date[d_i]) if d_i < len(spot_by_date) else float("nan")
                    features_d = build_hedge_features(
                        latent_z=z_d,
                        context=ctx_d,
                        net_option_delta_shares=float(net_option_delta_shares),
                        spot=spot_now_d,
                        spec=hedge_policy_bundle.feature_spec,
                    )
                    hedge_ratio = float(hedge_policy_bundle.predict_ratio(features_d, device=dev))

                hedge_shares = -hedge_ratio * float(net_option_delta_shares)
                hedge_shares = float(np.round(hedge_shares))
                if abs(hedge_shares) >= float(cfg.hedge_underlying_min_abs_shares):
                    hedge_shares = float(
                        np.clip(
                            hedge_shares,
                            -float(cfg.hedge_underlying_max_shares),
                            float(cfg.hedge_underlying_max_shares),
                        )
                    )

                    spot_now_d = float(spot_by_date[int(d)]) if int(d) < len(spot_by_date) else float("nan")
                    spot_next_d = float(spot_by_date[int(d) + 1]) if int(d) + 1 < len(spot_by_date) else float("nan")
                    if not np.isfinite(spot_now_d) or spot_now_d <= 0.0:
                        spot_now_d = float(np.nanmedian(spot[idx_arr])) if option_leg_idx else float("nan")
                    if not np.isfinite(spot_next_d) or spot_next_d <= 0.0:
                        spot_next_d = float(spot_now_d)

                    if not np.isfinite(spot_now_d) or spot_now_d <= 0.0:
                        hedge_shares = 0.0
                    else:
                        hedge_capital_required = float(abs(hedge_shares) * spot_now_d * underlying_margin_rate)
                        if enforce_portfolio_constraints and (
                            (buying_power_used_options_day + hedge_capital_required) > (buying_power_limit_day + 1e-9)
                        ):
                            capital_rejected_hedges_day += 1
                            capital_rejected_hedges_total += 1
                            hedge_shares = 0.0
                            hedge_capital_required = 0.0
                        else:
                            buying_power_used_hedge_day = float(hedge_capital_required)
                            hedge_slip = float(cfg.hedge_underlying_slippage_bps) / 1e4
                            hedge_pnl = _underlying_pnl(
                                spot_now=spot_now_d,
                                spot_next=spot_next_d,
                                shares=hedge_shares,
                                slippage=hedge_slip,
                            )

                            day_legs.append(
                                {
                                    "trade_key": f"HEDGE_{str(day.at[0, 'date'])}",
                                    "date": str(day.at[0, "date"]),
                                    "date_next": str(day.at[0, "date_next"]),
                                    "instrument": "UNDERLYING",
                                    "leg_role": "DELTA_HEDGE",
                                    "contract_idx": -1,
                                    "symbol": underlying_symbol,
                                    "side": "LONG" if hedge_shares > 0 else "SHORT",
                                    "shares": float(abs(hedge_shares)),
                                    "spot_now": spot_now_d,
                                    "spot_next": spot_next_d,
                                    "pnl": float(hedge_pnl),
                                    "delta": 1.0,
                                    "delta_shares": float(hedge_shares),
                                }
                            )

        hedge_rows.append(
            {
                "date": str(day.at[0, "date"]),
                "date_next": str(day.at[0, "date_next"]),
                "net_option_delta_shares": float(net_option_delta_shares),
                "hedge_shares": float(hedge_shares),
                "post_hedge_delta_shares": float(net_option_delta_shares + hedge_shares),
                "hedge_pnl": float(hedge_pnl),
            }
        )

        day_total_pnl = float(day_options_pnl_total + hedge_pnl)
        equity_end_day = float(equity_start_day + day_total_pnl)
        capital_rows.append(
            {
                "date": day_date,
                "date_next": day_date_next,
                "equity_start": float(equity_start_day),
                "equity_end": float(equity_end_day),
                "buying_power_limit": float(buying_power_limit_day),
                "buying_power_used_options": float(buying_power_used_options_day),
                "buying_power_used_hedge": float(buying_power_used_hedge_day),
                "buying_power_used_total": float(buying_power_used_options_day + buying_power_used_hedge_day),
                "capital_rejected_trades": int(capital_rejected_trades_day),
                "capital_rejected_hedges": int(capital_rejected_hedges_day),
            }
        )
        running_equity = float(equity_end_day)

        leg_rows.extend(day_legs)

    if trade_rows:
        trades = (
            pd.DataFrame(trade_rows)[keep_cols]
            .sort_values(["date", "selection_score"], ascending=[True, False])
            .reset_index(drop=True)
        )
    else:
        trades = pd.DataFrame(columns=keep_cols)

    all_days = pd.DataFrame({"date": dates[:-1]})

    hedge_df = pd.DataFrame(hedge_rows) if hedge_rows else pd.DataFrame(
        columns=["date", "date_next", "net_option_delta_shares", "hedge_shares", "post_hedge_delta_shares", "hedge_pnl"]
    )
    hedge_by_day = (
        hedge_df.groupby("date", as_index=False).agg(
            hedge_pnl=("hedge_pnl", "sum"),
            net_option_delta_shares=("net_option_delta_shares", "sum"),
            hedge_shares=("hedge_shares", "sum"),
            post_hedge_delta_shares=("post_hedge_delta_shares", "sum"),
        )
        if not hedge_df.empty
        else pd.DataFrame(columns=["date", "hedge_pnl", "net_option_delta_shares", "hedge_shares", "post_hedge_delta_shares"])
    )
    capital_df = (
        pd.DataFrame(capital_rows)
        if capital_rows
        else pd.DataFrame(
            columns=[
                "date",
                "date_next",
                "equity_start",
                "equity_end",
                "buying_power_limit",
                "buying_power_used_options",
                "buying_power_used_hedge",
                "buying_power_used_total",
                "capital_rejected_trades",
                "capital_rejected_hedges",
            ]
        )
    )
    capital_by_day = (
        capital_df.groupby("date", as_index=False).agg(
            equity_start=("equity_start", "last"),
            equity_end=("equity_end", "last"),
            buying_power_limit=("buying_power_limit", "last"),
            buying_power_used_options=("buying_power_used_options", "sum"),
            buying_power_used_hedge=("buying_power_used_hedge", "sum"),
            buying_power_used_total=("buying_power_used_total", "sum"),
            capital_rejected_trades=("capital_rejected_trades", "sum"),
            capital_rejected_hedges=("capital_rejected_hedges", "sum"),
        )
        if not capital_df.empty
        else pd.DataFrame(
            columns=[
                "date",
                "equity_start",
                "equity_end",
                "buying_power_limit",
                "buying_power_used_options",
                "buying_power_used_hedge",
                "buying_power_used_total",
                "capital_rejected_trades",
                "capital_rejected_hedges",
            ]
        )
    )

    if trades.empty:
        daily = all_days.copy()
        daily["options_pnl"] = 0.0
        daily["options_pnl_gross"] = 0.0
        daily["fees"] = 0.0
        daily["contracts"] = 0.0
        daily["hedge_pnl"] = 0.0
        daily["pnl"] = 0.0
        daily["trades"] = 0
        daily["net_delta_proxy"] = 0.0
        daily["gross_delta_proxy"] = 0.0
        daily["side_imbalance"] = 0.0
        daily["cp_imbalance"] = 0.0
        daily["net_option_delta_shares"] = 0.0
        daily["hedge_shares"] = 0.0
        daily["post_hedge_delta_shares"] = 0.0
    else:
        opt_daily = trades.groupby("date", as_index=False).agg(
            options_pnl=("pnl", "sum"),
            options_pnl_gross=("pnl_gross", "sum"),
            fees=("fees", "sum"),
            contracts=("contracts", "sum"),
            trades=("pnl", "size"),
        )
        hedge_proxy = trades.groupby("date", as_index=False).agg(
            net_delta_proxy=("delta_proxy", "sum"),
            gross_delta_proxy=("delta_proxy", lambda s: float(np.abs(s).sum())),
            long_trades=("side", lambda s: int((s == "LONG").sum())),
            short_trades=("side", lambda s: int((s == "SHORT").sum())),
            call_trades=("call_put", lambda s: int((s == "C").sum())),
            put_trades=("call_put", lambda s: int((s == "P").sum())),
        )
        hedge_proxy["side_imbalance"] = (hedge_proxy["long_trades"] - hedge_proxy["short_trades"]).abs().astype(float)
        hedge_proxy["cp_imbalance"] = (hedge_proxy["call_trades"] - hedge_proxy["put_trades"]).abs().astype(float)

        daily = all_days.merge(opt_daily, on="date", how="left")
        daily = daily.merge(hedge_by_day, on="date", how="left")
        daily = daily.merge(
            hedge_proxy[["date", "net_delta_proxy", "gross_delta_proxy", "side_imbalance", "cp_imbalance"]],
            on="date",
            how="left",
        )

        daily["options_pnl"] = pd.to_numeric(daily.get("options_pnl"), errors="coerce").fillna(0.0)
        daily["options_pnl_gross"] = pd.to_numeric(daily.get("options_pnl_gross"), errors="coerce").fillna(0.0)
        daily["fees"] = pd.to_numeric(daily.get("fees"), errors="coerce").fillna(0.0)
        daily["contracts"] = pd.to_numeric(daily.get("contracts"), errors="coerce").fillna(0.0)
        daily["hedge_pnl"] = pd.to_numeric(daily.get("hedge_pnl"), errors="coerce").fillna(0.0)
        daily["pnl"] = daily["options_pnl"] + daily["hedge_pnl"]
        daily["trades"] = pd.to_numeric(daily.get("trades"), errors="coerce").fillna(0).astype(int)
        daily["net_delta_proxy"] = pd.to_numeric(daily.get("net_delta_proxy"), errors="coerce").fillna(0.0)
        daily["gross_delta_proxy"] = pd.to_numeric(daily.get("gross_delta_proxy"), errors="coerce").fillna(0.0)
        daily["side_imbalance"] = pd.to_numeric(daily.get("side_imbalance"), errors="coerce").fillna(0.0)
        daily["cp_imbalance"] = pd.to_numeric(daily.get("cp_imbalance"), errors="coerce").fillna(0.0)
        daily["net_option_delta_shares"] = pd.to_numeric(daily.get("net_option_delta_shares"), errors="coerce").fillna(0.0)
        daily["hedge_shares"] = pd.to_numeric(daily.get("hedge_shares"), errors="coerce").fillna(0.0)
        daily["post_hedge_delta_shares"] = pd.to_numeric(daily.get("post_hedge_delta_shares"), errors="coerce").fillna(0.0)

    daily = daily.merge(capital_by_day, on="date", how="left")
    for col in (
        "buying_power_used_options",
        "buying_power_used_hedge",
        "buying_power_used_total",
        "capital_rejected_trades",
        "capital_rejected_hedges",
    ):
        daily[col] = pd.to_numeric(daily.get(col), errors="coerce").fillna(0.0)

    initial_capital = float(cfg.initial_capital)
    daily["equity"] = initial_capital + pd.to_numeric(daily["pnl"], errors="coerce").fillna(0.0).cumsum()
    equity_start_series = daily["equity"].shift(1).fillna(initial_capital).astype(float)
    equity_end_series = daily["equity"].astype(float)
    if "equity_start" in daily.columns:
        eq_start_existing = pd.to_numeric(daily["equity_start"], errors="coerce")
    else:
        eq_start_existing = pd.Series(np.nan, index=daily.index, dtype=float)
    if "equity_end" in daily.columns:
        eq_end_existing = pd.to_numeric(daily["equity_end"], errors="coerce")
    else:
        eq_end_existing = pd.Series(np.nan, index=daily.index, dtype=float)
    daily["equity_start"] = np.where(np.isfinite(eq_start_existing), eq_start_existing, equity_start_series)
    daily["equity_end"] = np.where(np.isfinite(eq_end_existing), eq_end_existing, equity_end_series)

    derived_bp_limit = np.clip(equity_start_series.to_numpy(dtype=float), 0.0, None) * buying_power_leverage
    if "buying_power_limit" in daily.columns:
        bp_existing = pd.to_numeric(daily["buying_power_limit"], errors="coerce")
    else:
        bp_existing = pd.Series(np.nan, index=daily.index, dtype=float)
    daily["buying_power_limit"] = np.where(np.isfinite(bp_existing), bp_existing, derived_bp_limit)
    daily["buying_power_limit"] = pd.to_numeric(daily["buying_power_limit"], errors="coerce").fillna(0.0)
    daily["buying_power_used_options"] = pd.to_numeric(daily.get("buying_power_used_options"), errors="coerce").fillna(0.0)
    daily["buying_power_used_hedge"] = pd.to_numeric(daily.get("buying_power_used_hedge"), errors="coerce").fillna(0.0)
    daily["buying_power_used_total"] = pd.to_numeric(daily.get("buying_power_used_total"), errors="coerce").fillna(0.0)
    daily["capital_rejected_trades"] = (
        pd.to_numeric(daily.get("capital_rejected_trades"), errors="coerce").fillna(0.0).astype(int)
    )
    daily["capital_rejected_hedges"] = (
        pd.to_numeric(daily.get("capital_rejected_hedges"), errors="coerce").fillna(0.0).astype(int)
    )
    daily["buying_power_utilization"] = (
        pd.to_numeric(daily["buying_power_used_total"], errors="coerce")
        / pd.to_numeric(daily["buying_power_limit"], errors="coerce").replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    pnl = daily["pnl"].astype(float)
    equity = daily["equity"].astype(float)
    hedge_abs_net = daily["net_delta_proxy"].abs().astype(float)
    hedge_gross = daily["gross_delta_proxy"].astype(float)
    hedge_side_imb = daily["side_imbalance"].astype(float)
    hedge_cp_imb = daily["cp_imbalance"].astype(float)
    peak_abs = equity.cummax()
    max_drawdown_abs = float((equity - peak_abs).min()) if len(equity) else 0.0

    peak_rel = peak_abs.replace(0.0, np.nan)
    drawdown_rel = (equity - peak_rel) / peak_rel
    max_drawdown = float(np.nanmin(drawdown_rel.to_numpy())) if np.isfinite(drawdown_rel).any() else 0.0

    daily_vol = float(pnl.std(ddof=0)) if len(pnl) else 0.0
    daily_mean = float(pnl.mean()) if len(pnl) else 0.0
    # Sharpe should be computed on returns, not raw dollars.
    equity_start = equity.shift(1).fillna(initial_capital)
    daily_ret = (pnl / equity_start.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    risk_free_rate_annual = float(getattr(cfg, "risk_free_rate_annual", 0.045))
    if not np.isfinite(risk_free_rate_annual):
        risk_free_rate_annual = 0.045
    risk_free_rate_daily = _annual_to_daily_rate(risk_free_rate_annual, periods_per_year=252)
    daily_excess_ret = daily_ret - risk_free_rate_daily

    ret_vol = float(daily_ret.std(ddof=0)) if len(daily_ret) else 0.0
    ret_mean = float(daily_ret.mean()) if len(daily_ret) else 0.0
    excess_ret_vol = float(daily_excess_ret.std(ddof=0)) if len(daily_excess_ret) else 0.0
    excess_ret_mean = float(daily_excess_ret.mean()) if len(daily_excess_ret) else 0.0
    daily_sharpe_rf0 = float(np.sqrt(252.0) * ret_mean / ret_vol) if ret_vol > 0 else 0.0
    daily_sharpe = float(np.sqrt(252.0) * excess_ret_mean / excess_ret_vol) if excess_ret_vol > 0 else 0.0
    ret_skew, ret_kurt = sample_skew_kurtosis(daily_excess_ret.to_numpy(dtype=float))
    psr_sr0_0 = probabilistic_sharpe_ratio(
        daily_excess_ret.to_numpy(dtype=float),
        benchmark_sharpe_ann=0.0,
        periods_per_year=252,
    )

    total_pnl = float(pnl.sum())
    capital_rejected_trades_sum = int(pd.to_numeric(daily.get("capital_rejected_trades"), errors="coerce").fillna(0.0).sum())
    capital_rejected_hedges_sum = int(pd.to_numeric(daily.get("capital_rejected_hedges"), errors="coerce").fillna(0.0).sum())
    bp_util = pd.to_numeric(daily.get("buying_power_utilization"), errors="coerce").fillna(0.0)
    bp_used = pd.to_numeric(daily.get("buying_power_used_total"), errors="coerce").fillna(0.0)
    summary = {
        "backtest_version": _BACKTEST_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_path": dataset_path.as_posix(),
        "underlying_symbol": underlying_symbol,
        "strategy_mode": strategy_mode,
        "days": int(len(daily)),
        "candidates": int(len(candidates)),
        "trades": int(len(trades)),
        "initial_capital": initial_capital,
        "final_equity": float(equity.iloc[-1]) if len(equity) else initial_capital,
        "total_pnl": total_pnl,
        "total_fees": float(daily.get("fees", pd.Series(dtype=float)).sum()),
        "total_options_pnl_gross": float(daily.get("options_pnl_gross", pd.Series(dtype=float)).sum()),
        "total_options_pnl": float(daily.get("options_pnl", pd.Series(dtype=float)).sum()),
        "total_hedge_pnl": float(daily.get("hedge_pnl", pd.Series(dtype=float)).sum()),
        "avg_daily_pnl": daily_mean,
        "daily_vol_pnl": daily_vol,
        "daily_return_mean": ret_mean,
        "daily_return_vol": ret_vol,
        "daily_excess_return_mean": excess_ret_mean,
        "daily_excess_return_vol": excess_ret_vol,
        "risk_free_rate_annual": risk_free_rate_annual,
        "risk_free_rate_daily": risk_free_rate_daily,
        "daily_sharpe": daily_sharpe,
        "daily_sharpe_rf0": daily_sharpe_rf0,
        "return_skew": float(ret_skew) if np.isfinite(ret_skew) else float("nan"),
        "return_kurtosis": float(ret_kurt) if np.isfinite(ret_kurt) else float("nan"),
        "psr_sr0_0": float(psr_sr0_0) if np.isfinite(psr_sr0_0) else float("nan"),
        "max_drawdown": max_drawdown,
        "max_drawdown_abs": max_drawdown_abs,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "enforce_portfolio_constraints": bool(enforce_portfolio_constraints),
        "buying_power_leverage": float(buying_power_leverage),
        "option_short_margin_rate": float(option_short_margin_rate),
        "underlying_margin_rate": float(underlying_margin_rate),
        "capital_rejected_trades": int(capital_rejected_trades_sum),
        "capital_rejected_hedges": int(capital_rejected_hedges_sum),
        "capital_rejected_trades_loop_count": int(capital_rejected_trades_total),
        "capital_rejected_hedges_loop_count": int(capital_rejected_hedges_total),
        "buying_power_utilization_avg": float(bp_util.mean()) if len(bp_util) else 0.0,
        "buying_power_utilization_max": float(bp_util.max()) if len(bp_util) else 0.0,
        "buying_power_used_avg": float(bp_used.mean()) if len(bp_used) else 0.0,
        "buying_power_used_max": float(bp_used.max()) if len(bp_used) else 0.0,
        "fill_gate": float(cfg.fill_gate),
        "fill_model": str(cfg.fill_model),
        "max_trades_per_day": int(cfg.max_trades_per_day),
        "max_contracts_per_trade": int(cfg.max_contracts_per_trade),
        "total_contracts": float(pd.to_numeric(trades.get("contracts"), errors="coerce").fillna(0.0).sum()),
        "volume_participation_rate": float(cfg.volume_participation_rate),
        "open_interest_participation_rate": float(cfg.open_interest_participation_rate),
        "spread_cross_fraction": float(cfg.spread_cross_fraction),
        "option_commission_per_contract": float(cfg.option_commission_per_contract),
        "option_fee_per_contract": float(cfg.option_fee_per_contract),
        "min_edge_to_cost_ratio": float(cfg.min_edge_to_cost_ratio),
        "selector_edge_clip_quantile": float(cfg.selector_edge_clip_quantile),
        "selector_mid_norm_floor": float(cfg.selector_mid_norm_floor),
        "selector_signal_soft_cap": float(cfg.selector_signal_soft_cap),
        "selector_long_score_scale": float(cfg.selector_long_score_scale),
        "selector_long_abs_signal_cap": float(cfg.selector_long_abs_signal_cap),
        "selector_allow_long_puts": bool(cfg.selector_allow_long_puts),
        "min_dte": int(cfg.min_dte),
        "max_dte": int(cfg.max_dte),
        "min_moneyness": float(cfg.min_moneyness),
        "max_moneyness": float(cfg.max_moneyness),
        "max_rel_spread": float(cfg.max_rel_spread),
        "vertical_wing_width_pct_target": float(cfg.vertical_wing_width_pct_target),
        "vertical_wing_width_pct_min": float(cfg.vertical_wing_width_pct_min),
        "vertical_wing_width_pct_max": float(cfg.vertical_wing_width_pct_max),
        "vertical_wing_max_premium_ratio": float(cfg.vertical_wing_max_premium_ratio),
        "vertical_wing_fill_gate": float(cfg.vertical_wing_fill_gate),
        "vertical_wing_max_rel_spread": float(cfg.vertical_wing_max_rel_spread),
        "vertical_wing_min_moneyness": float(cfg.vertical_wing_min_moneyness),
        "vertical_wing_max_moneyness": float(cfg.vertical_wing_max_moneyness),
        "vertical_skip_if_no_wing": bool(cfg.vertical_skip_if_no_wing),
        "hedge_max_net_delta_ratio": float(cfg.hedge_max_net_delta_ratio),
        "hedge_relaxed_net_delta_ratio": float(cfg.hedge_relaxed_net_delta_ratio),
        "hedge_max_net_delta_abs": float(cfg.hedge_max_net_delta_abs),
        "hedge_max_side_imbalance_ratio": float(cfg.hedge_max_side_imbalance_ratio),
        "hedge_avg_abs_net_delta_proxy": float(hedge_abs_net.mean()) if len(hedge_abs_net) else 0.0,
        "hedge_max_abs_net_delta_proxy": float(hedge_abs_net.max()) if len(hedge_abs_net) else 0.0,
        "hedge_avg_gross_delta_proxy": float(hedge_gross.mean()) if len(hedge_gross) else 0.0,
        "hedge_avg_side_imbalance": float(hedge_side_imb.mean()) if len(hedge_side_imb) else 0.0,
        "hedge_avg_cp_imbalance": float(hedge_cp_imb.mean()) if len(hedge_cp_imb) else 0.0,
        "hedge_underlying_delta": bool(cfg.hedge_underlying_delta),
        "hedge_underlying_ratio": float(cfg.hedge_underlying_ratio),
        "hedge_policy": str(getattr(cfg, "hedge_policy", "fixed")),
        "hedge_policy_path": str(getattr(cfg, "hedge_policy_path", None)) if getattr(cfg, "hedge_policy_path", None) else None,
        "hedge_underlying_min_abs_shares": float(cfg.hedge_underlying_min_abs_shares),
        "hedge_underlying_max_shares": int(cfg.hedge_underlying_max_shares),
        "hedge_underlying_slippage_bps": float(cfg.hedge_underlying_slippage_bps),
        "hedge_avg_abs_net_option_delta_shares": float(daily.get("net_option_delta_shares", pd.Series(dtype=float)).abs().mean()),
        "hedge_avg_abs_post_hedge_delta_shares": float(daily.get("post_hedge_delta_shares", pd.Series(dtype=float)).abs().mean()),
        "hedge_avg_abs_hedge_shares": float(daily.get("hedge_shares", pd.Series(dtype=float)).abs().mean()),
        "slippage_bps": float(cfg.slippage_bps),
        "signal_abs_gate": float(cfg.signal_abs_gate),
        "num_workers": int(cfg.num_workers),
        "parallel_backend": "sequential",
        "inference_batch_size": int(cfg.inference_batch_size),
    }

    legs_df = pd.DataFrame(leg_rows) if leg_rows else pd.DataFrame()
    if not legs_df.empty:
        sort_cols = [c for c in ["date", "trade_key", "leg_role", "instrument"] if c in legs_df.columns]
        if sort_cols:
            legs_df = legs_df.sort_values(sort_cols).reset_index(drop=True)

    _write_parquet_with_fallback(trades, bt_dir / "trades.parquet")
    _write_parquet_with_fallback(daily, bt_dir / "daily.parquet")
    _write_parquet_with_fallback(legs_df, bt_dir / "legs.parquet")
    _write_parquet_with_fallback(hedge_df, bt_dir / "hedges.parquet")
    (bt_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return bt_dir
