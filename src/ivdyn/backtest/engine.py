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

_OPRA_EXPIRY_RE = re.compile(r"^O:[A-Z]+(?P<exp>\d{6})[CP]\d{8}$")
_BACKTEST_VERSION = "2026-02-17-selector-hedged-v3"


@dataclass(slots=True)
class BacktestConfig:
    run_dir: Path
    dataset_path: Path
    device: str | None = None
    num_workers: int = 0
    inference_batch_size: int = 65536

    fill_gate: float = 0.65
    slippage_bps: float = 7.5
    signal_abs_gate: float = 0.04
    max_trades_per_day: int = 5
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
    m = _OPRA_EXPIRY_RE.match(str(symbol))
    if not m:
        return ""
    return m.group("exp")


def _leg_pnl(
    *,
    mid_now: float,
    mid_next: float,
    rel_spread: float,
    slippage: float,
    side: int,
) -> float:
    rel_sp = float(np.clip(rel_spread, 0.0, 3.0))
    cost = slippage + 0.5 * rel_sp * 0.15
    entry = mid_now * (1.0 + side * cost)
    exit_ = mid_next * (1.0 - side * cost)
    return float(side * (exit_ - entry))


def _execution_cost_norm(mid_now: float, rel_spread: float, slippage: float) -> float:
    rel_sp = float(np.clip(rel_spread, 0.0, 3.0))
    return float(mid_now * (slippage + 0.5 * rel_sp * 0.15))


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


def _predict_contracts(
    *,
    model_bundle: ModelBundle,
    ds: dict[str, np.ndarray],
    dev: torch.device,
    batch_size: int,
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
    pred_now = np.empty(n_contracts, dtype=np.float32)
    pred_next = np.empty(n_contracts, dtype=np.float32)
    fill_prob = np.empty(n_contracts, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n_contracts, batch_size):
            j = min(i + batch_size, n_contracts)
            idx = slice(i, j)
            d = date_idx[idx]
            cf = torch.as_tensor(contract_scaled[idx], dtype=torch.float32, device=dev)
            zc_now = torch.as_tensor(z_now[d], dtype=torch.float32, device=dev)
            zc_next = torch.as_tensor(z_next[d], dtype=torch.float32, device=dev)

            p_now_scaled = to_numpy(model.forward_pricer(zc_now, cf)).reshape(-1, 1)
            p_next_scaled = to_numpy(model.forward_pricer(zc_next, cf)).reshape(-1, 1)
            pred_now[idx] = model_bundle.price_scaler.inverse_transform(p_now_scaled).reshape(-1)
            pred_next[idx] = model_bundle.price_scaler.inverse_transform(p_next_scaled).reshape(-1)
            logits = to_numpy(model.forward_execution_logit(zc_now, cf)).reshape(-1)
            fill_prob[idx] = _sigmoid(logits).astype(np.float32)

    return pred_now, pred_next, fill_prob


def run_backtest(cfg: BacktestConfig) -> Path:
    run_dir = cfg.run_dir.resolve()
    bt_dir = run_dir / "backtest"
    bt_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = cfg.dataset_path.resolve()
    ds = _load_dataset(dataset_path)

    dev = torch.device(cfg.device) if cfg.device else device_auto()
    model_path = run_dir / "model.pt"
    bundle = ModelBundle.load(model_path, device=dev)

    # Cache inference to allow fast strategy iteration.
    cache_path = bt_dir / "pred_cache.npz"
    cache_meta_path = bt_dir / "pred_cache_meta.json"
    use_cache = False
    if cache_path.exists() and cache_meta_path.exists():
        try:
            meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("dataset") == str(dataset_path)
                and meta.get("model") == str(model_path)
                and int(meta.get("dataset_mtime", -1)) == int(dataset_path.stat().st_mtime)
                and int(meta.get("model_mtime", -1)) == int(model_path.stat().st_mtime)
            ):
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
        )
        np.savez_compressed(
            cache_path,
            pred_now_norm=pred_now_norm.astype(np.float32),
            pred_next_norm=pred_next_norm.astype(np.float32),
            fill_prob=fill_prob.astype(np.float32),
        )
        cache_meta_path.write_text(
            json.dumps(
                {
                    "dataset": str(dataset_path),
                    "model": str(model_path),
                    "dataset_mtime": int(dataset_path.stat().st_mtime),
                    "model_mtime": int(model_path.stat().st_mtime),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    dates = ds["dates"].astype(str)
    n_dates = len(dates)

    date_idx = ds["contract_date_index"].astype(np.int32)
    symbol = ds["contract_symbol"].astype(str)
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
    cp_sign = features[:, fidx["cp_sign"]] if "cp_sign" in fidx else np.where(call_put == "C", 1.0, -1.0).astype(np.float32)
    cp_sign = np.where(np.isfinite(cp_sign), np.sign(cp_sign), np.where(call_put == "C", 1.0, -1.0)).astype(np.float32)
    cp_sign = np.where(cp_sign == 0.0, np.where(call_put == "C", 1.0, -1.0), cp_sign).astype(np.float32)

    moneyness = strike / np.clip(spot, 1e-6, None)
    date_next_idx = np.clip(date_idx + 1, 0, n_dates - 1)
    date_next_arr = dates[date_next_idx]

    next_key_mid: dict[tuple[int, str], float] = {}
    next_key_mid_norm: dict[tuple[int, str], float] = {}
    for i in range(len(symbol)):
        key = (int(date_idx[i]), str(symbol[i]))
        next_key_mid[key] = float(mid_now[i])
        next_key_mid_norm[key] = float(mid_now_norm[i])
    mid_next = np.full(len(symbol), np.nan, dtype=np.float32)
    mid_next_norm = np.full(len(symbol), np.nan, dtype=np.float32)
    for i in range(len(symbol)):
        k = (int(date_idx[i] + 1), str(symbol[i]))
        if k in next_key_mid:
            mid_next[i] = np.float32(next_key_mid[k])
            mid_next_norm[i] = np.float32(next_key_mid_norm[k])

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
        & np.char.startswith(symbol.astype(str), "O:SPY")
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

    side = np.full(len(symbol), "", dtype=object)
    long_mask = tradable & (edge < 0.0)
    short_mask = tradable & (edge > 0.0)

    side[short_mask] = "SHORT"
    side[long_mask] = "LONG"
    active = side != ""

    candidates = pd.DataFrame(
        {
            "date_idx": date_idx[active],
            "date": date_arr[active],
            "date_next": date_next_arr[active],
            "symbol": symbol[active],
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
            "cp_sign": cp_sign[active].astype(float),
            "spot": spot[active].astype(float),
            "side": side[active].astype(str),
        }
    )

    keep_cols = [
        "date",
        "date_next",
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
        "pnl_per_contract",
        "ev_per_contract",
        "risk_score",
        "execution_cost_per_contract",
        "execution_cost_ratio",
        "max_fill_distance",
        "contracts",
        "notional",
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
    contract_multiplier = 100.0
    trade_rows: list[dict[str, object]] = []
    for d in sorted(candidates["date_idx"].unique().tolist()):
        day = candidates[candidates["date_idx"] == d].reset_index(drop=True)
        if day.empty:
            continue

        day_cap = max(0, int(cfg.max_trades_per_day))
        if day_cap == 0:
            continue

        abs_edge_day = np.abs(day["edge_usd_per_share"].to_numpy(dtype=float))
        edge_cap = float(np.quantile(abs_edge_day, selector_edge_q)) if len(abs_edge_day) else 0.0
        if not np.isfinite(edge_cap) or edge_cap <= 0.0:
            edge_cap = float(np.nanmax(abs_edge_day)) if len(abs_edge_day) else 0.0
        edge_rank = np.minimum(abs_edge_day, edge_cap)

        fill_day = np.clip(day["fill_prob"].to_numpy(dtype=float), 0.0, 1.0)
        mid_now_day = np.clip(day["mid_now"].to_numpy(dtype=float), 0.0, None)
        rel_sp_day = np.clip(day["rel_spread"].to_numpy(dtype=float), 0.0, 3.0)
        abs_signal_day = np.abs(day["signal"].to_numpy(dtype=float))

        exec_cost_est = mid_now_day * (slippage + 0.5 * rel_sp_day * 0.15)
        edge_net = np.clip(edge_rank - 2.0 * exec_cost_est, 0.0, None)
        spread_penalty = 1.0 + rel_sp_day
        signal_penalty = 1.0 / (1.0 + (abs_signal_day / selector_signal_soft_cap))
        selection_score = edge_net * fill_day * signal_penalty / spread_penalty

        cp_day = np.clip(day["cp_sign"].to_numpy(dtype=float), -1.0, 1.0)
        side_day = day["side"].to_numpy(dtype=str)
        is_long_day = side_day == "LONG"
        long_allowed = ~is_long_day | (abs_signal_day <= selector_long_abs_signal_cap)
        if not bool(cfg.selector_allow_long_puts):
            long_allowed &= ~(is_long_day & (cp_day < 0.0))
        selection_score = np.where(is_long_day, selection_score * selector_long_score_scale, selection_score)
        selection_score = np.where(long_allowed, selection_score, 0.0)
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

        for i in selected_idx:
            i = int(i)
            side_lbl = str(day.at[i, "side"])
            cp = str(day.at[i, "call_put"])
            dte_i = int(day.at[i, "dte"])
            spot_i = float(day.at[i, "spot"])
            if not np.isfinite(spot_i) or spot_i <= 0.0:
                continue
            notional = float(spot_i * contract_multiplier)

            main_side = 1 if side_lbl == "LONG" else -1
            mid_now_main = float(day.at[i, "mid_now"])
            mid_next_main = float(day.at[i, "mid_next"])
            mid_now_main_norm = float(day.at[i, "mid_now_norm"])
            mid_next_main_norm = float(day.at[i, "mid_next_norm"])
            rel_sp_main = float(day.at[i, "rel_spread"])
            if not np.isfinite(mid_now_main) or not np.isfinite(mid_next_main):
                continue

            pnl_per_share = _leg_pnl(
                mid_now=mid_now_main,
                mid_next=mid_next_main,
                rel_spread=rel_sp_main,
                slippage=slippage,
                side=main_side,
            )
            exec_cost_per_share = _execution_cost_norm(mid_now_main, rel_sp_main, slippage)

            pnl_per_contract = float(pnl_per_share * contract_multiplier)
            signal_i = float(day.at[i, "signal"])
            edge_usd_per_share_i = float(day.at[i, "edge_usd_per_share"])
            ev_per_contract = float(np.abs(edge_usd_per_share_i) * contract_multiplier)
            execution_cost_per_contract = float(exec_cost_per_share * contract_multiplier)
            execution_cost_ratio = float(exec_cost_per_share / max(mid_now_main, 1e-12))
            max_fill_distance = float(np.clip(rel_sp_main * 0.25, 0.0, None))

            trade_rows.append(
                {
                    "date": str(day.at[i, "date"]),
                    "date_next": str(day.at[i, "date_next"]),
                    "symbol": str(day.at[i, "symbol"]),
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
                    "expiry": _opra_expiry(str(day.at[i, "symbol"])),
                    "pnl_per_contract": pnl_per_contract,
                    "ev_per_contract": ev_per_contract,
                    "risk_score": float(np.abs(edge_usd_per_share_i)),
                    "execution_cost_per_contract": execution_cost_per_contract,
                    "execution_cost_ratio": execution_cost_ratio,
                    "max_fill_distance": max_fill_distance,
                    "contracts": 1,
                    "notional": notional,
                    "pnl": pnl_per_contract,
                }
            )

    if trade_rows:
        trades = (
            pd.DataFrame(trade_rows)[keep_cols]
            .sort_values(["date", "selection_score"], ascending=[True, False])
            .reset_index(drop=True)
        )
    else:
        trades = pd.DataFrame(columns=keep_cols)

    all_days = pd.DataFrame({"date": dates[:-1]})
    if trades.empty:
        daily = all_days.copy()
        daily["pnl"] = 0.0
        daily["trades"] = 0
        daily["net_delta_proxy"] = 0.0
        daily["gross_delta_proxy"] = 0.0
        daily["side_imbalance"] = 0.0
        daily["cp_imbalance"] = 0.0
    else:
        daily = trades.groupby("date", as_index=False).agg(pnl=("pnl", "sum"), trades=("pnl", "size"))
        hedge_daily = trades.groupby("date", as_index=False).agg(
            net_delta_proxy=("delta_proxy", "sum"),
            gross_delta_proxy=("delta_proxy", lambda s: float(np.abs(s).sum())),
            long_trades=("side", lambda s: int((s == "LONG").sum())),
            short_trades=("side", lambda s: int((s == "SHORT").sum())),
            call_trades=("call_put", lambda s: int((s == "C").sum())),
            put_trades=("call_put", lambda s: int((s == "P").sum())),
        )
        hedge_daily["side_imbalance"] = (hedge_daily["long_trades"] - hedge_daily["short_trades"]).abs().astype(float)
        hedge_daily["cp_imbalance"] = (hedge_daily["call_trades"] - hedge_daily["put_trades"]).abs().astype(float)
        daily = all_days.merge(daily, on="date", how="left")
        daily = daily.merge(
            hedge_daily[["date", "net_delta_proxy", "gross_delta_proxy", "side_imbalance", "cp_imbalance"]],
            on="date",
            how="left",
        )
        daily["pnl"] = pd.to_numeric(daily["pnl"], errors="coerce").fillna(0.0)
        daily["trades"] = pd.to_numeric(daily["trades"], errors="coerce").fillna(0).astype(int)
        daily["net_delta_proxy"] = pd.to_numeric(daily["net_delta_proxy"], errors="coerce").fillna(0.0)
        daily["gross_delta_proxy"] = pd.to_numeric(daily["gross_delta_proxy"], errors="coerce").fillna(0.0)
        daily["side_imbalance"] = pd.to_numeric(daily["side_imbalance"], errors="coerce").fillna(0.0)
        daily["cp_imbalance"] = pd.to_numeric(daily["cp_imbalance"], errors="coerce").fillna(0.0)
    daily["equity"] = pd.to_numeric(daily["pnl"], errors="coerce").fillna(0.0).cumsum()

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
    daily_sharpe = float(np.sqrt(252.0) * daily_mean / daily_vol) if daily_vol > 0 else 0.0

    summary = {
        "backtest_version": _BACKTEST_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_path": dataset_path.as_posix(),
        "days": int(len(daily)),
        "candidates": int(len(candidates)),
        "trades": int(len(trades)),
        "total_pnl": float(pnl.sum()),
        "avg_daily_pnl": daily_mean,
        "daily_vol_pnl": daily_vol,
        "daily_sharpe": daily_sharpe,
        "max_drawdown": max_drawdown,
        "max_drawdown_abs": max_drawdown_abs,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "fill_gate": float(cfg.fill_gate),
        "max_trades_per_day": int(cfg.max_trades_per_day),
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
        "hedge_max_net_delta_ratio": float(cfg.hedge_max_net_delta_ratio),
        "hedge_relaxed_net_delta_ratio": float(cfg.hedge_relaxed_net_delta_ratio),
        "hedge_max_net_delta_abs": float(cfg.hedge_max_net_delta_abs),
        "hedge_max_side_imbalance_ratio": float(cfg.hedge_max_side_imbalance_ratio),
        "hedge_avg_abs_net_delta_proxy": float(hedge_abs_net.mean()) if len(hedge_abs_net) else 0.0,
        "hedge_max_abs_net_delta_proxy": float(hedge_abs_net.max()) if len(hedge_abs_net) else 0.0,
        "hedge_avg_gross_delta_proxy": float(hedge_gross.mean()) if len(hedge_gross) else 0.0,
        "hedge_avg_side_imbalance": float(hedge_side_imb.mean()) if len(hedge_side_imb) else 0.0,
        "hedge_avg_cp_imbalance": float(hedge_cp_imb.mean()) if len(hedge_cp_imb) else 0.0,
        "slippage_bps": float(cfg.slippage_bps),
        "signal_abs_gate": float(cfg.signal_abs_gate),
        "num_workers": int(cfg.num_workers),
        "parallel_backend": "sequential",
        "inference_batch_size": int(cfg.inference_batch_size),
    }

    _write_parquet_with_fallback(trades, bt_dir / "trades.parquet")
    _write_parquet_with_fallback(daily, bt_dir / "daily.parquet")
    (bt_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return bt_dir
