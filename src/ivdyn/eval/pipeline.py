"""Evaluation pipeline producing numerical and graphical artifacts."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for evaluation.") from exc

from ivdyn.eval.metrics import brier_score, mae, r2, rmse
from ivdyn.model import ModelBundle, device_auto, to_numpy
from ivdyn.surface import butterfly_violations, calendar_violations


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=True)
    out: dict[str, np.ndarray] = {}
    for k in npz.files:
        arr = npz[k]
        if arr.dtype == object:
            arr = arr.astype(str)
        out[k] = arr
    return out

# Evaluation reports are produced over the full supplied dataset.


def _resolve_num_workers(requested: int, n_tasks: int) -> int:
    if n_tasks <= 1:
        return 1
    if requested == 1:
        return 1
    if requested <= 0:
        cpu = os.cpu_count() or 1
        return max(1, min(cpu - 1, n_tasks))
    return max(1, min(requested, n_tasks))


def _noarb_for_day(
    obs_surface: np.ndarray,
    pred_surface: np.ndarray,
    x_grid: np.ndarray,
    tenor_days: np.ndarray,
) -> tuple[float, float, float, float]:
    cal_obs = float(calendar_violations(obs_surface[None, ...], tenor_days)[0])
    cal_pred = float(calendar_violations(pred_surface[None, ...], tenor_days)[0])
    bfly_obs = float(butterfly_violations(obs_surface[None, ...], x_grid, tenor_days)[0])
    bfly_pred = float(butterfly_violations(pred_surface[None, ...], x_grid, tenor_days)[0])
    return cal_obs, cal_pred, bfly_obs, bfly_pred


def evaluate(
    run_dir: Path,
    dataset_path: Path,
    device: str | None = None,
    num_workers: int = 0,
) -> Path:
    run_dir = run_dir.resolve()
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    ds = _load_dataset(dataset_path)

    dev = torch.device(device) if device else device_auto()
    bundle = ModelBundle.load(run_dir / "model.pt", device=dev)
    model = bundle.model.to(dev).eval()

    dates = ds["dates"].astype(str)
    n_dates = len(dates)
    x_grid = ds["x_grid"].astype(np.float32)
    tenor_days = ds["tenor_days"].astype(np.int32)

    iv_surface_obs = ds["iv_surface"].astype(np.float32)
    surface_flat = iv_surface_obs.reshape(n_dates, -1)
    surface_scaled = bundle.surface_scaler.transform(surface_flat)

    context = ds["context"].astype(np.float32)
    context_scaled = bundle.context_scaler.transform(context)

    contract_features = ds["contract_features"].astype(np.float32)
    contract_scaled = bundle.contract_scaler.transform(contract_features)
    contract_price_target = ds["contract_price_target"].astype(np.float32)
    contract_fill_target = ds["contract_fill_target"].astype(np.float32)
    contract_date_idx = ds["contract_date_index"].astype(np.int32)
    contract_symbol = ds["contract_symbol"].astype(str)
    contract_mid = ds["contract_mid"].astype(np.float32)
    contract_spot = ds["contract_spot"].astype(np.float32)
    contract_mid_norm = contract_mid / np.clip(contract_spot, 1e-6, None)

    test_dates = np.arange(int(n_dates), dtype=np.int32)
    # Use inference_mode for lower overhead than no_grad during evaluation.
    with torch.inference_mode():
        sf = torch.as_tensor(surface_scaled, dtype=torch.float32, device=dev)
        mu, _ = model.encode(sf)
        recon_scaled = model.decode(mu)
        z_all_t = torch.as_tensor(mu, dtype=torch.float32, device=dev)
        ctx_t = torch.as_tensor(context_scaled, dtype=torch.float32, device=dev)
        z_next_t = model.forward_dynamics(z_all_t, ctx_t)
        forecast_scaled = model.decode(z_next_t)

        recon_raw = bundle.surface_scaler.inverse_transform(to_numpy(recon_scaled)).reshape(iv_surface_obs.shape)
        forecast_raw = bundle.surface_scaler.inverse_transform(to_numpy(forecast_scaled)).reshape(iv_surface_obs.shape)
        z_all = to_numpy(mu)
        z_next = to_numpy(z_next_t)

        cf = torch.as_tensor(contract_scaled, dtype=torch.float32, device=dev)
        z_contract = torch.as_tensor(z_all[contract_date_idx], dtype=torch.float32, device=dev)
        z_contract_next = torch.as_tensor(z_next[contract_date_idx], dtype=torch.float32, device=dev)
        price_scaled_pred = to_numpy(model.forward_pricer(z_contract, cf)).reshape(-1, 1)
        price_scaled_pred_next = to_numpy(model.forward_pricer(z_contract_next, cf)).reshape(-1, 1)
        exec_logit = to_numpy(model.forward_execution_logit(z_contract, cf)).reshape(-1)

    price_pred = bundle.price_scaler.inverse_transform(price_scaled_pred).reshape(-1)
    price_pred_next = bundle.price_scaler.inverse_transform(price_scaled_pred_next).reshape(-1)
    exec_prob = 1.0 / (1.0 + np.exp(-np.clip(exec_logit, -60.0, 60.0)))

    mask_test_contracts = np.isin(contract_date_idx, test_dates)

    y_true = contract_price_target[mask_test_contracts]
    y_pred = price_pred[mask_test_contracts]
    e_true = contract_fill_target[mask_test_contracts]
    e_prob = exec_prob[mask_test_contracts]

    metrics = {
        "test_contracts": int(mask_test_contracts.sum()),
        "price_rmse": rmse(y_pred, y_true),
        "price_mae": mae(y_pred, y_true),
        "price_r2_same_day": r2(y_pred, y_true),
        # Expose primary R^2 as next-day predictive quality.
        "price_r2": float("nan"),
        "exec_brier": brier_score(e_prob, e_true),
        "exec_positive_rate": float(np.mean(e_true)) if len(e_true) else float("nan"),
    }

    next_key_mid_norm: dict[tuple[int, str], float] = {}
    for i in range(len(contract_symbol)):
        next_key_mid_norm[(int(contract_date_idx[i]), str(contract_symbol[i]))] = float(contract_mid_norm[i])

    target_next_price_norm = np.full(len(contract_symbol), np.nan, dtype=np.float32)
    for i in range(len(contract_symbol)):
        k = (int(contract_date_idx[i] + 1), str(contract_symbol[i]))
        if k in next_key_mid_norm:
            target_next_price_norm[i] = np.float32(next_key_mid_norm[k])

    pred_next_return = (price_pred_next - contract_price_target) / np.clip(contract_price_target, 1e-6, None)
    target_next_return = (target_next_price_norm - contract_price_target) / np.clip(contract_price_target, 1e-6, None)

    mask_test_next = mask_test_contracts & np.isfinite(target_next_price_norm)
    y_next_true = target_next_price_norm[mask_test_next]
    y_next_pred = price_pred_next[mask_test_next]
    r_next_true = target_next_return[mask_test_next]
    r_next_pred = pred_next_return[mask_test_next]

    if len(y_next_true) > 0:
        metrics["next_test_contracts"] = int(len(y_next_true))
        metrics["next_price_rmse"] = rmse(y_next_pred, y_next_true)
        metrics["next_price_mae"] = mae(y_next_pred, y_next_true)
        metrics["next_price_r2"] = r2(y_next_pred, y_next_true)
        metrics["price_r2"] = metrics["next_price_r2"]
        metrics["price_r2_source"] = "next_day"
        metrics["next_return_directional_acc"] = float(np.mean(np.sign(r_next_pred) == np.sign(r_next_true)))
        metrics["next_return_corr"] = float(np.corrcoef(r_next_pred, r_next_true)[0, 1]) if len(r_next_true) > 1 else float("nan")

        # Baseline: "carry" (predict tomorrow's option price is today's).
        y_next_base = contract_price_target[mask_test_next]
        metrics["next_price_rmse_baseline_midcarry"] = rmse(y_next_base, y_next_true)
        metrics["next_price_mae_baseline_midcarry"] = mae(y_next_base, y_next_true)
        metrics["next_price_r2_baseline_midcarry"] = r2(y_next_base, y_next_true)
    else:
        metrics["next_test_contracts"] = 0
        metrics["next_price_rmse"] = float("nan")
        metrics["next_price_mae"] = float("nan")
        metrics["next_price_r2"] = float("nan")
        # Fallback for edge cases where no next-day targets are available.
        metrics["price_r2"] = metrics["price_r2_same_day"]
        metrics["price_r2_source"] = "same_day_fallback"
        metrics["next_return_directional_acc"] = float("nan")
        metrics["next_return_corr"] = float("nan")

    # Baseline for execution: constant probability = empirical positive rate.
    if len(e_true) > 0 and np.isfinite(metrics.get("exec_positive_rate", np.nan)):
        p0 = float(np.clip(metrics["exec_positive_rate"], 1e-6, 1.0 - 1e-6))
        metrics["exec_brier_baseline_constant"] = brier_score(np.full_like(e_true, p0, dtype=float), e_true)
    else:
        metrics["exec_brier_baseline_constant"] = float("nan")

    pred_test = recon_raw[test_dates]
    obs_test = iv_surface_obs[test_dates]
    # Same-day reconstruction quality (helps diagnose representation learning).
    metrics["surface_iv_rmse"] = rmse(pred_test, obs_test)
    metrics["surface_iv_mae"] = mae(pred_test, obs_test)
    metrics["surface_recon_iv_rmse"] = metrics["surface_iv_rmse"]
    metrics["surface_recon_iv_mae"] = metrics["surface_iv_mae"]

    # 1-step ahead surface forecast quality (this is the metric that matters for
    # trading and any forward-looking strategy).
    forecast_entry_idx = test_dates[test_dates < (n_dates - 1)]
    forecast_target_idx = forecast_entry_idx + 1
    metrics["surface_forecast_days"] = int(len(forecast_entry_idx))
    if len(forecast_entry_idx) > 0:
        pred_forecast = forecast_raw[forecast_entry_idx]
        obs_forecast = iv_surface_obs[forecast_target_idx]
        metrics["surface_forecast_iv_rmse"] = rmse(pred_forecast, obs_forecast)
        metrics["surface_forecast_iv_mae"] = mae(pred_forecast, obs_forecast)

        # Baseline: persistence (tomorrow's surface = today's surface).
        base_forecast = iv_surface_obs[forecast_entry_idx]
        metrics["surface_forecast_iv_rmse_baseline_persistence"] = rmse(base_forecast, obs_forecast)
        metrics["surface_forecast_iv_mae_baseline_persistence"] = mae(base_forecast, obs_forecast)

        mse_model = float(np.mean((pred_forecast - obs_forecast) ** 2))
        mse_base = float(np.mean((base_forecast - obs_forecast) ** 2))
        metrics["surface_forecast_skill_mse_vs_persistence"] = (
            float(1.0 - (mse_model / mse_base)) if mse_base > 0 else float("nan")
        )
    else:
        metrics["surface_forecast_iv_rmse"] = float("nan")
        metrics["surface_forecast_iv_mae"] = float("nan")
        metrics["surface_forecast_iv_rmse_baseline_persistence"] = float("nan")
        metrics["surface_forecast_iv_mae_baseline_persistence"] = float("nan")
        metrics["surface_forecast_skill_mse_vs_persistence"] = float("nan")

    workers = _resolve_num_workers(num_workers, len(test_dates))
    parallel_backend = "sequential"
    if len(test_dates) == 0:
        cal_obs = np.array([], dtype=np.float32)
        cal_pred = np.array([], dtype=np.float32)
        bfly_obs = np.array([], dtype=np.float32)
        bfly_pred = np.array([], dtype=np.float32)
    elif workers <= 1:
        rows = [_noarb_for_day(obs_test[i], pred_test[i], x_grid, tenor_days) for i in range(len(test_dates))]
        cal_obs = np.array([r[0] for r in rows], dtype=np.float32)
        cal_pred = np.array([r[1] for r in rows], dtype=np.float32)
        bfly_obs = np.array([r[2] for r in rows], dtype=np.float32)
        bfly_pred = np.array([r[3] for r in rows], dtype=np.float32)
    else:
        executor_cls = ProcessPoolExecutor
        parallel_backend = "process"
        try:
            ex_obj = executor_cls(max_workers=workers)
        except (PermissionError, OSError):
            ex_obj = ThreadPoolExecutor(max_workers=workers)
            parallel_backend = "thread"
        with ex_obj as ex:
            futures = [
                ex.submit(_noarb_for_day, obs_test[i], pred_test[i], x_grid, tenor_days)
                for i in range(len(test_dates))
            ]
            rows = [f.result() for f in futures]
        cal_obs = np.array([r[0] for r in rows], dtype=np.float32)
        cal_pred = np.array([r[1] for r in rows], dtype=np.float32)
        bfly_obs = np.array([r[2] for r in rows], dtype=np.float32)
        bfly_pred = np.array([r[3] for r in rows], dtype=np.float32)

    metrics["calendar_violation_obs_mean"] = float(np.mean(cal_obs)) if len(cal_obs) else float("nan")
    metrics["calendar_violation_pred_mean"] = float(np.mean(cal_pred)) if len(cal_pred) else float("nan")
    metrics["butterfly_violation_obs_mean"] = float(np.mean(bfly_obs)) if len(bfly_obs) else float("nan")
    metrics["butterfly_violation_pred_mean"] = float(np.mean(bfly_pred)) if len(bfly_pred) else float("nan")

    # No-arbitrage diagnostics for 1-step ahead forecasts (t -> t+1) on test.
    if len(forecast_entry_idx) > 0:
        pred_forecast = forecast_raw[forecast_entry_idx]
        obs_forecast = iv_surface_obs[forecast_target_idx]

        rows_f = [
            _noarb_for_day(obs_forecast[i], pred_forecast[i], x_grid, tenor_days) for i in range(len(forecast_entry_idx))
        ]
        cal_obs_f = np.array([r[0] for r in rows_f], dtype=np.float32)
        cal_pred_f = np.array([r[1] for r in rows_f], dtype=np.float32)
        bfly_obs_f = np.array([r[2] for r in rows_f], dtype=np.float32)
        bfly_pred_f = np.array([r[3] for r in rows_f], dtype=np.float32)

        metrics["calendar_violation_forecast_obs_mean"] = float(np.mean(cal_obs_f)) if len(cal_obs_f) else float("nan")
        metrics["calendar_violation_forecast_pred_mean"] = float(np.mean(cal_pred_f)) if len(cal_pred_f) else float("nan")
        metrics["butterfly_violation_forecast_obs_mean"] = float(np.mean(bfly_obs_f)) if len(bfly_obs_f) else float("nan")
        metrics["butterfly_violation_forecast_pred_mean"] = float(np.mean(bfly_pred_f)) if len(bfly_pred_f) else float("nan")
    else:
        cal_obs_f = np.array([], dtype=np.float32)
        cal_pred_f = np.array([], dtype=np.float32)
        bfly_obs_f = np.array([], dtype=np.float32)
        bfly_pred_f = np.array([], dtype=np.float32)
        metrics["calendar_violation_forecast_obs_mean"] = float("nan")
        metrics["calendar_violation_forecast_pred_mean"] = float("nan")
        metrics["butterfly_violation_forecast_obs_mean"] = float("nan")
        metrics["butterfly_violation_forecast_pred_mean"] = float("nan")

    metrics["num_workers"] = workers
    metrics["parallel_backend"] = parallel_backend

    contract_df = pd.DataFrame(
        {
            "date_idx": contract_date_idx,
            "date": ds["contract_date"].astype(str),
            "symbol": ds["contract_symbol"].astype(str),
            "call_put": ds["contract_call_put"].astype(str),
            "dte": ds["contract_dte"].astype(int),
            "strike": ds["contract_strike"].astype(float),
            "spot": ds["contract_spot"].astype(float),
            "mid": ds["contract_mid"].astype(float),
            "target_price_norm": contract_price_target,
            "pred_price_norm": price_pred,
            "target_next_price_norm": target_next_price_norm,
            "pred_next_price_norm": price_pred_next,
            "target_next_return": target_next_return,
            "pred_next_return": pred_next_return,
            "target_fill": contract_fill_target,
            "pred_fill_prob": exec_prob,
        }
    )
    contract_df.to_parquet(eval_dir / "contract_predictions.parquet", index=False)

    noarb_dates = pd.DataFrame(
        {
            "date": dates[test_dates],
            "calendar_obs": cal_obs,
            "calendar_pred": cal_pred,
            "butterfly_obs": bfly_obs,
            "butterfly_pred": bfly_pred,
        }
    )
    noarb_dates.to_parquet(eval_dir / "noarb_test_dates.parquet", index=False)

    if len(forecast_entry_idx) > 0:
        noarb_forecast = pd.DataFrame(
            {
                "date_entry": dates[forecast_entry_idx],
                "date_target": dates[forecast_target_idx],
                "calendar_obs": cal_obs_f,
                "calendar_pred": cal_pred_f,
                "butterfly_obs": bfly_obs_f,
                "butterfly_pred": bfly_pred_f,
            }
        )
        noarb_forecast.to_parquet(eval_dir / "noarb_forecast_test_dates.parquet", index=False)

    latent = pd.DataFrame(z_all, columns=[f"z_{i}" for i in range(z_all.shape[1])])
    latent.insert(0, "date", dates)
    latent.to_parquet(eval_dir / "latent_states.parquet", index=False)

    np.savez_compressed(
        eval_dir / "surface_predictions.npz",
        dates=dates,
        iv_surface_obs=iv_surface_obs,
        iv_surface_pred=recon_raw.astype(np.float32),
        iv_surface_forecast=forecast_raw.astype(np.float32),
        x_grid=x_grid,
        tenor_days=tenor_days,
        test_date_index=test_dates,
        forecast_entry_index=forecast_entry_idx,
        forecast_target_index=forecast_target_idx,
    )

    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return eval_dir
