from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from spygen.models.flow import ConditionalSurfaceFlow
from spygen.utils.paths import ensure_dir


@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-6
    hidden_size: int = 128
    flow_layers: int = 4
    early_stopping_patience: int = 5
    target_mode: str = "delta_theta_raw"
    flow_transform: str = "spline"
    flow_bins: int = 8
    aux_price_weight: float = 1.0
    aux_iv_weight: float = 0.1
    aux_samples_train: int = 8
    aux_samples_eval: int = 32
    early_stop_metric: str = "iv_rmse"


def _infer_target_mode(ds: dict[str, np.ndarray], config_mode: str) -> str:
    if "target_mode" in ds:
        raw = ds["target_mode"]
        if raw.ndim == 0:
            return str(raw.item())
        if raw.size > 0:
            return str(raw.reshape(-1)[0])
    return config_mode


def _get_training_targets(
    ds: dict[str, np.ndarray], target_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    theta_level = ds.get("theta_raw_level", ds["theta_raw"]).astype(np.float32)
    theta_target = ds.get("theta_target_raw")
    if theta_target is not None:
        return theta_target.astype(np.float32), theta_level
    if target_mode == "delta_theta_raw":
        prev = np.vstack([theta_level[0], theta_level[:-1]])
        return (theta_level - prev).astype(np.float32), theta_level
    return theta_level, theta_level


def _normal_cdf_torch(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _normal_pdf_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_call_price_torch(x: torch.Tensor, t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    t_safe = torch.clamp(t, min=1.0 / 365.0)
    sigma_safe = torch.clamp(sigma, min=1e-6)
    vol = sigma_safe * torch.sqrt(t_safe)
    d1 = (-x + 0.5 * vol * vol) / vol
    d2 = d1 - vol
    return _normal_cdf_torch(d1) - torch.exp(x) * _normal_cdf_torch(d2)


def _norm_vega_torch(x: torch.Tensor, t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    t_safe = torch.clamp(t, min=1.0 / 365.0)
    sigma_safe = torch.clamp(sigma, min=1e-6)
    vol = sigma_safe * torch.sqrt(t_safe)
    d1 = (-x + 0.5 * vol * vol) / vol
    return _normal_pdf_torch(d1) * torch.sqrt(t_safe)


def _implied_vol_torch(
    call_norm: torch.Tensor,
    x_mesh: torch.Tensor,
    t_mesh: torch.Tensor,
    n_iter: int = 12,
) -> torch.Tensor:
    intrinsic = torch.clamp(1.0 - torch.exp(x_mesh), min=0.0)
    lower = intrinsic + 1e-6
    upper = torch.full_like(call_norm, 1.0 - 1e-6)
    target = torch.minimum(torch.maximum(call_norm, lower), upper)
    sigma = torch.full_like(target, 0.2)
    for _ in range(n_iter):
        price = _norm_call_price_torch(x_mesh, t_mesh, sigma)
        vega = _norm_vega_torch(x_mesh, t_mesh, sigma)
        sigma = torch.clamp(sigma - (price - target) / torch.clamp(vega, min=1e-6), 1e-4, 5.0)
    return sigma


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def train_flow_model(
    dataset_path: str | Path,
    output_dir: str | Path,
    config: TrainConfig,
    nx: int,
    nt: int,
) -> Path:
    set_seed(config.seed)
    ds = load_dataset_npz(dataset_path)
    context = ds["context"].astype(np.float32)
    target_mode = _infer_target_mode(ds, config.target_mode)
    theta_raw_target, theta_raw_level = _get_training_targets(ds, target_mode=target_mode)
    surfaces = ds["surface"].astype(np.float32)
    x_grid = ds["x_grid"].astype(np.float32)
    tenors_days = ds["tenors_days"].astype(np.float32)
    theta_raw_prev = ds.get("theta_raw_prev")
    if theta_raw_prev is None:
        theta_raw_prev = np.vstack([theta_raw_level[0], theta_raw_level[:-1]])
    theta_raw_prev = theta_raw_prev.astype(np.float32)

    n = context.shape[0]
    if n < 10:
        raise ValueError("Need at least 10 samples to train the flow")

    split = max(1, int(n * 0.8))
    x_train, x_val = context[:split], context[split:]
    y_train, y_val = theta_raw_target[:split], theta_raw_target[split:]
    surf_train, surf_val = surfaces[:split], surfaces[split:]
    base_train, base_val = theta_raw_prev[:split], theta_raw_prev[split:]

    train_dl = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train),
            torch.from_numpy(surf_train),
            torch.from_numpy(base_train),
        ),
        batch_size=min(config.batch_size, len(x_train)),
        shuffle=True,
    )
    val_dl = DataLoader(
        TensorDataset(
            torch.from_numpy(x_val),
            torch.from_numpy(y_val),
            torch.from_numpy(surf_val),
            torch.from_numpy(base_val),
        ),
        batch_size=min(config.batch_size, max(1, len(x_val))),
        shuffle=False,
    )

    model = ConditionalSurfaceFlow(
        theta_dim=theta_raw_target.shape[1],
        context_dim=context.shape[1],
        nx=nx,
        nt=nt,
        hidden_features=config.hidden_size,
        num_layers=config.flow_layers,
        transform_type=config.flow_transform,
        num_bins=config.flow_bins,
        target_mode=target_mode,
    )
    context_mean = x_train.mean(axis=0)
    context_std = x_train.std(axis=0)
    context_std = np.maximum(context_std, 1e-4)
    theta_mean = y_train.mean(axis=0)
    theta_std = y_train.std(axis=0)
    theta_std = np.maximum(theta_std, 1e-4)
    model.set_normalization_stats(
        context_mean=context_mean,
        context_std=context_std,
        theta_mean=theta_mean,
        theta_std=theta_std,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    x_mesh_np, t_mesh_np = np.meshgrid(x_grid, tenors_days / 365.0, indexing="ij")
    x_mesh = torch.as_tensor(x_mesh_np, dtype=torch.float32)
    t_mesh = torch.as_tensor(t_mesh_np, dtype=torch.float32)

    best_val = float("inf")
    best_val_nll = float("inf")
    best_val_price_rmse = float("inf")
    best_val_iv_rmse = float("inf")
    best_state = None
    patience = 0
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        train_losses: list[float] = []
        train_nll_losses: list[float] = []
        train_price_losses: list[float] = []
        train_iv_losses: list[float] = []
        for xb, yb, sb, bb in train_dl:
            optimizer.zero_grad(set_to_none=True)
            nll_loss = -model.log_prob(yb, context=xb).mean()

            base = bb if target_mode == "delta_theta_raw" else None
            mean_surface = model.conditional_mean_surface(
                xb,
                num_samples=max(1, config.aux_samples_train),
                base_theta_raw=base,
            )
            price_mse = functional.mse_loss(mean_surface, sb)

            if config.aux_iv_weight > 0.0:
                batch = int(mean_surface.shape[0])
                x_local = x_mesh.to(mean_surface.device).unsqueeze(0).expand(batch, -1, -1)
                t_local = t_mesh.to(mean_surface.device).unsqueeze(0).expand(batch, -1, -1)
                iv_pred = _implied_vol_torch(mean_surface, x_local, t_local)
                iv_obs = _implied_vol_torch(sb, x_local, t_local)
                iv_mse = functional.mse_loss(iv_pred, iv_obs)
            else:
                iv_mse = torch.zeros((), dtype=price_mse.dtype, device=price_mse.device)

            loss = nll_loss + config.aux_price_weight * price_mse + config.aux_iv_weight * iv_mse
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(float(loss.item()))
            train_nll_losses.append(float(nll_loss.item()))
            train_price_losses.append(float(torch.sqrt(price_mse).item()))
            train_iv_losses.append(float(torch.sqrt(iv_mse).item()))

        model.eval()
        with torch.no_grad():
            val_nll_losses = []
            val_price_rmse = []
            val_iv_rmse = []
            for xb, yb, sb, bb in val_dl:
                val_nll_losses.append(float((-model.log_prob(yb, context=xb).mean()).item()))
                base = bb if target_mode == "delta_theta_raw" else None
                val_mean_surface = model.conditional_mean_surface(
                    xb,
                    num_samples=max(1, config.aux_samples_eval),
                    base_theta_raw=base,
                )
                price_mse = functional.mse_loss(val_mean_surface, sb)
                val_price_rmse.append(float(torch.sqrt(price_mse).item()))

                batch = int(val_mean_surface.shape[0])
                x_local = x_mesh.to(val_mean_surface.device).unsqueeze(0).expand(batch, -1, -1)
                t_local = t_mesh.to(val_mean_surface.device).unsqueeze(0).expand(batch, -1, -1)
                iv_pred = _implied_vol_torch(val_mean_surface, x_local, t_local)
                iv_obs = _implied_vol_torch(sb, x_local, t_local)
                iv_rmse = torch.sqrt(functional.mse_loss(iv_pred, iv_obs))
                val_iv_rmse.append(float(iv_rmse.item()))

            val_nll = float(np.mean(val_nll_losses)) if val_nll_losses else float("inf")
            val_price = float(np.mean(val_price_rmse)) if val_price_rmse else float("inf")
            val_iv = float(np.mean(val_iv_rmse)) if val_iv_rmse else float("inf")

        if config.early_stop_metric == "iv_rmse":
            val_score = val_iv
        elif config.early_stop_metric == "price_rmse":
            val_score = val_price
        else:
            val_score = val_nll

        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "train_nll": float(np.mean(train_nll_losses))
                if train_nll_losses
                else float("nan"),
                "train_price_rmse": float(np.mean(train_price_losses))
                if train_price_losses
                else float("nan"),
                "train_iv_rmse": float(np.mean(train_iv_losses))
                if train_iv_losses
                else float("nan"),
                "val_score": val_score,
                "val_nll": val_nll,
                "val_price_rmse": val_price,
                "val_iv_rmse": val_iv,
            }
        )

        if val_score < best_val:
            best_val = val_score
            best_val_nll = val_nll
            best_val_price_rmse = val_price
            best_val_iv_rmse = val_iv
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir = ensure_dir(output_dir)
    ckpt_path = out_dir / "flow_latest.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "theta_dim": theta_raw_target.shape[1],
            "context_dim": context.shape[1],
            "nx": nx,
            "nt": nt,
            "train_config": asdict(config),
            "target_mode": target_mode,
            "best_val_score": best_val,
            "best_val_nll": best_val_nll,
            "best_val_price_rmse": best_val_price_rmse,
            "best_val_iv_rmse": best_val_iv_rmse,
            "context_mean": torch.from_numpy(context_mean),
            "context_std": torch.from_numpy(context_std),
            "theta_mean": torch.from_numpy(theta_mean),
            "theta_std": torch.from_numpy(theta_std),
        },
        ckpt_path,
    )
    metrics_path = out_dir / "train_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "target_mode": target_mode,
                "early_stop_metric": config.early_stop_metric,
                "best_val_score": best_val,
                "best_val_nll": best_val_nll,
                "best_val_price_rmse": best_val_price_rmse,
                "best_val_iv_rmse": best_val_iv_rmse,
                "history": history,
            },
            indent=2,
        )
    )
    return ckpt_path
