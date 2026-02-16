from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from spygen.models.deep_smoothing import DeepSmoothingSurfaceModel
from spygen.models.flow import ConditionalSurfaceFlow


def load_checkpoint(path: str | Path) -> ConditionalSurfaceFlow | DeepSmoothingSurfaceModel:
    payload = torch.load(path, map_location="cpu")
    model_type = str(payload.get("model_type", "flow")).strip().lower()
    if model_type == "deep_smoothing":
        x_grid = payload.get("x_grid")
        tenors_days = payload.get("tenors_days")
        if x_grid is None:
            x_grid = np.linspace(-0.3, 0.3, int(payload["nx"]), dtype=np.float32)
        if tenors_days is None:
            tenors_days = np.array([7, 14, 30, 60, 90, 180], dtype=np.float32)
        if hasattr(x_grid, "detach"):
            x_grid = x_grid.detach().cpu().numpy()
        if hasattr(tenors_days, "detach"):
            tenors_days = tenors_days.detach().cpu().numpy()
        train_cfg = payload.get("train_config", {})
        model = DeepSmoothingSurfaceModel(
            context_dim=int(payload["context_dim"]),
            nx=int(payload["nx"]),
            nt=int(payload["nt"]),
            x_grid=np.asarray(x_grid, dtype=np.float32),
            tenors_days=np.asarray(tenors_days, dtype=np.float32),
            hidden_size=int(train_cfg.get("ds_hidden_size", train_cfg.get("hidden_size", 128))),
            num_layers=int(train_cfg.get("ds_layers", 3)),
            dropout=float(train_cfg.get("ds_dropout", 0.1)),
            corr_scale=float(train_cfg.get("ds_corr_scale", 0.35)),
            prior_blend=float(train_cfg.get("ds_prior_blend", 0.2)),
            num_experts=int(train_cfg.get("ds_num_experts", 3)),
            min_sigma=float(train_cfg.get("ds_sigma_min", 0.003)),
            max_sigma=float(train_cfg.get("ds_sigma_max", 0.08)),
            sample_temperature=float(train_cfg.get("ds_sample_temperature", 1.0)),
            target_mode=str(payload.get("target_mode", "delta_theta_raw")),
        )
    else:
        model = ConditionalSurfaceFlow(
            theta_dim=int(payload["theta_dim"]),
            context_dim=int(payload["context_dim"]),
            nx=int(payload["nx"]),
            nt=int(payload["nt"]),
            hidden_features=int(payload.get("train_config", {}).get("hidden_size", 128)),
            num_layers=int(payload.get("train_config", {}).get("flow_layers", 4)),
            transform_type=str(payload.get("train_config", {}).get("flow_transform", "spline")),
            num_bins=int(payload.get("train_config", {}).get("flow_bins", 8)),
            target_mode=str(payload.get("target_mode", "theta_raw")),
        )
    model.load_state_dict(payload["model_state"], strict=False)
    if {
        "context_mean",
        "context_std",
        "theta_mean",
        "theta_std",
    }.issubset(payload.keys()):
        context_mean = np.asarray(payload["context_mean"], dtype=np.float32)
        context_std = np.asarray(payload["context_std"], dtype=np.float32)
        theta_mean = np.asarray(payload["theta_mean"], dtype=np.float32)
        theta_std = np.asarray(payload["theta_std"], dtype=np.float32)
        if hasattr(payload["context_mean"], "detach"):
            context_mean = payload["context_mean"].detach().cpu().numpy().astype(np.float32)
            context_std = payload["context_std"].detach().cpu().numpy().astype(np.float32)
            theta_mean = payload["theta_mean"].detach().cpu().numpy().astype(np.float32)
            theta_std = payload["theta_std"].detach().cpu().numpy().astype(np.float32)
        model.set_normalization_stats(
            context_mean=context_mean,
            context_std=context_std,
            theta_mean=theta_mean,
            theta_std=theta_std,
        )
    model.eval()
    return model


def log_likelihood(
    model: ConditionalSurfaceFlow | DeepSmoothingSurfaceModel,
    theta_raw: np.ndarray,
    context: np.ndarray,
    base_theta_raw: np.ndarray | None = None,
) -> np.ndarray:
    with torch.no_grad():
        x = torch.as_tensor(context, dtype=torch.float32)
        y = torch.as_tensor(theta_raw, dtype=torch.float32)
        if base_theta_raw is not None and bool(
            getattr(model, "supports_base_in_log_prob", False)
        ):
            base = torch.as_tensor(base_theta_raw, dtype=torch.float32)
            lp = model.log_prob(y, context=x, base_theta_raw=base)  # type: ignore[call-arg]
        else:
            lp = model.log_prob(y, context=x)  # type: ignore[call-arg]
    return lp.cpu().numpy()


def sample_surfaces(
    model: ConditionalSurfaceFlow | DeepSmoothingSurfaceModel,
    context: np.ndarray,
    n_samples: int = 64,
    base_theta_raw: np.ndarray | None = None,
) -> np.ndarray:
    with torch.no_grad():
        x = torch.as_tensor(context, dtype=torch.float32)
        base = (
            torch.as_tensor(base_theta_raw, dtype=torch.float32)
            if base_theta_raw is not None
            else None
        )
        surfaces = model.sample_surfaces(context=x, num_samples=n_samples, base_theta_raw=base)
    return surfaces.cpu().numpy()


def conditional_mean_surface(
    model: ConditionalSurfaceFlow | DeepSmoothingSurfaceModel,
    context: np.ndarray,
    n_samples: int = 64,
    base_theta_raw: np.ndarray | None = None,
) -> np.ndarray:
    with torch.no_grad():
        x = torch.as_tensor(context, dtype=torch.float32)
        base = (
            torch.as_tensor(base_theta_raw, dtype=torch.float32)
            if base_theta_raw is not None
            else None
        )
        mean = model.conditional_mean_surface(
            context=x,
            num_samples=n_samples,
            base_theta_raw=base,
        )
    return mean.cpu().numpy()
