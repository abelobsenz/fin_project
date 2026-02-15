from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from spygen.models.flow import ConditionalSurfaceFlow


def load_checkpoint(path: str | Path) -> ConditionalSurfaceFlow:
    payload = torch.load(path, map_location="cpu")
    model = ConditionalSurfaceFlow(
        theta_dim=int(payload["theta_dim"]),
        context_dim=int(payload["context_dim"]),
        nx=int(payload["nx"]),
        nt=int(payload["nt"]),
        hidden_features=int(payload.get("train_config", {}).get("hidden_size", 128)),
        num_layers=int(payload.get("train_config", {}).get("flow_layers", 4)),
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
    model: ConditionalSurfaceFlow, theta_raw: np.ndarray, context: np.ndarray
) -> np.ndarray:
    with torch.no_grad():
        x = torch.as_tensor(context, dtype=torch.float32)
        y = torch.as_tensor(theta_raw, dtype=torch.float32)
        lp = model.log_prob(y, context=x)
    return lp.cpu().numpy()


def sample_surfaces(
    model: ConditionalSurfaceFlow,
    context: np.ndarray,
    n_samples: int = 64,
) -> np.ndarray:
    with torch.no_grad():
        x = torch.as_tensor(context, dtype=torch.float32)
        surfaces = model.sample_surfaces(context=x, num_samples=n_samples)
    return surfaces.cpu().numpy()


def conditional_mean_surface(
    model: ConditionalSurfaceFlow,
    context: np.ndarray,
    n_samples: int = 64,
) -> np.ndarray:
    samples = sample_surfaces(model, context=context, n_samples=n_samples)
    return samples.mean(axis=1)
