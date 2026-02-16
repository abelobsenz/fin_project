from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import RandomPermutation

from spygen.models.decoder import ArbitrageFreeDecoder


class ConditionalSurfaceFlow(nn.Module):
    def __init__(
        self,
        theta_dim: int,
        context_dim: int,
        nx: int,
        nt: int,
        hidden_features: int = 128,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        transforms = []
        for _ in range(num_layers):
            transforms.append(RandomPermutation(features=theta_dim))
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=theta_dim,
                    hidden_features=hidden_features,
                    context_features=context_dim,
                )
            )
        transform = CompositeTransform(transforms)
        distribution = StandardNormal([theta_dim])
        self.flow = Flow(transform=transform, distribution=distribution)
        self.decoder = ArbitrageFreeDecoder(nx=nx, nt=nt)
        self.theta_dim = theta_dim
        self.context_dim = context_dim
        self.nx = nx
        self.nt = nt
        self._context_mean: np.ndarray | None = None
        self._context_std: np.ndarray | None = None
        self._theta_mean: np.ndarray | None = None
        self._theta_std: np.ndarray | None = None

    def set_normalization_stats(
        self,
        *,
        context_mean: np.ndarray,
        context_std: np.ndarray,
        theta_mean: np.ndarray,
        theta_std: np.ndarray,
    ) -> None:
        self._context_mean = np.asarray(context_mean, dtype=np.float32)
        self._context_std = np.asarray(context_std, dtype=np.float32)
        self._theta_mean = np.asarray(theta_mean, dtype=np.float32)
        self._theta_std = np.asarray(theta_std, dtype=np.float32)

    def _normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        if self._context_mean is None or self._context_std is None:
            return torch.clamp(context, min=-8.0, max=8.0)
        mean = torch.as_tensor(self._context_mean, device=context.device, dtype=context.dtype)
        std = torch.as_tensor(self._context_std, device=context.device, dtype=context.dtype)
        normalized = (context - mean) / std
        return torch.clamp(normalized, min=-8.0, max=8.0)

    def _normalize_theta(self, theta_raw: torch.Tensor) -> torch.Tensor:
        if self._theta_mean is None or self._theta_std is None:
            return torch.clamp(theta_raw, min=-8.0, max=8.0)
        mean = torch.as_tensor(self._theta_mean, device=theta_raw.device, dtype=theta_raw.dtype)
        std = torch.as_tensor(self._theta_std, device=theta_raw.device, dtype=theta_raw.dtype)
        normalized = (theta_raw - mean) / std
        return torch.clamp(normalized, min=-8.0, max=8.0)

    def _denormalize_theta(self, theta_norm: torch.Tensor) -> torch.Tensor:
        if self._theta_mean is None or self._theta_std is None:
            return theta_norm
        mean = torch.as_tensor(self._theta_mean, device=theta_norm.device, dtype=theta_norm.dtype)
        std = torch.as_tensor(self._theta_std, device=theta_norm.device, dtype=theta_norm.dtype)
        bounded = torch.clamp(theta_norm, min=-8.0, max=8.0)
        return bounded * std + mean

    def log_prob(self, theta_raw: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        theta_norm = self._normalize_theta(theta_raw)
        context_norm = self._normalize_context(context)
        return self.flow.log_prob(theta_norm, context=context_norm)

    def sample_theta_raw(self, context: torch.Tensor, num_samples: int) -> torch.Tensor:
        context_norm = self._normalize_context(context)
        samples = self.flow.sample(num_samples=num_samples, context=context_norm)
        if context_norm.dim() != 2:
            raise ValueError("context must be rank-2 tensor [batch, context_dim]")
        batch = int(context_norm.shape[0])

        # nflows shape may vary by version:
        # - (batch, num_samples, features)
        # - (batch*num_samples, features)
        # - (num_samples, features) when batch=1
        if samples.dim() == 3:
            if samples.shape[0] == num_samples and samples.shape[1] == batch:
                samples = samples.permute(1, 0, 2).contiguous()
            elif samples.shape[0] != batch or samples.shape[1] != num_samples:
                raise RuntimeError(
                    f"Unexpected 3D sample shape {tuple(samples.shape)} "
                    f"for batch={batch}, num_samples={num_samples}"
                )
        elif samples.dim() == 2:
            n_rows, feat = int(samples.shape[0]), int(samples.shape[1])
            if n_rows == batch * num_samples:
                samples = samples.view(batch, num_samples, feat)
            elif batch == 1 and n_rows == num_samples:
                samples = samples.unsqueeze(0)
            else:
                raise RuntimeError(
                    f"Unexpected 2D sample shape {tuple(samples.shape)} "
                    f"for batch={batch}, num_samples={num_samples}"
                )
        else:
            raise RuntimeError(f"Unexpected sample rank: {samples.dim()}")
        return self._denormalize_theta(samples)

    def sample_surfaces(self, context: torch.Tensor, num_samples: int) -> torch.Tensor:
        samples = self.sample_theta_raw(context=context, num_samples=num_samples)
        batch, n, feat = samples.shape
        decoded = self.decoder(samples.view(batch * n, feat))
        return decoded.view(batch, n, self.nx, self.nt)

    def conditional_mean_surface(
        self, context: torch.Tensor, num_samples: int = 64
    ) -> torch.Tensor:
        sampled = self.sample_surfaces(context=context, num_samples=num_samples)
        return sampled.mean(dim=1)
