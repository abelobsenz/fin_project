from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from spygen.models.decoder import ArbitrageFreeDecoder


class DeepSmoothingSurfaceModel(nn.Module):
    """Hybrid prior-corrector model inspired by Deep Smoothing (NeurIPS 2020).

    The network predicts a multiplicative correction over a prior surface and
    applies soft shape projections to stabilize training on noisy EOD data.
    """

    supports_base_in_log_prob: bool = True

    def __init__(
        self,
        *,
        context_dim: int,
        nx: int,
        nt: int,
        x_grid: np.ndarray,
        tenors_days: np.ndarray,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        corr_scale: float = 0.35,
        prior_blend: float = 0.2,
        num_experts: int = 3,
        min_sigma: float = 0.003,
        max_sigma: float = 0.08,
        sample_temperature: float = 1.0,
        target_mode: str = "delta_theta_raw",
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.nx = int(nx)
        self.nt = int(nt)
        self.target_mode = target_mode
        self.corr_scale = float(corr_scale)
        self.prior_blend = float(prior_blend)
        self.num_experts = max(1, int(num_experts))
        self.min_sigma = float(max(1e-6, min_sigma))
        self.max_sigma = float(max(self.min_sigma + 1e-6, max_sigma))
        self.sample_temperature = float(max(0.0, sample_temperature))

        layers: list[nn.Module] = []
        in_features = self.context_dim
        for _ in range(max(1, num_layers)):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.correction_heads = nn.ModuleList(
            [nn.Linear(in_features, self.nx * self.nt) for _ in range(self.num_experts)]
        )
        self.gating_head = (
            nn.Linear(in_features, self.num_experts) if self.num_experts > 1 else None
        )
        self.log_sigma_head = nn.Linear(in_features, self.nx * self.nt)

        self.decoder = ArbitrageFreeDecoder(nx=self.nx, nt=self.nt)

        x = np.asarray(x_grid, dtype=np.float32).reshape(1, self.nx, 1)
        tenors = np.asarray(tenors_days, dtype=np.float32).reshape(1, 1, self.nt)
        tenor_years = tenors / 365.0

        intrinsic = np.maximum(0.0, 1.0 - np.exp(x))
        # Smooth baseline decays with tenor and moneyness distance.
        base_template = intrinsic * np.exp(-0.6 * tenor_years) + 0.12 * np.exp(
            -np.abs(x) / 0.22
        )
        base_template = np.clip(base_template, intrinsic + 1e-6, 1.0 - 1e-6)

        error_weights = np.exp(-np.abs(x) / 0.2) * np.sqrt(np.maximum(tenor_years, 1.0 / 365.0))
        error_weights = error_weights / np.maximum(error_weights.mean(), 1e-8)

        self.register_buffer("x_grid_tensor", torch.from_numpy(x))
        self.register_buffer("tenor_years", torch.from_numpy(tenor_years))
        self.register_buffer("intrinsic_surface", torch.from_numpy(intrinsic))
        self.register_buffer("prior_template", torch.from_numpy(base_template))
        self.register_buffer("error_weights", torch.from_numpy(error_weights))

        self._context_mean: np.ndarray | None = None
        self._context_std: np.ndarray | None = None

    @staticmethod
    def _clip_surface(
        surface: torch.Tensor,
        *,
        lower: torch.Tensor,
    ) -> torch.Tensor:
        upper = torch.full_like(surface, 1.0 - 1e-6)
        return torch.minimum(torch.maximum(surface, lower + 1e-6), upper)

    def set_normalization_stats(
        self,
        *,
        context_mean: np.ndarray,
        context_std: np.ndarray,
        theta_mean: np.ndarray,
        theta_std: np.ndarray,
    ) -> None:
        _ = (theta_mean, theta_std)
        self._context_mean = np.asarray(context_mean, dtype=np.float32)
        self._context_std = np.asarray(context_std, dtype=np.float32)

    def _normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        if self._context_mean is None or self._context_std is None:
            return torch.clamp(context, min=-8.0, max=8.0)
        mean = torch.as_tensor(self._context_mean, device=context.device, dtype=context.dtype)
        std = torch.as_tensor(self._context_std, device=context.device, dtype=context.dtype)
        normalized = (context - mean) / torch.clamp(std, min=1e-4)
        return torch.clamp(normalized, min=-8.0, max=8.0)

    def _base_prior(
        self,
        batch: int,
        *,
        base_theta_raw: torch.Tensor | None = None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if base_theta_raw is not None:
            prior = self.decoder(base_theta_raw).view(batch, self.nx, self.nt)
        else:
            prior = self.prior_template.expand(batch, -1, -1)
        intrinsic = self.intrinsic_surface.to(device=device, dtype=dtype)
        return self._clip_surface(prior, lower=intrinsic)

    def _shape_projection(self, surface: torch.Tensor) -> torch.Tensor:
        intrinsic = self.intrinsic_surface.to(device=surface.device, dtype=surface.dtype)
        projected = self._clip_surface(surface, lower=intrinsic)

        # Alternate simple projections to keep calendar + strike monotone + strike convex.
        for _ in range(2):
            # Calendar monotonicity.
            projected = torch.cummax(projected, dim=2)[0]
            # Strike monotonicity.
            projected = torch.cummin(projected, dim=1)[0]
            # Strike convexity via monotone first differences.
            projected = self._enforce_convex_in_strike(projected)
            projected = self._clip_surface(projected, lower=intrinsic)
        return projected

    def _enforce_convex_in_strike(self, surface: torch.Tensor) -> torch.Tensor:
        if self.nx < 3:
            return surface
        pieces: list[torch.Tensor] = []
        for j in range(self.nt):
            c = surface[:, :, j]
            d = c[:, 1:] - c[:, :-1]
            # Calls decrease in strike.
            d = torch.minimum(d, torch.zeros_like(d))
            # Convexity: first differences should be nondecreasing.
            d = torch.cummax(d, dim=1)[0]
            c0 = c[:, :1]
            c_proj = torch.cat([c0, c0 + torch.cumsum(d, dim=1)], dim=1)
            pieces.append(c_proj)
        return torch.stack(pieces, dim=2)

    def _mixture_correction(self, features: torch.Tensor) -> torch.Tensor:
        batch = int(features.shape[0])
        if self.num_experts == 1:
            return self.correction_heads[0](features).view(batch, self.nx, self.nt)
        experts = torch.stack(
            [head(features).view(batch, self.nx, self.nt) for head in self.correction_heads],
            dim=1,
        )
        logits = self.gating_head(features)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1).unsqueeze(-1)
        return torch.sum(weights * experts, dim=1)

    def _surface_sigma(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.log_sigma_head(features).view(int(features.shape[0]), self.nx, self.nt)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * torch.sigmoid(raw)
        return torch.clamp(sigma, min=self.min_sigma, max=self.max_sigma)

    def _predict_surface_and_sigma(
        self,
        context: torch.Tensor,
        *,
        base_theta_raw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.dim() != 2:
            raise ValueError("context must be rank-2 [batch, context_dim]")
        batch = int(context.shape[0])
        x = self._normalize_context(context)
        features = self.backbone(x)
        prior = self._base_prior(
            batch,
            base_theta_raw=base_theta_raw,
            device=context.device,
            dtype=context.dtype,
        )
        correction_raw = self._mixture_correction(features)
        correction = 1.0 + self.corr_scale * torch.tanh(correction_raw)
        pred = prior * correction
        pred = (1.0 - self.prior_blend) * pred + self.prior_blend * prior
        pred = self._shape_projection(pred)
        sigma = self._surface_sigma(features).to(device=pred.device, dtype=pred.dtype)
        return pred, sigma

    def forward_surface(
        self,
        context: torch.Tensor,
        *,
        base_theta_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred, _ = self._predict_surface_and_sigma(context, base_theta_raw=base_theta_raw)
        return pred

    def _target_surface_from_theta(
        self,
        theta_raw: torch.Tensor,
        *,
        base_theta_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if theta_raw.dim() != 2:
            raise ValueError("theta_raw must be rank-2 [batch, theta_dim]")
        if self.target_mode == "delta_theta_raw":
            if base_theta_raw is None:
                raise ValueError("base_theta_raw is required for delta_theta_raw mode")
            theta_level = theta_raw + base_theta_raw
        else:
            theta_level = theta_raw
        target = self.decoder(theta_level).view(theta_raw.shape[0], self.nx, self.nt)
        intrinsic = self.intrinsic_surface.to(device=target.device, dtype=target.dtype)
        return self._clip_surface(target, lower=intrinsic)

    def log_prob(
        self,
        theta_raw: torch.Tensor,
        context: torch.Tensor,
        *,
        base_theta_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred, sigma = self._predict_surface_and_sigma(context, base_theta_raw=base_theta_raw)
        target = self._target_surface_from_theta(theta_raw, base_theta_raw=base_theta_raw)
        weights = self.error_weights.to(device=pred.device, dtype=pred.dtype)
        sq = ((pred - target) / torch.clamp(sigma, min=1e-6)) ** 2
        nll_grid = 0.5 * (sq + 2.0 * torch.log(torch.clamp(sigma, min=1e-6)))
        weighted = nll_grid * weights
        nll = weighted.mean(dim=(1, 2))
        return -nll

    def sample_surfaces(
        self,
        context: torch.Tensor,
        num_samples: int,
        base_theta_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = int(max(1, num_samples))
        pred, sigma = self._predict_surface_and_sigma(context, base_theta_raw=base_theta_raw)
        if n == 1 or self.sample_temperature <= 0.0:
            return pred.unsqueeze(1).repeat(1, n, 1, 1)
        noise = torch.randn(
            pred.shape[0],
            n,
            self.nx,
            self.nt,
            dtype=pred.dtype,
            device=pred.device,
        )
        draws = pred.unsqueeze(1) + self.sample_temperature * sigma.unsqueeze(1) * noise
        flat = draws.view(pred.shape[0] * n, self.nx, self.nt)
        repaired = self._shape_projection(flat)
        return repaired.view(pred.shape[0], n, self.nx, self.nt)

    def conditional_mean_surface(
        self,
        context: torch.Tensor,
        num_samples: int = 64,
        base_theta_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = num_samples
        return self.forward_surface(context, base_theta_raw=base_theta_raw)

    def soft_arb_penalty(
        self,
        surface: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cal = torch.relu(surface[:, :, :-1] - surface[:, :, 1:]).mean()
        if self.nx < 3:
            zero = torch.zeros((), device=surface.device, dtype=surface.dtype)
            return cal, zero, zero
        d2 = surface[:, 2:, :] - 2.0 * surface[:, 1:-1, :] + surface[:, :-2, :]
        butterfly = torch.relu(-d2).mean()
        edge_curvature = (d2[:, [0, -1], :] ** 2).mean()
        return cal, butterfly, edge_curvature
