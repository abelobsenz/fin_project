from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as nn_functional


class ArbitrageFreeDecoder(nn.Module):
    """Decode unconstrained parameters into static-arbitrage-safe call surfaces.

    Parameterization per tenor uses nonnegative components:
    - tail value at highest strike
    - magnitude of final negative slope
    - nonnegative second differences across strike
    """

    def __init__(self, nx: int, nt: int, eps: float = 1e-6, max_call_norm: float = 1.0) -> None:
        super().__init__()
        self.nx = nx
        self.nt = nt
        self.param_dim = nx * nt
        self.eps = eps
        self.max_call_norm = max_call_norm

    def _decode_increment_curve(self, params: torch.Tensor) -> torch.Tensor:
        # params: (batch, nx)
        tail_value = nn_functional.softplus(params[:, 0]) + self.eps
        tail_slope_mag = nn_functional.softplus(params[:, 1]) + self.eps
        second = nn_functional.softplus(params[:, 2:]) + self.eps

        slopes: list[torch.Tensor] = [torch.empty(0, device=params.device)] * (self.nx - 1)
        slopes[-1] = -tail_slope_mag
        for i in range(self.nx - 3, -1, -1):
            slopes[i] = slopes[i + 1] - second[:, i]

        curve: list[torch.Tensor] = [torch.empty(0, device=params.device)] * self.nx
        curve[-1] = tail_value
        for i in range(self.nx - 2, -1, -1):
            curve[i] = curve[i + 1] - slopes[i]

        out = torch.stack(curve, dim=1)
        return torch.clamp(out, min=self.eps)

    def decode_increments(self, raw_params: torch.Tensor) -> torch.Tensor:
        if raw_params.dim() != 2 or raw_params.size(1) != self.param_dim:
            raise ValueError(f"Expected raw params shape (batch, {self.param_dim})")
        batch = raw_params.size(0)
        shaped = raw_params.view(batch, self.nt, self.nx)
        curves = [self._decode_increment_curve(shaped[:, j, :]) for j in range(self.nt)]
        increments = torch.stack(curves, dim=2)  # (batch, nx, nt)
        return increments

    def forward(self, raw_params: torch.Tensor) -> torch.Tensor:
        increments = self.decode_increments(raw_params)
        surface = torch.cumsum(increments, dim=2)
        # Normalize by per-sample global max to keep normalized calls bounded
        # while preserving monotonicity/convexity/calendar shape constraints.
        max_val = surface.amax(dim=(1, 2), keepdim=True)
        scale = torch.clamp(max_val / self.max_call_norm, min=1.0)
        return torch.clamp(surface / scale, min=self.eps)
