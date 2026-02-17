"""PyTorch architecture for IV dynamics and option pricing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is required for ivdyn training. Install torch to continue."
    ) from exc

from ivdyn.model.scalers import ArrayScaler


@dataclass(slots=True)
class ModelConfig:
    latent_dim: int = 16
    vae_hidden: tuple[int, int] = (256, 128)
    dynamics_hidden: tuple[int, ...] = (128, 64)
    pricing_hidden: tuple[int, int] = (192, 96)
    execution_hidden: tuple[int, int] = (128, 64)
    dropout: float = 0.05


class FeedForward(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        d_prev = in_dim
        for d in hidden:
            layers += [nn.Linear(d_prev, d), nn.GELU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d_prev = d
        layers += [nn.Linear(d_prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IVDynamicsTorchModel(nn.Module):
    def __init__(
        self,
        surface_dim: int,
        context_dim: int,
        contract_dim: int,
        config: ModelConfig,
    ):
        super().__init__()
        self.surface_dim = int(surface_dim)
        self.context_dim = int(context_dim)
        self.contract_dim = int(contract_dim)
        self.config = config

        self.encoder = FeedForward(
            in_dim=self.surface_dim,
            hidden=config.vae_hidden,
            out_dim=2 * config.latent_dim,
            dropout=config.dropout,
        )
        self.decoder = FeedForward(
            in_dim=config.latent_dim,
            hidden=tuple(reversed(config.vae_hidden)),
            out_dim=self.surface_dim,
            dropout=config.dropout,
        )
        self.dynamics = FeedForward(
            in_dim=config.latent_dim + self.context_dim,
            hidden=config.dynamics_hidden,
            out_dim=config.latent_dim,
            dropout=config.dropout,
        )
        self.pricer = FeedForward(
            in_dim=config.latent_dim + self.contract_dim,
            hidden=config.pricing_hidden,
            out_dim=1,
            dropout=config.dropout,
        )
        self.execution = FeedForward(
            in_dim=config.latent_dim + self.contract_dim,
            hidden=config.execution_hidden,
            out_dim=1,
            dropout=config.dropout,
        )

    def encode(self, surface_scaled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(surface_scaled)
        l = self.config.latent_dim
        mu = out[:, :l]
        logvar = out[:, l:].clamp(min=-8.0, max=8.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward_dynamics(self, z_prev: torch.Tensor, context_scaled: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat([z_prev, context_scaled], dim=1))

    def forward_pricer(self, z: torch.Tensor, contract_scaled: torch.Tensor) -> torch.Tensor:
        return self.pricer(torch.cat([z, contract_scaled], dim=1))

    def forward_execution_logit(self, z: torch.Tensor, contract_scaled: torch.Tensor) -> torch.Tensor:
        return self.execution(torch.cat([z, contract_scaled], dim=1))


@dataclass(slots=True)
class ModelBundle:
    model: IVDynamicsTorchModel
    surface_scaler: ArrayScaler
    context_scaler: ArrayScaler
    contract_scaler: ArrayScaler
    price_scaler: ArrayScaler

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "surface_dim": self.model.surface_dim,
            "context_dim": self.model.context_dim,
            "contract_dim": self.model.contract_dim,
            "config": asdict(self.model.config),
            "state_dict": self.model.state_dict(),
            "surface_scaler": self.surface_scaler.state(),
            "context_scaler": self.context_scaler.state(),
            "contract_scaler": self.contract_scaler.state(),
            "price_scaler": self.price_scaler.state(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path, device: str | torch.device | None = None) -> "ModelBundle":
        map_location = device or "cpu"
        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            # Older torch versions do not support weights_only argument.
            payload = torch.load(path, map_location=map_location)

        cfg = ModelConfig(**payload["config"])
        model = IVDynamicsTorchModel(
            surface_dim=int(payload["surface_dim"]),
            context_dim=int(payload["context_dim"]),
            contract_dim=int(payload["contract_dim"]),
            config=cfg,
        )
        model.load_state_dict(payload["state_dict"])
        model.to(map_location)
        model.eval()

        return cls(
            model=model,
            surface_scaler=ArrayScaler.from_state(payload["surface_scaler"]),
            context_scaler=ArrayScaler.from_state(payload["context_scaler"]),
            contract_scaler=ArrayScaler.from_state(payload["contract_scaler"]),
            price_scaler=ArrayScaler.from_state(payload["price_scaler"]),
        )


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()
