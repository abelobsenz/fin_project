"""Hedge policy models and bundle I/O.

We represent the policy as a small MLP that outputs a hedge ratio in
[0, max_ratio]. The hedge ratio is applied to the strategy's net option delta
exposure measured in *share* units:

    hedge_shares = - hedge_ratio * net_option_delta_shares

This matches the existing backtest semantics but allows the ratio to be
state-dependent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for hedge policies.") from exc


@dataclass(frozen=True, slots=True)
class HedgeFeatureSpec:
    """Controls what goes into the policy feature vector."""

    use_latent: bool = True
    use_context: bool = True
    use_net_delta: bool = True
    use_spot: bool = True


def build_hedge_features(
    *,
    latent_z: np.ndarray | None,
    context: np.ndarray | None,
    net_option_delta_shares: float,
    spot: float,
    spec: HedgeFeatureSpec,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    if spec.use_latent:
        if latent_z is None:
            raise ValueError("latent_z is required when use_latent=True")
        parts.append(np.asarray(latent_z, dtype=np.float32).reshape(-1))
    if spec.use_context:
        if context is None:
            raise ValueError("context is required when use_context=True")
        parts.append(np.asarray(context, dtype=np.float32).reshape(-1))
    if spec.use_net_delta:
        parts.append(np.asarray([float(net_option_delta_shares)], dtype=np.float32))
    if spec.use_spot:
        parts.append(np.asarray([float(spot)], dtype=np.float32))
    if not parts:
        raise ValueError("Empty hedge feature vector (spec disabled all inputs)")
    return np.concatenate(parts, axis=0).astype(np.float32)


class HedgeRatioMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, depth: int = 2, max_ratio: float = 1.25):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.max_ratio = float(max_ratio)

        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(int(depth)):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output in [0, max_ratio].
        return torch.sigmoid(self.net(x)).reshape(-1) * float(self.max_ratio)


@dataclass(slots=True)
class HedgePolicyBundle:
    """Serializable wrapper around the trained policy."""

    model: HedgeRatioMLP
    feature_spec: HedgeFeatureSpec
    feature_mean: np.ndarray
    feature_std: np.ndarray

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "version": "hedge_policy_v1",
            "model": {
                "input_dim": int(self.feature_mean.size),
                "hidden_dim": int(getattr(self.model.net[0], "out_features", 64)),
                "depth": int((len(self.model.net) - 1) // 2),
                "max_ratio": float(self.model.max_ratio),
                "state_dict": self.model.state_dict(),
            },
            "feature_spec": asdict(self.feature_spec),
            "feature_mean": np.asarray(self.feature_mean, dtype=np.float32),
            "feature_std": np.asarray(self.feature_std, dtype=np.float32),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path, device: str | torch.device | None = None) -> "HedgePolicyBundle":
        path = Path(path)
        map_location = device or "cpu"
        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=map_location)

        if payload.get("version") != "hedge_policy_v1":
            raise ValueError(f"Unsupported hedge policy bundle version: {payload.get('version')!r}")

        m = payload["model"]
        model = HedgeRatioMLP(
            input_dim=int(m["input_dim"]),
            hidden_dim=int(m.get("hidden_dim", 64)),
            depth=int(m.get("depth", 2)),
            max_ratio=float(m.get("max_ratio", 1.25)),
        )
        model.load_state_dict(m["state_dict"])
        model.to(map_location)
        model.eval()

        spec = HedgeFeatureSpec(**(payload.get("feature_spec") or {}))
        mean = np.asarray(payload["feature_mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(payload["feature_std"], dtype=np.float32).reshape(-1)
        std = np.where(std > 1e-12, std, 1.0).astype(np.float32)
        return cls(model=model, feature_spec=spec, feature_mean=mean, feature_std=std)

    def predict_ratio(self, features: np.ndarray, *, device: str | torch.device | None = None) -> float:
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != int(self.feature_mean.size):
            raise ValueError(f"Feature dim mismatch: got {x.shape[1]}, expected {int(self.feature_mean.size)}")

        x = (x - self.feature_mean.reshape(1, -1)) / self.feature_std.reshape(1, -1)
        dev = device or next(self.model.parameters()).device
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32, device=dev)
            r = self.model(xt).reshape(-1)
            return float(r.detach().cpu().numpy()[0])
