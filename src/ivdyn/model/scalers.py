"""Numpy/Torch compatible feature scalers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional import guard
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class ArrayScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-6) -> "ArrayScaler":
        mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        std = x.std(axis=0, keepdims=True).astype(np.float32)
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean, std=std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)

    def transform_torch(self, x):
        if torch is None:
            raise RuntimeError("torch is not available")
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return (x - mean) / std

    def inverse_transform_torch(self, x):
        if torch is None:
            raise RuntimeError("torch is not available")
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return x * std + mean

    def state(self) -> dict[str, np.ndarray]:
        return {"mean": self.mean.copy(), "std": self.std.copy()}

    @classmethod
    def from_state(cls, state: dict[str, np.ndarray]) -> "ArrayScaler":
        return cls(mean=np.asarray(state["mean"], dtype=np.float32), std=np.asarray(state["std"], dtype=np.float32))
