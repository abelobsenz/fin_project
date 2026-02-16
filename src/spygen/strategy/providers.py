from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from spygen.models.flow import ConditionalSurfaceFlow
from spygen.strategy.signals import project_residual


@dataclass(slots=True)
class SignalOutput:
    z: float
    residual: np.ndarray
    projections: dict[str, float]
    reference_surface: np.ndarray


class SignalProvider(ABC):
    def __init__(self, basis: dict[str, np.ndarray], residual_clip: float = 1.0) -> None:
        self.basis = basis
        self.residual_clip = residual_clip

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def prepare(
        self,
        *,
        surfaces: np.ndarray,
        context: np.ndarray,
        theta_raw: np.ndarray,
        model: ConditionalSurfaceFlow | None,
        theta_base_raw: np.ndarray | None = None,
        target_mode: str = "theta_raw",
    ) -> None:
        ...

    @abstractmethod
    def signal_for_day(self, idx: int) -> SignalOutput:
        ...

    def _build_output(self, observed: np.ndarray, reference: np.ndarray, z: float) -> SignalOutput:
        raw_residual = observed - reference
        residual = np.clip(raw_residual, -self.residual_clip, self.residual_clip)
        projections = project_residual(residual, self.basis)
        return SignalOutput(
            z=float(z),
            residual=residual,
            projections=projections,
            reference_surface=reference,
        )


class DeepFlowSignalProvider(SignalProvider):
    name = "deep_flow"

    def __init__(
        self,
        basis: dict[str, np.ndarray],
        n_samples: int = 32,
        residual_clip: float = 1.0,
    ) -> None:
        super().__init__(basis=basis, residual_clip=residual_clip)
        self.n_samples = n_samples
        self._surfaces: np.ndarray | None = None
        self._mean_surfaces: np.ndarray | None = None
        self._z: np.ndarray | None = None

    def prepare(
        self,
        *,
        surfaces: np.ndarray,
        context: np.ndarray,
        theta_raw: np.ndarray,
        model: ConditionalSurfaceFlow | None,
        theta_base_raw: np.ndarray | None = None,
        target_mode: str = "theta_raw",
    ) -> None:
        if model is None:
            raise ValueError("DeepFlowSignalProvider requires a trained model")
        x = torch.as_tensor(context, dtype=torch.float32)
        y = torch.as_tensor(theta_raw, dtype=torch.float32)
        base = (
            torch.as_tensor(theta_base_raw, dtype=torch.float32)
            if theta_base_raw is not None and target_mode == "delta_theta_raw"
            else None
        )
        with torch.no_grad():
            if base is not None and bool(getattr(model, "supports_base_in_log_prob", False)):
                log_prob = model.log_prob(
                    y,
                    context=x,
                    base_theta_raw=base,
                ).cpu().numpy()
            else:
                log_prob = model.log_prob(y, context=x).cpu().numpy()
            mean_surfaces = model.conditional_mean_surface(
                x,
                num_samples=self.n_samples,
                base_theta_raw=base,
            ).cpu().numpy()
        self._surfaces = surfaces
        self._mean_surfaces = mean_surfaces
        self._z = -log_prob

    def signal_for_day(self, idx: int) -> SignalOutput:
        if self._surfaces is None or self._mean_surfaces is None or self._z is None:
            raise RuntimeError("Provider not prepared")
        return self._build_output(
            observed=self._surfaces[idx],
            reference=self._mean_surfaces[idx],
            z=float(self._z[idx]),
        )


class ClimatologySignalProvider(SignalProvider):
    name = "climatology"

    def __init__(self, basis: dict[str, np.ndarray], residual_clip: float = 1.0) -> None:
        super().__init__(basis=basis, residual_clip=residual_clip)
        self._surfaces: np.ndarray | None = None

    def prepare(
        self,
        *,
        surfaces: np.ndarray,
        context: np.ndarray,
        theta_raw: np.ndarray,
        model: ConditionalSurfaceFlow | None,
        theta_base_raw: np.ndarray | None = None,
        target_mode: str = "theta_raw",
    ) -> None:
        _ = (context, theta_raw, model, theta_base_raw, target_mode)
        self._surfaces = surfaces

    def signal_for_day(self, idx: int) -> SignalOutput:
        if self._surfaces is None:
            raise RuntimeError("Provider not prepared")
        if idx == 0:
            reference = self._surfaces[0]
        else:
            reference = np.mean(self._surfaces[:idx], axis=0)
        observed = self._surfaces[idx]
        z = float(np.linalg.norm(observed - reference))
        return self._build_output(observed=observed, reference=reference, z=z)


class LastValueSignalProvider(SignalProvider):
    name = "last_value"

    def __init__(self, basis: dict[str, np.ndarray], residual_clip: float = 1.0) -> None:
        super().__init__(basis=basis, residual_clip=residual_clip)
        self._surfaces: np.ndarray | None = None

    def prepare(
        self,
        *,
        surfaces: np.ndarray,
        context: np.ndarray,
        theta_raw: np.ndarray,
        model: ConditionalSurfaceFlow | None,
        theta_base_raw: np.ndarray | None = None,
        target_mode: str = "theta_raw",
    ) -> None:
        _ = (context, theta_raw, model, theta_base_raw, target_mode)
        self._surfaces = surfaces

    def signal_for_day(self, idx: int) -> SignalOutput:
        if self._surfaces is None:
            raise RuntimeError("Provider not prepared")
        reference = self._surfaces[idx - 1] if idx > 0 else self._surfaces[0]
        observed = self._surfaces[idx]
        z = float(np.linalg.norm(observed - reference))
        return self._build_output(observed=observed, reference=reference, z=z)


class LinearContextSignalProvider(SignalProvider):
    name = "linear_context"

    def __init__(
        self,
        basis: dict[str, np.ndarray],
        residual_clip: float = 1.0,
        ridge: float = 1e-3,
        min_fit: int = 10,
    ) -> None:
        super().__init__(basis=basis, residual_clip=residual_clip)
        self.ridge = ridge
        self.min_fit = min_fit
        self._surfaces: np.ndarray | None = None
        self._context: np.ndarray | None = None

    def prepare(
        self,
        *,
        surfaces: np.ndarray,
        context: np.ndarray,
        theta_raw: np.ndarray,
        model: ConditionalSurfaceFlow | None,
        theta_base_raw: np.ndarray | None = None,
        target_mode: str = "theta_raw",
    ) -> None:
        _ = (theta_raw, model, theta_base_raw, target_mode)
        self._surfaces = surfaces
        self._context = context

    def signal_for_day(self, idx: int) -> SignalOutput:
        if self._surfaces is None or self._context is None:
            raise RuntimeError("Provider not prepared")

        observed = self._surfaces[idx]
        if idx < self.min_fit:
            reference = self._surfaces[idx - 1] if idx > 0 else self._surfaces[0]
            z = float(np.linalg.norm(observed - reference))
            return self._build_output(observed=observed, reference=reference, z=z)

        x_train = self._context[:idx]
        y_train = self._surfaces[:idx].reshape(idx, -1)
        x_train_aug = np.column_stack([x_train, np.ones(idx)])
        x_test_aug = np.concatenate([self._context[idx], np.ones(1)])

        xtx = x_train_aug.T @ x_train_aug
        reg = self.ridge * np.eye(xtx.shape[0])
        beta = np.linalg.solve(xtx + reg, x_train_aug.T @ y_train)
        pred = x_test_aug @ beta
        reference = pred.reshape(self._surfaces.shape[1:])

        z = float(np.linalg.norm(observed - reference))
        return self._build_output(observed=observed, reference=reference, z=z)


def make_signal_provider(
    provider_name: str,
    *,
    basis: dict[str, np.ndarray],
    n_samples: int,
    residual_clip: float,
) -> SignalProvider:
    name = provider_name.strip().lower()
    if name == "deep_flow":
        return DeepFlowSignalProvider(basis=basis, n_samples=n_samples, residual_clip=residual_clip)
    if name == "climatology":
        return ClimatologySignalProvider(basis=basis, residual_clip=residual_clip)
    if name == "last_value":
        return LastValueSignalProvider(basis=basis, residual_clip=residual_clip)
    if name == "linear_context":
        return LinearContextSignalProvider(basis=basis, residual_clip=residual_clip)
    raise ValueError(f"Unknown signal provider: {provider_name}")


def available_signal_providers() -> list[str]:
    return ["deep_flow", "climatology", "last_value", "linear_context"]
