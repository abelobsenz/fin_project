from __future__ import annotations

import torch

from spygen.models.flow import ConditionalSurfaceFlow


def test_conditional_mean_surface_shape_for_batched_context() -> None:
    torch.manual_seed(0)
    model = ConditionalSurfaceFlow(
        theta_dim=24,
        context_dim=5,
        nx=4,
        nt=6,
        hidden_features=16,
        num_layers=1,
    )
    context = torch.randn(3, 5)
    mean_surface = model.conditional_mean_surface(context=context, num_samples=7)
    assert tuple(mean_surface.shape) == (3, 4, 6)


def test_sample_theta_raw_handles_flattened_nflows_shape(monkeypatch) -> None:
    torch.manual_seed(1)
    model = ConditionalSurfaceFlow(
        theta_dim=24,
        context_dim=5,
        nx=4,
        nt=6,
        hidden_features=16,
        num_layers=1,
    )

    def fake_sample(*, num_samples: int, context: torch.Tensor) -> torch.Tensor:
        batch = int(context.shape[0])
        return torch.randn(batch * num_samples, model.theta_dim)

    monkeypatch.setattr(model.flow, "sample", fake_sample)
    context = torch.randn(4, 5)
    sampled = model.sample_theta_raw(context=context, num_samples=9)
    assert tuple(sampled.shape) == (4, 9, 24)
