from __future__ import annotations

import numpy as np
import torch

from spygen.models.deep_smoothing import DeepSmoothingSurfaceModel


def _make_model(target_mode: str = "delta_theta_raw") -> DeepSmoothingSurfaceModel:
    return DeepSmoothingSurfaceModel(
        context_dim=6,
        nx=9,
        nt=4,
        x_grid=np.linspace(-0.3, 0.3, 9, dtype=np.float32),
        tenors_days=np.array([7, 14, 30, 60], dtype=np.float32),
        hidden_size=24,
        num_layers=2,
        dropout=0.0,
        corr_scale=0.3,
        prior_blend=0.2,
        num_experts=3,
        min_sigma=0.002,
        max_sigma=0.05,
        sample_temperature=1.0,
        target_mode=target_mode,
    )


def test_deep_smoothing_shapes_and_arb_bounds() -> None:
    model = _make_model(target_mode="delta_theta_raw")
    context = torch.randn(5, 6)
    base_theta = torch.randn(5, 36)
    theta_delta = torch.randn(5, 36) * 0.05

    mean_surface = model.conditional_mean_surface(context, base_theta_raw=base_theta)
    assert mean_surface.shape == (5, 9, 4)

    samples = model.sample_surfaces(context, num_samples=7, base_theta_raw=base_theta)
    assert samples.shape == (5, 7, 9, 4)
    assert torch.isfinite(samples).all()
    sample_values = samples.detach()
    assert float(sample_values.min()) >= 0.0
    assert float(sample_values.max()) <= 1.0

    # With positive sampling temperature and heteroskedastic noise, draws should vary.
    assert not torch.allclose(samples[:, 0], samples[:, 1])

    lp = model.log_prob(theta_delta, context=context, base_theta_raw=base_theta)
    assert lp.shape == (5,)
    assert torch.isfinite(lp).all()


def test_deep_smoothing_theta_mode_without_base() -> None:
    model = _make_model(target_mode="theta_raw")
    context = torch.randn(3, 6)
    theta_raw = torch.randn(3, 36) * 0.1

    lp = model.log_prob(theta_raw, context=context)
    assert lp.shape == (3,)
    assert torch.isfinite(lp).all()
