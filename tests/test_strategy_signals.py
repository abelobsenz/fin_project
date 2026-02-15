from __future__ import annotations

import numpy as np

from spygen.strategy.basis import basis_vectors
from spygen.strategy.providers import ClimatologySignalProvider
from spygen.strategy.signals import project_residual


def test_project_residual_linear_scaling() -> None:
    rng = np.random.default_rng(3)
    x_grid = np.linspace(-0.3, 0.3, 21)
    tenors = [7, 14, 30, 60, 90, 180]
    basis = basis_vectors(x_grid=x_grid, tenor_days=tenors)
    residual = rng.normal(0.0, 0.03, size=(len(x_grid), len(tenors)))

    p1 = project_residual(residual, basis)
    p2 = project_residual(2.0 * residual, basis)

    for name in basis:
        assert np.isfinite(p1[name])
        assert np.isfinite(p2[name])
        assert np.isclose(p2[name], 2.0 * p1[name], rtol=1e-6, atol=1e-8)


def test_projection_notional_scale_is_comparable() -> None:
    x_grid = np.linspace(-0.3, 0.3, 21)
    tenors = [7, 14, 30, 60, 90, 180]
    basis = basis_vectors(x_grid=x_grid, tenor_days=tenors)

    magnitudes = []
    for name, vec in basis.items():
        proj = project_residual(vec, basis)
        magnitudes.append(abs(proj[name]))

    assert min(magnitudes) > 0.0
    assert max(magnitudes) / min(magnitudes) < 2.0


def test_provider_residual_clip_caps_tail() -> None:
    x_grid = np.linspace(-0.3, 0.3, 21)
    tenors = [7, 14, 30, 60, 90, 180]
    basis = basis_vectors(x_grid=x_grid, tenor_days=tenors)
    clip = 0.05
    provider = ClimatologySignalProvider(basis=basis, residual_clip=clip)

    surface0 = np.zeros((len(x_grid), len(tenors)), dtype=float)
    surface1 = np.full((len(x_grid), len(tenors)), 0.4, dtype=float)
    surfaces = np.stack([surface0, surface1], axis=0)
    context = np.zeros((2, 3), dtype=np.float32)
    theta = np.zeros((2, len(x_grid) * len(tenors)), dtype=np.float32)
    provider.prepare(surfaces=surfaces, context=context, theta_raw=theta, model=None)

    out = provider.signal_for_day(1)
    assert np.percentile(np.abs(out.residual), 99) <= clip + 1e-9
    for value in out.projections.values():
        assert np.isfinite(value)
