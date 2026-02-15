from __future__ import annotations

import numpy as np

from spygen.surface.arb_checks import is_arb_free
from spygen.surface.representation import (
    softplus_forward,
    softplus_inverse,
    surface_to_theta,
    theta_to_surface,
)


def test_theta_surface_roundtrip_stability() -> None:
    rng = np.random.default_rng(7)
    nx, nt = 21, 6
    theta = rng.uniform(0.01, 0.2, size=nx * nt)

    surface = theta_to_surface(theta, nx=nx, nt=nt)
    theta_roundtrip = surface_to_theta(surface)
    surface_roundtrip = theta_to_surface(theta_roundtrip, nx=nx, nt=nt)

    assert np.all(np.isfinite(surface_roundtrip))
    assert np.max(np.abs(surface_roundtrip - surface)) < 1e-4
    assert is_arb_free(surface_roundtrip, tol=1e-6)


def test_softplus_inverse_roundtrip() -> None:
    rng = np.random.default_rng(11)
    values = rng.uniform(1e-4, 2.0, size=128)
    raw = softplus_inverse(values)
    restored = softplus_forward(raw)

    assert np.allclose(restored, values, atol=1e-6, rtol=1e-6)
