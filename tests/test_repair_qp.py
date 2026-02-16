from __future__ import annotations

import numpy as np

from spygen.surface.arb_checks import is_arb_free
from spygen.surface.repair_qp import repair_surface_qp


def test_repair_qp_fixes_static_arbitrage() -> None:
    surface = np.array(
        [
            [0.35, 0.34, 0.36],
            [0.33, 0.32, 0.31],
            [0.34, 0.30, 0.29],
            [0.20, 0.25, 0.26],
        ],
        dtype=float,
    )
    x_grid = np.linspace(-0.3, 0.3, surface.shape[0])
    repaired = repair_surface_qp(surface, x_grid=x_grid, lambda_smooth=1e-3)
    assert is_arb_free(repaired.repaired)
    intrinsic = np.maximum(0.0, 1.0 - np.exp(x_grid))
    assert np.all(repaired.repaired >= intrinsic[:, None] - 1e-8)
    assert np.all(repaired.repaired <= 1.0 + 1e-8)
