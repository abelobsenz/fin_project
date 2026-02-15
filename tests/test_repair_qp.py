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
    repaired = repair_surface_qp(surface, lambda_smooth=1e-3)
    assert is_arb_free(repaired.repaired)
