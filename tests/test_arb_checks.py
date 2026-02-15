from __future__ import annotations

import numpy as np

from spygen.surface.arb_checks import arb_violation_counts, is_arb_free


def test_arb_violation_counts_detect_issues() -> None:
    surface = np.array(
        [
            [0.4, 0.5],
            [0.41, 0.49],
            [0.39, 0.48],
        ],
        dtype=float,
    )
    counts = arb_violation_counts(surface)
    assert counts["strike_monotonic"] > 0


def test_arb_checks_pass_on_clean_surface() -> None:
    x = np.linspace(0.5, 0.1, 5)
    surface = np.column_stack([x, x + 0.05, x + 0.1])
    assert is_arb_free(surface)
