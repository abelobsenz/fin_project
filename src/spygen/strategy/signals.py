from __future__ import annotations

import numpy as np


def dislocation_score(log_prob: float) -> float:
    return -float(log_prob)


def project_residual(residual: np.ndarray, basis: dict[str, np.ndarray]) -> dict[str, float]:
    proj: dict[str, float] = {}
    for name, vec in basis.items():
        # L1 normalization keeps projections closer to per-structure premium scale.
        denom = float(np.abs(vec).sum() + 1e-8)
        proj[name] = float(np.sum(residual * vec) / denom)
    return proj
