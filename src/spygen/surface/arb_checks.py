from __future__ import annotations

import numpy as np


def strike_monotonic_violations(surface: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    return np.diff(surface, axis=0) > tol


def strike_convexity_violations(surface: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    return surface[2:, :] - 2.0 * surface[1:-1, :] + surface[:-2, :] < -tol


def calendar_violations(surface: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    return np.diff(surface, axis=1) < -tol


def arb_violation_counts(surface: np.ndarray, tol: float = 1e-8) -> dict[str, int]:
    mono = strike_monotonic_violations(surface, tol=tol)
    conv = strike_convexity_violations(surface, tol=tol)
    cal = calendar_violations(surface, tol=tol)
    return {
        "strike_monotonic": int(np.sum(mono)),
        "strike_convex": int(np.sum(conv)),
        "calendar": int(np.sum(cal)),
    }


def is_arb_free(surface: np.ndarray, tol: float = 1e-8) -> bool:
    counts = arb_violation_counts(surface, tol=tol)
    return all(v == 0 for v in counts.values())
