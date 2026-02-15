from __future__ import annotations

import cvxpy as cp
import numpy as np

MIN_POS = 1e-5


def _project_increment_curve(curve: np.ndarray) -> np.ndarray:
    y = np.asarray(curve, dtype=float)
    n = y.shape[0]
    if n < 3:
        return np.maximum(y, MIN_POS)
    var = cp.Variable(n)
    obj = cp.sum_squares(var - y) + 1e-5 * cp.sum_squares(var[2:] - 2 * var[1:-1] + var[:-2])
    constraints = [var >= 0]
    constraints += [var[i + 1] <= var[i] for i in range(n - 1)]
    constraints += [var[i + 2] - 2 * var[i + 1] + var[i] >= 0 for i in range(n - 2)]
    cp.Problem(cp.Minimize(obj), constraints).solve(solver=cp.OSQP, warm_start=True, verbose=False)
    if var.value is None:
        return np.maximum(y, MIN_POS)
    return np.asarray(var.value, dtype=float)


def surface_to_increment_curves(surface: np.ndarray) -> np.ndarray:
    c = np.asarray(surface, dtype=float)
    increments = np.empty_like(c)
    increments[:, 0] = c[:, 0]
    if c.shape[1] > 1:
        increments[:, 1:] = c[:, 1:] - c[:, :-1]
    increments = np.maximum(increments, MIN_POS)
    projected = np.column_stack(
        [_project_increment_curve(increments[:, j]) for j in range(c.shape[1])]
    )
    return projected


def increment_curves_to_surface(increments: np.ndarray) -> np.ndarray:
    inc = np.asarray(increments, dtype=float)
    return np.cumsum(np.maximum(inc, 0.0), axis=1)


def increment_curves_to_theta(increments: np.ndarray) -> np.ndarray:
    inc = np.asarray(increments, dtype=float)
    nx, nt = inc.shape
    chunks: list[np.ndarray] = []
    for j in range(nt):
        curve = _project_increment_curve(inc[:, j])
        slopes = np.diff(curve)
        tail_value = max(curve[-1], MIN_POS)
        tail_slope_mag = max(-slopes[-1], MIN_POS)
        second = np.maximum(np.diff(slopes), MIN_POS)
        chunk = np.concatenate([[tail_value, tail_slope_mag], second])
        chunks.append(chunk)
    return np.concatenate(chunks)


def theta_to_increment_curves(theta: np.ndarray, nx: int, nt: int) -> np.ndarray:
    flat = np.asarray(theta, dtype=float)
    if flat.size != nx * nt:
        raise ValueError(f"Theta size mismatch: expected {nx * nt}, got {flat.size}")
    inc = np.zeros((nx, nt), dtype=float)
    cursor = 0
    for j in range(nt):
        params = np.maximum(flat[cursor : cursor + nx], MIN_POS)
        cursor += nx
        tail_value = params[0]
        tail_slope_mag = params[1]
        second = params[2:]

        slopes = np.empty(nx - 1, dtype=float)
        slopes[-1] = -tail_slope_mag
        for i in range(nx - 3, -1, -1):
            slopes[i] = slopes[i + 1] - second[i]

        curve = np.empty(nx, dtype=float)
        curve[-1] = tail_value
        for i in range(nx - 2, -1, -1):
            curve[i] = curve[i + 1] - slopes[i]
        inc[:, j] = np.maximum(curve, MIN_POS)
    return inc


def surface_to_theta(surface: np.ndarray) -> np.ndarray:
    inc = surface_to_increment_curves(surface)
    return increment_curves_to_theta(inc)


def theta_to_surface(theta: np.ndarray, nx: int, nt: int) -> np.ndarray:
    inc = theta_to_increment_curves(theta, nx=nx, nt=nt)
    return increment_curves_to_surface(inc)


def softplus_inverse(x: np.ndarray, eps: float = MIN_POS) -> np.ndarray:
    z = np.maximum(np.asarray(x, dtype=float), eps)
    return np.log(np.expm1(z))


def softplus_forward(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(np.asarray(x, dtype=float)))
