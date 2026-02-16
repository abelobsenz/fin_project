from __future__ import annotations

import warnings
from dataclasses import dataclass

import cvxpy as cp
import numpy as np


@dataclass(slots=True)
class RepairResult:
    repaired: np.ndarray
    duals: dict[str, np.ndarray]
    objective_value: float


def repair_surface_qp(
    raw_surface: np.ndarray,
    x_grid: np.ndarray | None = None,
    lambda_smooth: float = 1e-3,
    data_weight: float = 1.0,
) -> RepairResult:
    raw = np.asarray(raw_surface, dtype=float)
    nx, nt = raw.shape
    x = cp.Variable((nx, nt))

    objective = data_weight * cp.sum_squares(x - raw)
    if nx >= 3:
        objective += lambda_smooth * cp.sum_squares(x[2:, :] - 2 * x[1:-1, :] + x[:-2, :])
    if nt >= 3:
        objective += lambda_smooth * cp.sum_squares(x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2])

    mono_constraints = [x[i + 1, j] - x[i, j] <= 0 for j in range(nt) for i in range(nx - 1)]
    conv_constraints = [
        x[i + 2, j] - 2 * x[i + 1, j] + x[i, j] >= 0 for j in range(nt) for i in range(nx - 2)
    ]
    cal_constraints = [x[i, j + 1] - x[i, j] >= 0 for i in range(nx) for j in range(nt - 1)]
    if x_grid is None:
        intrinsic = np.zeros(nx, dtype=float)
    else:
        x_arr = np.asarray(x_grid, dtype=float)
        if x_arr.shape != (nx,):
            raise ValueError(f"x_grid shape mismatch: expected {(nx,)}, got {x_arr.shape}")
        intrinsic = np.maximum(0.0, 1.0 - np.exp(x_arr))

    bounds_constraints = [x >= intrinsic[:, None], x <= 1.0]
    constraints = mono_constraints + conv_constraints + cal_constraints + bounds_constraints

    problem = cp.Problem(cp.Minimize(objective), constraints)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Solution may be inaccurate.*")
        problem.solve(
            solver=cp.OSQP,
            warm_start=True,
            verbose=False,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=50_000,
            polishing=True,
        )

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
        raise RuntimeError(f"QP repair failed: status={problem.status}")

    duals = {
        "strike_monotonic": np.array([c.dual_value for c in mono_constraints], dtype=float),
        "strike_convex": np.array([c.dual_value for c in conv_constraints], dtype=float),
        "calendar": np.array([c.dual_value for c in cal_constraints], dtype=float),
    }
    return RepairResult(
        repaired=np.array(x.value, dtype=float),
        duals=duals,
        objective_value=float(problem.value),
    )
