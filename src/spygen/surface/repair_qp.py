from __future__ import annotations

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
    nonneg_constraints = [x >= 0]
    constraints = mono_constraints + conv_constraints + cal_constraints + nonneg_constraints

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)

    if x.value is None:
        raise RuntimeError("QP repair failed to produce a solution")

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
