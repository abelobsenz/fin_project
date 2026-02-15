from __future__ import annotations

import numpy as np


def sharpe_ratio(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    vol = float(r.std(ddof=1))
    if vol <= 1e-12:
        return 0.0
    return float(np.sqrt(252.0) * r.mean() / vol)


def max_drawdown(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    return float(dd.min())


def turnover_ratio(turnover: np.ndarray, equity: np.ndarray) -> float:
    t = np.asarray(turnover, dtype=float)
    e = np.asarray(equity, dtype=float)
    avg_eq = float(np.mean(np.maximum(np.abs(e), 1.0)))
    return float(t.sum() / avg_eq)
