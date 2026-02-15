from __future__ import annotations

import numpy as np


def discount_factor(rate: float, t: float) -> float:
    return float(np.exp(-rate * t))


def log_moneyness(strike: np.ndarray, forward: float) -> np.ndarray:
    return np.log(strike / forward)


def normalize_call(call_price: np.ndarray, discount: float, forward: float) -> np.ndarray:
    denom = max(1e-8, discount * forward)
    return call_price / denom


def denormalize_call(call_norm: np.ndarray, discount: float, forward: float) -> np.ndarray:
    return call_norm * discount * forward
