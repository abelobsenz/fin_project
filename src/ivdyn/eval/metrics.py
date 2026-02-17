"""Evaluation metrics helpers."""

from __future__ import annotations

import numpy as np


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(pred) - np.asarray(target)
    return float(np.sqrt(np.mean(diff * diff)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(pred) - np.asarray(target)
    return float(np.mean(np.abs(diff)))


def r2(pred: np.ndarray, target: np.ndarray) -> float:
    y = np.asarray(target)
    yhat = np.asarray(pred)
    denom = np.sum((y - y.mean()) ** 2)
    if denom <= 1e-12:
        return float("nan")
    num = np.sum((y - yhat) ** 2)
    return float(1.0 - num / denom)


def directional_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    p = np.asarray(pred)
    t = np.asarray(target)
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.sign(p) == np.sign(t)))


def brier_score(prob: np.ndarray, target: np.ndarray) -> float:
    p = np.clip(np.asarray(prob), 1e-6, 1.0 - 1e-6)
    y = np.asarray(target)
    return float(np.mean((p - y) ** 2))
