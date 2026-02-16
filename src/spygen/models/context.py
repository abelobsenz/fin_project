from __future__ import annotations

import numpy as np
import pandas as pd


def build_context_features(underlying: pd.DataFrame) -> pd.DataFrame:
    df = underlying.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])  # type: ignore[index]
        df = df.sort_values("date").set_index("date")
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

    returns = df["close"].pct_change()
    rv5 = returns.rolling(5).std() * np.sqrt(252.0)
    rv20 = returns.rolling(20).std() * np.sqrt(252.0)
    rv60_med = rv20.rolling(60, min_periods=20).median()

    context = pd.DataFrame(index=df.index)
    context["prev_return"] = returns.shift(1)
    context["rv5"] = rv5.shift(1)
    context["rv20"] = rv20.shift(1)
    context["rv_ratio"] = context["rv5"] / (context["rv20"].abs() + 1e-8)
    context["trend_20"] = (df["close"] / df["close"].rolling(20).mean() - 1.0).shift(1)
    context["rv_slope"] = (rv5 - rv20).shift(1)
    context["vol_regime"] = (rv20 > rv60_med).astype(float).shift(1)
    context["trend_5_20"] = (
        df["close"].rolling(5).mean() / (df["close"].rolling(20).mean() + 1e-8) - 1.0
    ).shift(1)

    return context.fillna(0.0)


def append_lagged_surface_pca_features(
    context: np.ndarray,
    repaired_surfaces: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    if n_components <= 0:
        return context.astype(np.float32), [], {}

    n = int(repaired_surfaces.shape[0])
    flat = repaired_surfaces.reshape(n, -1)
    lag = np.vstack([flat[0], flat[:-1]])
    lag_mean = lag.mean(axis=0, keepdims=True)
    centered = lag - lag_mean

    # Deterministic PCA via SVD (no sklearn dependency).
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(int(n_components), int(vt.shape[0]))
    comps = vt[:k]
    scores = centered @ comps.T
    scores = scores.astype(np.float32)

    var = singular_values**2
    denom = float(var.sum() + 1e-12)
    explained = (var[:k] / denom).astype(float)
    names = [f"lag_surface_pc{i + 1}" for i in range(k)]
    stats = {
        "lag_surface_pca_components": float(k),
        "lag_surface_pca_explained_total": float(np.sum(explained)),
    }
    for i, value in enumerate(explained, start=1):
        stats[f"lag_surface_pca_explained_pc{i}"] = float(value)

    merged = np.concatenate([context.astype(np.float32), scores], axis=1)
    return merged, names, stats
