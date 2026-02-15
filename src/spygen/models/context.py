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

    context = pd.DataFrame(index=df.index)
    context["prev_return"] = returns.shift(1)
    context["rv5"] = rv5.shift(1)
    context["rv20"] = rv20.shift(1)
    context["rv_ratio"] = context["rv5"] / (context["rv20"].abs() + 1e-8)
    context["trend_20"] = (df["close"] / df["close"].rolling(20).mean() - 1.0).shift(1)

    return context.fillna(0.0)
