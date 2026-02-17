"""Streamlit dashboard for model performance evidence."""

from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency guard
    px = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]
    PLOTLY_AVAILABLE = False


def _inject_style() -> None:
    st.markdown(
        """
<style>
:root {
  --bg0: #f2f7f4;
  --bg1: #dbe8df;
  --panel: #f9fbfa;
  --ink: #16231d;
  --accent: #0f766e;
  --accent2: #6b8f2a;
  --muted: #5c6b63;
}

.stApp {
  background:
    radial-gradient(1200px 500px at 0% -10%, #dff1e5 0%, rgba(223,241,229,0.2) 60%, transparent 100%),
    radial-gradient(1000px 400px at 100% 0%, #e2f0f2 0%, rgba(226,240,242,0.1) 55%, transparent 100%),
    linear-gradient(180deg, var(--bg0) 0%, #eff5f1 100%);
  color: var(--ink);
}

h1, h2, h3 {
  color: var(--ink);
  letter-spacing: -0.02em;
}

.block-note {
  border-left: 4px solid var(--accent);
  background: #f6fbf8;
  border-radius: 10px;
  padding: 10px 12px;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _resolve_run_dir() -> Path:
    default_base = Path("outputs/runs")
    latest_file = default_base / "latest.txt"
    env_default = os.environ.get("IVDYN_DEFAULT_RUN_DIR", "").strip()
    if env_default:
        default = env_default
    else:
        default = latest_file.read_text(encoding="utf-8").strip() if latest_file.exists() else ""

    with st.sidebar:
        st.header("Run Selection")
        run_dir_str = st.text_input("Run directory", value=default)
        dataset_str = st.text_input("Dataset path", value="outputs/datasets/dataset/dataset.npz")
        st.session_state["dataset_path"] = dataset_str

    return Path(run_dir_str)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_surface_eval(eval_dir: Path) -> dict[str, np.ndarray] | None:
    p = eval_dir / "surface_predictions.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _format_num(x: float | int | None, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except Exception:
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{digits}f}"


def _artifact_signature(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    sig: list[tuple[str, int, int]] = []
    for p in paths:
        if p.exists():
            s = p.stat()
            sig.append((str(p), int(s.st_mtime_ns), int(s.st_size)))
        else:
            sig.append((str(p), -1, -1))
    return tuple(sig)


def _compute_backtest_stats(daily: pd.DataFrame, trades: pd.DataFrame, bt_summary: dict) -> dict[str, float]:
    stats: dict[str, float] = {}
    if daily.empty:
        return stats

    pnl = pd.to_numeric(daily["pnl"], errors="coerce").fillna(0.0)
    if "equity" in daily.columns:
        equity = pd.to_numeric(daily["equity"], errors="coerce").fillna(0.0)
    else:
        equity = pnl.cumsum()
    peak = equity.cummax().replace(0.0, np.nan)
    drawdown = (equity - peak) / peak

    total_pnl = float(bt_summary.get("total_pnl", pnl.sum()))
    trades_n = float(bt_summary.get("trades", float(daily.get("trades", pd.Series(dtype=float)).sum())))
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()

    stats["total_pnl"] = total_pnl
    stats["daily_sharpe"] = float(bt_summary.get("daily_sharpe", np.nan))
    stats["max_drawdown"] = float(bt_summary.get("max_drawdown", drawdown.min() if len(drawdown) else np.nan))
    stats["win_rate"] = float(bt_summary.get("win_rate", (pnl > 0).mean()))
    stats["avg_daily_pnl"] = float(bt_summary.get("avg_daily_pnl", pnl.mean()))
    stats["pnl_p95"] = float(pnl.quantile(0.95))
    stats["pnl_p05"] = float(pnl.quantile(0.05))
    stats["best_day"] = float(pnl.max())
    stats["worst_day"] = float(pnl.min())
    stats["trades"] = trades_n
    stats["avg_trades_per_day"] = float(trades_n / max(1, len(daily)))
    stats["expectancy_per_trade"] = float(total_pnl / max(1.0, trades_n))

    if losses < 0:
        stats["profit_factor"] = float(wins / abs(losses))
    else:
        stats["profit_factor"] = float("nan")

    if not trades.empty and "pnl" in trades.columns:
        t_pnl = pd.to_numeric(trades["pnl"], errors="coerce").dropna()
        if not t_pnl.empty:
            stats["median_trade_pnl"] = float(t_pnl.median())
            stats["trade_win_rate"] = float((t_pnl > 0).mean())

    return stats


def _compute_prediction_stats(pred_test: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if pred_test.empty:
        return out

    err = pd.to_numeric(pred_test["pred_price_norm"], errors="coerce") - pd.to_numeric(
        pred_test["target_price_norm"], errors="coerce"
    )
    err = err.dropna()
    if err.empty:
        return out

    abs_err = err.abs()
    out["bias"] = float(err.mean())
    out["median_abs_error"] = float(abs_err.median())
    out["p90_abs_error"] = float(abs_err.quantile(0.90))
    out["p95_abs_error"] = float(abs_err.quantile(0.95))
    out["within_1pct_abs"] = float((abs_err <= 0.01).mean())
    out["within_2pct_abs"] = float((abs_err <= 0.02).mean())
    return out


def _surface_slice_df(
    obs_surface: np.ndarray,
    pred_surface: np.ndarray,
    x_grid: np.ndarray,
    tenor_days: np.ndarray,
    mode: str,
    selected: list[float],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if mode == "tenor":
        for tenor in selected:
            j = int(np.argmin(np.abs(tenor_days - tenor)))
            t_val = int(tenor_days[j])
            rows.append(
                pd.DataFrame(
                    {
                        "axis": x_grid,
                        "iv": obs_surface[:, j],
                        "series": f"{t_val}d",
                        "model": "Observed",
                    }
                )
            )
            rows.append(
                pd.DataFrame(
                    {
                        "axis": x_grid,
                        "iv": pred_surface[:, j],
                        "series": f"{t_val}d",
                        "model": "Predicted",
                    }
                )
            )
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["axis", "iv", "series", "model"])

    for x_val in selected:
        i = int(np.argmin(np.abs(x_grid - x_val)))
        x_pick = float(x_grid[i])
        rows.append(
            pd.DataFrame(
                {
                    "axis": tenor_days,
                    "iv": obs_surface[i, :],
                    "series": f"x={x_pick:.2f}",
                    "model": "Observed",
                }
            )
        )
        rows.append(
            pd.DataFrame(
                {
                    "axis": tenor_days,
                    "iv": pred_surface[i, :],
                    "series": f"x={x_pick:.2f}",
                    "model": "Predicted",
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["axis", "iv", "series", "model"])


def _surface_3d_figure(obs_surface: np.ndarray, pred_surface: np.ndarray, x_grid: np.ndarray, tenor_days: np.ndarray):
    if not PLOTLY_AVAILABLE:
        return None

    t_mesh, x_mesh = np.meshgrid(tenor_days.astype(float), x_grid.astype(float))

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=t_mesh,
            y=x_mesh,
            z=obs_surface,
            name="Observed",
            showscale=False,
            colorscale="Tealgrn",
            opacity=0.95,
        )
    )
    fig.add_trace(
        go.Surface(
            x=t_mesh,
            y=x_mesh,
            z=pred_surface,
            name="Predicted",
            showscale=False,
            colorscale="Solar",
            opacity=0.55,
            contours={"z": {"show": True, "usecolormap": False, "color": "#111111", "width": 1}},
        )
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        height=500,
        scene={
            "xaxis_title": "Tenor (days)",
            "yaxis_title": "Log-moneyness",
            "zaxis_title": "IV",
            "camera": {"eye": {"x": 1.4, "y": 1.4, "z": 0.9}},
        },
        legend={"orientation": "h", "x": 0.02, "y": 1.02},
    )
    return fig


def _slice_overlay_chart(df: pd.DataFrame, mode: str):
    if PLOTLY_AVAILABLE:
        x_title = "Log-moneyness x=ln(K/S)" if mode == "tenor" else "Tenor (days)"
        fig = px.line(
            df,
            x="axis",
            y="iv",
            color="series",
            line_dash="model",
            markers=True,
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.update_layout(margin={"l": 0, "r": 0, "t": 20, "b": 0}, height=360, xaxis_title=x_title, yaxis_title="Implied Volatility")
        return fig

    x_title = "Log-moneyness x=ln(K/S)" if mode == "tenor" else "Tenor (days)"
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("axis:Q", title=x_title),
            y=alt.Y("iv:Q", title="Implied Volatility"),
            color=alt.Color("series:N"),
            strokeDash=alt.StrokeDash("model:N"),
            tooltip=["model", "series", "axis", "iv"],
        )
        .properties(height=360)
    )


def _scalar_items(data: dict | None) -> list[tuple[str, float | int | str | bool]]:
    if not data:
        return []
    out: list[tuple[str, float | int | str | bool]] = []
    for k, v in data.items():
        if isinstance(v, (str, int, float, bool)):
            out.append((str(k), v))
    return out


def _render_pdf_table(ax, title: str, df: pd.DataFrame, max_rows: int = 22) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=11, loc="left")
    if df.empty:
        ax.text(0.01, 0.95, "No data available.", va="top", ha="left", fontsize=10)
        return

    show = df.head(max_rows).copy()
    for c in show.columns:
        if pd.api.types.is_numeric_dtype(show[c]):
            show[c] = show[c].map(lambda x: _format_num(x, 6))
        else:
            show[c] = show[c].astype(str)
    table = ax.table(
        cellText=show.to_numpy(),
        colLabels=[str(c) for c in show.columns],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.2)
    if len(df) > max_rows:
        ax.text(
            0.01,
            0.02,
            f"Showing first {max_rows} of {len(df)} rows.",
            va="bottom",
            ha="left",
            fontsize=8,
            color="#374151",
        )


def _build_pdf_report_bytes(run_dir: Path) -> bytes:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for PDF export") from exc

    eval_dir = run_dir / "evaluation"
    bt_dir = run_dir / "backtest"

    train_summary = _read_json(run_dir / "train_summary.json")
    metrics = _read_json(eval_dir / "metrics.json")
    bt_summary = _read_json(bt_dir / "summary.json")

    cfg = _read_json(run_dir / "train_config.json")
    hist_path = run_dir / "train_history.csv"
    pred_path = eval_dir / "contract_predictions.parquet"
    daily_path = bt_dir / "daily.parquet"
    trades_path = bt_dir / "trades.parquet"
    noarb_path = eval_dir / "noarb_test_dates.parquet"
    surf_path = eval_dir / "surface_predictions.npz"

    hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    daily = pd.read_parquet(daily_path) if daily_path.exists() else pd.DataFrame()
    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
    pred_df = pd.read_parquet(pred_path) if pred_path.exists() else pd.DataFrame()
    noarb = pd.read_parquet(noarb_path) if noarb_path.exists() else pd.DataFrame()
    surf = np.load(surf_path, allow_pickle=False) if surf_path.exists() else None

    if not daily.empty:
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        daily["pnl"] = pd.to_numeric(daily["pnl"], errors="coerce").fillna(0.0)
        if "equity" in daily.columns:
            daily["equity"] = pd.to_numeric(daily["equity"], errors="coerce").fillna(0.0)
        else:
            daily["equity"] = daily["pnl"].cumsum()
        if "options_pnl" in daily.columns:
            daily["options_pnl"] = pd.to_numeric(daily["options_pnl"], errors="coerce").fillna(0.0)
        else:
            daily["options_pnl"] = 0.0
        if "hedge_pnl" in daily.columns:
            daily["hedge_pnl"] = pd.to_numeric(daily["hedge_pnl"], errors="coerce").fillna(0.0)
        else:
            daily["hedge_pnl"] = 0.0
        for c in ("net_option_delta_shares", "hedge_shares", "post_hedge_delta_shares"):
            if c in daily.columns:
                daily[c] = pd.to_numeric(daily[c], errors="coerce").fillna(0.0)
            else:
                daily[c] = 0.0
        abs_net_delta = daily["net_option_delta_shares"].abs()
        abs_post_hedge_delta = daily["post_hedge_delta_shares"].abs()
        denom = abs_net_delta.replace(0.0, np.nan)
        daily["hedge_risk_reduction_pct"] = (
            ((abs_net_delta - abs_post_hedge_delta) / denom) * 100.0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        peak = daily["equity"].cummax().replace(0.0, np.nan)
        daily["drawdown"] = (daily["equity"] - peak) / peak
    bt_stats = _compute_backtest_stats(daily, trades, bt_summary) if not daily.empty else {}

    pred_test = pd.DataFrame()
    pred_stats: dict[str, float] = {}
    bucket_stats = pd.DataFrame()
    worst_errors = pd.DataFrame()
    if not pred_df.empty:
        pred_test = pred_df[pred_df["split"] == "test"].copy() if "split" in pred_df.columns else pred_df.copy()
        if not pred_test.empty:
            pred_test["error"] = pd.to_numeric(pred_test["pred_price_norm"], errors="coerce") - pd.to_numeric(
                pred_test["target_price_norm"], errors="coerce"
            )
            pred_test["abs_error"] = pred_test["error"].abs()
            pred_stats = _compute_prediction_stats(pred_test)
            if "dte" in pred_test.columns:
                pred_test["dte_bucket"] = pd.cut(
                    pred_test["dte"],
                    bins=[0, 14, 30, 60, 90, 180, 3650],
                    labels=["<=14", "15-30", "31-60", "61-90", "91-180", ">180"],
                    include_lowest=True,
                )
                bucket_stats = (
                    pred_test.groupby(["call_put", "dte_bucket"], dropna=False, observed=False)
                    .agg(
                        n=("error", "count"),
                        rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan),
                        mae=("abs_error", "mean"),
                        bias=("error", "mean"),
                    )
                    .reset_index()
                    .sort_values(["call_put", "dte_bucket"])
                )
            keep_cols = [
                c
                for c in [
                    "date",
                    "symbol",
                    "call_put",
                    "dte",
                    "target_price_norm",
                    "pred_price_norm",
                    "error",
                    "abs_error",
                ]
                if c in pred_test.columns
            ]
            if "abs_error" in pred_test.columns and keep_cols:
                worst_errors = pred_test.nlargest(25, "abs_error")[keep_cols]

    obs = pred_surface = x_grid = tenor_days = surf_dates = None
    fit_df = pd.DataFrame()
    if surf is not None:
        obs = surf["iv_surface_obs"].astype(np.float32)
        pred_surface = surf["iv_surface_pred"].astype(np.float32)
        x_grid = surf["x_grid"].astype(np.float32)
        tenor_days = surf["tenor_days"].astype(np.int32)
        surf_dates = pd.to_datetime(surf["dates"].astype(str))
        fit_df = pd.DataFrame(
            {
                "date": surf_dates,
                "surface_rmse": np.sqrt(np.mean((pred_surface - obs) ** 2, axis=(1, 2))),
                "surface_mae": np.mean(np.abs(pred_surface - obs), axis=(1, 2)),
            }
        )

    daily_desc = (
        daily["pnl"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_frame("daily_pnl").reset_index()
        if not daily.empty
        else pd.DataFrame()
    )
    grouped_trades = pd.DataFrame()
    if not trades.empty and {"side", "call_put", "pnl"}.issubset(trades.columns):
        t = trades.copy()
        t["pnl"] = pd.to_numeric(t["pnl"], errors="coerce")
        agg: dict[str, tuple[str, object]] = {
            "trades": ("pnl", "count"),
            "pnl_sum": ("pnl", "sum"),
            "pnl_mean": ("pnl", "mean"),
            "win_rate": ("pnl", lambda x: float((x > 0).mean()) if len(x) else np.nan),
        }
        if "signal" in t.columns:
            agg["signal_median"] = ("signal", "median")
        if "fill_prob" in t.columns:
            agg["fill_prob_mean"] = ("fill_prob", "mean")
        grouped_trades = (
            t.groupby(["side", "call_put"], dropna=False)
            .agg(**agg)
            .reset_index()
            .sort_values("pnl_sum", ascending=False)
        )

    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1: executive summary and key metrics
        kpi_rows = [
            ("Prediction", "Price RMSE", metrics.get("price_rmse")),
            ("Prediction", "Price MAE", metrics.get("price_mae")),
            ("Prediction", "Next-Day Price R2", metrics.get("next_price_r2", metrics.get("price_r2"))),
            ("Prediction", "Exec Brier", metrics.get("exec_brier")),
            ("Surface", "Surface RMSE", metrics.get("surface_iv_rmse")),
            ("Surface", "Surface MAE", metrics.get("surface_iv_mae")),
            ("Surface", "Calendar Viol Pred", metrics.get("calendar_violation_pred_mean")),
            ("Surface", "Butterfly Viol Pred", metrics.get("butterfly_violation_pred_mean")),
            ("Backtest", "Total PnL", bt_stats.get("total_pnl", bt_summary.get("total_pnl"))),
            ("Backtest", "Daily Sharpe", bt_stats.get("daily_sharpe", bt_summary.get("daily_sharpe"))),
            ("Backtest", "Max Drawdown", bt_stats.get("max_drawdown", bt_summary.get("max_drawdown"))),
            ("Backtest", "Profit Factor", bt_stats.get("profit_factor")),
            ("Backtest", "Trades", bt_stats.get("trades", bt_summary.get("trades"))),
            ("Backtest", "Options PnL", float(daily["options_pnl"].sum()) if not daily.empty else np.nan),
            ("Backtest", "Hedge PnL", float(daily["hedge_pnl"].sum()) if not daily.empty else np.nan),
            (
                "Backtest",
                "Avg Hedge Risk Reduction (%)",
                float(daily["hedge_risk_reduction_pct"].mean()) if not daily.empty else np.nan,
            ),
            ("Backtest", "Avg |Hedge Shares|", float(daily["hedge_shares"].abs().mean()) if not daily.empty else np.nan),
            ("Prediction", "Median |Error|", pred_stats.get("median_abs_error")),
            ("Prediction", "P95 |Error|", pred_stats.get("p95_abs_error")),
        ]
        kpi_df = pd.DataFrame(kpi_rows, columns=["section", "metric", "value"])

        run_lines = [
            f"Run: {run_dir.name}",
            f"Generated (UTC): {pd.Timestamp.utcnow().isoformat()}",
            f"Training model path: {train_summary.get('model_path', 'n/a')}",
            f"Dataset path: {train_summary.get('dataset_path', 'n/a')}",
            f"Training device: {train_summary.get('device', 'n/a')}",
        ]
        if not daily.empty:
            run_lines.append(
                f"Backtest window: {daily['date'].min().date()} to {daily['date'].max().date()} "
                f"({len(daily)} days)"
            )
        if not pred_test.empty:
            run_lines.append(f"Prediction test contracts: {len(pred_test)}")
        if surf_dates is not None and len(surf_dates):
            run_lines.append(
                f"Surface eval window: {surf_dates.min().date()} to {surf_dates.max().date()} ({len(surf_dates)} dates)"
            )

        fig = plt.figure(figsize=(11, 8.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
        left = fig.add_subplot(gs[0, 0])
        right = fig.add_subplot(gs[0, 1])
        left.axis("off")
        left.set_title("Run Summary", fontsize=12, loc="left")
        left.text(0.01, 0.98, "\n".join(run_lines), va="top", ha="left", fontsize=10)
        fig.suptitle(f"IV Dynamics Report: {run_dir.name}", fontsize=16, y=0.99)
        _render_pdf_table(right, "Key Metrics (UI Summary + Tabs)", kpi_df, max_rows=24)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: backtest charts
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
        fig.suptitle("Backtest & PnL Evidence", fontsize=14)
        if daily.empty:
            for ax in axes.ravel():
                ax.axis("off")
                ax.text(0.5, 0.5, "No backtest daily artifact.", ha="center", va="center", fontsize=11)
        else:
            axes[0, 0].plot(daily["date"], daily["equity"], color="#0f766e")
            axes[0, 0].set_title("Equity Curve")
            axes[0, 0].set_ylabel("Equity")

            axes[0, 1].fill_between(daily["date"], daily["drawdown"], 0.0, color="#b91c1c", alpha=0.35)
            axes[0, 1].set_title("Drawdown")
            axes[0, 1].set_ylabel("Drawdown")

            colors = np.where(daily["pnl"] >= 0.0, "#0f766e", "#b91c1c")
            axes[1, 0].bar(daily["date"], daily["pnl"], color=colors, width=1.6)
            axes[1, 0].set_title("Daily PnL")
            axes[1, 0].set_ylabel("PnL")

            axes[1, 1].hist(daily["pnl"].to_numpy(), bins=40, color="#334155", alpha=0.85)
            axes[1, 1].set_title("PnL Distribution")
            axes[1, 1].set_xlabel("Daily PnL")
            for ax in axes.ravel():
                ax.tick_params(axis="x", labelrotation=30)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: hedge diagnostics
        if not daily.empty:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
            fig.suptitle("Hedge Diagnostics", fontsize=14)

            hedge_colors = np.where(daily["hedge_shares"] >= 0.0, "#0f766e", "#b91c1c")
            axes[0, 0].bar(daily["date"], daily["hedge_shares"], color=hedge_colors, width=1.6)
            axes[0, 0].set_title("Hedge Amount Per Day (Shares)")
            axes[0, 0].set_ylabel("Shares")

            axes[0, 1].plot(daily["date"], daily["net_option_delta_shares"], label="Net option delta", color="#0f766e")
            axes[0, 1].plot(daily["date"], daily["post_hedge_delta_shares"], label="Post-hedge delta", color="#111827")
            axes[0, 1].set_title("Delta Exposure (Shares)")
            axes[0, 1].set_ylabel("Shares")
            axes[0, 1].legend(fontsize=8)

            axes[1, 0].bar(daily["date"], daily["hedge_pnl"], color="#d97706", alpha=0.7, width=1.6, label="Hedge PnL")
            ax2 = axes[1, 0].twinx()
            ax2.plot(daily["date"], daily["hedge_risk_reduction_pct"], color="#111827", linewidth=1.5, label="Risk reduction %")
            axes[1, 0].set_title("Hedge Contribution to PnL and Risk Reduction")
            axes[1, 0].set_ylabel("Hedge PnL")
            ax2.set_ylabel("Risk Reduction (%)")
            h1, l1 = axes[1, 0].get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            axes[1, 0].legend(h1 + h2, l1 + l2, fontsize=8, loc="best")

            axes[1, 1].hist(daily["hedge_risk_reduction_pct"].to_numpy(), bins=40, color="#334155", alpha=0.85)
            axes[1, 1].set_title("Risk Reduction Distribution")
            axes[1, 1].set_xlabel("Risk Reduction (%)")

            for ax in axes.ravel():
                ax.tick_params(axis="x", labelrotation=30)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Page 4: backtest numeric evidence
        fig = plt.figure(figsize=(11, 8.5))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.9])
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[2, 0])
        _render_pdf_table(ax0, "Daily PnL Summary", daily_desc)
        _render_pdf_table(ax1, "Trade Group Diagnostics (side x call_put)", grouped_trades)
        hedge_summary_rows: list[dict[str, object]] = []
        if not daily.empty:
            total_pnl = float(daily["pnl"].sum())
            hedge_total = float(daily["hedge_pnl"].sum())
            hedge_summary_rows = [
                {"metric": "options_pnl_total", "value": float(daily["options_pnl"].sum())},
                {"metric": "hedge_pnl_total", "value": hedge_total},
                {
                    "metric": "hedge_pnl_pct_of_total",
                    "value": (100.0 * hedge_total / total_pnl) if abs(total_pnl) > 1e-12 else np.nan,
                },
                {"metric": "avg_abs_hedge_shares", "value": float(daily["hedge_shares"].abs().mean())},
                {"metric": "avg_abs_net_option_delta_shares", "value": float(daily["net_option_delta_shares"].abs().mean())},
                {"metric": "avg_abs_post_hedge_delta_shares", "value": float(daily["post_hedge_delta_shares"].abs().mean())},
                {"metric": "avg_hedge_risk_reduction_pct", "value": float(daily["hedge_risk_reduction_pct"].mean())},
                {"metric": "median_hedge_risk_reduction_pct", "value": float(daily["hedge_risk_reduction_pct"].median())},
            ]
        _render_pdf_table(ax2, "Hedge Summary", pd.DataFrame(hedge_summary_rows))
        fig.suptitle("Backtest Numeric Evidence", fontsize=14)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 5: surface overlays (heatmaps + slices)
        if obs is not None and pred_surface is not None and x_grid is not None and tenor_days is not None and surf_dates is not None:
            i = len(obs) - 1
            err = pred_surface[i] - obs[i]
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
            fig.suptitle(f"Surface Overlays ({surf_dates[i].date()})", fontsize=14)

            im0 = axes[0, 0].imshow(obs[i], aspect="auto", origin="lower")
            axes[0, 0].set_title("Observed Surface")
            axes[0, 0].set_xlabel("Tenor idx")
            axes[0, 0].set_ylabel("Moneyness idx")
            plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)

            im1 = axes[0, 1].imshow(pred_surface[i], aspect="auto", origin="lower")
            axes[0, 1].set_title("Predicted Surface")
            axes[0, 1].set_xlabel("Tenor idx")
            axes[0, 1].set_ylabel("Moneyness idx")
            plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)

            im2 = axes[1, 0].imshow(err, aspect="auto", origin="lower")
            axes[1, 0].set_title("Surface Error (Pred - Obs)")
            axes[1, 0].set_xlabel("Tenor idx")
            axes[1, 0].set_ylabel("Moneyness idx")
            plt.colorbar(im2, ax=axes[1, 0], shrink=0.8)

            tenor_targets = [t for t in (30, 90) if len(tenor_days)]
            for t in tenor_targets:
                j = int(np.argmin(np.abs(tenor_days - t)))
                axes[1, 1].plot(x_grid, obs[i, :, j], label=f"Obs {int(tenor_days[j])}d")
                axes[1, 1].plot(x_grid, pred_surface[i, :, j], linestyle="--", label=f"Pred {int(tenor_days[j])}d")
            axes[1, 1].set_title("Slice Overlay by Tenor")
            axes[1, 1].set_xlabel("Log-moneyness")
            axes[1, 1].set_ylabel("IV")
            axes[1, 1].legend(fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
            for x_target in (-0.10, 0.0, 0.10):
                ix = int(np.argmin(np.abs(x_grid - x_target)))
                x_pick = float(x_grid[ix])
                axes[0].plot(tenor_days, obs[i, ix, :], label=f"Obs x={x_pick:.2f}")
                axes[0].plot(tenor_days, pred_surface[i, ix, :], linestyle="--", label=f"Pred x={x_pick:.2f}")
            axes[0].set_title("Slice Overlay by Moneyness")
            axes[0].set_xlabel("Tenor (days)")
            axes[0].set_ylabel("IV")
            axes[0].legend(fontsize=8)

            axes[1].scatter(obs[i].ravel(), pred_surface[i].ravel(), s=16, alpha=0.45)
            lim = float(max(np.max(obs[i]), np.max(pred_surface[i])))
            axes[1].plot([0.0, lim], [0.0, lim], color="black", linewidth=1.0)
            axes[1].set_title("Surface Point Fit: Pred vs Obs")
            axes[1].set_xlabel("Observed IV")
            axes[1].set_ylabel("Predicted IV")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Page 6: prediction error charts
        if not pred_test.empty:
            sample = pred_test.sample(min(len(pred_test), 8000), random_state=7)
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
            axes[0].scatter(sample["target_price_norm"], sample["pred_price_norm"], s=8, alpha=0.3)
            top = float(max(sample["target_price_norm"].max(), sample["pred_price_norm"].max()))
            axes[0].plot([0.0, top], [0.0, top], color="black", linewidth=1)
            axes[0].set_title("Predicted vs Observed Price")
            axes[0].set_xlabel("Observed")
            axes[0].set_ylabel("Predicted")

            axes[1].hist(pred_test["error"].dropna().to_numpy(), bins=70, color="#334155", alpha=0.85)
            axes[1].set_title("Error Distribution")
            axes[1].set_xlabel("Prediction Error")

            if not bucket_stats.empty:
                b = bucket_stats.copy()
                b["bucket"] = b["call_put"].astype(str) + ":" + b["dte_bucket"].astype(str)
                axes[2].barh(b["bucket"], b["rmse"], color="#0f766e", alpha=0.8)
                axes[2].set_title("RMSE by CP x DTE Bucket")
                axes[2].set_xlabel("RMSE")
            else:
                axes[2].axis("off")
                axes[2].text(0.5, 0.5, "No bucket diagnostics.", ha="center", va="center")
            fig.suptitle("Prediction Error Diagnostics", fontsize=14)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(11, 8.5))
            gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 0.9])
            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[1, 0])
            _render_pdf_table(ax0, "Prediction Bucket Stats", bucket_stats)
            _render_pdf_table(ax1, "Worst Contract Errors", worst_errors)
            fig.suptitle("Prediction Numeric Evidence", fontsize=14)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Page 7: training diagnostics charts
        if not hist.empty and "stage" in hist.columns and "epoch" in hist.columns:
            hist = hist.copy()
            hist["epoch"] = pd.to_numeric(hist["epoch"], errors="coerce")
            plot_cols = [
                c
                for c in [
                    "loss",
                    "recon_loss",
                    "val_recon_loss",
                    "dyn_loss",
                    "price_loss",
                    "exec_loss",
                    "val_dyn_loss",
                    "val_price_loss",
                    "val_exec_loss",
                    "kl_loss",
                    "calendar_loss",
                ]
                if c in hist.columns
            ]
            n_panels = min(6, len(plot_cols))
            if n_panels > 0:
                fig, axes = plt.subplots(2, 3, figsize=(11, 8.2))
                axes_flat = axes.ravel()
                for i in range(6):
                    ax = axes_flat[i]
                    if i >= n_panels:
                        ax.axis("off")
                        continue
                    col = plot_cols[i]
                    for stage in sorted(hist["stage"].dropna().astype(str).unique().tolist()):
                        sub = hist[hist["stage"].astype(str) == stage]
                        y = pd.to_numeric(sub[col], errors="coerce")
                        if y.notna().any():
                            ax.plot(sub["epoch"], y, label=stage)
                    ax.set_title(col)
                    ax.set_xlabel("Epoch")
                axes_flat[0].legend(fontsize=8)
                fig.suptitle("Training Diagnostics", fontsize=14)
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            by_stage = hist.groupby("stage", dropna=False).agg(
                epochs=("epoch", "count"),
                final_loss=("loss", "last") if "loss" in hist.columns else ("epoch", "count"),
                min_loss=("loss", "min") if "loss" in hist.columns else ("epoch", "count"),
                final_val_recon=("val_recon_loss", "last") if "val_recon_loss" in hist.columns else ("epoch", "count"),
            )
            fig = plt.figure(figsize=(11, 8.5))
            gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.1])
            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[1, 0])
            _render_pdf_table(ax0, "Training Stage Summary", by_stage.reset_index())
            ax1.axis("off")
            ax1.set_title("Training Config", fontsize=11, loc="left")
            cfg_text = json.dumps(cfg, indent=2) if cfg else "{}"
            ax1.text(0.01, 0.98, cfg_text, family="monospace", fontsize=8.2, va="top", ha="left")
            fig.suptitle("Training Numeric Evidence", fontsize=14)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Page 8: fits diagnostics
        if not fit_df.empty or not noarb.empty:
            fig, axes = plt.subplots(2, 1, figsize=(11, 8.2))
            fig.suptitle("Fit Diagnostics", fontsize=14)

            if not fit_df.empty:
                axes[0].plot(fit_df["date"], fit_df["surface_rmse"], label="surface_rmse", color="#0f766e")
                axes[0].plot(fit_df["date"], fit_df["surface_mae"], label="surface_mae", color="#7c3aed")
                axes[0].set_title("Surface Fit Over Time")
                axes[0].set_ylabel("Error")
                axes[0].legend()
                axes[0].tick_params(axis="x", labelrotation=30)
            else:
                axes[0].axis("off")
                axes[0].text(0.5, 0.5, "No surface fit artifact.", ha="center", va="center")

            if not noarb.empty and {"date", "calendar_obs", "calendar_pred", "butterfly_obs", "butterfly_pred"}.issubset(noarb.columns):
                nn = noarb.copy()
                nn["date"] = pd.to_datetime(nn["date"])
                axes[1].plot(nn["date"], nn["calendar_obs"], label="calendar_obs")
                axes[1].plot(nn["date"], nn["calendar_pred"], label="calendar_pred")
                axes[1].plot(nn["date"], nn["butterfly_obs"], label="butterfly_obs")
                axes[1].plot(nn["date"], nn["butterfly_pred"], label="butterfly_pred")
                axes[1].set_title("No-Arbitrage Diagnostics")
                axes[1].set_ylabel("Violation Rate")
                axes[1].legend(fontsize=8, ncol=2)
                axes[1].tick_params(axis="x", labelrotation=30)
            else:
                axes[1].axis("off")
                axes[1].text(0.5, 0.5, "No no-arbitrage diagnostics.", ha="center", va="center")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Page 9: raw report payloads
        payload = {
            "train_summary": train_summary,
            "train_config": cfg,
            "eval_metrics": metrics,
            "backtest_summary": bt_summary,
            "backtest_stats": bt_stats,
            "prediction_stats": pred_stats,
        }
        payload_rows: list[dict[str, object]] = []
        for section, values in payload.items():
            for k, v in _scalar_items(values if isinstance(values, dict) else {}):
                payload_rows.append({"section": section, "metric": k, "value": v})
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        _render_pdf_table(ax, "Raw JSON Scalars (all report sections)", pd.DataFrame(payload_rows), max_rows=55)
        fig.suptitle("Raw JSON Summary", fontsize=14)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buf.seek(0)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _build_pdf_report_cached(run_dir_str: str, signature: tuple[tuple[str, int, int], ...]) -> bytes:
    _ = signature
    return _build_pdf_report_bytes(Path(run_dir_str))


def _render_backtest_tab(bt_dir: Path) -> None:
    st.caption("Performance evidence focused on returns, risk, and distribution stability.")

    daily_path = bt_dir / "daily.parquet"
    trades_path = bt_dir / "trades.parquet"
    summary_path = bt_dir / "summary.json"

    if not daily_path.exists():
        st.info("Backtest artifacts not found. Run `ivdyn backtest ...` first.")
        return

    daily = pd.read_parquet(daily_path)
    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
    bt_summary = _read_json(summary_path)

    daily["date"] = pd.to_datetime(daily["date"])
    daily["pnl"] = pd.to_numeric(daily["pnl"], errors="coerce").fillna(0.0)
    daily["equity"] = pd.to_numeric(daily["equity"], errors="coerce").fillna(0.0)
    if "options_pnl" in daily.columns:
        daily["options_pnl"] = pd.to_numeric(daily["options_pnl"], errors="coerce").fillna(0.0)
    else:
        daily["options_pnl"] = 0.0
    if "hedge_pnl" in daily.columns:
        daily["hedge_pnl"] = pd.to_numeric(daily["hedge_pnl"], errors="coerce").fillna(0.0)
    else:
        daily["hedge_pnl"] = 0.0
    for c in ("net_option_delta_shares", "hedge_shares", "post_hedge_delta_shares"):
        if c in daily.columns:
            daily[c] = pd.to_numeric(daily[c], errors="coerce").fillna(0.0)
        else:
            daily[c] = 0.0
    abs_net_delta = daily["net_option_delta_shares"].abs()
    abs_post_hedge_delta = daily["post_hedge_delta_shares"].abs()
    denom = abs_net_delta.replace(0.0, np.nan)
    daily["hedge_risk_reduction_pct"] = (
        ((abs_net_delta - abs_post_hedge_delta) / denom) * 100.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    peak = daily["equity"].cummax().replace(0.0, np.nan)
    daily["drawdown"] = (daily["equity"] - peak) / peak

    stats = _compute_backtest_stats(daily, trades, bt_summary)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total PnL", f"{_format_num(stats.get('total_pnl'), 2)}")
    c2.metric("Sharpe", _format_num(stats.get("daily_sharpe"), 3))
    c3.metric("Max Drawdown", _format_num(stats.get("max_drawdown"), 3))
    c4.metric("Profit Factor", _format_num(stats.get("profit_factor"), 3))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Trades", f"{int(stats.get('trades', 0))}")
    c6.metric("Expectancy/Trade", _format_num(stats.get("expectancy_per_trade"), 2))
    c7.metric("Best Day", _format_num(stats.get("best_day"), 2))
    c8.metric("Worst Day", _format_num(stats.get("worst_day"), 2))

    col_a, col_b = st.columns(2)
    with col_a:
        eq = (
            alt.Chart(daily)
            .mark_line(color="#0f766e")
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("equity:Q", title="Equity"),
                tooltip=["date:T", "equity:Q", "pnl:Q", "trades:Q"],
            )
            .properties(height=280, title="Equity Curve")
        )
        st.altair_chart(eq, use_container_width=True)

        dd = (
            alt.Chart(daily)
            .mark_area(color="#b91c1c", opacity=0.35)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("drawdown:Q", title="Drawdown"),
                tooltip=["date:T", "drawdown:Q"],
            )
            .properties(height=190, title="Drawdown")
        )
        st.altair_chart(dd, use_container_width=True)

    with col_b:
        daily_bar = (
            alt.Chart(daily)
            .mark_bar()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("pnl:Q", title="Daily PnL"),
                color=alt.condition("datum.pnl >= 0", alt.value("#0f766e"), alt.value("#b91c1c")),
                tooltip=["date:T", "pnl:Q", "trades:Q"],
            )
            .properties(height=280, title="Daily PnL")
        )
        st.altair_chart(daily_bar, use_container_width=True)

        pnl_hist = (
            alt.Chart(daily)
            .mark_bar(opacity=0.85)
            .encode(
                alt.X("pnl:Q", bin=alt.Bin(maxbins=40), title="Daily PnL"),
                alt.Y("count():Q", title="Count"),
            )
            .properties(height=190, title="PnL Distribution")
        )
        st.altair_chart(pnl_hist, use_container_width=True)

    daily_desc = daily["pnl"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_frame("daily_pnl")
    st.dataframe(daily_desc, use_container_width=True)

    st.markdown("#### Hedge Diagnostics")
    h1, h2, h3 = st.columns(3)
    h1.metric("Hedge PnL", _format_num(float(daily["hedge_pnl"].sum()), 2))
    h2.metric("Avg Risk Reduction", f"{_format_num(float(daily['hedge_risk_reduction_pct'].mean()), 2)}%")
    h3.metric("Avg |Hedge Shares|", _format_num(float(daily["hedge_shares"].abs().mean()), 1))

    hedge_amt_chart = (
        alt.Chart(daily)
        .mark_bar()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("hedge_shares:Q", title="Hedge Shares"),
            color=alt.condition("datum.hedge_shares >= 0", alt.value("#0f766e"), alt.value("#b91c1c")),
            tooltip=[
                "date:T",
                alt.Tooltip("hedge_shares:Q", title="Hedge shares"),
                alt.Tooltip("net_option_delta_shares:Q", title="Option delta shares"),
                alt.Tooltip("post_hedge_delta_shares:Q", title="Post-hedge delta"),
            ],
        )
        .properties(height=220, title="Hedge Amount Per Day")
    )
    st.altair_chart(hedge_amt_chart, use_container_width=True)

    hedge_pnl_bar = (
        alt.Chart(daily)
        .mark_bar(color="#d97706", opacity=0.6)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("hedge_pnl:Q", title="Hedge PnL"),
            tooltip=[
                "date:T",
                alt.Tooltip("hedge_pnl:Q", title="Hedge PnL"),
                alt.Tooltip("options_pnl:Q", title="Options PnL"),
                alt.Tooltip("pnl:Q", title="Total PnL"),
            ],
        )
    )
    risk_reduction_line = (
        alt.Chart(daily)
        .mark_line(color="#111827", point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("hedge_risk_reduction_pct:Q", title="Delta Risk Reduction (%)"),
            tooltip=[
                "date:T",
                alt.Tooltip("hedge_risk_reduction_pct:Q", title="Risk reduction %"),
                alt.Tooltip("net_option_delta_shares:Q", title="Option delta shares"),
                alt.Tooltip("post_hedge_delta_shares:Q", title="Post-hedge delta"),
            ],
        )
    )
    st.altair_chart(
        alt.layer(hedge_pnl_bar, risk_reduction_line)
        .resolve_scale(y="independent")
        .properties(height=240, title="Hedge Contribution to PnL and Risk Reduction"),
        use_container_width=True,
    )

    if not trades.empty:
        trades = trades.copy()
        trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce")
        grouped = (
            trades.groupby(["side", "call_put"], dropna=False)
            .agg(
                trades=("pnl", "count"),
                pnl_sum=("pnl", "sum"),
                pnl_mean=("pnl", "mean"),
                win_rate=("pnl", lambda x: float((x > 0).mean()) if len(x) else np.nan),
                signal_median=("signal", "median"),
                fill_prob_mean=("fill_prob", "mean"),
            )
            .reset_index()
            .sort_values("pnl_sum", ascending=False)
        )
        st.dataframe(grouped, use_container_width=True, hide_index=True)


def _render_surface_tab(eval_dir: Path) -> None:
    st.caption("Interactive fit inspection: 3D surface overlay plus configurable slice overlays.")

    surf = _load_surface_eval(eval_dir)
    if surf is None:
        st.info("No surface prediction artifact found.")
        return

    dates = surf["dates"].astype(str)
    x_grid = surf["x_grid"].astype(np.float32)
    tenor_days = surf["tenor_days"].astype(np.int32)
    obs = surf["iv_surface_obs"].astype(np.float32)
    pred = surf["iv_surface_pred"].astype(np.float32)

    idx = st.slider("Select surface date", min_value=0, max_value=len(dates) - 1, value=len(dates) - 1)
    st.caption(f"Date: {dates[idx]}")

    err = pred[idx] - obs[idx]
    c1, c2, c3 = st.columns(3)
    c1.metric("Surface RMSE", _format_num(float(np.sqrt(np.mean(err**2)))))
    c2.metric("Surface MAE", _format_num(float(np.mean(np.abs(err)))))
    c3.metric("Max |Error|", _format_num(float(np.max(np.abs(err)))))

    if PLOTLY_AVAILABLE:
        fig3d = _surface_3d_figure(obs[idx], pred[idx], x_grid, tenor_days)
        st.plotly_chart(fig3d, use_container_width=True)
    else:
        st.warning("Install `plotly` to enable interactive 3D surface overlays.")

    overlay_mode = st.radio(
        "Slice overlay mode",
        ["By tenor", "By moneyness"],
        horizontal=True,
        key="surface_slice_mode",
    )

    if overlay_mode == "By tenor":
        tenor_options = [int(t) for t in tenor_days.tolist()]
        defaults = [t for t in (30, 90) if t in tenor_options]
        if not defaults:
            defaults = tenor_options[: min(2, len(tenor_options))]
        selected = st.multiselect("Tenor(s)", options=tenor_options, default=defaults, key="slice_tenors")
        selected = selected or [tenor_options[0]]
        mode = "tenor"
        df = _surface_slice_df(obs[idx], pred[idx], x_grid, tenor_days, mode=mode, selected=[float(x) for x in selected])
    else:
        x_options = [float(x) for x in x_grid.tolist()]
        defaults = []
        for target in (-0.10, 0.0, 0.10):
            defaults.append(float(x_options[int(np.argmin(np.abs(np.array(x_options) - target)))]))
        defaults = sorted(set(defaults))
        selected = st.multiselect(
            "Moneyness point(s)",
            options=x_options,
            default=defaults,
            format_func=lambda x: f"{x:.2f}",
            key="slice_x",
        )
        selected = selected or [0.0]
        mode = "moneyness"
        df = _surface_slice_df(obs[idx], pred[idx], x_grid, tenor_days, mode=mode, selected=[float(x) for x in selected])

    chart = _slice_overlay_chart(df, mode=mode)
    if PLOTLY_AVAILABLE and chart is not None and hasattr(chart, "to_plotly_json"):
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.altair_chart(chart, use_container_width=True)


def _render_prediction_tab(eval_dir: Path, metrics: dict) -> None:
    st.caption("Contract-level prediction quality with residual diagnostics and error concentration.")

    pred_path = eval_dir / "contract_predictions.parquet"
    if not pred_path.exists():
        st.info("No evaluation predictions found.")
        return

    pred = pd.read_parquet(pred_path)
    pred_test = pred[pred["split"] == "test"].copy()
    if pred_test.empty:
        st.info("No test split contracts in prediction artifact.")
        return

    pred_test["error"] = pd.to_numeric(pred_test["pred_price_norm"], errors="coerce") - pd.to_numeric(
        pred_test["target_price_norm"], errors="coerce"
    )
    pred_test["abs_error"] = pred_test["error"].abs()

    stat = _compute_prediction_stats(pred_test)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", _format_num(metrics.get("price_rmse")))
    c2.metric("MAE", _format_num(metrics.get("price_mae")))
    c3.metric("Next-Day R2", _format_num(metrics.get("next_price_r2", metrics.get("price_r2")), 3))
    c4.metric("Bias", _format_num(stat.get("bias")))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Median |Error|", _format_num(stat.get("median_abs_error")))
    c6.metric("P95 |Error|", _format_num(stat.get("p95_abs_error")))
    c7.metric("|Err|<=0.01", _format_num(100.0 * stat.get("within_1pct_abs", np.nan), 2) + "%")
    c8.metric("|Err|<=0.02", _format_num(100.0 * stat.get("within_2pct_abs", np.nan), 2) + "%")

    sample = pred_test.sample(min(len(pred_test), 8000), random_state=7)

    col_a, col_b = st.columns(2)
    with col_a:
        scatter = (
            alt.Chart(sample)
            .mark_circle(size=20, opacity=0.35)
            .encode(
                x=alt.X("target_price_norm:Q", title="Observed price"),
                y=alt.Y("pred_price_norm:Q", title="Predicted price"),
                color=alt.Color("call_put:N", scale=alt.Scale(scheme="set2")),
                tooltip=["date", "symbol", "dte", "target_price_norm", "pred_price_norm", "error"],
            )
            .properties(height=320, title="Predicted vs Observed")
        )
        top = float(max(sample["target_price_norm"].max(), sample["pred_price_norm"].max()))
        ref = alt.Chart(pd.DataFrame({"v": [0.0, top]})).mark_line(color="#111827").encode(x="v:Q", y="v:Q")
        st.altair_chart(scatter + ref, use_container_width=True)

    with col_b:
        hist = (
            alt.Chart(pred_test)
            .mark_bar(opacity=0.85)
            .encode(
                alt.X("error:Q", bin=alt.Bin(maxbins=70), title="Prediction error"),
                alt.Y("count():Q", title="Count"),
            )
            .properties(height=320, title="Error Distribution")
        )
        st.altair_chart(hist, use_container_width=True)

    pred_test["dte_bucket"] = pd.cut(
        pred_test["dte"],
        bins=[0, 14, 30, 60, 90, 180, 3650],
        labels=["<=14", "15-30", "31-60", "61-90", "91-180", ">180"],
        include_lowest=True,
    )

    bucket_stats = (
        pred_test.groupby(["call_put", "dte_bucket"], dropna=False, observed=False)
        .agg(
            n=("error", "count"),
            rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan),
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
        )
        .reset_index()
        .sort_values(["call_put", "dte_bucket"])
    )
    st.dataframe(bucket_stats, use_container_width=True, hide_index=True)


def _render_training_tab(run_dir: Path) -> None:
    st.caption("Model optimization diagnostics by stage with compact fit-over-time signals.")

    hist_path = run_dir / "train_history.csv"
    cfg_path = run_dir / "train_config.json"
    if not hist_path.exists():
        st.info("No training history found.")
        return

    hist = pd.read_csv(hist_path)
    hist["epoch"] = pd.to_numeric(hist["epoch"], errors="coerce")

    numeric_cols = [c for c in hist.columns if c not in {"stage", "epoch"} and pd.api.types.is_numeric_dtype(hist[c])]
    metric = st.selectbox("Training metric", options=numeric_cols, index=0 if numeric_cols else None)

    if metric:
        plot_df = hist[["stage", "epoch", metric]].dropna()
        if not plot_df.empty:
            ch = (
                alt.Chart(plot_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("epoch:Q"),
                    y=alt.Y(f"{metric}:Q", title=metric),
                    color=alt.Color("stage:N"),
                    tooltip=["stage", "epoch", metric],
                )
                .properties(height=320, title=f"{metric} by Stage")
            )
            st.altair_chart(ch, use_container_width=True)

    if "val_recon_loss" in hist.columns:
        recon_df = hist[["stage", "epoch", "val_recon_loss"]].dropna()
        if not recon_df.empty:
            recon_ch = (
                alt.Chart(recon_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("epoch:Q"),
                    y=alt.Y("val_recon_loss:Q", title="val_recon_loss"),
                    color=alt.Color("stage:N"),
                    tooltip=["stage", "epoch", "val_recon_loss"],
                )
                .properties(height=260, title="Validation Reconstruction")
            )
            st.altair_chart(recon_ch, use_container_width=True)

    by_stage = hist.groupby("stage", dropna=False).agg(
        epochs=("epoch", "count"),
        final_loss=("loss", "last") if "loss" in hist.columns else ("epoch", "count"),
        min_loss=("loss", "min") if "loss" in hist.columns else ("epoch", "count"),
        final_val_recon=("val_recon_loss", "last") if "val_recon_loss" in hist.columns else ("epoch", "count"),
    )
    st.dataframe(by_stage.reset_index(), use_container_width=True, hide_index=True)

    cfg = _read_json(cfg_path)
    if cfg:
        with st.expander("Training config"):
            st.json(cfg)


def _render_fits_tab(eval_dir: Path, metrics: dict) -> None:
    st.caption("Fit diagnostics: surface-level quality, no-arbitrage behavior, and temporal fit stability.")

    surf = _load_surface_eval(eval_dir)
    noarb_path = eval_dir / "noarb_test_dates.parquet"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Surface RMSE", _format_num(metrics.get("surface_iv_rmse")))
    c2.metric("Surface MAE", _format_num(metrics.get("surface_iv_mae")))
    c3.metric("Cal Viol (Pred)", _format_num(metrics.get("calendar_violation_pred_mean")))
    c4.metric("Bfly Viol (Pred)", _format_num(metrics.get("butterfly_violation_pred_mean")))

    if surf is not None:
        dates = pd.to_datetime(surf["dates"].astype(str))
        obs = surf["iv_surface_obs"].astype(np.float32)
        pred = surf["iv_surface_pred"].astype(np.float32)

        rmse_by_date = np.sqrt(np.mean((pred - obs) ** 2, axis=(1, 2)))
        mae_by_date = np.mean(np.abs(pred - obs), axis=(1, 2))

        fit_df = pd.DataFrame(
            {
                "date": dates,
                "surface_rmse": rmse_by_date,
                "surface_mae": mae_by_date,
            }
        )

        fit_chart = (
            alt.Chart(fit_df)
            .transform_fold(["surface_rmse", "surface_mae"], as_=["metric", "value"])
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T"),
                y=alt.Y("value:Q"),
                color=alt.Color("metric:N"),
                tooltip=["date:T", "metric:N", "value:Q"],
            )
            .properties(height=280, title="Surface Fit Over Time")
        )
        st.altair_chart(fit_chart, use_container_width=True)

    if noarb_path.exists():
        noarb = pd.read_parquet(noarb_path)
        noarb["date"] = pd.to_datetime(noarb["date"])
        long = noarb.melt(
            id_vars=["date"],
            value_vars=["calendar_obs", "calendar_pred", "butterfly_obs", "butterfly_pred"],
            var_name="series",
            value_name="value",
        )
        noarb_chart = (
            alt.Chart(long)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T"),
                y=alt.Y("value:Q", title="Violation Rate"),
                color=alt.Color("series:N"),
                tooltip=["date:T", "series:N", "value:Q"],
            )
            .properties(height=280, title="No-Arbitrage Diagnostics")
        )
        st.altair_chart(noarb_chart, use_container_width=True)


def render_dashboard(run_dir: Path) -> None:
    st.title("IV Dynamics Research Console")
    st.caption("Focused dashboard: backtest evidence, surface fit overlays, prediction errors, training diagnostics, and fit quality.")

    if not run_dir.exists():
        st.error(f"Run directory does not exist: {run_dir}")
        return

    eval_dir = run_dir / "evaluation"
    bt_dir = run_dir / "backtest"

    train_summary = _read_json(run_dir / "train_summary.json")
    metrics = _read_json(eval_dir / "metrics.json")
    bt_summary = _read_json(bt_dir / "summary.json")

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Price RMSE", _format_num(metrics.get("price_rmse")))
    top2.metric("Surface RMSE", _format_num(metrics.get("surface_iv_rmse")))
    top3.metric("Backtest PnL", _format_num(bt_summary.get("total_pnl"), 2))
    top4.metric("Backtest Sharpe", _format_num(bt_summary.get("daily_sharpe"), 3))

    report_paths = [
        run_dir / "train_history.csv",
        run_dir / "train_summary.json",
        run_dir / "train_config.json",
        eval_dir / "metrics.json",
        eval_dir / "contract_predictions.parquet",
        eval_dir / "noarb_test_dates.parquet",
        eval_dir / "surface_predictions.npz",
        bt_dir / "daily.parquet",
        bt_dir / "trades.parquet",
        bt_dir / "summary.json",
    ]
    with st.expander("Report", expanded=False):
        st.markdown(
            "<div class='block-note'>Comprehensive export: full backtest, surface overlays, prediction diagnostics, training diagnostics, fits, and raw summary payloads.</div>",
            unsafe_allow_html=True,
        )
        try:
            pdf_bytes = _build_pdf_report_cached(str(run_dir.resolve()), _artifact_signature(report_paths))
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"{run_dir.name}_report.pdf",
                mime="application/pdf",
            )
        except Exception as exc:
            st.warning(f"PDF export unavailable: {exc}")

    tabs = st.tabs(["Backtest & PnL", "Surface Overlays", "Prediction Errors", "Training Diagnostics", "Fits"])

    with tabs[0]:
        _render_backtest_tab(bt_dir)

    with tabs[1]:
        _render_surface_tab(eval_dir)

    with tabs[2]:
        _render_prediction_tab(eval_dir, metrics)

    with tabs[3]:
        _render_training_tab(run_dir)

    with tabs[4]:
        _render_fits_tab(eval_dir, metrics)

    with st.expander("Raw JSON"):
        st.write(
            {
                "train_summary": train_summary,
                "eval_metrics": metrics,
                "backtest_summary": bt_summary,
            }
        )


def main() -> None:
    st.set_page_config(page_title="IV Dynamics Dashboard", layout="wide")
    _inject_style()
    run_dir = _resolve_run_dir()
    render_dashboard(run_dir)


if __name__ == "__main__":
    main()
