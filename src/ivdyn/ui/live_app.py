"""Streamlit dashboard for live walk-forward outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_parquet_or_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def _discover_symbols(root: Path) -> list[str]:
    if not root.exists():
        return []
    out: list[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.lower() == "history":
            continue
        if (p / "sessions").is_dir():
            out.append(p.name.upper())
    return out


def _latest_session_dir(symbol_root: Path) -> Path | None:
    latest_path = symbol_root / "latest.txt"
    if latest_path.exists():
        try:
            target = Path(latest_path.read_text(encoding="utf-8").strip()).expanduser().resolve()
            if target.is_dir():
                return target
        except Exception:
            pass
    sessions_root = symbol_root / "sessions"
    if not sessions_root.is_dir():
        return None
    candidates = [p for p in sessions_root.glob("wf_*") if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _format_money(v: float | int | None) -> str:
    if v is None:
        return "n/a"
    try:
        x = float(v)
    except Exception:
        return "n/a"
    return f"${x:,.2f}"


def main() -> None:
    st.set_page_config(page_title="ivdyn ui2", layout="wide")
    st.title("ivdyn ui2: Live Walk-Forward")

    live_root = Path(os.environ.get("IVDYN_LIVE_OUTPUT_ROOT", "outputs/live_walkforward")).expanduser()
    symbols = _discover_symbols(live_root)
    if not symbols:
        st.warning(f"No live walk-forward records found under `{live_root}`.")
        st.stop()

    default_symbol = str(os.environ.get("IVDYN_LIVE_DEFAULT_SYMBOL", "")).upper().strip()
    default_idx = symbols.index(default_symbol) if default_symbol in symbols else 0
    with st.sidebar:
        st.header("Selection")
        symbol = st.selectbox("Symbol", options=symbols, index=default_idx)
        st.caption(f"Live root: `{live_root}`")

    symbol_root = live_root / symbol
    session_dir = _latest_session_dir(symbol_root)
    if session_dir is None:
        st.warning(f"No live sessions found for `{symbol}`.")
        st.stop()

    metadata = _read_json(session_dir / "metadata.json")
    bt_dir = session_dir / "backtest"
    daily = _read_parquet_or_csv(bt_dir / "daily.parquet")
    if "date" in daily.columns:
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        daily = daily.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    week_days = int(os.environ.get("IVDYN_LIVE_WF_WEEK_DAYS", "5") or "5")
    week_days = max(1, week_days)
    last_week = daily.tail(week_days).copy()
    last_week_pnl = float(pd.to_numeric(last_week.get("pnl"), errors="coerce").fillna(0.0).sum()) if not last_week.empty else 0.0
    last_week_trades = int(pd.to_numeric(last_week.get("trades"), errors="coerce").fillna(0.0).sum()) if not last_week.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Week PnL", _format_money(last_week_pnl))
    c2.metric("Last Week Trades", f"{last_week_trades}")
    c3.metric("As Of Date", str(metadata.get("asof_date", "n/a")))
    c4.metric("Last Trade Date", str(metadata.get("last_trade_date", "n/a")))

    st.caption(f"Latest session: `{session_dir}`")

    if last_week.empty:
        st.info("No daily walk-forward records available yet.")
    else:
        plot_df = last_week.copy()
        plot_df["date_str"] = plot_df["date"].dt.strftime("%Y-%m-%d")
        plot_df["pnl"] = pd.to_numeric(plot_df.get("pnl"), errors="coerce").fillna(0.0)
        plot_df["equity"] = pd.to_numeric(plot_df.get("equity"), errors="coerce")

        pnl_chart = (
            alt.Chart(plot_df)
            .mark_bar(color="#0f766e")
            .encode(
                x=alt.X("date_str:N", title="Date"),
                y=alt.Y("pnl:Q", title="Daily PnL"),
                tooltip=["date_str", "pnl", "trades", "equity"],
            )
            .properties(height=280)
        )
        st.altair_chart(pnl_chart, use_container_width=True)

        show_cols = [c for c in ["date", "pnl", "equity", "trades", "options_pnl", "hedge_pnl"] if c in last_week.columns]
        st.dataframe(last_week[show_cols], use_container_width=True)

    history_path = symbol_root / "history.csv"
    if history_path.exists():
        hist = pd.read_csv(history_path)
        if not hist.empty:
            st.subheader("Recent Sessions")
            st.dataframe(hist.tail(20), use_container_width=True)


if __name__ == "__main__":
    main()
