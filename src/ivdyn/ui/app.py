"""Streamlit dashboard focused on training and evaluation workflows."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

OPENAI_MODEL = "gpt-5"
RUN_REPORTS_KEY = "_te_run_reports"
ACTIVE_RUN_KEY = "_te_active_run_dir"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _inject_style() -> None:
    st.markdown(
        """
<style>
:root {
  --bg0: #f4f7f3;
  --panel: #fbfdf9;
  --ink: #1d2a23;
  --accent: #14532d;
  --line: #dbe4dc;
}

.stApp {
  font-family: "Avenir Next", "Segoe UI", sans-serif;
  background:
    radial-gradient(1100px 500px at -10% -20%, #dff3e4 0%, rgba(223,243,228,0.15) 60%, transparent 100%),
    radial-gradient(900px 380px at 110% -10%, #e4efe9 0%, rgba(228,239,233,0.10) 55%, transparent 100%),
    linear-gradient(180deg, var(--bg0) 0%, #eef3ef 100%);
  color: var(--ink);
}

div[data-testid="stMetric"] {
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--panel);
  padding: 6px 10px;
}

.block-note {
  border-left: 4px solid var(--accent);
  background: #f2f8f3;
  border-radius: 10px;
  padding: 10px 12px;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _load_dotenv() -> None:
    """Best-effort .env loader for direct `streamlit run` usage."""
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [Path.cwd() / ".env", repo_root / ".env"]
    seen: set[Path] = set()
    for candidate in candidates:
        p = candidate.resolve()
        if p in seen or not p.exists() or not p.is_file():
            continue
        seen.add(p)

        for raw_line in p.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue

            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                q = value[0]
                value = value[1:-1]
                if q == '"':
                    value = value.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
            else:
                comment_pos = value.find(" #")
                if comment_pos >= 0:
                    value = value[:comment_pos].rstrip()
            os.environ.setdefault(key, value)


def _ensure_state() -> None:
    if RUN_REPORTS_KEY not in st.session_state:
        st.session_state[RUN_REPORTS_KEY] = []
    if ACTIVE_RUN_KEY not in st.session_state:
        st.session_state[ACTIVE_RUN_KEY] = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _read_parquet_or_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def _format_value(v: Any, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, str):
        return v
    try:
        x = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def _discover_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    seen: set[Path] = set()
    for run_dir in root.glob("**/run_*"):
        if not run_dir.is_dir():
            continue
        p = run_dir.resolve()
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    out.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return out


def _extract_symbol_from_run_dir(run_dir: Path) -> str | None:
    parts = run_dir.resolve().parts
    for i, part in enumerate(parts):
        if part == "outputs" and i + 2 < len(parts) and parts[i + 2] == "runs":
            symbol = parts[i + 1].strip().upper()
            return symbol or None
    return None


def _run_recency(run_dir: Path) -> float:
    recency = run_dir.stat().st_mtime if run_dir.exists() else 0.0
    for rel in ("evaluation/metrics.json", "train_summary.json", "train_history.csv", "model.pt"):
        p = run_dir / rel
        if p.exists():
            try:
                recency = max(recency, p.stat().st_mtime)
            except Exception:
                continue
    return float(recency)


def _discover_latest_runs_by_symbol(outputs_root: Path) -> dict[str, Path]:
    by_symbol: dict[str, Path] = {}
    if not outputs_root.exists():
        return by_symbol

    for symbol_dir in sorted(outputs_root.iterdir()):
        if not symbol_dir.is_dir():
            continue
        runs_root = symbol_dir / "runs"
        if not runs_root.is_dir():
            continue

        candidates: list[Path] = []
        seen: set[Path] = set()

        for p in runs_root.glob("**/run_*"):
            if not p.is_dir():
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            candidates.append(rp)

        for latest_path in runs_root.glob("**/latest.txt"):
            try:
                target = Path(latest_path.read_text(encoding="utf-8").strip()).expanduser().resolve()
            except Exception:
                continue
            if not target.is_dir() or target in seen:
                continue
            seen.add(target)
            candidates.append(target)

        if not candidates:
            continue

        try:
            newest = max(candidates, key=_run_recency).resolve()
        except Exception:
            continue
        by_symbol[symbol_dir.name.upper()] = newest

    return by_symbol


def _artifact_manifest(run_dir: Path) -> pd.DataFrame:
    rel_paths = [
        "model.pt",
        "train_config.json",
        "train_summary.json",
        "train_history.csv",
        "latent_states.parquet",
        "evaluation/metrics.json",
        "evaluation/contract_predictions.parquet",
        "evaluation/noarb_test_dates.parquet",
        "evaluation/noarb_forecast_test_dates.parquet",
        "evaluation/surface_predictions.npz",
    ]
    rows: list[dict[str, Any]] = []
    for rel in rel_paths:
        p = run_dir / rel
        row: dict[str, Any] = {
            "artifact": rel,
            "exists": p.exists(),
            "size_bytes": np.nan,
            "modified_utc": "",
        }
        if p.exists():
            stat = p.stat()
            row["size_bytes"] = int(stat.st_size)
            row["modified_utc"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
        rows.append(row)
    return pd.DataFrame(rows)


def _build_artifact_report(run_dir: Path) -> dict[str, Any]:
    train_summary = _read_json(run_dir / "train_summary.json")
    train_config = _read_json(run_dir / "train_config.json")
    eval_metrics = _read_json(run_dir / "evaluation" / "metrics.json")
    hist_tail = _train_history_tail_rows(run_dir)
    manifest_df = _artifact_manifest(run_dir)

    dataset_path = str(train_summary.get("dataset_path", "")) if train_summary else ""
    status = "ok" if (train_summary or eval_metrics) else "missing_artifacts"
    updated_utc = datetime.fromtimestamp(_run_recency(run_dir), tz=timezone.utc).isoformat(timespec="seconds")

    return {
        "id": f"artifact_{int(datetime.now(tz=timezone.utc).timestamp() * 1000)}",
        "timestamp_utc": _utc_now(),
        "run_type": "artifact_review",
        "status": status,
        "dataset_path": dataset_path,
        "training_parameters": _json_safe(train_config),
        "evaluation_parameters": {"source": "artifacts"},
        "run_dir": str(run_dir.resolve()),
        "run_updated_utc": updated_utc,
        "train_summary": _json_safe(train_summary),
        "evaluation_metrics": _json_safe(eval_metrics),
        "train_history_tail": _json_safe(hist_tail),
        "artifact_manifest": _json_safe(manifest_df.to_dict(orient="records")),
        "error": "",
    }


def _extract_openai_text(payload: dict[str, Any]) -> str:
    def _text_from_value(v: Any) -> str:
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, dict):
            for k in ("value", "text", "output_text", "content"):
                if k in v:
                    t = _text_from_value(v.get(k))
                    if t:
                        return t
            return ""
        if isinstance(v, list):
            parts = [_text_from_value(x) for x in v]
            parts = [p for p in parts if p]
            return "\n\n".join(parts).strip()
        return ""

    chunks: list[str] = []

    raw = payload.get("output_text")
    if isinstance(raw, str) and raw.strip():
        chunks.append(raw.strip())
    elif isinstance(raw, list):
        parsed = _text_from_value(raw)
        if parsed:
            chunks.append(parsed)

    output = payload.get("output")
    if isinstance(output, list):
        for block in output:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "output_text":
                txt = _text_from_value(block.get("text"))
                if txt:
                    chunks.append(txt)
            content = block.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    txt = _text_from_value(item.get("text"))
                    if not txt:
                        txt = _text_from_value(item)
                    if txt:
                        chunks.append(txt)

    # Compatibility fallback for chat-completions shaped payloads.
    choices = payload.get("choices")
    if isinstance(choices, list):
        for ch in choices:
            if not isinstance(ch, dict):
                continue
            msg = ch.get("message")
            if isinstance(msg, dict):
                txt = _text_from_value(msg.get("content"))
                if txt:
                    chunks.append(txt)

    deduped: list[str] = []
    seen: set[str] = set()
    for c in chunks:
        cc = c.strip()
        if not cc or cc in seen:
            continue
        seen.add(cc)
        deduped.append(cc)
    return "\n\n".join(deduped).strip()


def _build_openai_feedback_prompt(report: dict[str, Any]) -> tuple[str, str]:
    system_text = (
        "You are a senior quantitative researcher reviewing an options-volatility ML system. "
        "Return specific, testable recommendations to improve model quality in training/evaluation workflows. "
        "Do not provide backtesting, portfolio allocation, or discretionary trading advice. "
        "Ground suggestions in financial microstructure and options theory (vol surface behavior, no-arbitrage, "
        "execution-probability calibration, and forecast skill). "
        "Avoid generic advice; each suggestion must include rationale, expected metric impact, and tradeoff/risk."
    )

    goal_context = {
        "primary_goal": "Improve training and evaluation quality for IV dynamics modeling.",
        "workflow_scope": "Training + evaluation only. Ignore backtesting recommendations.",
        "success_signals": [
            "Lower price_rmse",
            "Higher price_r2",
            "Lower exec_brier",
            "Lower surface_forecast_iv_rmse",
            "More stable validation losses in train_history",
        ],
        "constraints": [
            "Recommendations should be directly actionable in CLI-driven train/eval runs.",
            "Prefer changes that can be validated by artifacts already produced in this repo.",
        ],
    }

    financial_basis = {
        "domain": "US equity options implied-volatility dynamics",
        "state_representation": "Latent state learned from IV surface snapshots over x=ln(K/S) and tenor_days",
        "model_outputs": [
            "Option price target (normalized by spot)",
            "Execution/fill probability target",
            "Surface reconstruction and one-step-ahead surface forecast",
        ],
        "financial_principles": [
            "Smile/skew and term-structure behavior should be stable and economically plausible",
            "No-arbitrage diagnostics matter (calendar and butterfly violations)",
            "Near-ATM and short-tenor contracts can dominate risk and should be treated carefully",
            "Execution probability quality should be judged via calibration-oriented metrics like Brier score",
            "Forecast skill should be compared to persistence/carry baselines, not judged in isolation",
        ],
        "metric_interpretation": {
            "price_rmse": "Lower is better (same-day fit quality for normalized option price)",
            "price_r2": "Higher is better (especially next-day predictive quality where available)",
            "exec_brier": "Lower is better (better probability calibration for fills)",
            "surface_forecast_iv_rmse": "Lower is better (forward IV surface forecast quality)",
            "calendar_violation_*": "Lower is better (fewer calendar-arbitrage inconsistencies)",
            "butterfly_violation_*": "Lower is better (fewer convexity/arbitrage inconsistencies)",
        },
    }

    context = {
        "goal_context": goal_context,
        "financial_basis": financial_basis,
        "run_type": report.get("run_type"),
        "status": report.get("status"),
        "dataset_path": report.get("dataset_path"),
        "run_dir": report.get("run_dir"),
        "training_parameters": _json_safe(report.get("training_parameters", {})),
        "evaluation_parameters": _json_safe(report.get("evaluation_parameters", {})),
        "train_summary": _json_safe(report.get("train_summary", {})),
        "evaluation_metrics": _json_safe(report.get("evaluation_metrics", {})),
        "train_history_tail": _json_safe(report.get("train_history_tail", [])),
    }
    user_text = (
        "Given this run output, project goal, and financial basis, provide:\n"
        "1) Top 5 prioritized improvements (highest impact first)\n"
        "2) For each: financial/technical rationale, exact metrics expected to change, and expected direction\n"
        "3) Concrete next-run parameter changes (only fields that should be modified)\n"
        "4) Sanity checks to validate that improvements are real (including baseline checks and overfitting checks)\n"
        "5) Risks/failure-modes for each recommendation\n\n"
        "Keep the response structured and explicit. Use short sections and bullet points.\n\n"
        f"RUN_CONTEXT_JSON:\n{json.dumps(context, indent=2, sort_keys=True)}"
    )
    return system_text, user_text


def _request_openai_feedback(
    report: dict[str, Any],
    *,
    model: str,
    timeout_seconds: int,
) -> tuple[dict[str, Any], str | None, dict[str, Any] | None, str | None]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {}, None, None, "Missing OPENAI_API_KEY in environment or .env file."

    system_text, user_text = _build_openai_feedback_prompt(report)
    request_payload: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_text}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_text}],
            },
        ],
    }

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    read_timeout = max(30, int(os.environ.get("OPENAI_TIMEOUT_SECONDS", str(timeout_seconds))))
    connect_timeout = max(5, int(os.environ.get("OPENAI_CONNECT_TIMEOUT_SECONDS", "10")))
    max_retries = max(1, int(os.environ.get("OPENAI_MAX_RETRIES", "3")))
    retry_backoff_seconds = max(1.0, float(os.environ.get("OPENAI_RETRY_BACKOFF_SECONDS", "2.0")))

    last_error: str | None = None
    last_raw: dict[str, Any] | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=request_payload,
                timeout=(connect_timeout, read_timeout),
            )
        except (requests.exceptions.ReadTimeout, requests.exceptions.Timeout) as exc:
            last_error = (
                f"OpenAI request timed out on attempt {attempt}/{max_retries} "
                f"(connect={connect_timeout}s, read={read_timeout}s): {exc}"
            )
            if attempt < max_retries:
                time.sleep(retry_backoff_seconds * float(attempt))
                continue
            return request_payload, None, last_raw, last_error
        except requests.exceptions.ConnectionError as exc:
            last_error = f"OpenAI connection failed on attempt {attempt}/{max_retries}: {exc}"
            if attempt < max_retries:
                time.sleep(retry_backoff_seconds * float(attempt))
                continue
            return request_payload, None, last_raw, last_error
        except Exception as exc:
            return request_payload, None, last_raw, f"OpenAI request failed: {exc}"

        if resp.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
            last_error = f"OpenAI API temporary error ({resp.status_code}) on attempt {attempt}/{max_retries}."
            time.sleep(retry_backoff_seconds * float(attempt))
            continue

        if resp.status_code >= 400:
            body_snippet = resp.text.strip()[:1200]
            return request_payload, None, last_raw, f"OpenAI API error ({resp.status_code}): {body_snippet}"

        try:
            response_json = resp.json()
        except Exception as exc:
            return request_payload, None, last_raw, f"OpenAI API returned non-JSON response: {exc}"

        last_raw = response_json
        text = _extract_openai_text(response_json)
        if not text:
            text = "OpenAI responded without extractable text. Inspect raw response JSON."
        return request_payload, text, response_json, None

    return request_payload, None, last_raw, (last_error or "OpenAI request failed for unknown reason.")


def _train_history_tail_rows(run_dir: Path, n: int = 8) -> list[dict[str, Any]]:
    hist_path = run_dir / "train_history.csv"
    if not hist_path.exists():
        return []
    try:
        hist = pd.read_csv(hist_path)
    except Exception:
        return []
    if hist.empty:
        return []
    return hist.tail(n).to_dict(orient="records")


def _append_report(report: dict[str, Any]) -> None:
    reports = st.session_state.get(RUN_REPORTS_KEY)
    if not isinstance(reports, list):
        reports = []
    reports.append(report)
    st.session_state[RUN_REPORTS_KEY] = reports


def _render_training_diagnostics(run_dir: Path) -> None:
    st.subheader("Training Diagnostics")

    hist_path = run_dir / "train_history.csv"
    if not hist_path.exists():
        st.info("No training history found for selected run.")
        return

    hist = pd.read_csv(hist_path)
    if hist.empty:
        st.info("Training history file is empty.")
        return

    numeric_cols = [
        c for c in hist.columns if c not in {"stage", "epoch"} and pd.api.types.is_numeric_dtype(hist[c])
    ]
    if not numeric_cols:
        st.dataframe(hist.tail(50), use_container_width=True)
        return

    stage_options = ["all"] + sorted(str(x) for x in hist["stage"].dropna().unique()) if "stage" in hist.columns else ["all"]
    selected_stage = st.selectbox("Stage filter", options=stage_options, index=0, key="diag_train_stage")

    chart_df = hist.copy()
    if selected_stage != "all" and "stage" in chart_df.columns:
        chart_df = chart_df[chart_df["stage"] == selected_stage].copy()

    metric_pick = st.selectbox("Metric", options=numeric_cols, index=0, key="diag_train_metric")
    if chart_df.empty:
        st.info("No rows match the selected stage.")
        return

    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y(f"{metric_pick}:Q", title=metric_pick),
            color=alt.Color("stage:N") if "stage" in chart_df.columns else alt.value("#14532d"),
            tooltip=[c for c in ["stage", "epoch", metric_pick] if c in chart_df.columns],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(chart_df.tail(40), use_container_width=True)


def _render_eval_diagnostics(run_dir: Path) -> None:
    st.subheader("Evaluation Diagnostics")
    eval_dir = run_dir / "evaluation"

    metrics = _read_json(eval_dir / "metrics.json")
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price RMSE", _format_value(metrics.get("price_rmse"), 6))
        c2.metric("Price R2", _format_value(metrics.get("price_r2"), 6))
        c3.metric("Exec Brier", _format_value(metrics.get("exec_brier"), 6))
        c4.metric("Surface Forecast RMSE", _format_value(metrics.get("surface_forecast_iv_rmse"), 6))

        with st.expander("Full Metrics JSON", expanded=False):
            st.json(metrics)
    else:
        st.warning("No evaluation metrics found.")

    pred_df = _read_parquet_or_csv(eval_dir / "contract_predictions.parquet")
    if pred_df.empty:
        st.info("No contract_predictions artifact found.")
        return

    for col in ["target_price_norm", "pred_price_norm", "dte"]:
        if col in pred_df.columns:
            pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")

    pred_df = pred_df.dropna(subset=[c for c in ["target_price_norm", "pred_price_norm"] if c in pred_df.columns]).copy()
    if pred_df.empty:
        st.info("Predictions artifact has no valid numeric rows.")
        return

    pred_df["abs_error"] = (pred_df["pred_price_norm"] - pred_df["target_price_norm"]).abs()

    scatter = (
        alt.Chart(pred_df.sample(min(4000, len(pred_df)), random_state=7))
        .mark_circle(opacity=0.35, size=36, color="#14532d")
        .encode(
            x=alt.X("target_price_norm:Q", title="Target price (normalized)"),
            y=alt.Y("pred_price_norm:Q", title="Predicted price (normalized)"),
            tooltip=[c for c in ["date", "symbol", "dte", "target_price_norm", "pred_price_norm", "abs_error"] if c in pred_df.columns],
        )
        .properties(height=360)
    )
    st.altair_chart(scatter, use_container_width=True)

    hist = (
        alt.Chart(pred_df)
        .mark_bar(color="#166534")
        .encode(
            x=alt.X("abs_error:Q", bin=alt.Bin(maxbins=50), title="Absolute error"),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[alt.Tooltip("count():Q", title="Rows")],
        )
        .properties(height=260)
    )
    st.altair_chart(hist, use_container_width=True)

    if "dte" in pred_df.columns:
        bins = [0, 14, 30, 60, 90, 180, 3650]
        labels = ["<=14", "15-30", "31-60", "61-90", "91-180", ">180"]
        pred_df["dte_bucket"] = pd.cut(pred_df["dte"], bins=bins, labels=labels, include_lowest=True)
        bucket = (
            pred_df.groupby("dte_bucket", dropna=False, observed=False)
            .agg(n=("abs_error", "count"), mae=("abs_error", "mean"), p95=("abs_error", lambda x: float(x.quantile(0.95))))
            .reset_index()
        )
        st.dataframe(bucket, use_container_width=True, hide_index=True)


def _render_run_overview(run_dir: Path) -> None:
    if not run_dir.exists() or not run_dir.is_dir():
        st.info("Select a valid run directory in the sidebar.")
        return

    report = _build_artifact_report(run_dir)
    metrics = report.get("evaluation_metrics", {})
    train_summary = report.get("train_summary", {})
    symbol = _extract_symbol_from_run_dir(run_dir) or "n/a"

    st.markdown(
        (
            "<div class='block-note'>"
            f"<b>Symbol</b>: <code>{symbol}</code><br>"
            f"<b>Run directory</b>: <code>{report.get('run_dir', 'n/a')}</code><br>"
            f"<b>Last updated (UTC)</b>: <code>{report.get('run_updated_utc', 'n/a')}</code>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price RMSE", _format_value(metrics.get("price_rmse"), 6))
    c2.metric("Price R2", _format_value(metrics.get("price_r2"), 6))
    c3.metric("Exec Brier", _format_value(metrics.get("exec_brier"), 6))
    c4.metric("Surface Forecast RMSE", _format_value(metrics.get("surface_forecast_iv_rmse"), 6))
    c5.metric("Final Val Recon", _format_value(train_summary.get("final_val_recon"), 6))

    st.subheader("Training Summary")
    st.json(train_summary, expanded=False)

    st.subheader("Evaluation Metrics")
    st.json(metrics, expanded=False)

    training_parameters = report.get("training_parameters", {})
    st.subheader("Training Parameters (train_config.json)")
    st.json(training_parameters, expanded=False)

    tail = report.get("train_history_tail", [])
    if isinstance(tail, list) and tail:
        st.subheader("Train History Tail")
        st.dataframe(pd.DataFrame(tail), use_container_width=True, hide_index=True)

    manifest = report.get("artifact_manifest", [])
    if isinstance(manifest, list) and manifest:
        st.subheader("Artifact Manifest")
        st.dataframe(pd.DataFrame(manifest), use_container_width=True, hide_index=True)


def _render_openai_tab(run_dir: Path) -> None:
    st.subheader("OpenAI GPT-5 Improvement Suggestions")
    if not run_dir.exists() or not run_dir.is_dir():
        st.info("Select a valid run directory in the sidebar.")
        return

    timeout_seconds = int(os.environ.get("OPENAI_TIMEOUT_SECONDS", "240"))
    max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "3"))
    st.caption(f"Request settings: timeout={timeout_seconds}s, retries={max_retries}, model={OPENAI_MODEL}")

    if st.button("Generate Suggestions For Selected Run", type="primary", use_container_width=False):
        report = _build_artifact_report(run_dir)
        report["openai_model"] = OPENAI_MODEL
        with st.spinner("Requesting GPT-5 improvement suggestions..."):
            req_payload, suggestions, raw, err = _request_openai_feedback(
                report,
                model=OPENAI_MODEL,
                timeout_seconds=timeout_seconds,
            )
        report["openai_request_payload"] = req_payload
        report["openai_suggestions"] = suggestions
        report["openai_response_raw"] = raw
        report["openai_error"] = err
        _append_report(report)
        if err:
            st.error(err)
        else:
            st.success("OpenAI suggestions generated.")

    st.caption(f"Current selected run: `{run_dir.resolve()}`")

    reports = st.session_state.get(RUN_REPORTS_KEY, [])
    if not isinstance(reports, list):
        reports = []
    selected_run_reports = [r for r in reports if str(r.get("run_dir", "")) == str(run_dir.resolve())]
    if not selected_run_reports:
        st.info("No OpenAI suggestions generated yet for this run.")
        return

    for i, report in enumerate(reversed(selected_run_reports), start=1):
        label = f"{i}. {report.get('timestamp_utc', 'n/a')}"
        with st.expander(label, expanded=(i == 1)):
            st.write(f"Model selected: `{report.get('openai_model', OPENAI_MODEL)}`")

            req_payload = report.get("openai_request_payload")
            if isinstance(req_payload, dict) and req_payload:
                st.caption("Request payload sent to OpenAI API service")
                st.json(req_payload, expanded=False)
            else:
                st.caption("No OpenAI request payload captured for this run.")

            openai_error = report.get("openai_error")
            if openai_error:
                st.error(str(openai_error))

            suggestion_text = report.get("openai_suggestions")
            if isinstance(suggestion_text, str) and suggestion_text.strip():
                st.caption("Parsed response text")
                st.text_area(
                    "Suggestions",
                    value=suggestion_text,
                    height=320,
                    key=f"openai_suggestions_{report.get('id', i)}",
                    disabled=True,
                )

            raw = report.get("openai_response_raw")
            if isinstance(raw, dict) and raw:
                with st.expander("Raw OpenAI response JSON", expanded=False):
                    st.json(raw, expanded=False)


def main() -> None:
    _load_dotenv()
    _ensure_state()

    st.set_page_config(page_title="ivdyn: Training + Evaluation", layout="wide")
    _inject_style()

    st.title("ivdyn: Training + Evaluation Center")
    st.caption("Backtest UI removed. This view is now for training/evaluation artifact inspection.")

    outputs_root = Path("outputs")
    discovered_runs = _discover_run_dirs(outputs_root)
    run_options = [str(p) for p in discovered_runs]
    latest_runs_by_symbol = _discover_latest_runs_by_symbol(outputs_root)
    symbol_options = sorted(latest_runs_by_symbol)

    active_default = str(st.session_state.get(ACTIVE_RUN_KEY, "") or "")
    if not active_default:
        if latest_runs_by_symbol:
            newest_run = max(latest_runs_by_symbol.values(), key=_run_recency)
            active_default = str(newest_run)
        elif run_options:
            active_default = run_options[0]
    elif latest_runs_by_symbol and not Path(active_default).exists():
        newest_run = max(latest_runs_by_symbol.values(), key=_run_recency)
        active_default = str(newest_run)
    elif run_options and active_default not in run_options:
        active_default = run_options[0]

    with st.sidebar:
        st.header("Inspect Run")
        inspect_source_choices = ["Latest by symbol", "Manual run directory"]
        default_source_idx = 0 if symbol_options else 1
        inspect_source = st.radio("Inspect source", options=inspect_source_choices, index=default_source_idx)

        if inspect_source == "Latest by symbol" and symbol_options:
            default_symbol = _extract_symbol_from_run_dir(Path(active_default)) if active_default else None
            if default_symbol not in latest_runs_by_symbol:
                newest_run = max(latest_runs_by_symbol.values(), key=_run_recency)
                default_symbol = _extract_symbol_from_run_dir(newest_run) or symbol_options[0]
            default_symbol_idx = symbol_options.index(default_symbol) if default_symbol in symbol_options else 0
            selected_symbol = st.selectbox("Symbol (latest run)", options=symbol_options, index=default_symbol_idx)
            inspect_run_raw = str(latest_runs_by_symbol[selected_symbol])
            st.text_input("Resolved run directory", value=inspect_run_raw, disabled=True)
        else:
            inspect_run_raw = st.text_input("Run directory", value=active_default)
            if run_options:
                idx = run_options.index(active_default) if active_default in run_options else 0
                picked = st.selectbox("Recent discovered runs", options=run_options, index=idx)
                if picked and picked != inspect_run_raw:
                    inspect_run_raw = picked

    inspect_run = Path(inspect_run_raw).expanduser()
    if inspect_run.exists() and inspect_run.is_dir():
        st.session_state[ACTIVE_RUN_KEY] = str(inspect_run.resolve())

    tabs = st.tabs(["Run Overview", "Training Diagnostics", "Evaluation Diagnostics", "OpenAI Suggestions"])

    with tabs[0]:
        _render_run_overview(inspect_run)

    with tabs[1]:
        if not inspect_run.exists() or not inspect_run.is_dir():
            st.info("Select a valid run directory in the sidebar to view training diagnostics.")
        else:
            _render_training_diagnostics(inspect_run)

    with tabs[2]:
        if not inspect_run.exists() or not inspect_run.is_dir():
            st.info("Select a valid run directory in the sidebar to view evaluation diagnostics.")
        else:
            _render_eval_diagnostics(inspect_run)

    with tabs[3]:
        _render_openai_tab(inspect_run)


if __name__ == "__main__":
    main()
