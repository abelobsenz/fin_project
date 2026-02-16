from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from spygen.models.sampling import load_checkpoint
from spygen.surface.arb_checks import arb_violation_counts

DEFAULT_DATASET = Path("data/processed/dataset.npz")
DEFAULT_CHECKPOINT = Path("outputs/checkpoints/flow_latest.pt")


def _latest_dirs(base: Path, prefix: str) -> list[Path]:
    if not base.exists():
        return []
    return sorted([p for p in base.glob(f"{prefix}*") if p.is_dir()])


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def _load_dataset(path: str, cache_key: str) -> dict[str, np.ndarray]:
    _ = cache_key
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


@st.cache_resource(show_spinner=False)
def _load_model(path: str, cache_key: str):
    _ = cache_key
    return load_checkpoint(path)


def _file_cache_key(path: Path) -> str:
    if not path.exists():
        return "missing"
    stat = path.stat()
    return f"{int(stat.st_mtime_ns)}:{int(stat.st_size)}"


def _surface_fig(
    surface: np.ndarray,
    x_grid: np.ndarray,
    tenors_days: np.ndarray,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    im = ax.imshow(surface, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Tenor (days)")
    ax.set_ylabel("Log-moneyness x")

    xticks = np.arange(len(tenors_days))
    yticks = np.linspace(0, len(x_grid) - 1, 5, dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(x)) for x in tenors_days])
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{x_grid[i]:.2f}" for i in yticks])

    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    return fig


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_call_price(x: float, t: float, sigma: float) -> float:
    if t <= 0.0:
        return max(0.0, 1.0 - math.exp(x))
    vol_sqrt_t = max(sigma, 1e-8) * math.sqrt(t)
    d1 = (-x + 0.5 * vol_sqrt_t * vol_sqrt_t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return _normal_cdf(d1) - math.exp(x) * _normal_cdf(d2)


def _iv_from_norm_call(c_norm: float, x: float, t: float) -> float:
    intrinsic = max(0.0, 1.0 - math.exp(x))
    upper = 1.0
    target = min(max(float(c_norm), intrinsic + 1e-8), upper - 1e-8)
    lo, hi = 1e-4, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        model = _norm_call_price(x=x, t=t, sigma=mid)
        if abs(model - target) < 1e-6:
            return mid
        if model > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _surface_to_iv(surface: np.ndarray, x_grid: np.ndarray, tenors_days: np.ndarray) -> np.ndarray:
    iv = np.zeros_like(surface, dtype=float)
    for i, x in enumerate(x_grid):
        for j, tenor in enumerate(tenors_days):
            t = max(float(tenor) / 365.0, 1.0 / 365.0)
            iv[i, j] = _iv_from_norm_call(float(surface[i, j]), float(x), t)
    return iv


def _line_fig(
    observed: np.ndarray,
    mean_surface: np.ndarray,
    x_values: np.ndarray,
    x_label: str,
    title: str,
    y_label: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(x_values, observed, label="Observed (repaired)", linewidth=2)
    ax.plot(x_values, mean_surface, label="Conditional mean", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _surface_fig_3d(
    surface: np.ndarray,
    x_grid: np.ndarray,
    tenors_days: np.ndarray,
    title: str,
    z_label: str,
) -> plt.Figure:
    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111, projection="3d")

    tenor_mesh, x_mesh = np.meshgrid(tenors_days, x_grid)
    surf = ax.plot_surface(
        tenor_mesh,
        x_mesh,
        surface,
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
        alpha=0.95,
    )
    ax.set_title(title)
    ax.set_xlabel("Tenor (days)")
    ax.set_ylabel("Log-moneyness x")
    ax.set_zlabel(z_label)
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.1)
    fig.tight_layout()
    return fig


def _show_run_tables() -> None:
    outputs_dir = Path("outputs")

    eval_rows: list[dict[str, float | str]] = []
    for d in _latest_dirs(outputs_dir, "eval_"):
        summary = _read_json(d / "summary.json")
        if summary is None:
            continue
        eval_rows.append(
            {
                "run": d.name,
                "n_obs": summary.get("n_obs", 0),
                "mean_log_likelihood": summary.get("mean_log_likelihood", 0.0),
                "arb_pass_rate": summary.get("sample_arb_pass_rate", 0.0),
            }
        )

    bt_rows: list[dict[str, float | str]] = []
    for d in _latest_dirs(outputs_dir / "backtests", "run_"):
        summary = _read_json(d / "summary.json")
        if summary is None:
            continue
        bt_rows.append(
            {
                "run": d.name,
                "n_days": summary.get("n_days", 0),
                "total_pnl": summary.get("total_pnl", 0.0),
                "sharpe": summary.get("sharpe", 0.0),
                "turnover_ratio": summary.get("turnover_ratio", 0.0),
            }
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Eval runs")
        if eval_rows:
            st.dataframe(
                pd.DataFrame(eval_rows).sort_values("run", ascending=False),
                use_container_width=True,
            )
        else:
            st.info("No eval runs found.")

    with col2:
        st.subheader("Backtest runs")
        if bt_rows:
            st.dataframe(
                pd.DataFrame(bt_rows).sort_values("run", ascending=False),
                use_container_width=True,
            )
        else:
            st.info("No backtest runs found.")


def _show_latest_backtest_diagnostics() -> None:
    backtest_root = Path("outputs/backtests")
    runs = list(reversed(_latest_dirs(backtest_root, "run_")))
    if not runs:
        st.info("No backtest runs found for diagnostics.")
        return

    run_names = [r.name for r in runs]
    selected = st.selectbox("Backtest run", run_names, index=0, key="bt_diag_run")
    run_dir = backtest_root / selected

    pnl_attr = _read_json(run_dir / "pnl_attribution.json")
    gate_reasons = _read_json(run_dir / "gate_reasons.json")
    exec_summary = _read_json(run_dir / "execution_summary.json")
    run_meta = _read_json(run_dir / "run_metadata.json")
    unit_sanity = _read_json(run_dir / "unit_sanity.json")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("PnL attribution")
        st.json(pnl_attr if pnl_attr is not None else {})
    with c2:
        st.write("Gate reasons")
        st.json(gate_reasons if gate_reasons is not None else {})
    with c3:
        st.write("Execution summary")
        st.json(exec_summary if exec_summary is not None else {})

    st.write("Run metadata")
    st.json(run_meta if run_meta is not None else {})
    st.write("Unit sanity")
    st.json(unit_sanity if unit_sanity is not None else {})


def main() -> None:
    st.set_page_config(page_title="SPY Surface Model UI", layout="wide")
    st.title("SPY Arbitrage-Free Surface Model")
    st.caption("Observed repaired surface vs conditional mean + diagnostics")

    with st.sidebar:
        st.header("Inputs")
        dataset_path = Path(
            st.text_input("Dataset (.npz)", value=str(DEFAULT_DATASET))
        )
        checkpoint_path = Path(
            st.text_input("Checkpoint (.pt)", value=str(DEFAULT_CHECKPOINT))
        )
        n_samples = st.slider("Samples for conditional mean", min_value=8, max_value=256, value=64)
        if st.button("Reload dataset/model from disk", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    if not dataset_path.exists():
        st.error(f"Dataset not found: {dataset_path}")
        return
    if not checkpoint_path.exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
        return

    dataset_key = _file_cache_key(dataset_path)
    checkpoint_key = _file_cache_key(checkpoint_path)
    dataset = _load_dataset(str(dataset_path), dataset_key)
    required = {"dates", "context", "surface", "x_grid", "tenors_days"}
    missing = required.difference(dataset.keys())
    if missing:
        st.error(f"Dataset missing keys: {sorted(missing)}")
        return

    dates = dataset["dates"]
    surfaces = dataset["surface"]
    contexts = dataset["context"]
    theta_raw_target = (
        dataset["theta_target_raw"] if "theta_target_raw" in dataset else dataset["theta_raw"]
    )
    theta_raw_level = (
        dataset["theta_raw_level"] if "theta_raw_level" in dataset else dataset["theta_raw"]
    )
    mode_arr = dataset["target_mode"] if "target_mode" in dataset else None
    target_mode = str(mode_arr.reshape(-1)[0]) if mode_arr is not None else "theta_raw"
    if target_mode == "delta_theta_raw":
        if "theta_raw_prev" in dataset:
            theta_base_raw = dataset["theta_raw_prev"]
        else:
            theta_base_raw = np.vstack([theta_raw_level[0], theta_raw_level[:-1]])
    else:
        theta_base_raw = None
    x_grid = dataset["x_grid"]
    tenors_days = dataset["tenors_days"]

    idx = st.slider("Date index", min_value=0, max_value=len(dates) - 1, value=len(dates) - 1)
    date_label = str(dates[idx])
    st.caption(
        f"Loaded dataset: {dataset_path} | rows={len(dates)} | "
        f"range={dates[0]} to {dates[-1]}"
    )

    with st.spinner("Loading model and computing diagnostics..."):
        model = _load_model(str(checkpoint_path), checkpoint_key)
        ctx = torch.as_tensor(contexts[idx : idx + 1], dtype=torch.float32)
        trg = torch.as_tensor(theta_raw_target[idx : idx + 1], dtype=torch.float32)
        base = (
            torch.as_tensor(theta_base_raw[idx : idx + 1], dtype=torch.float32)
            if theta_base_raw is not None
            else None
        )

        with torch.no_grad():
            if base is not None and bool(getattr(model, "supports_base_in_log_prob", False)):
                log_prob = float(model.log_prob(trg, context=ctx, base_theta_raw=base).item())
            else:
                log_prob = float(model.log_prob(trg, context=ctx).item())
            mean_surface = model.conditional_mean_surface(
                ctx,
                num_samples=n_samples,
                base_theta_raw=base,
            ).cpu().numpy()[0]
            sample_surfaces = model.sample_surfaces(
                context=ctx,
                num_samples=min(n_samples, 64),
                base_theta_raw=base,
            ).cpu().numpy()[0]

    observed = np.asarray(surfaces[idx], dtype=float)
    residual = observed - mean_surface
    dislocation = -log_prob

    obs_arb = arb_violation_counts(observed)
    mean_arb = arb_violation_counts(mean_surface)
    sample_arb_pass = float(
        np.mean([
            sum(arb_violation_counts(s).values()) == 0 for s in sample_surfaces
        ])
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Date", date_label)
    c2.metric("Log-likelihood", f"{log_prob:,.2f}")
    c3.metric("Dislocation z", f"{dislocation:,.2f}")
    c4.metric("Sample arb pass", f"{100*sample_arb_pass:.1f}%")
    st.caption(f"Model target mode: `{target_mode}`")

    c5, c6 = st.columns(2)
    with c5:
        st.write("Observed arb violations", obs_arb)
    with c6:
        st.write("Conditional-mean arb violations", mean_arb)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.pyplot(_surface_fig(observed, x_grid, tenors_days, "Observed Repaired Surface"))
    with col2:
        st.pyplot(_surface_fig(mean_surface, x_grid, tenors_days, "Conditional Mean Surface"))
    with col3:
        st.pyplot(_surface_fig(residual, x_grid, tenors_days, "Residual (Observed - Mean)"))

    observed_iv = _surface_to_iv(observed, x_grid=x_grid, tenors_days=tenors_days)
    mean_iv = _surface_to_iv(mean_surface, x_grid=x_grid, tenors_days=tenors_days)
    iv_residual = observed_iv - mean_iv

    st.subheader("3D fit surface")
    view_type = st.selectbox(
        "3D surface value",
        ["Normalized call", "Implied volatility"],
        index=0,
        key="surface_3d_value",
    )
    surface_choice = st.selectbox(
        "3D surface selection",
        ["Observed", "Conditional mean", "Residual (Observed - Mean)"],
        index=0,
        key="surface_3d_select",
    )

    if view_type == "Normalized call":
        if surface_choice == "Observed":
            surf_data = observed
        elif surface_choice == "Conditional mean":
            surf_data = mean_surface
        else:
            surf_data = residual
        z_label = "Normalized call"
    else:
        if surface_choice == "Observed":
            surf_data = observed_iv
        elif surface_choice == "Conditional mean":
            surf_data = mean_iv
        else:
            surf_data = iv_residual
        z_label = "Implied volatility"

    st.pyplot(
        _surface_fig_3d(
            surf_data,
            x_grid=x_grid,
            tenors_days=tenors_days,
            title=f"{surface_choice} ({view_type})",
            z_label=z_label,
        )
    )

    st.subheader("Cross-sections")
    cross_mode = st.selectbox(
        "Cross-section axis",
        ["Tenor slice (vary x)", "Moneyness slice (vary tenor)"],
        index=0,
        key="cross_axis_mode",
    )

    if cross_mode == "Tenor slice (vary x)":
        tenor_labels = [str(int(t)) for t in tenors_days]
        selected_tenor = st.selectbox(
            "Tenor bucket (days)",
            tenor_labels,
            index=min(2, len(tenor_labels) - 1),
            key="cross_tenor_select",
        )
        tenor_idx = tenor_labels.index(selected_tenor)
        x_vals = np.asarray(x_grid, dtype=float)
        norm_obs = observed[:, tenor_idx]
        norm_model = mean_surface[:, tenor_idx]
        iv_obs_slice = observed_iv[:, tenor_idx]
        iv_model_slice = mean_iv[:, tenor_idx]
        x_label = "Log-moneyness x"
        norm_title = f"Normalized call cross-section at tenor {selected_tenor}D"
        iv_title = f"Implied vol cross-section at tenor {selected_tenor}D"
    else:
        x_labels = [f"{float(x):.2f}" for x in x_grid]
        selected_x = st.selectbox(
            "Log-moneyness x",
            x_labels,
            index=len(x_labels) // 2,
            key="cross_x_select",
        )
        x_idx = x_labels.index(selected_x)
        x_vals = np.asarray(tenors_days, dtype=float)
        norm_obs = observed[x_idx, :]
        norm_model = mean_surface[x_idx, :]
        iv_obs_slice = observed_iv[x_idx, :]
        iv_model_slice = mean_iv[x_idx, :]
        x_label = "Tenor (days)"
        norm_title = f"Normalized call term structure at x={selected_x}"
        iv_title = f"Implied vol term structure at x={selected_x}"

    cross_col1, cross_col2 = st.columns(2)
    with cross_col1:
        st.pyplot(
            _line_fig(
                observed=norm_obs,
                mean_surface=norm_model,
                x_values=x_vals,
                x_label=x_label,
                title=norm_title,
                y_label="Normalized call",
            )
        )
    with cross_col2:
        st.pyplot(
            _line_fig(
                observed=iv_obs_slice,
                mean_surface=iv_model_slice,
                x_values=x_vals,
                x_label=x_label,
                title=iv_title,
                y_label="Implied volatility",
            )
        )

    st.subheader("Implied volatility view")
    iv_col1, iv_col2, iv_col3 = st.columns(3)
    with iv_col1:
        st.pyplot(_surface_fig(observed_iv, x_grid, tenors_days, "Observed Implied Vol Surface"))
    with iv_col2:
        st.pyplot(_surface_fig(mean_iv, x_grid, tenors_days, "Model Implied Vol Surface"))
    with iv_col3:
        st.pyplot(_surface_fig(iv_residual, x_grid, tenors_days, "IV Residual (Observed - Model)"))

    st.subheader("Run summaries")
    _show_run_tables()
    st.subheader("Backtest diagnostics")
    _show_latest_backtest_diagnostics()


if __name__ == "__main__":
    main()
