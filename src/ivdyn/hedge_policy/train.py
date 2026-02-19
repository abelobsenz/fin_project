"""Training loop for a learned underlying hedge policy.

This is intentionally *not* full RL: the current ivdyn backtest holds option
positions from close-to-close (one day), and the underlying hedge is also
close-to-close. That makes hedging a per-day decision that can be learned as a
"contextual" policy from historical episodes.

We train a small MLP that maps (latent IV surface state + context + net option
delta + spot) -> hedge ratio.

Objective:
    maximize mean daily PnL - risk_aversion * std(daily PnL)

where daily PnL is computed using the historical next-day underlying move and
the option PnL from the backtest, with realistic underlying slippage costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for hedge policy training.") from exc

from ivdyn.hedge_policy.policy import HedgeFeatureSpec, HedgePolicyBundle, HedgeRatioMLP, build_hedge_features
from ivdyn.model import ModelBundle, device_auto, to_numpy


@dataclass(slots=True)
class HedgePolicyTrainConfig:
    run_dir: Path
    dataset_path: Path
    out_dir: Path | None = None
    device: str | None = None

    # Policy network.
    hidden_dim: int = 64
    depth: int = 2
    max_ratio: float = 1.25

    # Training.
    epochs: int = 400
    lr: float = 3e-3
    weight_decay: float = 1e-4
    train_frac: float = 0.7
    seed: int = 7

    # Objective.
    risk_aversion: float = 0.50
    # Underlying execution.
    underlying_slippage_bps: float = 1.0
    min_abs_shares: float = 20.0
    max_shares: float = 200.0
    smooth_deadzone_temp: float = 5.0

    feature_spec: HedgeFeatureSpec = HedgeFeatureSpec()


def _load_dataset_npz(path: Path) -> dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=True)
    out: dict[str, np.ndarray] = {}
    for k in npz.files:
        arr = npz[k]
        if arr.dtype == object:
            arr = arr.astype(str)
        out[k] = arr
    return out


def _read_backtest_table(bt_dir: Path, stem: str) -> pd.DataFrame:
    p = bt_dir / f"{stem}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    c = bt_dir / f"{stem}.csv"
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(f"Missing backtest artifact: {p} (or {c})")


def _compute_latents(bundle: ModelBundle, ds: dict[str, np.ndarray], dev: torch.device) -> np.ndarray:
    model = bundle.model.to(dev).eval()
    iv_surface = ds["iv_surface"].astype(np.float32)
    n_dates = iv_surface.shape[0]
    surface_flat = iv_surface.reshape(n_dates, -1)
    surface_scaled = bundle.surface_scaler.transform(surface_flat)
    with torch.no_grad():
        sf = torch.as_tensor(surface_scaled, dtype=torch.float32, device=dev)
        mu, _ = model.encode(sf)
        z_now = to_numpy(mu)
    return np.asarray(z_now, dtype=np.float32)


def _smooth_abs(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(x * x + eps)


def _soft_deadzone(x: torch.Tensor, *, min_abs: float, temp: float) -> torch.Tensor:
    """Smooth approximation of: x if |x|>=min_abs else 0."""
    if min_abs <= 0:
        return x
    t = float(max(temp, 1e-6))
    gate = torch.sigmoid((_smooth_abs(x) - float(min_abs)) / t)
    return x * gate


def _hedge_pnl_differentiable(
    *,
    spot_now: torch.Tensor,
    spot_next: torch.Tensor,
    shares: torch.Tensor,
    slippage_bps: float,
) -> torch.Tensor:
    """Differentiable approximation to underlying close-to-close PnL with costs.

    PnL ~= shares*(spot_next-spot_now) - 2*slip*spot_now*|shares|
    """
    slip = float(slippage_bps) / 1e4
    gross = shares * (spot_next - spot_now)
    tc = 2.0 * slip * spot_now * _smooth_abs(shares)
    return gross - tc


def _pnl_objective(pnl: torch.Tensor, risk_aversion: float) -> torch.Tensor:
    # Maximize mean - risk_aversion * std.
    mean = pnl.mean()
    std = pnl.std(unbiased=False)
    return mean - float(risk_aversion) * std


def train_hedge_policy(cfg: HedgePolicyTrainConfig) -> Path:
    run_dir = Path(cfg.run_dir).resolve()
    bt_dir = run_dir / "backtest"
    if not bt_dir.exists():
        raise RuntimeError(
            f"No backtest directory found at {bt_dir}. Run `ivdyn backtest ...` first to create trade episodes."
        )

    out_dir = (Path(cfg.out_dir) if cfg.out_dir else (run_dir / "hedge_policy" / _utc_tag())).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = _load_dataset_npz(Path(cfg.dataset_path).resolve())
    dates = ds["dates"].astype(str)
    spot_by_date = np.asarray(ds.get("spot"), dtype=np.float32).reshape(-1)
    context = np.asarray(ds.get("context"), dtype=np.float32)

    daily = _read_backtest_table(bt_dir, "daily")
    hedges = _read_backtest_table(bt_dir, "hedges")
    hedge_by_day = hedges.groupby("date", as_index=False).agg(net_option_delta_shares=("net_option_delta_shares", "sum"))

    # Join to a full per-date table (dates[:-1] are tradable days).
    day_df = pd.DataFrame({"date": dates[:-1]})
    day_df = day_df.merge(daily[["date", "options_pnl"]], on="date", how="left")
    day_df = day_df.merge(hedge_by_day, on="date", how="left")
    day_df["options_pnl"] = pd.to_numeric(day_df.get("options_pnl"), errors="coerce").fillna(0.0)
    day_df["net_option_delta_shares"] = pd.to_numeric(day_df.get("net_option_delta_shares"), errors="coerce").fillna(0.0)

    # Map date -> date_idx.
    date_to_idx = {str(d): i for i, d in enumerate(dates)}
    day_df["date_idx"] = day_df["date"].map(date_to_idx).astype(int)
    day_df = day_df.sort_values("date_idx").reset_index(drop=True)

    dev = torch.device(cfg.device) if cfg.device else device_auto()
    bundle = ModelBundle.load(run_dir / "model.pt", device=dev)
    z_now = _compute_latents(bundle, ds, dev)

    # Build features.
    feats: list[np.ndarray] = []
    for _, row in day_df.iterrows():
        d = int(row["date_idx"])
        spot = float(spot_by_date[d]) if d < len(spot_by_date) else float("nan")
        ctx = context[d] if d < len(context) else None
        zz = z_now[d] if d < len(z_now) else None
        feats.append(
            build_hedge_features(
                latent_z=zz,
                context=ctx,
                net_option_delta_shares=float(row["net_option_delta_shares"]),
                spot=spot,
                spec=cfg.feature_spec,
            )
        )

    X = np.stack(feats, axis=0).astype(np.float32)
    y_opt_pnl = day_df["options_pnl"].to_numpy(dtype=np.float32)
    delta_shares = day_df["net_option_delta_shares"].to_numpy(dtype=np.float32)
    spot_now = spot_by_date[day_df["date_idx"].to_numpy(dtype=int)].astype(np.float32)
    spot_next = spot_by_date[(day_df["date_idx"].to_numpy(dtype=int) + 1)].astype(np.float32)

    # Normalize features.
    feat_mean = X.mean(axis=0)
    feat_std = X.std(axis=0)
    feat_std = np.where(feat_std > 1e-12, feat_std, 1.0).astype(np.float32)
    Xn = (X - feat_mean) / feat_std

    # Split by time.
    n = Xn.shape[0]
    n_train = int(max(1, min(n - 1, round(float(cfg.train_frac) * n))))
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n)

    g = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    model = HedgeRatioMLP(
        input_dim=int(Xn.shape[1]),
        hidden_dim=int(cfg.hidden_dim),
        depth=int(cfg.depth),
        max_ratio=float(cfg.max_ratio),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    X_t = torch.as_tensor(Xn, dtype=torch.float32, device=dev)
    pnl_opt_t = torch.as_tensor(y_opt_pnl, dtype=torch.float32, device=dev)
    delta_t = torch.as_tensor(delta_shares, dtype=torch.float32, device=dev)
    spot_now_t = torch.as_tensor(spot_now, dtype=torch.float32, device=dev)
    spot_next_t = torch.as_tensor(spot_next, dtype=torch.float32, device=dev)

    history: list[dict[str, Any]] = []

    def eval_fixed_ratio(r: float, idx: np.ndarray) -> tuple[float, float, float]:
        rr = float(r)
        d = delta_t[idx]
        sh = -rr * d
        sh = torch.clamp(sh, -float(cfg.max_shares), float(cfg.max_shares))
        sh = _soft_deadzone(sh, min_abs=float(cfg.min_abs_shares), temp=float(cfg.smooth_deadzone_temp))
        hedge_pnl = _hedge_pnl_differentiable(spot_now=spot_now_t[idx], spot_next=spot_next_t[idx], shares=sh, slippage_bps=float(cfg.underlying_slippage_bps))
        total = pnl_opt_t[idx] + hedge_pnl
        mean = float(total.mean().detach().cpu())
        std = float(total.std(unbiased=False).detach().cpu())
        obj = float(_pnl_objective(total, float(cfg.risk_aversion)).detach().cpu())
        return mean, std, obj

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        r = model(X_t)
        # Convert to shares.
        sh = -r * delta_t
        sh = torch.clamp(sh, -float(cfg.max_shares), float(cfg.max_shares))
        sh = _soft_deadzone(sh, min_abs=float(cfg.min_abs_shares), temp=float(cfg.smooth_deadzone_temp))

        hedge_pnl = _hedge_pnl_differentiable(
            spot_now=spot_now_t,
            spot_next=spot_next_t,
            shares=sh,
            slippage_bps=float(cfg.underlying_slippage_bps),
        )
        total_pnl = pnl_opt_t + hedge_pnl

        train_pnl = total_pnl[train_idx]
        val_pnl = total_pnl[val_idx] if val_idx.size else None

        objective = _pnl_objective(train_pnl, float(cfg.risk_aversion))
        loss = -objective
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == int(cfg.epochs):
            model.eval()
            with torch.no_grad():
                r_eval = model(X_t)
                sh_eval = -r_eval * delta_t
                sh_eval = torch.clamp(sh_eval, -float(cfg.max_shares), float(cfg.max_shares))
                sh_eval = _soft_deadzone(sh_eval, min_abs=float(cfg.min_abs_shares), temp=float(cfg.smooth_deadzone_temp))
                hedge_eval = _hedge_pnl_differentiable(spot_now=spot_now_t, spot_next=spot_next_t, shares=sh_eval, slippage_bps=float(cfg.underlying_slippage_bps))
                total_eval = pnl_opt_t + hedge_eval
                tr = total_eval[train_idx]
                va = total_eval[val_idx] if val_idx.size else None

                rec: dict[str, Any] = {
                    "epoch": int(epoch),
                    "train_mean_pnl": float(tr.mean().cpu()),
                    "train_std_pnl": float(tr.std(unbiased=False).cpu()),
                    "train_objective": float(_pnl_objective(tr, float(cfg.risk_aversion)).cpu()),
                    "avg_ratio": float(r_eval.mean().cpu()),
                    "avg_abs_shares": float(_smooth_abs(sh_eval).mean().cpu()),
                }
                if va is not None:
                    rec.update(
                        {
                            "val_mean_pnl": float(va.mean().cpu()),
                            "val_std_pnl": float(va.std(unbiased=False).cpu()),
                            "val_objective": float(_pnl_objective(va, float(cfg.risk_aversion)).cpu()),
                        }
                    )
                history.append(rec)

    # Baselines.
    baselines = {}
    for name, r0 in {"ratio_0.0": 0.0, "ratio_0.85": 0.85, "ratio_1.0": 1.0}.items():
        baselines[name] = {
            "train": dict(zip(["mean_pnl", "std_pnl", "objective"], eval_fixed_ratio(r0, train_idx))),
            "val": dict(zip(["mean_pnl", "std_pnl", "objective"], eval_fixed_ratio(r0, val_idx))) if val_idx.size else None,
        }

    # Final learned evaluation.
    model.eval()
    with torch.no_grad():
        r_final = model(X_t)
        sh_final = -r_final * delta_t
        sh_final = torch.clamp(sh_final, -float(cfg.max_shares), float(cfg.max_shares))
        sh_final = _soft_deadzone(sh_final, min_abs=float(cfg.min_abs_shares), temp=float(cfg.smooth_deadzone_temp))
        hedge_final = _hedge_pnl_differentiable(spot_now=spot_now_t, spot_next=spot_next_t, shares=sh_final, slippage_bps=float(cfg.underlying_slippage_bps))
        total_final = pnl_opt_t + hedge_final
        learned = {
            "train": {
                "mean_pnl": float(total_final[train_idx].mean().cpu()),
                "std_pnl": float(total_final[train_idx].std(unbiased=False).cpu()),
                "objective": float(_pnl_objective(total_final[train_idx], float(cfg.risk_aversion)).cpu()),
            },
            "val": {
                "mean_pnl": float(total_final[val_idx].mean().cpu()),
                "std_pnl": float(total_final[val_idx].std(unbiased=False).cpu()),
                "objective": float(_pnl_objective(total_final[val_idx], float(cfg.risk_aversion)).cpu()),
            }
            if val_idx.size
            else None,
        }

    # Save bundle.
    bundle_out = HedgePolicyBundle(
        model=model.to("cpu"),
        feature_spec=cfg.feature_spec,
        feature_mean=np.asarray(feat_mean, dtype=np.float32),
        feature_std=np.asarray(feat_std, dtype=np.float32),
    )
    policy_path = out_dir / "hedge_policy.pt"
    bundle_out.save(policy_path)

    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_dir": run_dir.as_posix(),
        "dataset_path": str(Path(cfg.dataset_path).resolve()),
        "objective": {
            "type": "mean_minus_risk_std",
            "risk_aversion": float(cfg.risk_aversion),
        },
        "execution": {
            "underlying_slippage_bps": float(cfg.underlying_slippage_bps),
            "min_abs_shares": float(cfg.min_abs_shares),
            "max_shares": float(cfg.max_shares),
        },
        "feature_spec": {
            "use_latent": bool(cfg.feature_spec.use_latent),
            "use_context": bool(cfg.feature_spec.use_context),
            "use_net_delta": bool(cfg.feature_spec.use_net_delta),
            "use_spot": bool(cfg.feature_spec.use_spot),
        },
        "baselines": baselines,
        "learned": learned,
        "policy_path": policy_path.as_posix(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "README.txt").write_text(
        """Learned hedge policy trained by ivdyn.

To use this policy in a backtest:

  ivdyn backtest --run-dir <RUN_DIR> --dataset <DATASET> \\
    --hedge-policy learned --hedge-policy-path hedge_policy/</...>/hedge_policy.pt
""",
        encoding="utf-8",
    )

    return out_dir


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
