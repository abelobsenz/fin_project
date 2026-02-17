"""PyTorch training pipeline for IV dynamics architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for training. Install torch first.") from exc

from ivdyn.model import IVDynamicsTorchModel, ModelBundle, ModelConfig, device_auto, to_numpy
from ivdyn.model.scalers import ArrayScaler
from ivdyn.utils import make_run_dir


@dataclass(slots=True)
class TrainingConfig:
    out_dir: Path
    seed: int = 7
    train_frac: float = 0.70
    val_frac: float = 0.15

    latent_dim: int = 16

    vae_epochs: int = 120
    vae_batch_size: int = 32
    vae_lr: float = 2e-3
    vae_kl_beta: float = 0.02
    noarb_lambda: float = 0.2

    head_epochs: int = 100
    dyn_batch_size: int = 64
    contract_batch_size: int = 2048
    head_lr: float = 1e-3

    joint_epochs: int = 30
    joint_lr: float = 5e-4
    joint_contract_batch_size: int = 4096
    joint_dyn_lambda: float = 1.0
    joint_price_lambda: float = 1.0
    joint_exec_lambda: float = 0.25

    weight_decay: float = 1e-5
    price_risk_weight: float = 1.0
    exec_risk_weight: float = 0.5
    risk_focus_abs_x: float = 0.06
    risk_focus_tau_days: float = 20.0
    exec_label_smoothing: float = 0.03
    exec_logit_l2: float = 2e-4


def _load_dataset(dataset_path: Path) -> dict[str, np.ndarray]:
    z = np.load(dataset_path, allow_pickle=True)
    out: dict[str, np.ndarray] = {}
    for k in z.files:
        arr = z[k]
        if arr.dtype == object:
            arr = arr.astype(str)
        out[k] = arr
    return out


def _date_splits(n_dates: int, train_frac: float, val_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i_train = max(2, int(n_dates * train_frac))
    i_val = max(i_train + 1, int(n_dates * (train_frac + val_frac)))
    i_val = min(i_val, n_dates - 1)
    return np.arange(i_train), np.arange(i_train, i_val), np.arange(i_val, n_dates)


def _contract_splits(contract_date_idx: np.ndarray, tr: np.ndarray, va: np.ndarray, te: np.ndarray):
    idx = np.arange(len(contract_date_idx))
    c_tr = idx[np.isin(contract_date_idx, tr)]
    c_va = idx[np.isin(contract_date_idx, va)]
    c_te = idx[np.isin(contract_date_idx, te)]
    return c_tr, c_va, c_te


def _recon_raw_from_scaled(
    recon_scaled: torch.Tensor,
    surface_scaler: ArrayScaler,
    nx: int,
    nt: int,
) -> torch.Tensor:
    mean = torch.as_tensor(surface_scaler.mean, dtype=recon_scaled.dtype, device=recon_scaled.device)
    std = torch.as_tensor(surface_scaler.std, dtype=recon_scaled.dtype, device=recon_scaled.device)
    return (recon_scaled * std + mean).view(-1, nx, nt)


def _calendar_penalty_torch(
    recon_scaled: torch.Tensor,
    surface_scaler: ArrayScaler,
    nx: int,
    nt: int,
    tenor_days: np.ndarray,
) -> torch.Tensor:
    recon_raw = _recon_raw_from_scaled(recon_scaled, surface_scaler, nx, nt)
    tau = torch.as_tensor(tenor_days.astype(np.float32) / 365.0, dtype=recon_scaled.dtype, device=recon_scaled.device)
    tau = tau.view(1, 1, -1)

    total_var = recon_raw.pow(2) * tau
    viol = torch.relu(total_var[:, :, :-1] - total_var[:, :, 1:])
    return viol.pow(2).mean()


def _iter_batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    if len(indices) == 0:
        return []
    order = rng.permutation(indices)
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _eval_recon(model: IVDynamicsTorchModel, surface_scaled: torch.Tensor, idx: np.ndarray) -> float:
    if len(idx) == 0:
        return float("nan")
    with torch.no_grad():
        mu, _ = model.encode(surface_scaled[idx])
        recon = model.decode(mu)
        return float(F.mse_loss(recon, surface_scaled[idx]).item())


def _contract_risk_focus_weights(
    *,
    features: np.ndarray,
    feature_names: list[str],
    price_risk_weight: float,
    exec_risk_weight: float,
    risk_focus_abs_x: float,
    risk_focus_tau_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(features)
    price_w = np.ones(n, dtype=np.float32)
    exec_w = np.ones(n, dtype=np.float32)

    req = {"x", "tau", "cp_sign"}
    if not req.issubset(set(feature_names)):
        return price_w, exec_w

    ix_x = feature_names.index("x")
    ix_tau = feature_names.index("tau")
    ix_cp = feature_names.index("cp_sign")

    abs_x = np.abs(features[:, ix_x].astype(np.float64))
    tau = np.clip(features[:, ix_tau].astype(np.float64), 1e-6, None)
    cp_sign = features[:, ix_cp].astype(np.float64)

    x_scale = max(float(risk_focus_abs_x), 1e-4)
    tau_scale = max(float(risk_focus_tau_days) / 365.0, 1e-6)

    near_atm = np.exp(-np.square(abs_x / x_scale))
    short_tenor = np.exp(-np.square(tau / tau_scale))
    is_put = (cp_sign < 0.0).astype(np.float64)
    focus = is_put * near_atm * short_tenor

    price_w += np.clip(float(price_risk_weight), 0.0, None) * focus
    exec_w += np.clip(float(exec_risk_weight), 0.0, None) * focus

    return price_w.astype(np.float32), exec_w.astype(np.float32)


def _weighted_smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    beta: float = 0.02,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def _smooth_binary_targets(target: torch.Tensor, smoothing: float) -> torch.Tensor:
    s = float(np.clip(smoothing, 0.0, 0.25))
    if s <= 0.0:
        return target
    return target * (1.0 - s) + 0.5 * s


def _weighted_bce_with_logits(
    pred_logit: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(pred_logit, target, reduction="none")
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def train(dataset_path: Path, cfg: TrainingConfig) -> Path:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    ds = _load_dataset(dataset_path)

    dates = ds["dates"].astype(str)
    iv_surface = ds["iv_surface"].astype(np.float32)
    context = ds["context"].astype(np.float32)
    features = ds["contract_features"].astype(np.float32)
    feature_names = ds.get("contract_feature_names", np.array([], dtype=str)).astype(str).tolist()
    price_target = ds["contract_price_target"].astype(np.float32)
    fill_target = ds["contract_fill_target"].astype(np.float32)
    date_idx = ds["contract_date_index"].astype(np.int32)
    tenor_days = ds["tenor_days"].astype(np.int32)

    n_dates, nx, nt = iv_surface.shape
    surface_flat = iv_surface.reshape(n_dates, nx * nt)

    tr_dates, va_dates, te_dates = _date_splits(n_dates, cfg.train_frac, cfg.val_frac)
    c_tr, c_va, c_te = _contract_splits(date_idx, tr_dates, va_dates, te_dates)

    surface_scaler = ArrayScaler.fit(surface_flat[tr_dates])
    context_scaler = ArrayScaler.fit(context[tr_dates])
    contract_scaler = ArrayScaler.fit(features[c_tr])
    price_scaler = ArrayScaler.fit(price_target[c_tr].reshape(-1, 1))

    surface_scaled_np = surface_scaler.transform(surface_flat)
    context_scaled_np = context_scaler.transform(context)
    feature_scaled_np = contract_scaler.transform(features)
    price_scaled_np = price_scaler.transform(price_target.reshape(-1, 1)).reshape(-1)

    price_weight_np, exec_weight_np = _contract_risk_focus_weights(
        features=features,
        feature_names=feature_names,
        price_risk_weight=cfg.price_risk_weight,
        exec_risk_weight=cfg.exec_risk_weight,
        risk_focus_abs_x=cfg.risk_focus_abs_x,
        risk_focus_tau_days=cfg.risk_focus_tau_days,
    )

    device = device_auto()

    surface_scaled = _to_tensor(surface_scaled_np, device)
    context_scaled = _to_tensor(context_scaled_np, device)
    feature_scaled = _to_tensor(feature_scaled_np, device)
    price_scaled = _to_tensor(price_scaled_np.reshape(-1, 1), device)
    fill_t = _to_tensor(fill_target.reshape(-1, 1), device)
    price_w_t = _to_tensor(price_weight_np.reshape(-1, 1), device)
    exec_w_t = _to_tensor(exec_weight_np.reshape(-1, 1), device)

    model_cfg = ModelConfig(latent_dim=cfg.latent_dim)
    model = IVDynamicsTorchModel(
        surface_dim=surface_scaled.shape[1],
        context_dim=context_scaled.shape[1],
        contract_dim=feature_scaled.shape[1],
        config=model_cfg,
    ).to(device)

    hist_rows: list[dict[str, object]] = []

    # -------------------------- Stage 1: VAE pretrain --------------------------
    vae_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    opt_vae = torch.optim.AdamW(vae_params, lr=cfg.vae_lr, weight_decay=cfg.weight_decay)

    for epoch in range(1, cfg.vae_epochs + 1):
        model.train()
        losses = []
        recon_losses = []
        kl_losses = []
        cal_losses = []

        for batch in _iter_batches(tr_dates, cfg.vae_batch_size, rng):
            x = surface_scaled[batch]
            mu, logvar = model.encode(x)
            z = model.reparameterize(mu, logvar)
            recon = model.decode(z)

            recon_loss = F.mse_loss(recon, x)
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            cal = _calendar_penalty_torch(recon, surface_scaler, nx, nt, tenor_days)
            loss = recon_loss + cfg.vae_kl_beta * kl + cfg.noarb_lambda * cal

            opt_vae.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae_params, 2.5)
            opt_vae.step()

            losses.append(float(loss.item()))
            recon_losses.append(float(recon_loss.item()))
            kl_losses.append(float(kl.item()))
            cal_losses.append(float(cal.item()))

        model.eval()
        val_recon = _eval_recon(model, surface_scaled, va_dates)

        hist_rows.append(
            {
                "stage": "vae",
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "recon_loss": float(np.mean(recon_losses)),
                "kl_loss": float(np.mean(kl_losses)),
                "calendar_loss": float(np.mean(cal_losses)),
                "val_recon_loss": float(val_recon),
            }
        )

    # ---------------------- Stage 2: heads on frozen latent --------------------
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = False

    head_params = list(model.dynamics.parameters()) + list(model.pricer.parameters()) + list(model.execution.parameters())
    opt_head = torch.optim.AdamW(head_params, lr=cfg.head_lr, weight_decay=cfg.weight_decay)

    model.eval()
    with torch.no_grad():
        z_all, _ = model.encode(surface_scaled)

    dyn_train_idx = tr_dates[1:]

    for epoch in range(1, cfg.head_epochs + 1):
        model.train()
        dyn_losses = []
        price_losses = []
        exec_losses = []

        for batch in _iter_batches(dyn_train_idx, cfg.dyn_batch_size, rng):
            z_prev = z_all[batch - 1]
            ctx = context_scaled[batch]
            z_t = z_all[batch]

            pred = model.forward_dynamics(z_prev, ctx)
            dyn_loss = F.mse_loss(pred, z_t)

            opt_head.zero_grad(set_to_none=True)
            dyn_loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 2.0)
            opt_head.step()

            dyn_losses.append(float(dyn_loss.item()))

        for batch in _iter_batches(c_tr, cfg.contract_batch_size, rng):
            z = z_all[date_idx[batch]]
            feat = feature_scaled[batch]
            p_t = price_scaled[batch]
            e_t = fill_t[batch]
            p_w = price_w_t[batch]
            e_w = exec_w_t[batch]

            p_pred = model.forward_pricer(z, feat)
            p_loss = _weighted_smooth_l1_loss(p_pred, p_t, p_w, beta=0.02)

            e_pred = model.forward_execution_logit(z, feat)
            e_t_s = _smooth_binary_targets(e_t, cfg.exec_label_smoothing)
            e_loss = _weighted_bce_with_logits(e_pred, e_t_s, e_w) + cfg.exec_logit_l2 * torch.mean(
                e_pred.pow(2)
            )

            loss = p_loss + cfg.joint_exec_lambda * e_loss

            opt_head.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 2.0)
            opt_head.step()

            price_losses.append(float(p_loss.item()))
            exec_losses.append(float(e_loss.item()))

        model.eval()
        with torch.no_grad():
            val_dyn = np.nan
            if len(va_dates) > 1:
                idx = va_dates[1:]
                pred = model.forward_dynamics(z_all[idx - 1], context_scaled[idx])
                val_dyn = float(F.mse_loss(pred, z_all[idx]).item())

            val_price = np.nan
            val_exec = np.nan
            if len(c_va) > 0:
                zv = z_all[date_idx[c_va]]
                fv = feature_scaled[c_va]
                pv = price_scaled[c_va]
                ev = fill_t[c_va]
                val_price = float(F.mse_loss(model.forward_pricer(zv, fv), pv).item())
                val_exec = float(F.binary_cross_entropy_with_logits(model.forward_execution_logit(zv, fv), ev).item())

        hist_rows.append(
            {
                "stage": "heads",
                "epoch": epoch,
                "dyn_loss": float(np.mean(dyn_losses) if dyn_losses else np.nan),
                "price_loss": float(np.mean(price_losses) if price_losses else np.nan),
                "exec_loss": float(np.mean(exec_losses) if exec_losses else np.nan),
                "val_dyn_loss": float(val_dyn),
                "val_price_loss": float(val_price),
                "val_exec_loss": float(val_exec),
            }
        )

    # ------------------------- Stage 3: joint fine-tune ------------------------
    for p in model.encoder.parameters():
        p.requires_grad = True
    for p in model.decoder.parameters():
        p.requires_grad = True

    joint_params = list(model.parameters())
    opt_joint = torch.optim.AdamW(joint_params, lr=cfg.joint_lr, weight_decay=cfg.weight_decay)

    local_idx_map = np.full(n_dates, -1, dtype=np.int32)
    local_idx_map[tr_dates] = np.arange(len(tr_dates), dtype=np.int32)

    for epoch in range(1, cfg.joint_epochs + 1):
        model.train()

        x = surface_scaled[tr_dates]
        ctx = context_scaled[tr_dates]

        mu, logvar = model.encode(x)
        z = mu
        recon = model.decode(z)

        recon_loss = F.mse_loss(recon, x)
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        cal = _calendar_penalty_torch(recon, surface_scaler, nx, nt, tenor_days)

        dyn_loss = torch.tensor(0.0, device=device)
        if len(tr_dates) > 1:
            dyn_pred = model.forward_dynamics(z[:-1], ctx[1:])
            dyn_loss = F.mse_loss(dyn_pred, z[1:])

        p_loss = torch.tensor(0.0, device=device)
        e_loss = torch.tensor(0.0, device=device)
        if len(c_tr) > 0:
            pick_size = min(len(c_tr), cfg.joint_contract_batch_size)
            pick = rng.choice(c_tr, size=pick_size, replace=False)
            local = local_idx_map[date_idx[pick]]
            local_t = torch.as_tensor(local, dtype=torch.long, device=device)

            zc = z[local_t]
            fc = feature_scaled[pick]
            pt = price_scaled[pick]
            et = fill_t[pick]
            pw = price_w_t[pick]
            ew = exec_w_t[pick]

            p_pred = model.forward_pricer(zc, fc)
            p_loss = _weighted_smooth_l1_loss(p_pred, pt, pw, beta=0.02)

            e_pred = model.forward_execution_logit(zc, fc)
            et_s = _smooth_binary_targets(et, cfg.exec_label_smoothing)
            e_loss = _weighted_bce_with_logits(e_pred, et_s, ew) + cfg.exec_logit_l2 * torch.mean(
                e_pred.pow(2)
            )

        loss = (
            recon_loss
            + cfg.vae_kl_beta * kl
            + cfg.noarb_lambda * cal
            + cfg.joint_dyn_lambda * dyn_loss
            + cfg.joint_price_lambda * p_loss
            + cfg.joint_exec_lambda * e_loss
        )

        opt_joint.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(joint_params, 2.5)
        opt_joint.step()

        model.eval()
        with torch.no_grad():
            val_recon = _eval_recon(model, surface_scaled, va_dates)

        hist_rows.append(
            {
                "stage": "joint",
                "epoch": epoch,
                "loss": float(loss.item()),
                "recon_loss": float(recon_loss.item()),
                "kl_loss": float(kl.item()),
                "calendar_loss": float(cal.item()),
                "dyn_loss": float(dyn_loss.item()),
                "price_loss": float(p_loss.item()),
                "exec_loss": float(e_loss.item()),
                "val_recon_loss": float(val_recon),
            }
        )

    # Save artifacts
    run_dir = make_run_dir(cfg.out_dir, prefix="run")

    bundle = ModelBundle(
        model=model.eval().cpu(),
        surface_scaler=surface_scaler,
        context_scaler=context_scaler,
        contract_scaler=contract_scaler,
        price_scaler=price_scaler,
    )
    model_path = run_dir / "model.pt"
    bundle.save(model_path)

    hist = pd.DataFrame(hist_rows)
    hist.to_csv(run_dir / "train_history.csv", index=False)

    with torch.no_grad():
        model_cpu = bundle.model
        sf = torch.as_tensor(surface_scaled_np, dtype=torch.float32)
        mu_all, _ = model_cpu.encode(sf)
        latent = to_numpy(mu_all)

    latent_df = pd.DataFrame(latent, columns=[f"z_{i}" for i in range(latent.shape[1])])
    latent_df.insert(0, "date", dates)
    latent_df.to_parquet(run_dir / "latent_states.parquet", index=False)

    split_info = {
        "train_date_idx": tr_dates.tolist(),
        "val_date_idx": va_dates.tolist(),
        "test_date_idx": te_dates.tolist(),
        "train_dates": [dates[i] for i in tr_dates],
        "val_dates": [dates[i] for i in va_dates],
        "test_dates": [dates[i] for i in te_dates],
        "n_contracts_train": int(len(c_tr)),
        "n_contracts_val": int(len(c_va)),
        "n_contracts_test": int(len(c_te)),
    }
    (run_dir / "split_info.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    cfg_payload = asdict(cfg)
    cfg_payload["out_dir"] = str(cfg.out_dir)
    cfg_payload["device"] = str(device)
    (run_dir / "train_config.json").write_text(json.dumps(cfg_payload, indent=2), encoding="utf-8")

    summary = {
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "device": str(device),
        "final_val_recon": float(hist[hist["stage"] == "joint"]["val_recon_loss"].iloc[-1]) if (hist["stage"] == "joint").any() else None,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return run_dir
