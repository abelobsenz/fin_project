from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from spygen.models.flow import ConditionalSurfaceFlow
from spygen.utils.paths import ensure_dir


@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-6
    hidden_size: int = 128
    flow_layers: int = 4
    early_stopping_patience: int = 5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def train_flow_model(
    dataset_path: str | Path,
    output_dir: str | Path,
    config: TrainConfig,
    nx: int,
    nt: int,
) -> Path:
    set_seed(config.seed)
    ds = load_dataset_npz(dataset_path)
    context = ds["context"].astype(np.float32)
    theta_raw = ds["theta_raw"].astype(np.float32)

    n = context.shape[0]
    if n < 10:
        raise ValueError("Need at least 10 samples to train the flow")

    split = max(1, int(n * 0.8))
    x_train, x_val = context[:split], context[split:]
    y_train, y_val = theta_raw[:split], theta_raw[split:]

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=min(config.batch_size, len(x_train)),
        shuffle=True,
    )
    val_dl = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=min(config.batch_size, max(1, len(x_val))),
        shuffle=False,
    )

    model = ConditionalSurfaceFlow(
        theta_dim=theta_raw.shape[1],
        context_dim=context.shape[1],
        nx=nx,
        nt=nt,
        hidden_features=config.hidden_size,
        num_layers=config.flow_layers,
    )
    context_mean = x_train.mean(axis=0)
    context_std = x_train.std(axis=0)
    context_std = np.maximum(context_std, 1e-4)
    theta_mean = y_train.mean(axis=0)
    theta_std = y_train.std(axis=0)
    theta_std = np.maximum(theta_std, 1e-4)
    model.set_normalization_stats(
        context_mean=context_mean,
        context_std=context_std,
        theta_mean=theta_mean,
        theta_std=theta_std,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_val = float("inf")
    best_state = None
    patience = 0

    for _epoch in range(config.epochs):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad(set_to_none=True)
            loss = -model.log_prob(yb, context=xb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in val_dl:
                val_losses.append(float((-model.log_prob(yb, context=xb).mean()).item()))
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir = ensure_dir(output_dir)
    ckpt_path = out_dir / "flow_latest.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "theta_dim": theta_raw.shape[1],
            "context_dim": context.shape[1],
            "nx": nx,
            "nt": nt,
            "train_config": asdict(config),
            "best_val_nll": best_val,
            "context_mean": torch.from_numpy(context_mean),
            "context_std": torch.from_numpy(context_std),
            "theta_mean": torch.from_numpy(theta_mean),
            "theta_std": torch.from_numpy(theta_std),
        },
        ckpt_path,
    )
    metrics_path = out_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps({"best_val_nll": best_val}, indent=2))
    return ckpt_path
