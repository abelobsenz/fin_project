from __future__ import annotations

from pathlib import Path

from spygen.pipeline import (
    backtest_from_config,
    build_dataset_range,
    eval_checkpoint,
    synth_data_range,
    train_from_config,
)


def test_end_to_end_synthetic_pipeline(tmp_path: Path) -> None:
    config = {
        "paths": {
            "raw_dir": str(tmp_path / "data" / "raw"),
            "processed_dir": str(tmp_path / "data" / "processed"),
            "underlying_path": str(tmp_path / "data" / "underlying" / "spy_eod.parquet"),
            "outputs_dir": str(tmp_path / "outputs"),
        },
        "surface": {"x_min": -0.3, "x_max": 0.3, "nx": 15, "tenors_days": [7, 14, 30, 60, 90, 180]},
        "repair": {"lambda_smooth": 1e-3, "data_weight": 1.0},
        "train": {
            "seed": 7,
            "batch_size": 16,
            "epochs": 1,
            "lr": 1e-3,
            "weight_decay": 1e-6,
            "hidden_size": 32,
            "flow_layers": 1,
            "early_stopping_patience": 1,
        },
        "strategy": {
            "threshold": 0.0,
            "n_samples": 4,
            "slippage_bps": 3.0,
            "max_contracts": 5,
            "max_notional": 10000,
            "max_spread": 10.0,
        },
        "synth": {"seed": 7, "bad_quote_prob": 0.03, "strike_points": 15},
    }

    start = "2024-01-02"
    end = "2024-01-26"
    synth_data_range(start=start, end=end, config=config)
    ds_path = build_dataset_range(start=start, end=end, config=config)
    assert ds_path.exists()

    ckpt = train_from_config(config, dataset_path=ds_path)
    assert ckpt.exists()

    eval_dir = eval_checkpoint(ckpt, config, dataset_path=ds_path, n_samples=4)
    assert (eval_dir / "summary.json").exists()

    bt_dir = backtest_from_config(ckpt, config, dataset_path=ds_path)
    assert (bt_dir / "summary.json").exists()
