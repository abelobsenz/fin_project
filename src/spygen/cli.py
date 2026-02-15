from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from spygen.pipeline import (
    backtest_from_config,
    build_dataset_range,
    eval_checkpoint,
    fetch_underlying_range,
    load_config,
    run_sanity,
    synth_data_range,
    train_from_config,
    walkforward_from_config,
)
from spygen.utils.logging import configure_logging

app = typer.Typer(help="SPY arbitrage-free surface generative modeling CLI")
DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


@app.command("fetch-underlying")
def fetch_underlying_cmd(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    end: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    symbol: Annotated[str, typer.Option(help="Underlying ticker")] = "SPY",
    config: Annotated[Path, typer.Option(help="Config path")] = DEFAULT_CONFIG_PATH,
) -> None:
    configure_logging()
    cfg = load_config(config)
    try:
        out = fetch_underlying_range(start=start, end=end, symbol=symbol, config=cfg)
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Saved underlying parquet: {out}")


@app.command("synth-data")
def synth_data_cmd(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    end: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    config: Annotated[Path, typer.Option(help="Config path")] = DEFAULT_CONFIG_PATH,
) -> None:
    configure_logging()
    cfg = load_config(config)
    synth_data_range(start=start, end=end, config=cfg)
    typer.echo(f"Synthetic chains written under {cfg['paths']['raw_dir']}")


@app.command("build-dataset")
def build_dataset_cmd(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    end: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    config: Annotated[Path, typer.Option(help="Config path")] = DEFAULT_CONFIG_PATH,
) -> None:
    configure_logging()
    cfg = load_config(config)
    ds = build_dataset_range(start=start, end=end, config=cfg)
    typer.echo(f"Dataset written: {ds}")


@app.command("train")
def train_cmd(
    config: Annotated[Path, typer.Option(help="Config path")] = DEFAULT_CONFIG_PATH,
) -> None:
    configure_logging()
    cfg = load_config(config)
    ckpt = train_from_config(cfg)
    typer.echo(f"Checkpoint written: {ckpt}")


@app.command("eval")
def eval_cmd(
    checkpoint: Annotated[Path, typer.Option(exists=True, help="Checkpoint path")],
    config: Annotated[Path, typer.Option(help="Config path")] = DEFAULT_CONFIG_PATH,
) -> None:
    configure_logging()
    cfg = load_config(config)
    out = eval_checkpoint(checkpoint_path=checkpoint, config=cfg)
    typer.echo(f"Evaluation outputs: {out}")


@app.command("backtest")
def backtest_cmd(
    checkpoint: Annotated[Path, typer.Option(exists=True, help="Checkpoint path")],
    config: Annotated[Path, typer.Option(help="Config path")] = DEFAULT_CONFIG_PATH,
) -> None:
    configure_logging()
    cfg = load_config(config)
    out = backtest_from_config(checkpoint_path=checkpoint, config=cfg)
    typer.echo(f"Backtest outputs: {out}")


@app.command("sanity")
def sanity_cmd(
    config: Annotated[Path, typer.Option(help="Config path")] = DEFAULT_CONFIG_PATH,
) -> None:
    configure_logging()
    cfg = load_config(config)
    outputs = run_sanity(cfg)
    for key, value in outputs.items():
        typer.echo(f"{key}: {value}")


@app.command("walkforward")
def walkforward_cmd(
    config: Annotated[Path, typer.Option(help="Config path")] = DEFAULT_CONFIG_PATH,
) -> None:
    configure_logging()
    cfg = load_config(config)
    out = walkforward_from_config(cfg)
    typer.echo(f"Walk-forward outputs: {out}")


def main() -> None:
    app()
