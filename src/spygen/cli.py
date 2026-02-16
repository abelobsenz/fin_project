from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from spygen.pipeline import (
    backtest_from_config,
    build_dataset_range,
    clear_market_data,
    collect_eod_chains_asof,
    collect_eod_chains_asof_massive,
    collect_eod_chains_asof_massive_flatfile,
    eval_checkpoint,
    fetch_market_data_range_massive,
    fetch_underlying_range,
    fetch_underlying_range_massive,
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


@app.command("fetch-underlying-massive")
def fetch_underlying_massive_cmd(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    end: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    symbol: Annotated[str, typer.Option(help="Underlying ticker")] = "SPY",
    config: Annotated[Path, typer.Option(help="Config path")] = DEFAULT_CONFIG_PATH,
) -> None:
    configure_logging()
    cfg = load_config(config)
    try:
        out = fetch_underlying_range_massive(start=start, end=end, symbol=symbol, config=cfg)
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Saved underlying parquet (Massive): {out}")


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


@app.command("collect-chains")
def collect_chains_cmd(
    asof: Annotated[str, typer.Option(help="YYYY-MM-DD (run after close for EOD)")],
    symbol: Annotated[str, typer.Option(help="Underlying ticker")] = "SPY",
    tenors: Annotated[str, typer.Option(help="Comma-separated tenor days")] = "7,14,30,60,90,180",
    greeks: Annotated[bool, typer.Option(help="Request greeks/IV in chain response")] = False,
    config: Annotated[Path, typer.Option(help="Config path")] = DEFAULT_CONFIG_PATH,
) -> None:
    configure_logging()
    cfg = load_config(config)
    tenor_days = [int(x.strip()) for x in tenors.split(",") if x.strip()]
    try:
        out = collect_eod_chains_asof(
            asof=asof,
            symbol=symbol,
            tenors_days=tenor_days,
            greeks=greeks,
            config=cfg,
        )
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Saved option chain parquet: {out}")


@app.command("collect-chains-massive")
def collect_chains_massive_cmd(
    asof: Annotated[str, typer.Option(help="YYYY-MM-DD (run after close for EOD)")],
    symbol: Annotated[str, typer.Option(help="Underlying ticker")] = "SPY",
    tenors: Annotated[str, typer.Option(help="Comma-separated tenor days")] = "7,14,30,60,90,180",
    config: Annotated[Path, typer.Option(help="Config path")] = DEFAULT_CONFIG_PATH,
) -> None:
    configure_logging()
    cfg = load_config(config)
    tenor_days = [int(x.strip()) for x in tenors.split(",") if x.strip()]
    try:
        out = collect_eod_chains_asof_massive(
            asof=asof,
            symbol=symbol,
            tenors_days=tenor_days,
            config=cfg,
        )
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Saved option chain parquet (Massive): {out}")


@app.command("collect-chains-massive-flatfile")
def collect_chains_massive_flatfile_cmd(
    asof: Annotated[str, typer.Option(help="YYYY-MM-DD (EOD day to collect)")],
    symbol: Annotated[str, typer.Option(help="Underlying ticker")] = "SPY",
    tenors: Annotated[str, typer.Option(help="Comma-separated tenor days")] = "7,14,30,60,90,180",
    config: Annotated[Path, typer.Option(help="Config path")] = DEFAULT_CONFIG_PATH,
) -> None:
    configure_logging()
    cfg = load_config(config)
    tenor_days = [int(x.strip()) for x in tenors.split(",") if x.strip()]
    out = collect_eod_chains_asof_massive_flatfile(
        asof=asof,
        symbol=symbol,
        tenors_days=tenor_days,
        config=cfg,
    )
    typer.echo(f"Saved option chain parquet (Massive flatfiles): {out}")


@app.command("fetch-market-data-massive")
def fetch_market_data_massive_cmd(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    end: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    symbol: Annotated[str, typer.Option(help="Underlying ticker")] = "SPY",
    tenors: Annotated[str, typer.Option(help="Comma-separated tenor days")] = "7,14,30,60,90,180",
    options_source: Annotated[
        str, typer.Option(help="Options source: flatfiles or api")
    ] = "flatfiles",
    clean: Annotated[bool, typer.Option(help="Delete existing local data files first")] = True,
    stop_on_error: Annotated[bool, typer.Option(help="Stop immediately when a day fails")] = False,
    config: Annotated[Path, typer.Option(help="Config path")] = DEFAULT_CONFIG_PATH,
) -> None:
    configure_logging()
    cfg = load_config(config)
    tenor_days = [int(x.strip()) for x in tenors.split(",") if x.strip()]
    summary = fetch_market_data_range_massive(
        start=start,
        end=end,
        symbol=symbol,
        tenors_days=tenor_days,
        config=cfg,
        clean=clean,
        stop_on_error=stop_on_error,
        options_source=options_source,
    )
    typer.echo(
        "Massive pull complete: "
        f"succeeded={summary['days_succeeded']}/{summary['days_attempted']} "
        f"failed={summary['days_failed']} "
        f"aligned_underlying_rows={summary['underlying_rows_aligned']}"
    )
    typer.echo(f"Summary: {summary['summary_path']}")


@app.command("clear-market-data")
def clear_market_data_cmd(
    config: Annotated[Path, typer.Option(help="Config path")] = DEFAULT_CONFIG_PATH,
) -> None:
    configure_logging()
    cfg = load_config(config)
    removed = clear_market_data(cfg)
    typer.echo(
        "Cleared market data files: "
        f"raw={removed['raw']} underlying={removed['underlying']} "
        f"processed={removed['processed']} cache={removed['cache']}"
    )


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
