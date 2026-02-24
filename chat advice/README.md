# ivdyn

`ivdyn` is an end-to-end research and backtesting system for learning implied-volatility dynamics from options data, forecasting option prices/fill likelihood, and simulating execution-aware options strategies with costs and delta-hedge diagnostics.

It includes:
- Data ingestion from Massive-compatible sources (REST snapshots and OPRA day aggs flatfiles)
- Surface construction with liquidity-aware gridding and static no-arbitrage diagnostics
- A PyTorch latent-dynamics model (surface encoder/decoder + dynamics + pricing head + execution head)
- Evaluation and backtest pipelines that write reproducible artifacts
- A Streamlit UI for single-run analysis and all-symbol portfolio reporting (including PDF export)

## 1. System Overview

The project is organized as a pipeline:

1. Pull or prepare options + underlying data
2. Build a training dataset (`dataset.npz`) with surfaces, context, and sampled contracts
3. Train model (`model.pt`) and persist training diagnostics
4. Evaluate model on dataset and generate fit/prediction/no-arbitrage artifacts
5. Backtest strategy with realistic costs/fill assumptions and optional underlying delta hedge
6. Inspect results in UI and export PDFs (single-run and all-symbol)

## 2. Core Model Design

Model implementation: `src/ivdyn/model/torch_system.py`

The architecture has 5 connected components:
- `encoder`: maps normalized IV surface -> latent state `z_t`
- `decoder`: reconstructs surface from latent state
- `dynamics`: predicts next latent state from `(z_t, context_t)`
- `pricer`: predicts normalized option price from `(z_t, contract_features)`
- `execution`: predicts execution/fill logit from `(z_t, contract_features)`

Training pipeline: `src/ivdyn/training/pipeline.py`

Training is staged:
- Stage 1 (`vae`): surface reconstruction + KL + calendar penalty
- Stage 2 (`heads`): dynamics/pricer/execution heads (with frozen encoder/decoder)
- Stage 3 (`joint`): joint fine-tuning of full model

## 3. Repository Layout

- `src/ivdyn/data`: plugins, schema normalization, dataset builder
- `src/ivdyn/surface`: surface interpolation/repair + no-arbitrage diagnostics
- `src/ivdyn/model`: PyTorch model + scalers + model bundle save/load
- `src/ivdyn/training`: training pipeline
- `src/ivdyn/eval`: evaluation metrics and artifact generation
- `src/ivdyn/backtest`: strategy simulator, cost model, hedge diagnostics
- `src/ivdyn/ui`: Streamlit research console + PDF reporting
- `src/ivdyn/utils/cli.py`: all CLI commands and defaults

## 4. Installation

Requirements:
- Python `>=3.11`
- PyTorch-compatible environment for CPU/CUDA/MPS

Install in editable mode:

```bash
pip install -e .
```

Entry point:

```bash
ivdyn --help
```

## 5. Environment Variables

`ivdyn` autoloads `.env` from repo root or current working directory.

Common keys:
- `MASSIVE_API_KEY` (or `POLYGON_API_KEY`)
- `MASSIVE_FLATFILES_ACCESS_KEY`
- `MASSIVE_FLATFILES_SECRET_ACCESS_KEY`
- `MASSIVE_FLATFILES_ENDPOINT_URL`
- `MASSIVE_FLATFILES_BUCKET`
- `MASSIVE_FLATFILES_PREFIX`

## 6. Data Sources and Expected Paths

### Canonical symbol data

- Options raw snapshots/parquets:
  - `data/symbols/<SYMBOL>/options/raw/YYYY-MM-DD.parquet`
- Underlying EOD bars:
  - `data/symbols/<SYMBOL>/underlying/<symbol>_eod.parquet`

### Full OPRA day-agg cache

- `data/options_source/us_options_opra/day_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz`

### Data pulling commands

Underlying:

```bash
ivdyn pull-underlying-massive \
  --data-root data \
  --symbol SPY \
  --start-date 2024-01-01 \
  --end-date 2025-12-31
```

Massive options snapshots:

```bash
ivdyn pull-massive \
  --data-root data \
  --symbol SPY \
  --start-date 2024-01-01 \
  --end-date 2025-12-31
```

Flatfile OPRA day aggs:

```bash
ivdyn pull-flatfiles \
  --data-root data \
  --start-date 2024-01-01 \
  --end-date 2025-12-31
```

Extract one symbol from flatfiles to canonical per-day parquets:

```bash
ivdyn pull-options-symbol \
  --data-root data \
  --symbol SPY \
  --start-date 2024-01-01 \
  --end-date 2025-12-31
```

## 7. Dataset Build

Builder: `src/ivdyn/data/loaders.py`

Creates:
- Surface tensors (`iv_surface`, `price_surface`, `liq_surface`)
- Context features (returns/realized vol/surface summary/no-arb stats)
- Contract-level samples and targets

Primary command:

```bash
ivdyn build-dataset \
  --data-root data \
  --out-dir outputs/datasets/SPY_2024train \
  --symbol SPY \
  --plugin massive_raw_parquet \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --num-workers 0
```

Plugins:
- `massive_raw_parquet` (default)
- `massive_flatfile_aggs`
- `massive_rest` (requires API key)

Outputs in `--out-dir`:
- `dataset.npz`
- `contracts.parquet`
- `noarb_by_date.parquet`
- `dataset_preview.parquet`
- `dataset_meta.json`

## 8. Train

Command:

```bash
ivdyn train \
  --dataset outputs/datasets/SPY_2024train/dataset.npz \
  --out-dir outputs/SPY/runs/runs_2024train_full
```

Training outputs (new run directory):
- `model.pt`
- `train_history.csv`
- `latent_states.parquet`
- `train_config.json`
- `train_summary.json`
- plus `latest.txt` in the parent run root

## 9. Evaluate

Command:

```bash
ivdyn evaluate \
  --run-dir outputs/SPY/runs/runs_2024train_full/run_<TIMESTAMP> \
  --dataset outputs/datasets/SPY_2025exec/dataset.npz \
  --num-workers 0
```

If `--run-dir` is omitted, CLI falls back to `outputs/runs/latest.txt` (or newest `run_*` in `outputs/runs`).

If `--dataset` is omitted, it falls back to `dataset_path` in `train_summary.json`.

Evaluation outputs under `run_dir/evaluation/`:
- `metrics.json`
- `contract_predictions.parquet`
- `noarb_test_dates.parquet`
- `noarb_forecast_test_dates.parquet` (when forecast pairs exist)
- `latent_states.parquet`
- `surface_predictions.npz`

## 10. Backtest

Command:

```bash
ivdyn backtest \
  --run-dir outputs/SPY/runs/runs_2024train_full/run_<TIMESTAMP> \
  --dataset outputs/datasets/SPY_2025exec/dataset.npz \
  --num-workers 0
```

If `--run-dir` is omitted, CLI falls back to `outputs/runs/latest.txt` (or newest `run_*` in `outputs/runs`).

Backtest outputs under `run_dir/backtest/`:
- `trades.parquet`
- `daily.parquet`
- `legs.parquet`
- `hedges.parquet`
- `summary.json`

### Trading mechanics (implemented)

- Contract eligibility via DTE/moneyness/spread/fill/signal gates
- Selection with edge-vs-cost filtering and daily trade cap
- Strategy modes:
  - `single` (one option leg)
  - `vertical` (anchor + wing)
- Fill model:
  - `assume` (always filled)
  - `expected` (PnL/costs scaled by fill probability)
- Explicit option costs and spread/slippage execution model
- Optional daily underlying delta hedge with hedge diagnostics

### Current important CLI defaults (from parser)

- `--fill-gate 0.45`
- `--slippage-bps 10`
- `--spread-cross-fraction 0.75`
- `--option-commission-per-contract 0.65`
- `--option-fee-per-contract 0.05`
- `--min-edge-to-cost-ratio 1.75`
- `--volume-participation-rate 0.02`
- `--max-trades-per-day 5`
- `--strategy-mode vertical`
- `--hedge-underlying-delta` enabled by default
- `--hedge-underlying-ratio 1.0`
- `--hedge-underlying-min-abs-shares 20`
- `--hedge-underlying-max-shares 200`

## 11. UI (Streamlit)

Launch:

```bash
ivdyn ui --run-dir outputs/SPY/runs/runs_2024train_full/run_<TIMESTAMP>
```

If `--run-dir` is omitted, UI defaults to `outputs/runs/latest.txt` unless `IVDYN_DEFAULT_RUN_DIR` is set.

UI app: `src/ivdyn/ui/app.py`

### Live walk-forward (`ui2`)

Run live walk-forward explicitly with a short command:

```bash
ivdyn wf --symbol SPY
```

This command:

- infers a target symbol (from `--symbol` or latest run; override with `IVDYN_LIVE_SYMBOLS`)
- checks whether today's options/underlying data exists under `data/symbols/<SYMBOL>/...`
- attempts a Massive pull for missing files
- builds/refreshes a live dataset
- runs a short walk-forward backtest window and writes results to:
  - `outputs/live_walkforward/<SYMBOL>/sessions/wf_<TIMESTAMP>/backtest/...`
  - `outputs/live_walkforward/<SYMBOL>/history.csv`

This keeps live records separate from normal run backtests (`<run_dir>/backtest`).

Launch the dedicated live dashboard:

```bash
ivdyn ui2 --symbol SPY
```

UI2 app: `src/ivdyn/ui/live_app.py`

Useful env switches:

- `IVDYN_LIVE_FORCE_RUN=1` to rerun even when today is already processed
- `IVDYN_LIVE_SYMBOLS=SPY,QQQ` to force symbols processed each run
- `IVDYN_LIVE_OUTPUT_ROOT=outputs/live_walkforward` to relocate live artifacts

Key features:
- Sidebar run source selector:
  - `Latest by stock` (auto-discovers latest run under `outputs/<SYMBOL>/runs`)
  - `Manual run directory`
- Professional dashboard layout with tabs:
  - `Backtest & PnL`
  - `Surface Overlays`
  - `Prediction Errors`
  - `Training Diagnostics`
  - `Fits`
- Costs/execution metrics surfaced in KPI cards and tables
- Single-run PDF export
- All-symbol portfolio PDF export (`Download All Symbols PDF`)

All-symbol report summarizes latest run per symbol with metrics such as:
- `total_pnl`, `daily_sharpe`, `max_drawdown`, `total_fees`
- cost settings (`slippage_bps`, `spread_cross_fraction`, commissions/fees, etc.)

## 12. Recommended Multi-Symbol Workflow

Example symbols:

```bash
SYMS="AAPL MSFT QQQ SPY"
```

Train on 2024, execute/backtest on 2025 for each symbol:

```bash
for S in $SYMS; do
  ivdyn build-dataset --data-root data --symbol "$S" --plugin massive_raw_parquet --start-date 2024-01-01 --end-date 2024-12-31 --out-dir "outputs/datasets/${S}_2024train"
  ivdyn train --dataset "outputs/datasets/${S}_2024train/dataset.npz" --out-dir "outputs/${S}/runs/runs_2024train_full"
  ivdyn build-dataset --data-root data --symbol "$S" --plugin massive_raw_parquet --start-date 2025-01-01 --end-date 2025-12-31 --out-dir "outputs/datasets/${S}_2025exec"
  RUN=$(cat "outputs/${S}/runs/runs_2024train_full/latest.txt")
  ivdyn evaluate --run-dir "$RUN" --dataset "outputs/datasets/${S}_2025exec/dataset.npz" --num-workers 0
  ivdyn backtest --run-dir "$RUN" --dataset "outputs/datasets/${S}_2025exec/dataset.npz" --num-workers 0
done
```

Then open UI on any one run (all-symbol PDF scans `outputs/<SYMBOL>/runs` automatically):

```bash
ivdyn ui --run-dir "$(cat outputs/SPY/runs/runs_2024train_full/latest.txt)"
```

## 13. Where to Change Parameters

### Runtime (recommended)

Set params per experiment via CLI flags on `ivdyn backtest` (or `train`, `build-dataset`).

### Default values in code

- CLI defaults: `src/ivdyn/utils/cli.py` in `_build_parser()`
- Backtest config dataclass: `src/ivdyn/backtest/engine.py` (`BacktestConfig`)

If you want permanent new defaults for flags like:
- `--fill-gate`
- `--slippage-bps`
- `--spread-cross-fraction`
- `--min-edge-to-cost-ratio`

edit them in `src/ivdyn/utils/cli.py` under the `backtest` parser block.

## 14. Artifacts Cheat Sheet

Dataset dir:
- `dataset.npz`: model-ready tensors
- `dataset_meta.json`: dataset summary/config

Run dir root:
- `model.pt`: trained model bundle
- `train_history.csv`: per-epoch training logs
- `train_config.json`: resolved training config
- `train_summary.json`: summary with dataset/model path

Run dir `evaluation/`:
- `metrics.json`: numeric eval metrics
- `contract_predictions.parquet`: contract-level predictions
- `surface_predictions.npz`: observed/reconstructed/forecast surfaces

Run dir `backtest/`:
- `summary.json`: main strategy + cost + hedge KPIs and config snapshot
- `daily.parquet`: day-level PnL/equity/cost/hedge series
- `trades.parquet`: selected trade ideas
- `legs.parquet`: leg-level detail (anchor/wing/underlying hedge)
- `hedges.parquet`: explicit daily hedge records

## 15. Performance Notes

- `build-dataset --num-workers 0` auto-uses parallel workers
- `evaluate --num-workers 0` parallelizes no-arbitrage diagnostics
- `backtest --inference-batch-size 65536` controls inference throughput
- `backtest` uses prediction cache files in `backtest/pred_cache*.npz/json`

## 16. Troubleshooting

- `No run directory found`: pass `--run-dir` explicitly
- `No dataset argument and no train_summary.json`: pass `--dataset` explicitly
- Missing parquet engine: install `pyarrow` (already listed dependency)
- Flatfile build errors on missing underlying closes: run `pull-underlying-massive` for same date range
- UI shows empty sections: ensure `evaluate` and `backtest` were run for that run directory

## 17. Quick Command Reference

```bash
ivdyn train ...
ivdyn build-dataset ...
ivdyn evaluate ...
ivdyn backtest ...
ivdyn ui ...
ivdyn pull-underlying-massive ...
ivdyn pull-massive ...
ivdyn pull-flatfiles ...
ivdyn pull-options-symbol ...
```

For full args:

```bash
ivdyn <command> --help
```
