# spy-arbfree-surface-gen

Arbitrage-free deep generative modeling of SPY option surfaces (EOD) plus a simple synthetic-data backtest for a surface dislocation mean-reversion strategy.

## What This Repo Does
- Fetches SPY EOD underlying candles from Tradier `markets/history`.
- Supports Tradier option endpoints (`lookup`, `expirations`, `chains`, `quotes`) for live snapshot / forward collection workflows.
- Builds normalized call surfaces on a fixed log-moneyness/tenor grid.
- Runs static-arbitrage checks and repairs surfaces with a convex QP (`cvxpy` + `OSQP`).
- Converts repaired surfaces into nonnegative increment-curve parameters.
- Trains a conditional normalizing flow (`PyTorch` + `nflows`) for `p(theta | context)`.
- Backtests a simple EOD dislocation strategy using bid/ask-aware one-day holding rules.

## Tradier Data Notes
- External data provider is **Tradier only**.
- Tradier historical options for expired contracts are not available (see Tradier historical-data note), so offline backtests in this repo are synthetic by default.
- CI never performs network calls; Tradier integration tests run on saved JSON fixtures.

## Quickstart (Offline Synthetic)
```bash
python -m pip install -e .[dev]
ruff check .
pytest
python -m spygen sanity
```

## Real SPY Underlying Fetch (Tradier)
1. Set token:
```bash
export TRADIER_TOKEN="..."
```
2. Fetch EOD bars:
```bash
python -m spygen fetch-underlying --start 2024-01-01 --end 2024-12-31 --symbol SPY
```

## Main CLI
```bash
python -m spygen synth-data --start 2024-01-02 --end 2024-03-29
python -m spygen build-dataset --start 2024-01-02 --end 2024-03-29
python -m spygen train --config configs/default.yaml
python -m spygen eval --checkpoint outputs/checkpoints/flow_latest.pt
python -m spygen backtest --checkpoint outputs/checkpoints/flow_latest.pt
python -m spygen walkforward --config configs/default.yaml
python -m spygen sanity
```

Debug/loose trading mode (for plumbing validation):
```bash
python -m spygen backtest --checkpoint outputs/checkpoints/flow_latest.pt --config configs/debug_loose.yaml
```

## Lightweight UI (Model Diagnostics)
Install UI extras:
```bash
python -m pip install -e '.[ui]'
```

Launch dashboard (from the same venv used for training):
```bash
PYTHONPATH=src .venv/bin/python -m streamlit run src/spygen/ui_app.py
```

UI shows:
- observed repaired surface vs model conditional-mean surface
- residual heatmap and tenor slice comparison
- per-date log-likelihood / dislocation score
- static-arbitrage checks on observed and generated samples
- implied-volatility view: observed IV surface vs model-implied IV surface
- eval/backtest run summary tables
- latest backtest diagnostics (`pnl_attribution.json`, `gate_reasons.json`, `execution_summary.json`)

## Backtest Artifacts
Each backtest run under `outputs/backtests/run_*` writes:
- `daily.parquet`: daily PnL/equity/turnover
- `trade_blotter.parquet`: trade-level attribution (`edge_gross`, `edge_net`, `spread_paid`, `fill_slippage`, `fees`, exposure proxies)
- `pnl_attribution.json`: costs, edge stats, hit rate, tail losses, per-structure decomposition
- `gate_reasons.json`: aggregated gate reject/accept counters by structure
- `execution_summary.json`: spread/slippage distributions and spread-gate skip rate
- `unit_sanity.json`: edge-vs-cost unit check diagnostics
- `events.jsonl`: structured per-decision events
- `run_metadata.json`: config snapshot, seed, git SHA

## Project Caveats
- EOD model only; no intraday microstructure realism.
- Execution is simplified (bid/ask + slippage heuristic, one-day hold).
- Synthetic options are for testing and research workflow only.
- Not financial advice.
