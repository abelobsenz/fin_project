# spy-arbfree-surface-gen

Arbitrage-free deep generative modeling of SPY option surfaces (EOD) plus a simple synthetic-data backtest for a surface dislocation mean-reversion strategy.

## What This Repo Does
- Fetches SPY EOD underlying candles from Tradier `markets/history`.
- Fetches SPY EOD underlying candles from Tradier or Massive.
- Supports Tradier option endpoints (`lookup`, `expirations`, `chains`, `quotes`) for live snapshot / forward collection workflows.
- Supports Massive market-data endpoints for bars, options contracts, and options chain snapshots.
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
python -m spygen collect-chains --asof 2026-02-15 --symbol SPY --greeks
```

## Massive API Setup
1. Set key:
```bash
export MASSIVE_API_KEY="..."
export MASSIVE_FILES_ACCESS_KEY_ID="..."
export MASSIVE_FILES_SECRET_ACCESS_KEY="..."
```
2. Fetch EOD bars:
```bash
python -m spygen fetch-underlying-massive --start 2024-01-01 --end 2024-12-31 --symbol SPY
```
3. Collect EOD-style option chains into `data/raw`:
```bash
python -m spygen collect-chains-massive --asof 2026-02-15 --symbol SPY --tenors 7,14,30,60,90,180
```
Notes:
- Massive pulls are entitlement-guarded with `massive.max_history_days` (default `730`).
- If a requested date is older than entitlement, the command raises a clear error.
- For underlying range pulls, start date is clipped to entitlement cutoff when needed.
- For historical options collection at scale, prefer flat files with `collect-chains-massive-flatfile` or `fetch-market-data-massive --options-source flatfiles`.

Fetch an aligned underlying + options dataset over a period:
```bash
python -m spygen fetch-market-data-massive \
  --start 2025-01-01 \
  --end 2025-12-31 \
  --symbol SPY \
  --tenors 7,14,30,60,90,180 \
  --options-source flatfiles \
  --clean
```
This single command fetches underlying bars, collects daily chains for matching dates, and re-aligns `data/underlying/spy_eod.parquet` to only successful chain days.
If fallback collection is too noisy/slow, lower `massive.fallback_max_contracts_per_expiry` and/or `massive.fallback_strike_band_pct` in config.

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

Collect Tradier live chains directly into `data/raw/YYYY-MM-DD.parquet` (same format used by dataset build/backtest):
```bash
python scripts/collect_eod_chains.py --asof 2026-02-15 --symbol SPY --greeks
```

Collect Massive chains / bars using standalone scripts:
```bash
python scripts/fetch_spy_eod_massive.py --start 2024-01-01 --end 2024-12-31 --symbol SPY
python scripts/collect_eod_chains_massive.py --asof 2026-02-15 --symbol SPY
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
- interactive 3D surface view (normalized call or implied vol; observed/model/residual)
- residual heatmap and tenor slice comparison
- dropdown-driven cross-sections (tenor slice or moneyness term-structure)
- per-date log-likelihood / dislocation score
- static-arbitrage checks on observed and generated samples
- implied-volatility view: observed IV surface vs model-implied IV surface
- eval/backtest run summary tables
- latest backtest diagnostics (`pnl_attribution.json`, `gate_reasons.json`, `execution_summary.json`)

## Backtest Artifacts
Each backtest run under `outputs/backtests/run_*` writes:
- `daily.parquet`: daily PnL/equity/turnover
- `trade_blotter.parquet`: trade-level attribution (`edge_gross_usd`, `edge_net_usd`, `spread_paid`, `fill_slippage`, `fees`, exposure proxies)
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
