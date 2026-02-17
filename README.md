# ivdyn

`ivdyn` is a from-scratch package for learning implied-volatility and option-pricing dynamics from Massive.com datasets.

## What It Implements

- Massive-compatible data plugins (raw parquet snapshots, local OPRA flatfile aggs, REST hooks).
- Surface builder with liquidity-aware gridding and static no-arbitrage diagnostics.
- Multi-stage deep model (VAE encoder/decoder + latent dynamics + pricing head + execution head) implemented in PyTorch.
- Evaluation and backtest-ready artifacts.
- Streamlit dashboard with graphical and numerical model evidence.

## Quickstart

```bash
pip install -e .
ivdyn pull-underlying-massive --data-root data --symbol SPY --start-date 2024-07-01 --end-date 2024-12-31
ivdyn pull-flatfiles --data-root data --start-date 2024-07-01 --end-date 2024-12-31
ivdyn pull-massive --data-root data --symbol SPY --start-date 2024-07-01 --end-date 2024-12-31
ivdyn build-dataset --data-root data --out-dir outputs/dataset
ivdyn build-dataset --data-root data --out-dir outputs/dataset_2024h2 --plugin massive_flatfile_aggs --start-date 2024-07-01 --end-date 2024-12-31
ivdyn train --dataset outputs/dataset/dataset.npz --out-dir outputs/runs
ivdyn evaluate --run-dir outputs/runs/<RUN_ID> --dataset outputs/dataset/dataset.npz
ivdyn backtest --run-dir outputs/runs/<RUN_ID> --dataset outputs/dataset/dataset.npz
ivdyn ui --run-dir outputs/runs/<RUN_ID>
```

`evaluate`, `backtest`, and `ui` default to the most recent run in `outputs/runs` when `--run-dir` is omitted.

## Data Inputs

Expected folders:

- `data/raw/*.parquet` and `data/raw/*.metadata.json` (Massive-derived chain snapshots)
- `data/underlying/spy_eod.parquet` (underlying EOD prices)
- Optional cache: `data/massive_cache/flatfiles/...`

`pull-underlying-massive` fetches true underlying daily bars into `data/underlying/<symbol>_eod.parquet`.

`pull-massive` uses the Massive REST snapshot endpoint and writes daily files to `data/raw`.
Set `MASSIVE_API_KEY` (or pass `--api-key`).

`pull-flatfiles` uses Massive's S3-compatible flatfiles endpoint and writes day files under:
`data/massive_cache/flatfiles/us_options_opra/day_aggs_v1/YYYY/MM/*.csv.gz`.
Credentials are read from `.env` keys:
`MASSIVE_FLATFILES_ACCESS_KEY`, `MASSIVE_FLATFILES_SECRET_ACCESS_KEY`,
`MASSIVE_FLATFILES_ENDPOINT_URL`, `MASSIVE_FLATFILES_BUCKET`, `MASSIVE_FLATFILES_PREFIX`.

## Notes

- Training uses PyTorch with automatic device selection (CUDA/MPS/CPU).
- The prior experimental numpy trainer was removed; `ivdyn` now has a single PyTorch training path.
- Architecture is designed to be tradeable/backtestable later via modular execution and strategy hooks.

## Performance Tuning

- Dataset build supports parallel day processing:
  - `ivdyn build-dataset ... --num-workers 0` (`0` = auto)
- Evaluation supports parallel no-arbitrage diagnostics:
  - `ivdyn evaluate ... --num-workers 0`
- Backtest supports:
  - parallel day simulation via `--num-workers`
  - batched contract inference via `--inference-batch-size` (default `65536`)
- UI supports surface overlays and PDF report export (requires `matplotlib`, included in project dependencies).
- UI includes tabbed layout focused on PnL evidence, interactive 3D/slice surface overlays, prediction errors, training diagnostics, and fit quality.
