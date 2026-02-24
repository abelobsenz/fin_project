# Data Format: Day vs Minute Aggs

This note summarizes the file formats currently present in this repo for OPRA day/minute aggregates and the canonical schema used by the workflow.

## Source flatfiles (from `data/options_source/us_options_opra`)

Path patterns:
- Day aggs: `day_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz`
- Minute aggs: `minute_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz`

Observed header (both day and minute files):
- `ticker,volume,open,close,high,low,window_start,transactions`

Observed sample date: `2025-03-20`
- Day file rows: `242013`; unique tickers: `242013`; max rows per ticker: `1`
- Minute file rows: `1965579`; unique tickers: `242013`; max rows per ticker: `405`

Interpretation:
- Day aggs are one row per contract for the day.
- Minute aggs are many rows per contract (one row per minute bucket).

## Symbol-level parquet snapshots

Observed paths for SPY:
- Day: `data/symbols/SPY/options/day_aggs/YYYY-MM-DD.parquet`
- Minute: `data/symbols/SPY/options/minute_aggs/YYYY-MM-DD.parquet`

Observed columns in both:
- `date, expiry, dte, call_put, symbol, strike, bid, ask, mid, last, volume, open_interest, underlying_close, delta, gamma, theta, vega, iv`

This column set matches the canonical schema from `src/ivdyn/data/schemas.py` (`CANONICAL_CHAIN_COLUMNS`).

Observed sample date: `2025-03-20`
- Day parquet rows: `1017`; max rows per option symbol: `1`
- Minute parquet rows: `126294`; max rows per option symbol: `405`

Important detail:
- In this canonical parquet schema there is no explicit minute timestamp column (for example `window_start` is not present). Minute granularity is reflected by repeated rows per option symbol/date.

## Dataset metadata differences (day vs minute workflows)

From included sample metadata files:
- Day dataset plugin: `massive_flatfile_aggs`
- Minute dataset plugin: `massive_flatfile_minute_aggs`

Minute dataset contract features include additional intraday features not in day dataset:
- `intraday_ret_oc`
- `intraday_range_frac`
- `intraday_rv_1m`
- `intraday_vwap_dev`
- `intraday_volume_cv`
- `intraday_log_bar_count`
- `intraday_log_volume_per_bar`

