from __future__ import annotations

import argparse

from spygen.pipeline import fetch_underlying_range_massive, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch SPY EOD candles from Massive")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = fetch_underlying_range_massive(
        start=args.start,
        end=args.end,
        symbol=args.symbol,
        config=cfg,
    )
    print(f"Saved underlying parquet (Massive): {out}")


if __name__ == "__main__":
    main()
