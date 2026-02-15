from __future__ import annotations

import argparse

from spygen.pipeline import fetch_underlying_range, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch SPY EOD candles from Tradier")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fetch_underlying_range(start=args.start, end=args.end, symbol=args.symbol, config=cfg)


if __name__ == "__main__":
    main()
