from __future__ import annotations

import argparse
from datetime import datetime

from spygen.pipeline import collect_eod_chains_asof, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect EOD-style SPY option chains from Tradier into data/raw"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument(
        "--asof",
        default=datetime.now().date().isoformat(),
        help="YYYY-MM-DD; run after market close for true EOD snapshot",
    )
    parser.add_argument("--tenors", default="7,14,30,60,90,180")
    parser.add_argument("--greeks", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tenor_days = [int(x.strip()) for x in args.tenors.split(",") if x.strip()]
    out = collect_eod_chains_asof(
        asof=args.asof,
        symbol=args.symbol,
        tenors_days=tenor_days,
        greeks=args.greeks,
        config=cfg,
    )
    print(f"Saved option chain parquet: {out}")


if __name__ == "__main__":
    main()
