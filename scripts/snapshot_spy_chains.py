from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from spygen.integrations.tradier import TradierClient, TradierConfig
from spygen.pipeline import load_config
from spygen.utils.paths import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot live SPY option chains from Tradier")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--max-expiries", type=int, default=3)
    parser.add_argument("--greeks", action="store_true")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg.get("tradier", {})
    client = TradierClient(
        TradierConfig(
            base_url=tcfg.get("base_url", "https://api.tradier.com"),
            token_env_var=tcfg.get("token_env_var", "TRADIER_TOKEN"),
            cache_enabled=bool(tcfg.get("cache_enabled", True)),
            cache_dir=Path(tcfg.get("cache_dir", "data/tradier_cache")),
            connect_timeout=float(tcfg.get("connect_timeout", 10.0)),
            read_timeout=float(tcfg.get("read_timeout", 30.0)),
            max_retries=int(tcfg.get("max_retries", 4)),
            backoff_base=float(tcfg.get("backoff_base", 0.5)),
        )
    )

    expiries = client.get_option_expirations(symbol=args.symbol)[: args.max_expiries]
    frames = []
    for exp in expiries:
        chain = client.get_option_chain(symbol=args.symbol, expiration=exp, greeks=args.greeks)
        chain["asof"] = datetime.utcnow().date()
        frames.append(chain)

    if not frames:
        print("No chains returned")
        return

    out_dir = ensure_dir("data/raw_live")
    out = pd.concat(frames, ignore_index=True)
    out_file = out_dir / f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet"
    out.to_parquet(out_file, index=False)
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
