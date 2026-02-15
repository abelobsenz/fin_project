from __future__ import annotations

import argparse

from spygen.pipeline import backtest_from_config, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EOD backtest")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = backtest_from_config(checkpoint_path=args.checkpoint, config=cfg)
    print(f"Backtest results in {out}")


if __name__ == "__main__":
    main()
