from __future__ import annotations

import argparse

from spygen.pipeline import load_config, synth_data_range


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic SPY option chain data")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    synth_data_range(start=args.start, end=args.end, config=cfg)


if __name__ == "__main__":
    main()
