from __future__ import annotations

import argparse

from spygen.pipeline import build_dataset_range, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed model dataset from raw chains")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = build_dataset_range(start=args.start, end=args.end, config=cfg)
    print(f"Saved dataset {out}")


if __name__ == "__main__":
    main()
