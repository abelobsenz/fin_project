from __future__ import annotations

import argparse

from spygen.pipeline import load_config, train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train conditional flow model")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = train_from_config(cfg)
    print(f"Saved checkpoint {ckpt}")


if __name__ == "__main__":
    main()
