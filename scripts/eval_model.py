from __future__ import annotations

import argparse

from spygen.pipeline import eval_checkpoint, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained flow model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = eval_checkpoint(checkpoint_path=args.checkpoint, config=cfg)
    print(f"Saved eval outputs {out}")


if __name__ == "__main__":
    main()
