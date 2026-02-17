#!/usr/bin/env python3
from __future__ import annotations

import sys

from ivdyn.cli import main


if __name__ == "__main__":
    main(["ui", *sys.argv[1:]])
