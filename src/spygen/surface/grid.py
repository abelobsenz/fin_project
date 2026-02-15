from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SurfaceGrid:
    x_min: float
    x_max: float
    nx: int
    tenors_days: list[int]

    @property
    def x(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.nx)

    @property
    def tenors_years(self) -> np.ndarray:
        return np.array(self.tenors_days, dtype=float) / 365.0

    @classmethod
    def from_config(cls, config: dict) -> SurfaceGrid:
        return cls(
            x_min=float(config["x_min"]),
            x_max=float(config["x_max"]),
            nx=int(config["nx"]),
            tenors_days=[int(x) for x in config["tenors_days"]],
        )
