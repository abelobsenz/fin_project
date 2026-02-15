from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class OptionLeg:
    name: str
    target_dte: int
    target_x: float
    call_put: str
    weight: float


@dataclass(slots=True)
class TradeStructure:
    name: str
    legs: list[OptionLeg]


def default_trade_structures() -> dict[str, TradeStructure]:
    return {
        "rr": TradeStructure(
            name="rr",
            legs=[
                OptionLeg("rr_left", target_dte=30, target_x=-0.15, call_put="C", weight=1.0),
                OptionLeg("rr_right", target_dte=30, target_x=0.15, call_put="C", weight=-1.0),
            ],
        ),
        "fly": TradeStructure(
            name="fly",
            legs=[
                OptionLeg("fly_left", target_dte=30, target_x=-0.15, call_put="C", weight=1.0),
                OptionLeg("fly_atm", target_dte=30, target_x=0.0, call_put="C", weight=-2.0),
                OptionLeg("fly_right", target_dte=30, target_x=0.15, call_put="C", weight=1.0),
            ],
        ),
        "calendar": TradeStructure(
            name="calendar",
            legs=[
                OptionLeg("cal_short", target_dte=14, target_x=0.0, call_put="C", weight=-1.0),
                OptionLeg("cal_long", target_dte=60, target_x=0.0, call_put="C", weight=1.0),
            ],
        ),
    }


def basis_vectors(x_grid: np.ndarray, tenor_days: list[int]) -> dict[str, np.ndarray]:
    nx = len(x_grid)
    nt = len(tenor_days)

    def idx_x(target: float) -> int:
        return int(np.argmin(np.abs(x_grid - target)))

    def idx_t(target: int) -> int:
        return int(np.argmin(np.abs(np.array(tenor_days) - target)))

    rr = np.zeros((nx, nt))
    rr[idx_x(-0.15), idx_t(30)] = 1.0
    rr[idx_x(0.15), idx_t(30)] = -1.0

    fly = np.zeros((nx, nt))
    fly[idx_x(-0.15), idx_t(30)] = 1.0
    fly[idx_x(0.0), idx_t(30)] = -2.0
    fly[idx_x(0.15), idx_t(30)] = 1.0

    cal = np.zeros((nx, nt))
    cal[idx_x(0.0), idx_t(14)] = -1.0
    cal[idx_x(0.0), idx_t(60)] = 1.0

    return {"rr": rr, "fly": fly, "calendar": cal}
