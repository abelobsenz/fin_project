from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TypedDict


@dataclass(slots=True)
class OptionQuoteRow:
    date: date
    expiry: date
    dte: int
    strike: float
    call_put: str
    bid: float
    ask: float
    mid: float
    volume: int
    open_interest: int
    underlying_close: float


class OptionQuoteDict(TypedDict):
    date: str
    expiry: str
    dte: int
    strike: float
    call_put: str
    bid: float
    ask: float
    mid: float
    volume: int
    open_interest: int
    underlying_close: float
