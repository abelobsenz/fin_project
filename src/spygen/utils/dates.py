from __future__ import annotations

from datetime import date, datetime

import pandas as pd


def parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def business_days(start: str | date, end: str | date) -> list[date]:
    s = parse_date(start)
    e = parse_date(end)
    days = pd.bdate_range(s, e)
    return [d.date() for d in days]
