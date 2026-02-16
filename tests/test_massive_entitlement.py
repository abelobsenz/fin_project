from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from spygen.pipeline import _clip_massive_date_range, _validate_massive_asof


def _today():
    return datetime.now(UTC).date()


def test_clip_massive_date_range_clips_old_start() -> None:
    today = _today()
    cfg = {"massive": {"max_history_days": 10}}
    start = today - timedelta(days=30)
    end = today - timedelta(days=1)
    adj_start, adj_end, cutoff = _clip_massive_date_range(start, end, config=cfg, today=today)
    assert adj_end == end
    assert adj_start == cutoff
    assert adj_start == today - timedelta(days=10)


def test_clip_massive_date_range_raises_if_all_old() -> None:
    today = _today()
    cfg = {"massive": {"max_history_days": 10}}
    start = today - timedelta(days=30)
    end = today - timedelta(days=20)
    with pytest.raises(ValueError, match="entitlement window"):
        _clip_massive_date_range(start, end, config=cfg, today=today)


def test_validate_massive_asof_rejects_old_date() -> None:
    today = _today()
    cfg = {"massive": {"max_history_days": 10}}
    old_asof = today - timedelta(days=20)
    with pytest.raises(ValueError, match="entitlement window"):
        _validate_massive_asof(asof_d=old_asof, config=cfg, today=today)
