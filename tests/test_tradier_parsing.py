from __future__ import annotations

import json
from pathlib import Path

from spygen.integrations.tradier import (
    parse_expirations,
    parse_lookup,
    parse_market_history,
    parse_option_chain,
    parse_quotes,
)

FIX = Path("tests/fixtures")


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text())


def test_parse_market_history_fixture() -> None:
    df = parse_market_history(_load("tradier_history_spy.json"))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert float(df.iloc[0]["close"]) == 477.1


def test_parse_lookup_fixture() -> None:
    rows = parse_lookup(_load("tradier_lookup_spy.json"))
    assert len(rows) == 2
    assert rows[0]["rootsymbol"] == "SPY"


def test_parse_expirations_fixture() -> None:
    exp = parse_expirations(_load("tradier_expirations_spy.json"))
    assert len(exp) == 3
    assert exp[0].isoformat() == "2024-02-16"


def test_parse_chain_fixture() -> None:
    df = parse_option_chain(_load("tradier_chain_spy.json"))
    assert {"symbol", "option_type", "strike", "expiration_date"}.issubset(df.columns)
    assert len(df) == 2
    assert float(df.iloc[0]["delta"]) == 0.52


def test_parse_quotes_fixture() -> None:
    q = parse_quotes(_load("tradier_quotes_spy.json"))
    assert "SPY" in q
    assert "SPY240216C00470000" in q
