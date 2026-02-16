from __future__ import annotations

from datetime import date

from spygen.integrations.massive import (
    parse_options_chain_snapshot,
    parse_options_contracts,
    parse_stock_aggs,
)


def test_parse_stock_aggs() -> None:
    payload = {
        "results": [
            {"t": 1704153600000, "o": 470.1, "h": 471.2, "l": 469.8, "c": 470.9, "v": 1234567}
        ]
    }
    df = parse_stock_aggs(payload)
    assert len(df) == 1
    assert float(df.iloc[0]["close"]) == 470.9


def test_parse_options_contracts() -> None:
    payload = {
        "results": [
            {
                "ticker": "O:SPY240119C00470000",
                "underlying_ticker": "SPY",
                "expiration_date": "2024-01-19",
                "strike_price": 470.0,
                "contract_type": "call",
            }
        ]
    }
    df = parse_options_contracts(payload)
    assert len(df) == 1
    assert str(df.iloc[0]["underlying_ticker"]) == "SPY"


def test_parse_options_chain_snapshot() -> None:
    payload = {
        "results": [
            {
                "details": {
                    "ticker": "O:SPY240119C00470000",
                    "expiration_date": "2024-01-19",
                    "strike_price": 470.0,
                    "contract_type": "call",
                },
                "last_quote": {"bid": 2.1, "ask": 2.3},
                "last_trade": {"price": 2.2},
                "day": {"volume": 100},
                "open_interest": 250,
                "greeks": {"delta": 0.51, "gamma": 0.02, "vega": 0.11, "theta": -0.08},
                "implied_volatility": 0.19,
            }
        ]
    }
    df = parse_options_chain_snapshot(
        payload,
        asof=date(2024, 1, 2),
        fallback_underlying_close=470.5,
    )
    assert len(df) == 1
    row = df.iloc[0]
    assert row["call_put"] == "C"
    assert abs(float(row["mid"]) - 2.2) < 1e-9
    assert int(row["dte"]) > 0
