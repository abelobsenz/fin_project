from __future__ import annotations

from datetime import date

from spygen.integrations.massive_flatfiles import (
    MassiveFlatFileClient,
    MassiveFlatFilesConfig,
    parse_option_symbol_osi,
)


def test_parse_option_symbol_osi() -> None:
    parsed = parse_option_symbol_osi("O:SPY250117C00600000")
    assert parsed is not None
    assert parsed["underlying"] == "SPY"
    assert parsed["call_put"] == "C"
    assert parsed["expiry"] == date(2025, 1, 17)
    assert abs(float(parsed["strike"]) - 600.0) < 1e-12


def test_parse_option_symbol_osi_rejects_invalid() -> None:
    assert parse_option_symbol_osi("SPY") is None


def test_object_key_for_date() -> None:
    client = MassiveFlatFileClient(
        MassiveFlatFilesConfig(
            prefix="us_options_opra/day_aggs_v1",
            cache_enabled=False,
        )
    )
    key = client.object_key_for_date(date(2025, 2, 13))
    assert key == "us_options_opra/day_aggs_v1/2025/02/2025-02-13.csv.gz"
