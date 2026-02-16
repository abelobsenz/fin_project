from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

import spygen.pipeline as pipeline


def test_collect_eod_chains_asof_massive_falls_back_to_open_close(
    tmp_path: Path,
    monkeypatch,
) -> None:
    today = datetime.now(UTC).date()
    asof = today - timedelta(days=1)

    class FakeMassiveClient:
        def __init__(self, _config=None) -> None:
            pass

        def get_stock_aggs(self, symbol: str, start, end) -> pd.DataFrame:
            _ = (symbol, start, end)
            idx = pd.Index([asof], name="date")
            return pd.DataFrame(
                {
                    "open": [500.0],
                    "high": [501.0],
                    "low": [499.0],
                    "close": [500.0],
                    "volume": [1_000_000.0],
                },
                index=idx,
            )

        def list_options_contracts(
            self,
            *,
            underlying_ticker: str,
            as_of,
            expiration_gte,
            expiration_lte,
            limit: int = 1000,
        ) -> pd.DataFrame:
            _ = (underlying_ticker, as_of, expiration_gte, expiration_lte, limit)
            return pd.DataFrame(
                {
                    "ticker": [
                        f"O:SPY{(asof + timedelta(days=7)).strftime('%y%m%d')}C00500000",
                        f"O:SPY{(asof + timedelta(days=30)).strftime('%y%m%d')}P00500000",
                    ],
                    "expiration_date": [asof + timedelta(days=7), asof + timedelta(days=30)],
                    "strike_price": [500.0, 500.0],
                    "contract_type": ["call", "put"],
                }
            )

        def get_option_open_close(
            self,
            *,
            options_ticker: str,
            as_of,
            adjusted: bool = True,
        ) -> dict:
            _ = (options_ticker, as_of, adjusted)
            return {"close": 2.0, "volume": 100, "open_interest": 200}

    monkeypatch.setattr(pipeline, "MassiveClient", FakeMassiveClient)

    cfg = {
        "paths": {"raw_dir": str(tmp_path / "raw")},
        "massive": {
            "max_history_days": 730,
            "fallback_rel_spread": 0.1,
            "fallback_strike_band_pct": 0.5,
            "fallback_max_contracts_per_expiry": 50,
            "fallback_min_mid": 0.01,
        },
    }

    out_path = pipeline.collect_eod_chains_asof_massive(
        asof=asof.isoformat(),
        symbol="SPY",
        tenors_days=[7, 30],
        config=cfg,
    )

    assert out_path.exists()
    rows = pd.read_parquet(out_path)
    assert len(rows) == 2
    assert set(rows["call_put"]) == {"C", "P"}
    assert float(rows["bid"].min()) >= 0.0
    assert float(rows["ask"].min()) > float(rows["bid"].min())

    meta_path = out_path.with_suffix(".metadata.json")
    meta = json.loads(meta_path.read_text())
    assert meta["source"] == "massive_open_close_fallback"


def test_fetch_market_data_range_massive_aligns_underlying_to_chain_days(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    underlying_path = tmp_path / "underlying" / "spy_eod.parquet"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    underlying_path.parent.mkdir(parents=True, exist_ok=True)

    days = [
        datetime(2025, 1, 2).date(),
        datetime(2025, 1, 3).date(),
        datetime(2025, 1, 6).date(),
    ]

    def fake_fetch_underlying_range_massive(
        start: str,
        end: str,
        symbol: str,
        config: dict,
    ) -> Path:
        _ = (start, end, symbol, config)
        pd.DataFrame(
            {
                "date": days,
                "open": [500.0, 501.0, 502.0],
                "high": [501.0, 502.0, 503.0],
                "low": [499.0, 500.0, 501.0],
                "close": [500.5, 501.5, 502.5],
                "volume": [1_000_000, 1_200_000, 1_100_000],
            }
        ).to_parquet(underlying_path, index=False)
        return underlying_path

    def fake_collect_eod_chains_asof_massive(
        asof: str,
        symbol: str,
        tenors_days: list[int],
        config: dict,
    ) -> Path:
        _ = (symbol, tenors_days, config)
        if asof.endswith("03"):
            raise ValueError("forced failure")
        day = pd.to_datetime(asof).date()
        out = raw_dir / f"{asof}.parquet"
        pd.DataFrame(
            {
                "date": [day],
                "expiry": [day + timedelta(days=7)],
                "dte": [7],
                "call_put": ["C"],
                "symbol": ["O:SPY250110C00500000"],
                "strike": [500.0],
                "bid": [2.0],
                "ask": [2.2],
                "mid": [2.1],
                "volume": [100],
                "open_interest": [1000],
                "underlying_close": [500.5],
            }
        ).to_parquet(out, index=False)
        return out

    monkeypatch.setattr(
        pipeline,
        "fetch_underlying_range_massive",
        fake_fetch_underlying_range_massive,
    )
    monkeypatch.setattr(
        pipeline,
        "collect_eod_chains_asof_massive",
        fake_collect_eod_chains_asof_massive,
    )

    cfg = {
        "paths": {
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "underlying_path": str(underlying_path),
        },
        "massive": {"max_history_days": 730},
    }

    summary = pipeline.fetch_market_data_range_massive(
        start="2025-01-01",
        end="2025-01-10",
        symbol="SPY",
        tenors_days=[7, 30],
        config=cfg,
        clean=False,
        stop_on_error=False,
        options_source="api",
    )

    assert summary["days_attempted"] == 3
    assert summary["days_succeeded"] == 2
    assert summary["days_failed"] == 1
    assert summary["underlying_rows_aligned"] == 2

    aligned = pd.read_parquet(underlying_path)
    got = sorted(pd.to_datetime(aligned["date"]).dt.date.astype(str).tolist())
    assert got == ["2025-01-02", "2025-01-06"]
