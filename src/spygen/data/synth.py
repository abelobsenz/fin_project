from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from spygen.finance.black import bs_price
from spygen.utils.dates import business_days
from spygen.utils.paths import ensure_dir

TARGET_DTES = [7, 14, 30, 60, 90, 180]


@dataclass(slots=True)
class SynthConfig:
    seed: int = 123
    strike_points: int = 21
    bad_quote_prob: float = 0.02


def _expiry_calendar(start: date, end: date, max_dte: int = 240) -> list[date]:
    cal: list[date] = []
    d = start
    while d <= end + timedelta(days=max_dte):
        if d.weekday() == 4:  # Friday
            cal.append(d)
        d += timedelta(days=1)
    return cal


def generate_underlying_series(
    start: date,
    end: date,
    seed: int = 123,
    s0: float = 450.0,
    mu: float = 0.07,
    sigma: float = 0.18,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = business_days(start, end)
    dt = 1.0 / 252.0
    prices = [s0]
    for _ in range(1, len(days)):
        z = rng.normal()
        next_price = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        prices.append(float(next_price))
    return pd.DataFrame({"date": days, "close": prices})


def _pick_expiries(current: date, expiry_cal: list[date], rng: np.random.Generator) -> list[date]:
    out: list[date] = []
    for dte in TARGET_DTES:
        jitter = int(rng.integers(-2, 3))
        target = current + timedelta(days=max(2, dte + jitter))
        candidates = [e for e in expiry_cal if e > current]
        expiry = min(candidates, key=lambda e: abs((e - target).days))
        if expiry not in out:
            out.append(expiry)
    out.sort()
    return out


def _smile_vol(log_moneyness: np.ndarray, t: float, day_factor: float) -> np.ndarray:
    base = 0.16 + 0.03 * np.exp(-2.5 * t) + 0.01 * day_factor
    skew = -0.11 * log_moneyness
    smile = 0.35 * (log_moneyness**2)
    term = 0.06 / np.sqrt(t + 0.02)
    vol = base + skew + smile + term
    return np.clip(vol, 0.07, 1.2)


def _symbol(expiry: date, call_put: str, strike: float) -> str:
    return f"SPY{expiry.strftime('%y%m%d')}{call_put}{int(round(strike * 1000)):08d}"


def _synth_spread_abs(mid: float, wing: float, t: float, rng: np.random.Generator) -> float:
    short_penalty = 0.015 / np.sqrt(max(t, 0.02))
    wing_penalty = 0.02 * min(3.0, wing / 0.1)
    rel = np.clip(0.008 + short_penalty + wing_penalty, 0.01, 0.35)
    rel *= float(np.clip(1.0 + rng.normal(0.0, 0.08), 0.75, 1.35))
    return float(max(0.01, mid * rel))


def generate_daily_chain(
    asof: date,
    spot: float,
    expiry_cal: list[date],
    rng: np.random.Generator,
    strike_points: int,
    bad_quote_prob: float,
    rate: float = 0.01,
) -> pd.DataFrame:
    expiries = _pick_expiries(asof, expiry_cal, rng)
    strikes = np.linspace(0.7 * spot, 1.3 * spot, strike_points)
    rows: list[dict[str, float | int | str | date]] = []

    for expiry in expiries:
        dte = max(1, (expiry - asof).days)
        t = dte / 365.0
        forward = spot * np.exp(rate * t)
        x = np.log(strikes / forward)
        day_factor = np.sin(asof.toordinal() / 50.0)
        vols = _smile_vol(x, t, day_factor)

        for strike, sigma in zip(strikes, vols, strict=True):
            call_mid_clean = bs_price(
                spot=spot,
                strike=float(strike),
                t=t,
                r=rate,
                sigma=float(sigma),
                option_type="C",
            )
            put_mid_clean = bs_price(
                spot=spot,
                strike=float(strike),
                t=t,
                r=rate,
                sigma=float(sigma),
                option_type="P",
            )

            for call_put, clean_mid in (("C", call_mid_clean), ("P", put_mid_clean)):
                noisy_mid = max(0.01, clean_mid * (1.0 + rng.normal(0.0, 0.01)))
                wing = abs(np.log(strike / spot))
                spread = _synth_spread_abs(noisy_mid, wing=wing, t=t, rng=rng)

                if rng.random() < bad_quote_prob:
                    noisy_mid = max(0.01, noisy_mid * (1.0 + rng.normal(0.0, 0.25)))
                    spread *= float(rng.uniform(1.2, 2.5))

                bid = max(0.0, noisy_mid - 0.5 * spread)
                ask = max(bid + 0.01, noisy_mid + 0.5 * spread)
                liquidity = np.exp(-4.0 * wing) / (1.0 + 2.0 * t)
                volume = int(rng.poisson(40 * liquidity + 1))
                oi = int(rng.poisson(300 * liquidity + 5))

                rows.append(
                    {
                        "date": asof,
                        "expiry": expiry,
                        "dte": dte,
                        "strike": float(strike),
                        "call_put": call_put,
                        "bid": float(bid),
                        "ask": float(ask),
                        "mid": float((bid + ask) * 0.5),
                        "volume": max(0, volume),
                        "open_interest": max(0, oi),
                        "underlying_close": float(spot),
                        "symbol": _symbol(expiry, call_put, float(strike)),
                    }
                )

    return pd.DataFrame(rows)


def write_synthetic_dataset(
    start: date,
    end: date,
    raw_dir: str | Path,
    underlying_path: str | Path,
    config: SynthConfig,
) -> None:
    rng = np.random.default_rng(config.seed)
    raw_root = ensure_dir(raw_dir)
    underlying = generate_underlying_series(start, end, seed=config.seed)
    ensure_dir(Path(underlying_path).parent)
    underlying.to_parquet(underlying_path, index=False)

    expiry_cal = _expiry_calendar(start, end)
    for row in underlying.itertuples(index=False):
        asof = row.date
        chain = generate_daily_chain(
            asof=asof,
            spot=float(row.close),
            expiry_cal=expiry_cal,
            rng=rng,
            strike_points=config.strike_points,
            bad_quote_prob=config.bad_quote_prob,
        )
        out_file = raw_root / f"{asof.isoformat()}.parquet"
        chain.to_parquet(out_file, index=False)
        meta = {
            "source": "synthetic",
            "date": asof.isoformat(),
            "generated_at": datetime.now(UTC).isoformat(),
            "rows": int(len(chain)),
            "seed": config.seed,
            "bad_quote_prob": config.bad_quote_prob,
        }
        (raw_root / f"{asof.isoformat()}.metadata.json").write_text(json.dumps(meta, indent=2))
