from __future__ import annotations

from spygen.finance.black import bs_price, implied_vol_bisection


def test_bs_price_put_call_sanity() -> None:
    call = bs_price(spot=100, strike=100, t=1.0, r=0.01, sigma=0.2, option_type="C")
    put = bs_price(spot=100, strike=100, t=1.0, r=0.01, sigma=0.2, option_type="P")
    assert call > 0
    assert put > 0
    assert call > put


def test_implied_vol_recovers_input() -> None:
    target_sigma = 0.31
    price = bs_price(spot=105, strike=100, t=0.5, r=0.02, sigma=target_sigma, option_type="C")
    iv = implied_vol_bisection(price, spot=105, strike=100, t=0.5, r=0.02, option_type="C")
    assert abs(iv - target_sigma) < 1e-3
