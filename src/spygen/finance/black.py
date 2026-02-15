from __future__ import annotations

import math


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(
    spot: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
    option_type: str = "C",
) -> float:
    if t <= 0:
        intrinsic = (
            max(0.0, spot - strike)
            if option_type.upper() == "C"
            else max(0.0, strike - spot)
        )
        return intrinsic
    sigma = max(1e-8, sigma)
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc = math.exp(-r * t)

    if option_type.upper() == "C":
        return spot * _norm_cdf(d1) - strike * disc * _norm_cdf(d2)
    return strike * disc * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def implied_vol_bisection(
    price: float,
    spot: float,
    strike: float,
    t: float,
    r: float,
    option_type: str = "C",
    low: float = 1e-4,
    high: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    if t <= 0:
        return 0.0
    lo, hi = low, high
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        model = bs_price(spot=spot, strike=strike, t=t, r=r, sigma=mid, option_type=option_type)
        err = model - price
        if abs(err) < tol:
            return mid
        if err > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
