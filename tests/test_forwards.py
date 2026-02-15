from __future__ import annotations

import pandas as pd

from spygen.finance.black import bs_price
from spygen.finance.forwards import infer_forward_from_parity


def test_infer_forward_from_parity() -> None:
    spot = 100.0
    rate = 0.01
    t = 30 / 365
    strikes = [95.0, 100.0, 105.0]
    rows = []
    for k in strikes:
        c = bs_price(spot=spot, strike=k, t=t, r=rate, sigma=0.2, option_type="C")
        p = bs_price(spot=spot, strike=k, t=t, r=rate, sigma=0.2, option_type="P")
        rows.append({"strike": k, "mid": c, "call_put": "C", "dte": 30})
        rows.append({"strike": k, "mid": p, "call_put": "P", "dte": 30})
    chain = pd.DataFrame(rows)
    f = infer_forward_from_parity(chain, rate=rate, spot=spot)
    assert abs(f - spot * (1.0 + rate * t)) < 0.5
