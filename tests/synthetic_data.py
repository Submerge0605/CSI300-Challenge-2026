"""Invented observations for interface checks, not an alpha benchmark."""
import numpy as np
import pandas as pd
from submission.base_features import BARRA, FACTORS

def packet(n_dates=140, n_stocks=16):
    rng = np.random.default_rng(2026)
    dates = pd.bdate_range('2020-01-01', periods=n_dates)
    codes = [f'SYN{i:03}' for i in range(n_stocks)]
    keys = pd.MultiIndex.from_product([dates, codes], names=['日期', '代码']).to_frame(index=False)
    keys['日期'] = keys['日期'].dt.strftime('%Y%m%d').astype('int64')
    close = (20 * np.exp(np.cumsum(rng.normal(0, .01, (n_dates, n_stocks)), axis=0))).ravel()
    op = close * np.exp(rng.normal(0, .002, len(keys)))
    market = keys.assign(open=op, close=close,
                         high=np.maximum(op, close) * 1.005,
                         low=np.minimum(op, close) * .995,
                         volume=rng.uniform(1e5, 5e5, len(keys)),
                         amount=rng.uniform(2e6, 1e7, len(keys)),
                         is_suspend=0, is_st=0)
    factors = keys.copy()
    for name in FACTORS:
        factors[name] = rng.normal(0, .1, len(keys))
    for name in ['total_market_cap', 'float_market_cap']:
        factors[name] = rng.uniform(1e8, 1e10, len(keys))
    for name in ['pe_ttm', 'pb', 'ps_ttm']:
        factors[name] = rng.uniform(1, 30, len(keys))
    factors['list_days'] = 1000
    factors['is_new'] = 0
    factors['is_st'] = 0
    barra = keys.copy()
    for name in BARRA:
        barra[name] = rng.normal(0, 1, len(keys))
    barra['is_st'] = 0
    return {'MarketData': market, 'Factor33_winsor': factors, 'Barra_factor': barra}
