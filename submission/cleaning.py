"""Causal, shared input sanitation. Never inspect labels or fill across dates."""
import numpy as np
import pandas as pd

KEY=['日期','代码']

def clean_table(frame, name):
    d=frame.drop(columns=['is_st'],errors='ignore').copy(deep=True)
    if not all(c in d for c in KEY):raise ValueError(f'{name}: missing keys')
    if d[KEY].isna().any().any():raise ValueError(f'{name}: null keys')
    dates=pd.to_numeric(d['日期'],errors='raise')
    if not np.isfinite(dates).all() or not dates.eq(np.floor(dates)).all():raise ValueError(f'{name}: invalid date key')
    d['日期']=dates.astype('int64')
    pd.to_datetime(d['日期'].astype(str),format='%Y%m%d',errors='raise')
    if not d['代码'].map(lambda x:isinstance(x,str) and bool(x.strip())).all():raise ValueError(f'{name}: invalid stock key')
    d=d.drop_duplicates().reset_index(drop=True)
    if d.duplicated(KEY).any():raise ValueError(f'{name}: conflicting duplicate stock/date')
    numeric=[c for c in d.select_dtypes(include='number') if c!='日期']
    if numeric:d[numeric]=d[numeric].replace([np.inf,-np.inf],np.nan)
    if name=='MarketData':
        cols=['open','high','low','close'];prices=d[cols];tol=1e-8*np.maximum(prices.abs().max(axis=1),1)
        invalid=prices.le(0).any(axis=1)|(d['high']+tol<prices[['open','close','low']].max(axis=1))|(d['low']-tol>prices[['open','close','high']].min(axis=1))
        d.loc[invalid,cols]=np.nan
        for c in ['volume','amount']:d.loc[d[c].lt(0),c]=np.nan
    elif name=='MarketDataRawOpen':d.loc[d['open'].le(0),'open']=np.nan
    elif name=='Factor33_winsor':
        for c in ['pe_ttm','pb','ps_ttm']:d.loc[d[c].le(0),c]=np.nan
    return d

def clean_tables(data):
    return {name:clean_table(data[name],name) for name in ['MarketData','Factor33_winsor','Barra_factor']}
