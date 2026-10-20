"""Fixed observable transformations. No learned parameter or retained history changes."""
import numpy as np
import pandas as pd
RISK=['bar_size_r','bar_beta_r','bar_momentum_r','bar_residvol_r','bar_nlsize_r','bar_btop_r','bar_liquidity_r','bar_earnings_r','bar_growth_r','bar_leverage_r']
VARIANTS=['stfree','smooth25','smooth50','risk25','volhalf','capacity25','return25','riskreturn25']
def rank(a,d):return pd.Series(np.asarray(a)).groupby(pd.Series(np.asarray(d))).rank(pct=True).to_numpy()
def transform(frame,extra,variant,compact=None):
    if variant not in VARIANTS:raise ValueError('Unregistered transformation')
    day=frame['日期'].reset_index(drop=True);score=rank(extra,day)
    if variant.startswith('smooth'):
        cal=dict(zip(np.sort(day.unique()),range(day.nunique())));pos=day.map(cal)
        temp=pd.DataFrame({'code':frame['代码'].to_numpy(),'pos':pos,'score':score})
        g=temp.groupby('code',sort=False);past=g.score.shift(1).where(pos.sub(g.pos.shift(1)).eq(1)).fillna(temp.score).to_numpy()
        lag=.25 if variant=='smooth25' else .5;score=(score+lag*past)/(1+lag)
    elif variant=='risk25':
        score=score.copy()
        for date,idx in day.groupby(day,sort=False).indices.items():
            b=frame.iloc[idx][RISK].to_numpy(dtype=float);b=b-b.mean(axis=0)
            component=b@np.linalg.solve(b.T@b+.001*np.eye(len(RISK)),b.T@(score[idx]-score[idx].mean()))
            score[idx]-=.25*component
    elif variant=='volhalf':score=(score-.5)/np.sqrt(np.maximum(frame.k_vol20.fillna(.02).to_numpy(),.005))
    elif variant in ['capacity25','return25','riskreturn25']:
        if compact is None:raise ValueError('Missing compact family')
        score=.75*score+.25*rank(compact,day)
    return rank(score,day)
