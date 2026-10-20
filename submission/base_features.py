"""Causal cross-sectional features shared by training and inference."""
import numpy as np
import pandas as pd

KEY = ['日期', '代码']
BARRA = dict(zip(['市值','贝塔','动量','残差波动','非线性市值','账面市值比','流动性','盈利','成长','杠杆'],
                 ['bar_size','bar_beta','bar_momentum','bar_residvol','bar_nlsize','bar_btop','bar_liquidity','bar_earnings','bar_growth','bar_leverage']))
FACTORS = ['turn','change_ratio','daily_return','momentum_5','reversal_5','volatility_5',
           'sma_20','ema_20','macd_diff_12_26_9','macd_dea_12_26_9','macd_hist_12_26_9',
           'rsi_12','kdj_k_9_3_3','kdj_d_9_3_3','bias_20','cci_14','atr_14',
           'total_market_cap','float_market_cap','pe_ttm','pb','ps_ttm','roe_avg_ttm',
           'roa_avg_ttm','gross_profit_rate_ttm','net_profit_rate_ttm','debt_to_asset_lf',
           'current_ratio_lf','beta_000300SH_22','list_days','is_new']
BASE = FACTORS + list(BARRA.values())
LONG = ['c_mom_60','c_mom_119','c_vol_60','c_vol_119','c_ma60','c_ma120',
        'c_amihud_20','c_hlrange_20','c_overnight_20','c_intraday_20','c_maxdd_60',
        'c_volratio_20','c_skew_60','c_turn_std_20']

def neutralize(scores, exposure_frame, weight):
    """Contemporaneous risk projection; no fitted model or state is retained."""
    rank=pd.Series(scores).rank(pct=True).to_numpy()
    if weight==0: return rank
    exposure=exposure_frame[[v+'_r' for v in BARRA.values()]].to_numpy(dtype=float)
    exposure=exposure-exposure.mean(axis=0)
    gram=exposure.T@exposure+0.001*np.eye(exposure.shape[1])
    component=exposure@np.linalg.solve(gram,exposure.T@(rank-rank.mean()))
    return rank-weight*component

def normalize(data):
    bar = data['Barra_factor'].drop(columns=['名称'], errors='ignore').rename(columns=BARRA)
    market = data['MarketData'].drop(columns=['名称'], errors='ignore').rename(columns={'open':'adj_open'})
    factors = data['Factor33_winsor'].drop(columns=['名称','is_suspend','is_st'], errors='ignore')
    # Membership is observable at t. Never remove a feature row using a future label.
    return bar.merge(market,on=KEY,how='left',validate='1:1').merge(
        factors,on=KEY,how='left',validate='1:1').sort_values(['代码','日期']).reset_index(drop=True)

def build_features(panel):
    d=panel.sort_values(['代码','日期']).reset_index(drop=True).copy()
    dates=np.sort(d['日期'].unique())
    d['_pos']=d['日期'].map(dict(zip(dates,range(len(dates)))))
    g=d.groupby('代码',sort=False)
    def contiguous(lag):
        return d['_pos'].sub(g['_pos'].shift(lag)).eq(lag)
    def rolling(s,w,method,lag=None):
        grouped=s.groupby(d['代码'],sort=False)
        def calculate(x):
            a=x.to_numpy(dtype=float); result=np.full(len(a),np.nan)
            if len(a)<w: return result
            v=np.lib.stride_tricks.sliding_window_view(a,w)
            if method=='mean': values=np.mean(v,axis=1)
            elif method=='std': values=np.std(v,axis=1,ddof=1)
            elif method=='max': values=np.max(v,axis=1)
            elif method=='skew':
                centered=v-v.mean(axis=1,keepdims=True)
                sd=np.std(v,axis=1,ddof=1)
                with np.errstate(divide='ignore',invalid='ignore'):
                    values=w/((w-1)*(w-2))*np.sum((centered/sd[:,None])**3,axis=1)
                values=np.where(sd==0,0.0,values)
            else: raise ValueError(method)
            result[w-1:]=values; return result
        # Each statistic depends on its exact window, avoiding cumulative
        # floating-point drift between full-history training and truncated inference.
        r=grouped.transform(calculate)
        return r.where(contiguous(w-1 if lag is None else lag))
    close=d['close']; op=d['adj_open']; prev=g['close'].shift(1)
    ret=(close/prev-1).where(contiguous(1))
    intra=close/op-1
    overnight=(op/prev-1).where(contiguous(1))
    hl=(d['high']-d['low'])/close
    for w in [60,119]:
        d[f'c_mom_{w}']=(close/g['close'].shift(w)-1).where(contiguous(w))
        d[f'c_vol_{w}']=rolling(ret,w,'std',lag=w)
    for w in [60,120]: d[f'c_ma{w}']=close/rolling(close,w,'mean')-1
    d['c_amihud_20']=rolling(ret.abs()/(d['amount'].abs()+1),20,'mean',lag=20)
    d['c_hlrange_20']=rolling(hl,20,'mean')
    d['c_overnight_20']=rolling(overnight,20,'mean',lag=20)
    d['c_intraday_20']=rolling(intra,20,'mean')
    d['c_maxdd_60']=close/rolling(close,60,'max')-1
    d['c_volratio_20']=d['volume']/rolling(d['volume'],20,'mean')
    d['c_skew_60']=rolling(ret,60,'skew',lag=60)
    d['c_turn_std_20']=rolling(d['turn'],20,'std')
    extra=[]
    def add(name,value):
        d[name]=value; extra.append(name)
    add('intraday_1',intra); add('overnight_1',overnight)
    add('range_1',hl); add('close_location',(2*close-d['high']-d['low'])/(d['high']-d['low']).replace(0,np.nan))
    for w in [2,5,10,20]:
        add(f'return_{w}',(close/g['close'].shift(w)-1).where(contiguous(w)))
        add(f'intraday_{w}',rolling(intra,w,'mean'))
        add(f'overnight_{w}',rolling(overnight,w,'mean',lag=w))
        add(f'range_{w}',rolling(hl,w,'mean'))
        add(f'volume_ratio_{w}',d['volume']/rolling(d['volume'],w,'mean'))
    for w in [5,20]: add(f'return_vol_{w}',rolling(ret,w,'std',lag=w))
    for name in ['macd_diff_12_26_9','macd_dea_12_26_9','macd_hist_12_26_9','atr_14']:
        add(name+'_relative',d[name]/close)
    # New price-derived features use ratios rather than extra price levels.
    raw=d[BASE+LONG+extra].replace([np.inf,-np.inf],np.nan).copy()
    for c in ['pe_ttm','pb','ps_ttm']: raw.loc[raw[c]<=0,c]=np.nan
    ranked=raw.groupby(d['日期']).rank(pct=True)-0.5
    ranked=ranked.fillna(0).astype('float32').add_suffix('_r')
    return pd.concat([d[KEY],ranked],axis=1)
