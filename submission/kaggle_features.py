"""Causal adaptations of primary Kaggle writeups, independent finite windows."""
import numpy as np
import pandas as pd
KEY=['日期','代码']
def extras(panel, ranked, peers=True):
    d=panel.sort_values(['代码','日期']).reset_index(drop=True)
    assert ranked[KEY].equals(d[KEY])
    cal=dict(zip(np.sort(d['日期'].unique()),range(d['日期'].nunique())))
    pieces=[]
    for code,g in d.groupby('代码',sort=False):
        n=len(g);pos=g['日期'].map(cal).to_numpy();c=g.close.to_numpy(float);o=g.adj_open.to_numpy(float)
        r=np.r_[np.nan,c[1:]/c[:-1]-1];r[1:][np.diff(pos)!=1]=np.nan
        values={'k_return1':r,'k_intra':c/o-1,'k_range':(g.high-g.low).to_numpy()/c}
        def window(a,w,fn):
            out=np.full(n,np.nan)
            if n>=w:
                v=np.lib.stride_tricks.sliding_window_view(a,w);out[w-1:]=fn(v)
                ok=np.r_[np.zeros(w-1,dtype=bool),pos[w-1:]-pos[:n-w+1]==w-1]
                out[~ok]=np.nan
            return out
        for w in [5,20,60]:
            lag=np.full(n,np.nan)
            if n>w:
                good=pos[w:]-pos[:-w]==w;lag[w:]=np.where(good,np.log(c[w:]/c[:-w]),np.nan)
            values[f'k_logret{w}']=lag
            values[f'k_logma{w}']=np.log(c/window(c,w,lambda x:x.mean(axis=1)))
            values[f'k_vol{w}']=window(r,w,lambda x:x.std(axis=1,ddof=1))
            t=np.arange(w)-(w-1)/2;logc=np.log(c)
            values[f'k_slope{w}']=window(logc,w,lambda x:(x*t).sum(axis=1)/(t*t).sum())
            values[f'k_trenderror{w}']=window(logc,w,lambda x:np.sqrt(np.mean((x-x.mean(axis=1,keepdims=True)-((x*t).sum(axis=1)/(t*t).sum())[:,None]*t)**2,axis=1)))
            # Finite EMA is exactly reproducible from the supplied120date window.
            weights=(1-2/(w+1))**np.arange(w-1,-1,-1);weights/=weights.sum()
            values[f'k_ema{w}']=np.log(c/window(c,w,lambda x:x@weights))
        for w in [16,49]:
            a=2*window(c,w//2,lambda x:x.mean(axis=1))-window(c,w,lambda x:x.mean(axis=1))
            hull=window(a,int(np.sqrt(w)),lambda x:x.mean(axis=1))
            values[f'k_hull{w}']=c/hull-1
        # Second differences and persistent own-history deviations.
        for name in ['k_return1','k_intra','k_logret5','k_vol20']:
            a=values[name];first=np.r_[np.nan,np.diff(a)];first[1:][np.diff(pos)!=1]=np.nan
            values[name+'_diff']=first
            values[name+'_diff2']=np.r_[np.nan,np.diff(first)]
            values[name+'_dev20']=a-window(a,20,lambda x:x.mean(axis=1))
        pieces.append(pd.DataFrame(values,index=g.index))
    raw=pd.concat(pieces).sort_index()
    for name in ['turn','volatility_5','daily_return','momentum_5','bias_20','roe_avg_ttm','roa_avg_ttm','gross_profit_rate_ttm','net_profit_rate_ttm','debt_to_asset_lf','beta_000300SH_22']:
        raw['k_'+name]=pd.to_numeric(d[name],errors='coerce')
    for name in ['total_market_cap','float_market_cap','pe_ttm','pb','ps_ttm','list_days','volume','amount']:
        a=pd.to_numeric(d[name],errors='coerce');raw['k_log_'+name]=np.log1p(a.where(a>=0))
    for name in [c for c in d if c.startswith('bar_')]:raw['k_'+name]=d[name]
    raw=raw.replace([np.inf,-np.inf],np.nan).clip(-1e6,1e6)
    temporal=[c for c in raw if any(k in c for k in ['hull','diff','dev20','slope','trenderror','ema'])]
    market_cols=['k_return1','k_intra','k_range','k_logret5','k_logret20','k_logret60','k_logma20','k_vol20','k_turn','k_bias_20','k_log_volume','k_log_total_market_cap']
    context={}
    for c in market_cols:
        gr=raw[c].groupby(d['日期']);mean=gr.transform('mean');sd=gr.transform('std')
        context[c+'_market']=mean;context[c+'_dispersion']=sd;context[c+'_relative']=raw[c]-mean
    rr=ranked[[c for c in ranked if c.endswith('_r') and c not in ['is_st_r','c_mom_119_r','c_vol_119_r','c_ma120_r']]].to_numpy()
    context.update(k_row_mean=rr.mean(axis=1),k_row_std=rr.std(axis=1),k_row_q10=np.quantile(rr,.1,axis=1),k_row_q50=np.quantile(rr,.5,axis=1),k_row_q90=np.quantile(rr,.9,axis=1))
    f=pd.concat([raw,pd.DataFrame(context,index=d.index)],axis=1)
    if peers:
        neighbor={f'k_peer{k}_{c}':np.full(len(d),np.nan) for k in [5,20] for c in ['k_return1','k_logret5','k_logret20','k_vol20','k_log_volume']}
        distance_names=['momentum_5_r','volatility_5_r','bar_size_r','turn_r','beta_000300SH_22_r','return_20_r']
        coords=ranked[distance_names].to_numpy(dtype=np.float32)
        for day,idx in d.groupby('日期',sort=False).indices.items():
            a=coords[idx];dist=np.maximum((a*a).sum(axis=1)[:,None]+(a*a).sum(axis=1)[None,:]-2*a@a.T,0);np.fill_diagonal(dist,np.inf)
            nearest=np.argsort(dist,axis=1,kind='stable')[:,:20]
            for k in [5,20]:
                for c in ['k_return1','k_logret5','k_logret20','k_vol20','k_log_volume']:
                    b=raw[c].to_numpy()[idx][nearest[:,:k]];valid=np.isfinite(b);den=valid.sum(axis=1)
                    neighbor[f'k_peer{k}_{c}'][idx]=np.divide(np.where(valid,b,0).sum(axis=1),den,out=np.full(len(idx),np.nan),where=den>0)
        f=pd.concat([f,pd.DataFrame(neighbor)],axis=1)
    f=f.astype('float32');out=pd.concat([d[KEY],f],axis=1)
    return out,dict(raw=list(raw.columns),market=list(context),temporal=temporal,peers=list(neighbor) if peers else [])

def sector_features(frame,ranked,mapping):
    sector=frame['代码'].map(mapping).fillna('UNKNOWN')
    cols=['k_return1','k_logret5','k_logret20','k_vol20','k_log_volume']
    result=frame.copy()
    for c in cols:
        mean=result[c].groupby([result['日期'],sector]).transform('mean')
        result[c+'_sector']=mean;result[c+'_sector_relative']=result[c]-mean
    for c in ['momentum_5_r','volatility_5_r','bar_size_r','turn_r','return_20_r']:
        result[c+'_sector_rank']=ranked[c].groupby([ranked['日期'],sector]).rank(pct=True)-.5
    return result,sector
