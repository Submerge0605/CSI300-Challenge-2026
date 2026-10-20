"""Causal OHLCV ratios inspired by Qlib Alpha158, implemented with independent windows.
Reference: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py
No Qlib dependency, external observations or fitted feature transforms.
"""
import numpy as np
import pandas as pd

def alpha_features(panel):
    d=panel.sort_values(['代码','日期']).reset_index(drop=True)
    parts=[]
    calendar=dict(zip(np.sort(d['日期'].unique()),range(d['日期'].nunique())))
    for code,g in d.groupby('代码',sort=False):
        close=g.close.to_numpy(float);op=g.adj_open.to_numpy(float)
        high=g.high.to_numpy(float);low=g.low.to_numpy(float);vol=g.volume.to_numpy(float)
        pos=g['日期'].map(calendar).to_numpy();n=len(g)
        spread=np.where(high>low,high-low,np.nan)
        values={'a_body_range':(close-op)/spread,'a_upper_open':(high-np.maximum(close,op))/op,
            'a_upper_range':(high-np.maximum(close,op))/spread,
            'a_lower_open':(np.minimum(close,op)-low)/op,
            'a_lower_range':(np.minimum(close,op)-low)/spread,
            'a_shift_open':(2*close-high-low)/op}
        ret=np.r_[np.nan,close[1:]/close[:-1]-1];ret[1:][np.diff(pos)!=1]=np.nan
        logv=np.log1p(np.maximum(vol,0));dv=np.r_[np.nan,np.diff(logv)]
        dv[1:][np.diff(pos)!=1]=np.nan
        def corr(a,b):
            ac=a-a.mean(axis=1,keepdims=True);bc=b-b.mean(axis=1,keepdims=True)
            denom=np.sqrt(np.sum(ac*ac,axis=1)*np.sum(bc*bc,axis=1))
            return np.divide(np.sum(ac*bc,axis=1),denom,out=np.full(len(a),np.nan),where=denom>1e-15)
        for w in [5,20,60]:
            if n<w:
                for name in ['slope','rsqr','resi','high','low','rsv','tsrank','pvcor','rvcor','upcount','sump','vsump']:
                    values[f'a_{name}_{w}']=np.full(n,np.nan)
                continue
            def view(a): return np.lib.stride_tricks.sliding_window_view(a,w)
            c=view(close);h=view(high);l=view(low);v=view(vol);r=view(ret)
            t=np.arange(w)-(w-1)/2;cm=c.mean(axis=1);cc=c-cm[:,None]
            slope=np.sum(cc*t,axis=1)/np.sum(t*t);ss=np.sum(cc*cc,axis=1)
            rsqr=np.divide(slope*slope*np.sum(t*t),ss,out=np.full(len(c),np.nan),where=ss>1e-15)
            hi=h.max(axis=1);lo=l.min(axis=1);delta=hi-lo
            va=v*np.abs(r);vr=np.where(r>0,va,0).sum(axis=1)
            den=np.abs(r).sum(axis=1);vden=va.sum(axis=1)
            outputs=dict(slope=slope/close[w-1:],rsqr=rsqr,
                resi=(c[:,-1]-cm-slope*t[-1])/close[w-1:],high=hi/close[w-1:],low=lo/close[w-1:],
                rsv=np.divide(c[:,-1]-lo,delta,out=np.full(len(c),np.nan),where=delta>0),
                tsrank=(np.sum(c<c[:,-1,None],axis=1)+.5*np.sum(c==c[:,-1,None],axis=1))/w,
                pvcor=corr(c,view(logv)),rvcor=corr(r,view(dv)),
                upcount=np.where(np.isnan(r),np.nan,r>0).mean(axis=1),
                sump=np.divide(np.maximum(r,0).sum(axis=1),den,out=np.full(len(c),np.nan),where=den>0),
                vsump=np.divide(vr,vden,out=np.full(len(c),np.nan),where=vden>0))
            for name,a in outputs.items():
                result=np.full(n,np.nan);result[w-1:]=a
                lag=w if name in ['rvcor','upcount','sump','vsump'] else w-1
                valid=np.r_[np.zeros(lag,dtype=bool),(pos[lag:]-pos[:-lag])==lag]
                result[~valid]=np.nan;values[f'a_{name}_{w}']=result
        f=pd.DataFrame(values,index=g.index)
        parts.append(f)
    raw=pd.concat(parts).sort_index().replace([np.inf,-np.inf],np.nan)
    ranked=raw.groupby(d['日期']).rank(pct=True).sub(.5).fillna(0).astype('float32').add_suffix('_r')
    return pd.concat([d[['日期','代码']],ranked],axis=1)
