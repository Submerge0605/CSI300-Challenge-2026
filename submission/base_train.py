"""Reproduce the frozen three-seed compact ensemble from supplied 2016-2022 data only."""
import argparse,json
from pathlib import Path
import lightgbm as lgb
import pandas as pd
import numpy as np
try:
    from .features import normalize,build_features
    from .cleaning import clean_table
except ImportError:
    from features import normalize,build_features
    from cleaning import clean_table

PARAMS=dict(objective='regression',learning_rate=.025,num_leaves=15,max_depth=4,
    min_data_in_leaf=500,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,
    lambda_l2=20,verbosity=-1,num_threads=4,seed=42,deterministic=True,force_col_wise=True)

def load_data(root,end=20221231):
    if not 20160101<=end<=20221231:raise ValueError('Only supplied 2016-2022 data is allowed')
    root=Path(root)
    def read(name):
        paths=[p for p in sorted(root.glob(name+'/*/*/*.parquet')) if 201601<=int(p.stem)<=end//100]
        if not paths:raise FileNotFoundError(name)
        frame=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
        return frame.loc[frame['日期'].between(20160101,end)].copy()
    tables={name:clean_table(read(name),name) for name in ['MarketData','Factor33_winsor','Barra_factor']}
    raw=clean_table(read('MarketDataRawOpen'),'MarketDataRawOpen').rename(columns={'open':'raw_open'})
    m=tables['MarketData'][['日期','代码','is_suspend']].merge(raw,on=['日期','代码'],validate='1:1')
    m=m.sort_values(['代码','日期']).reset_index(drop=True)
    days=np.sort(m['日期'].unique());positions=dict(zip(days,range(len(days))))
    pos=m['日期'].map(positions);g=m.groupby('代码',sort=False)
    valid=pd.Series(True,index=m.index)
    for lag in [1,2]:
        valid &= g['日期'].shift(-lag).map(positions).eq(pos+lag)
        price=g['raw_open'].shift(-lag)
        valid &= g['is_suspend'].shift(-lag).eq(0)&price.gt(0)&np.isfinite(price)
    y=g['raw_open'].shift(-2)/g['raw_open'].shift(-1)-1
    labels=m[['日期','代码']].assign(y=y.where(valid&np.isfinite(y)),exit_date=g['日期'].shift(-2))
    features=build_features(normalize(tables))
    labels=features[['日期','代码']].merge(labels,on=['日期','代码'],how='left',validate='1:1')
    return features,labels,tables

def fit_model(X,L,cutoff,seed=42):
    mask=L.y.notna()&L.exit_date.le(cutoff)
    cols=[c for c in X if c.endswith('_r') and c not in ['c_mom_119_r','c_vol_119_r','c_ma120_r']]
    target=L.loc[mask,'y'].groupby(L.loc[mask,'日期']).rank(pct=True)-.5
    model=lgb.train(PARAMS|{'seed':seed},lgb.Dataset(X.loc[mask,cols],label=target,feature_name=cols),num_boost_round=250)
    return model,cols,int(mask.sum())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True)
    ap.add_argument('--out-dir',default=str(Path(__file__).parent/'model'));ap.add_argument('--cutoff',type=int,default=20221231)
    a=ap.parse_args();X,L,_=load_data(a.data_root,a.cutoff)
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True);specs=[]
    for i,seed in enumerate([42,43,44]):
        model,cols,n=fit_model(X,L,a.cutoff,seed);file=f'lgb_{i}.txt';model.save_model(str(out/file))
        specs.append(dict(file=file,features=cols,weight=1/3,kind='compactbag3',rounds=250,params=PARAMS|{'seed':seed},train_rows=n))
    meta=dict(models=specs,label='raw_open(t+2)/raw_open(t+1)-1',neutral_weight=0,
        aggregation='mean_daily_percentile',train_cutoff=a.cutoff,max_lookback=120,public_data_only=True)
    (out/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(train_rows=n,features=len(cols),models=3,cutoff=a.cutoff)),flush=True)
if __name__=='__main__':main()
