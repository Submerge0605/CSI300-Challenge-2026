"""Deterministic offline refit; original data is read only and is_st discarded."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd,lightgbm as lgb
try:
    from .base_train import load_data,PARAMS
    from .features import normalize
    from .kaggle_features import extras
    from .postprocess import VARIANTS
except ImportError:
    from base_train import load_data,PARAMS
    from features import normalize
    from kaggle_features import extras
    from postprocess import VARIANTS
def main():
    policy=Path(__file__).parent/'deployment_policy.json';default=json.loads(policy.read_text())['variant'] if policy.exists() else 'stfree'
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);ap.add_argument('--cutoff',type=int,default=20221231);ap.add_argument('--out-dir',default=str(Path(__file__).parent/'model'));ap.add_argument('--variant',choices=['confidence'],default=default);a=ap.parse_args()
    X,L,tables=load_data(a.data_root,a.cutoff);E,g=extras(normalize(tables),X,peers=False);base=[c for c in X if c.endswith('_r') and c not in ['is_st_r','c_mom_119_r','c_vol_119_r','c_ma120_r']];F=pd.concat([X[base],E[g['raw']+g['market']]],axis=1);assert not any('is_st' in c for c in F)
    mask=L.y.notna()&L.exit_date.le(a.cutoff);target=L.loc[mask,'y'].groupby(L.loc[mask,'日期']).rank(pct=True).to_numpy()-.5;dest=Path(a.out_dir);dest.mkdir(parents=True,exist_ok=True);models=[]
    families=['extra','utility']
    for family in families:
        params=PARAMS|dict(num_threads=3)
        if family in ['extra','utility']:params.update(extra_trees=True,num_leaves=63,max_depth=8,min_data_in_leaf=500);rounds=350
        else:params.update(num_leaves=15,max_depth=4,min_data_in_leaf=1000);rounds=250
        family_target=target
        if family=='utility':
            y=pd.Series(L.loc[mask,'y'].to_numpy());dates=L.loc[mask,'日期'].reset_index(drop=True)
            if a.variant=='riskreturn25':y/=np.maximum(F.loc[mask,'k_vol20'].fillna(.02).to_numpy(),.005)
            group=y.groupby(dates);family_target=((y-group.transform('mean'))/group.transform('std').clip(lower=1e-6)).clip(-3,3).fillna(0).to_numpy()*.2
        for seed in [42,43,44]:
            m=lgb.train(params|dict(seed=seed),lgb.Dataset(F.loc[mask],label=family_target),num_boost_round=rounds);file=f'{family}_{seed}.txt';m.save_model(str(dest/file));models.append(dict(file=file,family=family,features=list(F.columns),seed=seed,params=params|dict(seed=seed),rounds=rounds,train_rows=int(mask.sum())))
            print(json.dumps(dict(event='fit',family=family,seed=seed,rows=int(mask.sum()),features=F.shape[1])),flush=True)
    meta=dict(kind='stfree_extra_trees',variant=a.variant,models=models,train_cutoff=a.cutoff,max_lookback=120,label='raw_open(t+2)/raw_open(t+1)-1',is_st_policy='never read',inference_updates=False,pristine_holdout=False)
    (dest/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
