"""Stateless <=120-date forecast with fixed ensemble disagreement penalty."""
from pathlib import Path
import json
import numpy as np,pandas as pd,lightgbm as lgb
try:
    from .features import normalize,build_features
    from .kaggle_features import extras
    from .postprocess import rank
except ImportError:
    from features import normalize,build_features
    from kaggle_features import extras
    from postprocess import rank
KEY=['日期','代码']
class InferenceModel:
    def __init__(self,model_dir:str):
        root=Path(model_dir);self.meta=json.loads((root/'meta.json').read_text(encoding='utf-8'))
        if self.meta['kind']!='stfree_extra_trees' or self.meta['variant']!='confidence':raise ValueError('Unknown deployment policy')
        self.models=[(s,lgb.Booster(model_file=str(root/s['file']))) for s in self.meta['models']]
        for s,m in self.models:
            if m.feature_name()!=s['features'] or any('is_st' in c for c in s['features']):raise ValueError('Invalid feature schema')
        if len(self.models)!=6 or any(sum(s['family']==f for s,m in self.models)!=3 for f in ['extra','utility']):raise ValueError('Expected six models')
    def predict(self,as_of_date:str,data:dict[str,pd.DataFrame])->pd.DataFrame:
        day=int(as_of_date.replace('-',''));days=np.sort(data['Barra_factor']['日期'].unique());days=days[days<=day][-120:]
        if not len(days) or days[-1]!=day:raise ValueError('Missing signal day')
        packet={k:v.loc[v['日期'].isin(days)].drop(columns=['is_st'],errors='ignore').copy() for k,v in data.items() if k in ['MarketData','Factor33_winsor','Barra_factor']}
        panel=normalize(packet);X=build_features(panel);E,_=extras(panel,X,peers=False);matrix=pd.concat([X,E.drop(columns=KEY)],axis=1);cur=matrix.loc[matrix['日期'].eq(day)].reset_index(drop=True);scores={}
        for s,m in self.models:scores.setdefault(s['family'],[]).append(rank(m.predict(cur[s['features']],num_threads=1),cur['日期']))
        extra=rank(np.mean(scores['extra'],axis=0),cur['日期']);utility=rank(np.mean(scores['utility'],axis=0),cur['日期']);disagreement=.75*np.std(scores['extra'],axis=0)+.25*np.std(scores['utility'],axis=0)
        result=rank(.75*extra+.25*utility-.25*disagreement,cur['日期']);codes=cur['代码'].to_numpy()
        if len(codes)==0 or len(set(codes))!=len(codes) or not np.isfinite(result).all():raise ValueError('Invalid prediction')
        return pd.DataFrame(dict(date=as_of_date,code=codes,score=result))
