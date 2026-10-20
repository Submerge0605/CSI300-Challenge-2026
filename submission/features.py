"""Shared causal feature construction for training and stateless inference."""
try:
    from .base_features import normalize as base_normalize, build_features as base_features, neutralize
    from .cleaning import clean_tables
    from .alpha_features import alpha_features
except ImportError:
    from base_features import normalize as base_normalize, build_features as base_features, neutralize
    from cleaning import clean_tables
    from alpha_features import alpha_features

def normalize(data):
    return base_normalize(clean_tables(data))

def build_features(panel):
    base=base_features(panel)
    alpha=alpha_features(panel)
    return base.merge(alpha,on=['日期','代码'],how='left',validate='1:1',sort=False)
