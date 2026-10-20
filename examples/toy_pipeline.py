"""Educational synthetic panel only: no competition data, SDK or saved weights."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline


def make_toy_panel(seed=2026, n_days=360, n_stocks=24):
    rng = np.random.default_rng(seed)
    # Synthetic business days are NOT the real Chinese exchange calendar.
    days = pd.bdate_range("2020-01-01", periods=n_days)
    codes = [f"S{i:03d}" for i in range(n_stocks)]
    common = rng.normal(0.0001, 0.004, (n_days, 1))
    shocks = common + rng.normal(0, 0.009, (n_days, n_stocks))
    opens = rng.uniform(8, 60, n_stocks) * np.exp(np.cumsum(shocks, axis=0))
    closes = opens * np.exp(rng.normal(0, 0.003, opens.shape))
    volumes = rng.lognormal(11.0, 0.6, opens.shape)
    frames = []
    for j, code in enumerate(codes):
        frames.append(pd.DataFrame({
            "date": days, "code": code, "open_raw": opens[:, j],
            "close": closes[:, j], "volume": volumes[:, j],
        }))
    return pd.concat(frames, ignore_index=True), days


def add_features_and_labels(panel):
    d = panel.sort_values(["code", "date"]).reset_index(drop=True).copy()
    group = d.groupby("code", sort=False)
    d["ret1"] = group["close"].pct_change(fill_method=None)
    d["mom5"] = group["close"].pct_change(5, fill_method=None)
    d["vol10"] = d.groupby("code")["ret1"].transform(
        lambda s: s.rolling(10, min_periods=10).std(ddof=1)
    )
    ma20 = group["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    d["ma20_gap"] = d["close"] / ma20 - 1
    d["log_volume"] = np.log1p(d["volume"])
    raw_features = ["ret1", "mom5", "vol10", "ma20_gap", "log_volume"]
    features = []
    for name in raw_features:
        ranked = name + "_rank"
        d[ranked] = d.groupby("date")[name].rank(pct=True) - 0.5
        features.append(ranked)
    # Every synthetic stock has every date: row-shift equals calendar-shift here.
    # Real missing-date data require global calendar continuity checks.
    entry = group["open_raw"].shift(-1)
    exit_ = group["open_raw"].shift(-2)
    d["y"] = exit_ / entry - 1
    d["exit_date"] = group["date"].shift(-2)
    d["y_rank"] = d.groupby("date")["y"].rank(pct=True) - 0.5
    return d, features


def daily_rank_ic(frame):
    def one_day(g):
        if len(g) < 3 or g["score"].nunique() < 2 or g["y"].nunique() < 2:
            return np.nan
        return g["score"].rank().corr(g["y"].rank())
    clean = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["score", "y"])
    return clean.groupby("date", sort=True)[["score", "y"]].apply(one_day)


def toy_top_k_proxy(frame, k=5):
    # Educational return proxy, NOT the competition execution engine.
    r = frame.pivot(index="date", columns="code", values="y").sort_index()
    score = frame.pivot(index="date", columns="code", values="score").reindex_like(r)
    if r.isna().any().any():
        raise ValueError("This simplified proxy requires a complete synthetic panel")
    selected = score.rank(axis=1, method="first", ascending=False).le(k)
    w = selected.astype(float).div(selected.sum(axis=1), axis=0)
    delta = w.diff()
    delta.iloc[0] = w.iloc[0]     # Start from cash; no final liquidation assumed.
    buys = delta.clip(lower=0).sum(axis=1)
    sells = (-delta.clip(upper=0)).sum(axis=1)
    turnover = buys + sells
    # Toy assumptions only, NOT current legal tax rates or hidden-platform costs.
    cost = 0.0003 * turnover + 0.001 * sells
    gross = (w * r).sum(axis=1)
    net = gross - cost
    benchmark = r.mean(axis=1)   # Synthetic equal-weight reference, not CSI300.
    active = net - benchmark
    n = len(net)
    nav = (1 + net).cumprod()
    bench_nav = (1 + benchmark).cumprod()
    cagr = nav.iloc[-1] ** (252 / n) - 1
    bench_cagr = bench_nav.iloc[-1] ** (252 / n) - 1
    tracking_error = active.std(ddof=1) * np.sqrt(252)
    ir = (cagr - bench_cagr) / tracking_error if tracking_error > 0 else np.nan
    path = np.r_[1.0, nav.to_numpy()]
    drawdown = 1 - path / np.maximum.accumulate(path)
    return {
        "toy_net_CAGR": float(cagr),
        "toy_IR_CAGR_definition": float(ir),
        "toy_max_drawdown": float(drawdown.max()),
        "toy_cumulative_two_sided_turnover": float(turnover.sum()),
    }


def main():
    panel, days = make_toy_panel()
    d, features = add_features_and_labels(panel)
    train_cutoff = days[239]
    valid_start = days[250]
    train_mask = d["date"].le(train_cutoff) & d["exit_date"].le(train_cutoff) & d["y"].notna()
    valid_mask = d["date"].ge(valid_start) & d["y"].notna()
    train, valid = d.loc[train_mask].copy(), d.loc[valid_mask].copy()
    assert train["exit_date"].max() <= train_cutoff < valid_start
    assert set(features).isdisjoint({"y", "y_rank", "exit_date", "open_raw"})

    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    ridge.fit(train[features], train["y_rank"])

    trees = lgb.train(
        {"objective": "regression", "learning_rate": 0.05, "num_leaves": 15,
         "min_data_in_leaf": 40, "lambda_l2": 10.0, "seed": 2026,
         "num_threads": 1, "verbosity": -1, "deterministic": True, "force_col_wise": True},
        lgb.Dataset(train[features].astype("float32"), label=train["y_rank"]),
        num_boost_round=60,
    )
    print("SYNTHETIC EDUCATION ONLY; no deliberately planted predictable alpha.")
    print("train rows:", len(train), "validation rows:", len(valid))
    for name, model in [("Ridge", ridge), ("LightGBM", trees)]:
        scored = valid[["date", "code", "y"]].copy()
        scored["score"] = model.predict(valid[features])
        ic = daily_rank_ic(scored)
        print(name, "mean daily RankIC:", round(float(ic.mean()), 6))
        print(toy_top_k_proxy(scored))


if __name__ == "__main__":
    main()
