# 参考资料

## 特征与竞赛方案

- [Microsoft Qlib Alpha158](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py)：量价形态、趋势与滚动统计的设计参考。
- [Ubiquant 第7名方案：Wenrui Kong](https://www.kaggle.com/competitions/ubiquant-market-prediction/discussion/338293)：Extra Trees与特征工程的研究线索。
- [Ubiquant 第8名方案：kaggle-doer](https://www.kaggle.com/competitions/ubiquant-market-prediction/discussion/338236)：市场状态、截面表示与历史变化。
- [G-Research Crypto 第9名方案：bturan19](https://www.kaggle.com/competitions/g-research-crypto-forecasting/discussion/324180)：平滑、量价与时间窗口的参考。
- [Optiver 第1名方案：nyanp](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970)：同伴信息与特征构造的实验来源。
- [完整24篇方案与实验索引](kaggle_ideas.md)：包括JPX、Ubiquant、G-Research Crypto、Optiver、Jane Street、Two Sigma等比赛。

这些来源提供了研究方向；本仓库的输入、日频标签、模型规模和120日推理窗口按本赛规则实现。

## 训练与验证

- [LightGBM参数文档](https://lightgbm.readthedocs.io/en/stable/Parameters.html)：额外随机阈值、采样、叶约束与确定性参数。
- [LightGBM Booster API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html)：加载、预测与离线refit。
- [scikit-learn时间序列验证](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)：按时间划分的基本做法。
- [The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)：多轮回测选择带来的偏差。
- [The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)：非正态收益和多重试验下的绩效评价。
- [WILDS](https://proceedings.mlr.press/v139/koh21a.html)：分布变化下的模型评估。

## 工具与发布

- [skd-backtest](https://github.com/HenryZ16/skd-backtest)：比赛本地回测工具，指标定义及执行细节以对应版本源码为准。
- [冠军公开仓库](https://github.com/Star6azeR/CSI300-Challenge-2026/)：赛后材料组织参考。
- [比赛平台](https://datatech.geekpie.club/c/csi300-2026)：平台成绩来源。

代码生成、资料整理和实验审查使用了LLM与Agent辅助。最后选包还参考了本地StartLux-Decision的有限选项建议；决定仍依据归档实验和检查结果。
