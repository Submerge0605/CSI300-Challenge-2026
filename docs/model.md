# v10模型说明

v10由两个监督目标、六个LightGBM Booster和一个固定的截面融合组成。六份权重沿用v9，v10改变的是预测聚合方式。

## 标签和输入

对信号日$t$、股票$i$，raw-open监督标签为：

$$y_{i,t}=O^{raw}_{i,t+2}/O^{raw}_{i,t+1}-1.$$

训练要求两个端点开盘价有限且为正、非停牌，并检查它们是全市场日历上的连续交易日。训练截止日还要覆盖标签的退出日期。

推理读取`MarketData`、`Factor33_winsor`和`Barra_factor`。训练另外读取`MarketDataRawOpen`构造标签。三张输入表按`日期、代码`合并，以Barra表的当日成员形成预测范围。

每次推理在输入中选取截至信号日最近最多120个市场日期。共享清洗会丢弃`is_st`，对冲突主键报错，并处理不合法数值。源码见[`cleaning.py`](../submission/cleaning.py)。

## 229维特征

| 通道 | 维数 | 主要内容 |
| --- | ---: | --- |
| 截面排名 | 124 | 组织方因子、Barra暴露、量价比例、趋势和滚动统计 |
| 原值及市场状态 | 105 | 对数收益、均线偏离、有限EMA、趋势斜率、波动、差分、当日均值和离散度 |

先按股票计算历史统计，再按日期构造截面表示。截面排名使用`rank(pct=True)-0.5`，缺失排名填0；原值通道保留LightGBM可以处理的缺失值。有限窗口实现让完整训练历史和截断推理使用同一计算定义。

Alpha158式形态与滚动特征见[`alpha_features.py`](../submission/alpha_features.py)；新增量价及市场通道见[`kaggle_features.py`](../submission/kaggle_features.py)。v10调用`peers=False`，没有使用该文件中保留的邻居或行业辅助实验。

## 两个监督目标

排名分支以同一信号日有效H1标签的截面百分位排名减0.5为目标：

$$y^{rank}_{i,t}=rank_t(y_{i,t})-0.5.$$

utility分支保留收益幅度的一部分：

$$y^{utility}_{i,t}=0.2\cdot clip\left(\frac{y_{i,t}-\bar y_t}{\max(s_t,10^{-6})},-3,3\right).$$

$s_t$采用同一信号日有效股票H1收益的截面样本标准差；缺失结果填0。两个分支都使用平方误差回归。utility目标是标准化收益，组合IR由后续配置、执行与回测计算。

## 固定配方

| 参数 | 值 |
| --- | --- |
| `objective` | `regression` |
| `extra_trees` | `True` |
| `learning_rate` | 0.025 |
| `num_leaves` / `max_depth` | 63 / 8 |
| `min_data_in_leaf` | 500 |
| `feature_fraction` / `bagging_fraction` | 0.8 / 0.8 |
| `bagging_freq` | 1 |
| `lambda_l2` | 20 |
| `num_boost_round` | 350 |
| seeds | 42、43、44 |
| 训练线程 | 3 |
| 每个模型成熟训练行数 | 499,799 |
| 最终训练截止日 | 2022-12-31 |

`extra_trees=True`是LightGBM的额外随机阈值机制，模型仍按梯度提升训练。每个Booster有350棵树，六个模型合计2,100棵。

## 融合顺序

令$R_t$表示日截面百分位排名，$e_s$和$u_s$分别是两个分支在种子$s$下的原始输出。先逐模型排名：

$$E_s=R_t(e_s),\qquad U_s=R_t(u_s).$$

再分别平均并重新排名：

$$A=R_t\left(\frac13\sum_s E_s\right),\qquad B=R_t\left(\frac13\sum_s U_s\right).$$

用种子间总体标准差构造分歧项：

$$D=0.75\,std_{pop}(E_s)+0.25\,std_{pop}(U_s).$$

最终输出：

$$score=R_t(0.75A+0.25B-0.25D).$$

较大的种子分歧会降低股票位置。这是一个固定的敏感度惩罚，标准差使用`ddof=0`。比例、排名顺序和特征列顺序都属于发布模型的定义，源码见[`inference.py`](../submission/inference.py)。

## 文件身份

[`release_manifest.json`](../release_manifest.json)保存计算源码、配置、权重与meta的SHA-256。原历史提交ZIP的SHA-256为：

```text
85e10d7462d027ff760c09a49ad26df460fcebcf836e97e271769287eb34a432
```

公开仓库补充了文档和测试，并替换了旧README；计算源码与模型权重保持原字节。运行`python scripts/verify_artifacts.py`可检查当前副本。
