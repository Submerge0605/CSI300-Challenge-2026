# 数据与预测接口

推理使用三个表，重新训练还需要原始开盘价表。原始行情、标签缓存与比赛SDK不随仓库发布；数据和SDK由使用者通过有权限的来源取得。

## 表与主键

所有表使用共同主键：

- `日期`：整数`YYYYMMDD`，例如`20220707`，各表类型一致。
- `代码`：非空字符串，各表编码一致，保留前导零及原有后缀。

每个表的`日期、代码`组合应唯一。清洗会合并完全相同的重复行，遇到同键但内容冲突的行则报错。数值列可含NaN；缺一整列与列中有缺失值不同，模型不会自动补齐必要字段。

|表名|推理必要列|训练增加的要求|
|---|---|---|
|`MarketData`|日期、代码、open、high、low、close、volume、amount|还需要`is_suspend`，0表示未停牌|
|`Factor33_winsor`|日期、代码及下面的31列|相同|
|`Barra_factor`|日期、代码及下面的10列|相同|
|`MarketDataRawOpen`|推理不使用|日期、代码、open|

`名称`可省略。`is_st`会丢弃，不作为特征或股票筛选依据；Factor33表的`is_suspend`也会丢弃，训练标签使用MarketData中的停牌状态。其余额外列应避免在表之间产生同名冲突。

### MarketData

`open、high、low、close`按数据提供方的原有价格口径输入；源码会把该表的`open`重命名为`adj_open`。`volume、amount`为非负数值，量额单位与训练数据保持一致。不要用`MarketDataRawOpen`替换这张特征表，也不要在接入时另行改变复权口径。

非正价格、明显不成立的OHLC关系、负成交量或成交额、无穷值，会按`submission/cleaning.py`转为缺失。正常的负收益仍然保留。

### Factor33_winsor

当前模型路径需要以下31列；表名不代表必须把所有原始因子都用于预测。

```text
turn change_ratio daily_return momentum_5 reversal_5 volatility_5
sma_20 ema_20 macd_diff_12_26_9 macd_dea_12_26_9 macd_hist_12_26_9
rsi_12 kdj_k_9_3_3 kdj_d_9_3_3 bias_20 cci_14 atr_14
total_market_cap float_market_cap pe_ttm pb ps_ttm
roe_avg_ttm roa_avg_ttm gross_profit_rate_ttm net_profit_rate_ttm
debt_to_asset_lf current_ratio_lf beta_000300SH_22 list_days is_new
```

这些字段均为数值。`pe_ttm、pb、ps_ttm`的非正值按原特征规则置为缺失；这不是对所有财务负值的通用删除规则。

### Barra_factor

```text
市值 贝塔 动量 残差波动 非线性市值 账面市值比 流动性 盈利 成长 杠杆
```

Barra表是合并的左表，决定每个日期可观察到哪些股票。请提供各历史日期当时的完整同伴集合。截面排名和市场均值依赖这个集合，不能只给准备买入的股票，或用最新成分列表重写过去。

### MarketDataRawOpen

仅重新训练需要此表。其`open`是原始开盘价，用于构造信号日`t`之后的标签：

```text
y(t) = raw_open(t+2) / raw_open(t+1) - 1
```

两个端点须为连续市场交易日、价格有限且为正，并且未停牌；标签退出日期不得晚于训练cutoff。该标签与特征表中的复权价格分开处理。

## 训练文件目录

从仓库根运行训练时，`--data-root ./data`对应：

```text
data/
  MarketData/2016/01/201601.parquet
  Factor33_winsor/2016/01/201601.parquet
  Barra_factor/2016/01/201601.parquet
  MarketDataRawOpen/2016/01/201601.parquet
  ...其余年月...
```

读取器匹配`<表名>/*/*/*.parquet`，文件名主体须为整数`YYYYMM`。训练范围从20160101开始，到`--cutoff`结束；原版支持的cutoff数值范围为20160101–20221231。

## Python推理接口

```python
from submission.inference import InferenceModel

model = InferenceModel(model_dir='submission/model')
tables = {
    'MarketData': market_frame,
    'Factor33_winsor': factor_frame,
    'Barra_factor': barra_frame,
}
prediction = model.predict(as_of_date='2022-07-07', data=tables)
```

三个值都是pandas DataFrame，字段按上面的定义准备。`as_of_date`用`YYYY-MM-DD`字符串，并且必须出现在Barra日期中。代码按Barra日期选取截至信号日的最后120个市场日期，再统一截取三个表；额外历史和之后的日期不会进入该次特征计算。短历史的部分滚动特征会缺失，原接口没有“至少120日”的强制断言。

日期序列来自提供的Barra表，不会自动恢复漏掉的整天。使用真实市场日历，保留缺失值和原始成员关系，不把缺日压缩为连续观测。

返回值是三列DataFrame：

|列|含义|
|---|---|
|`date`|请求的信号日字符串|
|`code`|该日可观察股票代码，每个代码一行|
|`score`|有限的日度百分位排名，范围`(0,1]`，并列取平均名次|

`score`用于相对排序，不是收益预测百分数或下单权重。原始比赛同伴集合约300只；接口没有把行数写死。机器可读字段定义见[data_schema.json](data_schema.json)。
