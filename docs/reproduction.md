# 运行与复现

本仓库保留v10的固定六模型评分：extra与utility各三个LightGBM Booster，种子42、43、44，每个350轮，使用相同顺序的229个特征。最方便的起点是验证已有权重，再运行合成输入测试。

## 安装并验证

建议使用Python3.11和独立虚拟环境，从仓库根执行：

```bash
python -m venv .venv
```

激活环境：Linux/macOS用`source .venv/bin/activate`；Windows PowerShell用`.venv/Scripts/Activate.ps1`。然后运行：

```bash
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py
python -m unittest discover -s tests -v
```

`requirements.txt`列出推理需要的NumPy、pandas和LightGBM。LightGBM的SciPy等传递依赖由pip安装；精确版本以文件为准。重新训练另用包含五项依赖的`requirements-training.txt`，其中增加Parquet读取所需的PyArrow并明确SciPy版本。

完整性脚本核对归档文件；合成测试检查接口、排名与输入不变性。合成股票和价格用于验证代码行为，测试通过不等于重现了历史IC或组合收益。

## 用已有模型预测

公开接口是`submission.inference.InferenceModel`。传入三个DataFrame及信号日，得到`date、code、score`。字段、日历和股票池定义见[data_schema.md](data_schema.md)。推理只使用CPU，不需要Torch或比赛SDK。

每个种子先把原始预测转成当日百分位。两个family分别平均种子排名并再次排名，得到E和U；种子间总体标准差形成分歧D：

```text
E = rank(mean(extra_seed_percentiles))
U = rank(mean(utility_seed_percentiles))
D = 0.75 * std(extra_seed_percentiles, ddof=0)
  + 0.25 * std(utility_seed_percentiles, ddof=0)
score = rank(0.75 * E + 0.25 * U - 0.25 * D)
```

输入同伴集合、特征列顺序和这些rank的先后都是模型的一部分。更改它们就不再是同一个评分函数。

## 从授权数据重新训练

先按数据文档布置四张表，再安装训练依赖：

```bash
python -m pip install -r requirements-training.txt
python submission/train.py --help
python submission/train.py --data-root ./data --cutoff 20221231 --variant confidence --out-dir artifacts/retrained
```

使用新的空输出目录。`train.py`默认会写入包内`model`目录，因此命令显式给出`--out-dir`，保留随仓库提供的权重。

正式入口是`submission/train.py`。`base_train.py`提供共享数据加载与参数，同时保留旧三模型CLI，不是v10的复现入口。`deployment_policy.json`中的`confidence`策略应随代码保留。

训练按cutoff过滤数据，再只使用已经退出、端点有效的H1标签。extra学习当天收益百分位减0.5；utility学习当天标准化收益，标准差下限1e-6，裁剪±3后乘0.2。六个模型使用相同特征集合，训练完成后把模型文本和`meta.json`写入指定目录。

重新训练的模型可以这样载入：

```python
from submission.inference import InferenceModel

model = InferenceModel(model_dir='artifacts/retrained')
```

`verify_artifacts.py`核对随仓库发布的基准文件；新训练目录应另存自己的版本、数据快照说明与SHA清单，不要覆盖基准清单来“通过”检查。比较两次训练时，先核对标签与成熟行、特征顺序、库版本、线程和实际模型输出，再解释模型文件差异。

## 复现分为三层

|层次|所需材料|检查什么|
|---|---|---|
|代码与合成输入|仓库代码、依赖及已有权重|加载、输出schema、评分公式、120日截取与不变性|
|真实数据重建和重训|四张授权数据表与对应数据快照|特征/标签、成熟行、六模型训练与真实窗口评分|
|年度OOF和组合回测|对应年度cutoff模型、授权SDK与实际配置|样本外预测、持仓执行、费用与指标|

全2022截止日训练的发布模型不能用来评价2018–2022并称为OOF；年度OOF需要每折仅使用更早的成熟标签。项目本地年份均已反复研究，复核这些年份不等于获得新的未见测试集。

实际组合回测需在使用者单独取得的SDK环境运行，并记录其版本、基准、持有与调仓频率、约束、佣金及滑点。SDK不属于推理依赖，仓库的合成测试也不代替账户与成交回放。
