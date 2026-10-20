# v10训练与推理包

本目录保存v10的原始计算源码、配置和六份LightGBM权重。模型文件、计算源码与历史提交一致；这个README为公开版重新整理。

```python
from submission.inference import InferenceModel

model = InferenceModel("submission/model")
scores = model.predict(as_of_date, data)
```

训练入口是`train.py`。`base_train.py`保留了早期模型的辅助入口，v10重训使用下面的命令：

```bash
python submission/train.py --data-root ./data --cutoff 20221231 --variant confidence --out-dir artifacts/retrained
```

请从仓库根目录运行，将重训文件写入新的输出目录。部署用到的`deployment_policy.json`与`model/meta.json`已一同保留。

| 文件 | 作用 |
| --- | --- |
| `inference.py` | 120日裁剪、加载模型、固定融合与输出 |
| `train.py` | 两目标、三种子的离线训练 |
| `base_train.py` | 数据读取、raw-open标签与成熟性检查 |
| `cleaning.py` | 主键、重复记录、无穷值、价格与ST处理 |
| `features.py` | 共享特征接口 |
| `base_features.py` | 基础因子排名与有限窗口统计 |
| `alpha_features.py` | Alpha158启发的量价比例特征 |
| `kaggle_features.py` | 量价、差分、市场状态特征 |
| `postprocess.py` | 日内排名及历史研究辅助方法 |
| `model/` | 六个Booster、特征顺序与参数 |

[完整模型说明](../docs/model.md) · [数据格式](../docs/data_schema.md) · [复现指南](../docs/reproduction.md)
