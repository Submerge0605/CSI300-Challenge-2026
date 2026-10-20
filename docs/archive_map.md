# 资料整理索引

这次整理把比赛期的模型、实验记录和赛后笔记汇成一个能直接阅读、运行的仓库。本地研究档案原样保留。

| 材料 | 仓库入口 |
| --- | --- |
| 推荐公开的 v10 训练、特征与推理代码 | [submission](../submission/) |
| 原六模型与特征顺序 | [model](../submission/model/)、[文件指纹](../release_manifest.json) |
| 模型结构与训练目标 | [model.md](model.md) |
| 安装、输入格式、训练与回测 | [reproduction.md](reproduction.md)、[data_schema.md](data_schema.md) |
| v2–v14 的版本变化与后期失败实验 | [research.md](research.md) |
| 逐版本公榜与年度回测汇总 | [results.json](../experiments/results.json) |
| Word 中24篇 Kaggle 方案的迁移与实验记录 | [kaggle_ideas.md](kaggle_ideas.md) |
| 赛后金融与机器学习学习笔记 | [learning_guide.md](learning_guide.md) |
| 可独立运行的合成数据练习 | [toy_pipeline.py](../examples/toy_pipeline.py) |
| 冻结权重和120日推理契约检查 | [tests](../tests/) |
| 来源和许可 | [references.md](references.md)、[THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md) |

原始行情、逐股票标签与预测、特征缓存、回放账户明细、虚拟环境、重复解压目录和第三方下载副本留在本地。公开仓库保留方法、汇总结果和可复现入口。

整理时扫描了 25,399 个研究文件（不含虚拟环境、Python缓存、Node依赖和本次发布工作目录）。旧版本代码及原始回测收据没有随首版全部上传；版本与结果摘要记录了它们的出处和指纹。
