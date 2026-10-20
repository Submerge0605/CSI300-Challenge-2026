# 来源与许可说明

本项目原创实现、文档和六份自训LightGBM权重采用MIT许可。竞赛原始数据由组织方提供，使用时遵守其条款。

## Qlib

`submission/alpha_features.py`中的量价形态与滚动统计参考了Microsoft Qlib的Alpha158设计，并保留源码中的引用。项目使用独立有限窗口实现，不依赖Qlib运行。

- 项目：https://github.com/microsoft/qlib
- 参考文件：https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py
- 许可：MIT，Copyright Microsoft Corporation；副本见[`licenses/Qlib-MIT.txt`](licenses/Qlib-MIT.txt)。

## Kaggle方案

量价表示、市场上下文、有限EMA与模型对照参考了Kaggle作者的公开方案。阅读与本地改编的对应关系见[`docs/kaggle_ideas.md`](docs/kaggle_ideas.md)。仓库收录自己的实现和实验摘要，原作者的Notebook、下载附件及预训练权重不随包分发。

## 比赛工具与参考

比赛由DataTech与TopK AI量化实验室组织。主办方baseline用于理解接口与建立对照。回测工具[`skd-backtest`](https://github.com/HenryZ16/skd-backtest)采用Apache-2.0，需单独取得；本仓库未打包SDK本体。

[`Star6azeR/CSI300-Challenge-2026`](https://github.com/Star6azeR/CSI300-Challenge-2026/)提供了发布结构参考，未复制其模型或代码。

## Python依赖

NumPy、pandas、LightGBM、PyArrow和SciPy通过依赖文件安装，各自保留原许可证。教学练习还使用scikit-learn。仓库不分发这些库的安装包。
