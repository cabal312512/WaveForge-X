# WaveForge-X

OFDM、OTFS、AFDM 的纯 CPU 通信研究平台，整合波形自适应、有限计算预算、
迭代接收机与可验证状态复用。代码、实验设置、实际结果和负结果保留在同一个公开版本中。

| 主题 | 主要观察 | 详细记录（英文） |
|---|---|---|
| 波形自适应 | 减少切换有作用，但未超过最佳固定方案；修正探测顺序后未超过简单探测后固定 | [Adaptation](docs/adaptation.md) |
| 接收机预算 | 存在真实 BER—成本取舍；联合选择器未胜出，成本为固定配置的 2.49 倍 | [Receiver budgets](docs/receiver-budget.md) |
| 判决感知接收 | 有效误差集合限制相对指定 LMMSE 参考的位分歧；准备成本可能抵消迭代收益 | [Receiver study](docs/research.md) |
| 状态复用 | 合法复用可减少准备；新增相位/细化策略未普遍超过强缓存基线 | [State reuse](docs/reuse.md) |

[完整实验记录](docs/experiment-record.md)说明统计单位、开发/确认划分、原始数据覆盖范围和局限。
运算量模型不等于 CPU 周期，Python 实测时间不等于硬件时延。判决一致性不是零 BER；
`floating_point_certified=False`。TDL 是有限 FIR 的标准 profile-based 模型，未宣称完整符合标准。

使用 Python 3.11/3.12 和 NumPy/SciPy，不需要硬件、CUDA 或深度学习框架。
环境、依赖和缓存留在项目内。运行命令见 [英文首页](README.md)与[复现说明](docs/reproduce.md)。

MIT License；版权所有 cabal312512。人工智能参与代码、诊断及技术文档制作，
结果来自实际运行记录；失败情况与无收益结论同样保留。
