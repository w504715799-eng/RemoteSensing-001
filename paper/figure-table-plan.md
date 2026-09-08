# 图表计划

| 产物 | 内容 | 当前状态／数据角色 |
|---|---|---|
| T1 | 数据角色、ROI 数、模型调用成本 | 正文第2节数据角色及3.2节调用表；开发／正式流水线时间见第5节 |
| T2 | 三评分 R9/R1 AURC 与 rho | 已生成 development_scores.csv；仅 development |
| F1 | 开发 R9/R1 十点风险—覆盖曲线，含随机解析参考 | 已生成 `figures/development_risk_coverage.pdf` |
| T3 | B 阈值／覆盖／校准准则；C 均值／覆盖／UCB／终局 | 已生成 calibration_evaluation.csv；两种 bound 分列，不是同义置信曲线 |
| T4 | Spain 两子集五评分配对比较和缺失分母 | 已生成spain_scores/paired/curves/transfer/structure五CSV |
| F2 | 外部固定阈值覆盖与损失 | 已生成spain_threshold_transfer.pdf；仅K5使用原阈值 |
| F3 | 预设 ROI 的预测、参考、分数、风险 | 已生成spain_fixed_examples.pdf；每子集首两ROI，不换样本 |
| F4 | 外部R1/R9十点风险—覆盖曲线 | 已生成spain_risk_coverage.pdf；等ROI描述性均值，无置信区间 |

图注必须标明 ROI 聚合、误差窗口、覆盖网格、方向和开发选择偏差。
F1 不画独立性未经证明的像素 bootstrap 区间。CSV 为科学数值事实源；PDF 元数据
不要求字节确定性，但轴、图例和数据必须可复建。
