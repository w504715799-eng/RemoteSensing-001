# 第一篇论文：已发布证据清单

核对日期：2026-09-07。基线 Git：`68821ee59500f53caef242f958c48615e36a6465`。
仅读取已提交 JSON 和源码；未访问 B/C 原始数据、缓存或 ledger，也未重新执行 C。
下列三份结果的原始字节均与 canonical JSON 一致，SHA-256 已核对。

| 主张 | 数据角色 | 公式／字段 | 文件 | SHA-256 | 支持 | 不支持 |
|---|---|---|---|---|---|---|
| K5 在开发集有更低 R9 AURC | development，120 ROI，用于选择 | 逐 ROI 十点 AURC 的等权均值 | `artifacts/phase2b3a/sen2naipv2-development-score-audit-v1.json` | `5bb0e5138d6ed1df6c65744556be02ccd48b77d3288df39630d16fbd9cd2dce9` | 有限外部研究值得继续 | 未见数据上的优势或显著性 |
| B 冻结阈值与覆盖 | calibration，120 ROI，用于拟合 | threshold, coverage, risk_bound | `artifacts/phase2b3b/sen2naipv2-calibration-conformal-v1.json` | `5fe01daaaa27443d5289cc366eacbe1fb628492c25bab16fadd164d3a8eb3174` | 预定拟合已完成 | 校准覆盖可直接迁移 |
| C 经验达标但不确定 | internal_test，120 ROI，已消费终局 | statistics.mean_loss, risk_ucb, phase_decision | `artifacts/phase2b3c/sen2naipv2-internal-test-evaluation-v1.json` | `9f267e459908a28e6fd254de9cc1d7d2352bf23b672366a9eda9839d53a0e115` | 忠实报告终局与证据不足 | 风险保证成立或允许重新调参 |

辅助流程证据（SHA-256；均为对应目录内完整文件名）：

- A `sen2naipv2-development-score-cache-audit-v1.json`：`d61c36e2180a2dc3468d4d9aba083ac0925d163ac2bb910e0227138e9fa249f1`。
- A `sen2naipv2-development-score-acceptance-v1.json`：`34741fe788cac6e28c6d8b1ce2fd96335b608e1b3e6ffb29e82ac064a2118227`。
- B `sen2naipv2-calibration-conformal-cache-audit-v1.json`：`40aff8aaccb27719d958d42026d68d8efc0c7f79479e3bd99777053ee1d6399f`。
- B `sen2naipv2-calibration-conformal-acceptance-v1.json`：`ce7f1a91e9954235e289dfc6eececb67261ff57110caaf15fe6640a0fb3b69ab`。
- C `sen2naipv2-internal-test-evaluation-cache-audit-v1.json`：`4bf048b222d4993adb0fd70b0273f60a994ecb7b3539e58ed3b2d57dcf4f43ff`。
- C `sen2naipv2-internal-test-evaluation-acceptance-v1.json`：`6fda78e13b13f78201487043fe9f679b432b692216778d172e7f36bf2d9ded21`。

## 可引用数字

| 方法 | 开发 R9 AURC | 开发 R9 rho | 开发 R1 AURC | 开发 R1 rho |
|---|---:|---:|---:|---:|
| LR 重投影残差 | 0.0101288090 | 0.4688257674 | 0.0099522002 | 0.2974127147 |
| 三模型分歧 | 0.0096675805 | 0.5790021349 | 0.0092596192 | 0.4040862072 |
| LDSR K5 方差 | 0.0095736511 | 0.5986411877 | 0.0092195973 | 0.3959881512 |

R9 AURC 相对 LR 下降约 5.48%，相对三模型约 0.97%；这是开发结果，不是外部效应。
R1 下 K5 的 rho 低于三模型，说明指标和误差尺度影响排序，不能只选有利列。
B 阈值 `7.970395366024563e-06`，覆盖 `0.5832304954528809`，
校准准则 `0.04999883134510526`。
C 覆盖 `0.5741024335225423`，ROI 平均损失 `0.04198510404780997`，
上界 `0.0779855394819395`，目标 `0.05`，终局 `empirically_met_but_inconclusive`。

## 原检查点 B 复核

1. 三个方法的开发集聚合选择风险均随覆盖率增加而增加，支持拒绝高分像素降低平均风险；
   不代表每个 ROI 都单调。
2. C 的平均损失低于目标，但上界超标；“经验违反率”不能从均值达标推成逐 ROI
   违反概率达标。此项不宣告完整通过。
3. 开发 K5 的 R9 AURC 优于随机解析基线和 LR，支持误差排序；没有新增幻觉标签证据。

第 1、3 项支持继续精简实证路线，不支持恢复普遍保证或增加下游任务。
四 ROI 的 K5/K25 冒烟只支持有限采样稳定性。指标完整定义见
[论文指标字典](../../paper/metric-dictionary.md)。
