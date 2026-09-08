# 本地研究执行检查点

2026-09-07。已产出证据清单、指标字典、论文骨架、图表规格、邻域适配规格、
Spain 元数据审计和统计草案。现已实现只读渲染器与邻域评分，生成三份 CSV 和一份 PDF；
现已在云端 development 上完成两个候选的 CPU 比较及全量复算，选中 3×3、sigma=1；
详见 [结果报告](first-paper-neighborhood-development.md)。未运行外部实验。

实现均先经历缺失功能导致的测试失败，再通过合成及固定证据测试。
development 范围已按固定清单与 A 审计核验：120 ROI、240 影像、600 K5 预测。
两轮均复用缓存，无新增推理，没有访问 B/C 像素或缓存。两轮科学 JSON 字节一致。
下一步是解决 Spain 版本化文本合同与独立外部入口，而非直接启动 GPU。
Spain 候选已在像素访问前改为 100 包及同版本 CSV；L2A 波段和 HRharm 协调关系已由
固定数据卡确认。仍缺包内行对应核验方案、解包安全及 nodata／异常范围规则，不能冻结。
48 ROI 文本投影已生成；OpenSR 1.3.3 已核验裁边、软分类与缺失语义，图像级结构
诊断的配准失败规则仍待实现。没有再次连接云端或读取 Spain 像素。

新结果及运行记录已保存到云端持久化存储，本地只接收数值证据，用户已获知可关机。
base 环境经用户授权新增 pytest 8.4.2、iniconfig 2.3.0；未新建环境。
本地相关测试 78 passed，云端相关测试 28 passed。
已记录 CPU 后处理时间，尚无新推理的单图耗时／显存实测，不能宣称 GPU 预算完成。

## 2026-09-08 本地进展

新增 `spain_inputs.py`：对已解码数组进行精确 shape／dtype／数值校验、RGBN 选择、
饱和数量记录与归一化；将固定成员元数据绑定到包内行位置，不使用 DataFrame 索引
或排序假设。23 项新测试先失败后通过，相关测试合计 110 passed，43 条既有弃用警告。
处理规则是预定研究合同，不代表真实包 dtype／nodata 已被验证。
没有下载或解包 Spain，没有连接云端，没有改动原 SPOT loader 或 A/B/C 产物。
仍需安全解码／固定包哈希入口、五评分汇总与外部协议最终验收，尚未到 GPU 阶段。

## 2026-09-08 五评分核心检查点

新增 `src/trustsr/evaluation/external_scores.py`：四张 HR-free 分数图（LR、三模型、
K5、固定 w3 邻域）及随机解析期望，共用中心预测计算 R1/R9 十点曲线、AURC、rho；
随机 rho 为 null，不伪装成一次有随机相关性的实现。旧阈值仅应用 K5，单独记录
覆盖、ROI 最大 R9 损失与全拒绝。新输出修正高风险排除比例的名称，不改旧字段。

主比较汇总要求每个固定 ROI 恰有结果或明确的整 ROI 失败，拒绝缺项、重复、清单外
成员和非有限 AURC；无有效配对返回 null，不返回零。此函数只汇总主比较，尚未覆盖
完整运行器的逐方法部分失败、所有次要指标的子集均值及 HR 来源计数。

16 项新测试先失败后通过；相关回归命令：

```sh
.venv/bin/python -m pytest tests/data/test_spain_inputs.py tests/paper tests/risk tests/data/test_local_data_policy.py --override-ini addopts='' -q
```

结果 126 passed，43 条既有 PyTorch JIT 弃用警告。没有 Spain 像素访问、云端连接
或旧实验重跑。模型身份与 seed 顺序仍由未来运行器认证，数组核心不声称认证预测来源。

## 限定硬件测速入口

`scripts/paper/benchmark_inference.py` 已实现；11 项 CPU 测试验证固定 3 ROI 选择、
缺失／错 split 拒绝、真实 bicubic 输出计时与哈希、非法输出拒绝、无 CUDA 时无数据
访问或输出，以及历史数据／模型／仓库输出路径保护。相关回归合计 137 passed，
43 条既有警告；CLI `--help` 可运行。真实 GPU 分支尚未执行，不以合成测试冒充硬件测试。

预算口径更正：冻结 A 中 SEN2SRLite 是 CPU，不能把它的耗时称为 GPU 推理耗时。
固定范围、调用数、30 分钟外层硬超时与测量限制详见 [预算规格](first-paper-compute-budget.md)。
下一资源步骤是短时 GPU 测量；安全解包和完整外部运行器仍未完成，Spain 未获本入口访问。

只读代码审阅未发现 Critical／Important 问题；独立合成成功路径检查确认 3 次 bicubic、
4 次 SEN2SRLite、16 次 LDSR 及 seed 顺序。该成功路径探针尚未收入自动回归测试；
现有自动测试覆盖 CPU 测量核心和无 CUDA／无效输入停止路径。真实硬件仍待测。

最终全仓验证：`.venv/bin/python -m pytest --override-ini addopts='' -q` 返回
**2487 passed, 43 warnings in 778.23s**。新代码 Ruff、Git diff whitespace 和 staged
data-policy 检查通过；A/B/C 产物、旧 CLI、模型适配器及冻结评价代码无 diff。
本轮仍未连接云端；完成的是 GPU 测速准备，不是外部研究或最终论文。

## 后续 GPU 测速完成

用户提供服务器后，按 `49a7e39` 完成固定范围测速，云端 27 项测试通过；环境未改。
实际结果、来源 SHA 和限制见 [测速报告](first-paper-inference-timing.md)。模型推理成本
已可作限定外推，但完整流水线开销、SEN2SRLite 新旧哈希差异与安全解包仍待处理。
不将这次完成计作最终外部协议冻结；无 Spain 访问，用户已获知可关机保留数据盘。
