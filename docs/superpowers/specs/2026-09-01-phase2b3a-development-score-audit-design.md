# Phase 2B3-A：Development-only 不确定性评分审计设计

**日期：** 2026-09-01  
**状态：** 书面规格已批准，进入实施计划

**上位路线：** `docs/superpowers/specs/2026-08-26-trustworthy-sentinel2-sr-roadmap.md`  
**直接前置阶段：** `docs/superpowers/specs/2026-08-31-phase2b2b-development-three-model-smoke.md`  
**前置代码提交：** `2455c1e07686076561b181c753cc61a5ed440222`

## 1. 文献复核与阶段定位

下列组合不能作为本项目的论文首创点：

- 黑盒生成式超分辨率、多次随机推理方差、局部风险与 conformal 可信掩膜已经由
  NeurIPS 2025 工作系统提出；
- LDSR-S2 已报告多次生成得到的像素级不确定性，并在官方 OpenSR 实现中公开相关接口；
- 近期公开但尚未形成同行评议论文的 `opensr-conformal-uq` 项目，已经把多次 OpenSR
  推理、局部平滑和 conformal UQ 用于 Sen2NAIP 小样本实验；
- SEN2SR 已覆盖 Sentinel-2 超分辨率的低频辐射和空间一致性约束。

因此，本阶段不直接拟合 conformal 阈值，也不把“LDSR 方差 + conformal”称作创新。
本阶段只回答一个更窄、可证伪的问题：

> 在冻结的真实 Sentinel-2/NAIP 跨传感器 development ROI 上，哪一种不使用 HR 的
> 不确定性代理，能够稳定地排序 LDSR-S2 的局部四波段重建风险；这种排序是否随成像
> 日期差和 LR/HR 相关性分层而失效？

候选论文价值来自跨传感器真实域的失效审计、ROI 级统计和预注册选择规则，而不是再次
发明通用方差或 conformal 公式。只有本阶段冻结出合格评分后，才分别设计：

- Phase 2B3-B：只使用 `calibration` ROI 拟合 conformal 阈值；
- Phase 2B3-C：冻结后只运行一次 `internal_test` 验收。

## 2. 范围

### 2.1 包含

- 只评估 LDSR-S2 的一个冻结中心重建；
- 三种不使用 HR 的评分：LDSR 多次采样方差、LR 重投影一致性残差、三模型分歧；
- 4 个 development ROI 上的 `K=5`/`K=25` 稳定性与成本冒烟；
- 通过冒烟后，在全部 120 个 development ROI 上比较合格候选；
- 以完整 ROI 为统计单位的相关性、风险—覆盖率和分层诊断；
- 确定性的评分选择规则、缓存、回放、审计和小型结果产物；
- CPU 先行、GPU 分阶段开启的执行方式。

### 2.2 不包含

- 不使用 `calibration` 或 `internal_test` 像素、HR 标签或指标选择评分；
- 不拟合 conformal 阈值，不输出可信掩膜，不宣称风险保证；
- 不增加光谱角、高频幻觉或复合遥感风险；
- 不训练、微调或改变 Bicubic、SEN2SRLite、LDSR-S2 主干；
- 不设计分组/Mondrian、加权或跨域 conformal；
- 不实现选择性回退或下游任务；
- 不把像素当成独立统计样本，不报告像素级显著性检验；
- 不下载 190 GB 级数据到本地，不把影像、模型或大缓存提交到 Git；
- 不固定云端 GPU 型号、UUID 或云厂商。

这些内容必须分别经过后续规格审批，防止一个阶段同时选择评分、校准阈值并测试结果。

## 3. 冻结数据与防泄漏协议

### 3.1 唯一数据来源

使用 Phase 2B1-B 已冻结的 360 行全资产清单：

```text
POST_MANIFEST_SHA256 = c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a
```

像素加载继续服从 Phase 2B2-A 输入合同：RGBN 顺序 `[B04, B03, B02, B08]`，LR
`(4, 128, 128)`，HR `(4, 512, 512)`，`torch.float32`，反射率 `[0, 1]`，不静默裁剪。

### 3.2 A1 冒烟集合

只使用既有 `select_development_smoke_records()` 返回的 4 个 ROI：

- `split == "development"`；
- `selection_round == 1`；
- `days_between == -1`；
- `correlation_bin` 按 `0, 1, 2, 3` 各一个；
- 4 个不同 `sample_id` 和 `spatial_group_id`。

这 4 个 ROI 只用于工程正确性、随机评分稳定性和资源测量，不用于选择论文结论。

### 3.3 A2 完整 development 集合

从冻结清单中按清单顺序筛选全部且仅有 `split == "development"` 的 120 个 ROI。加载前
必须验证：

- 恰好 120 个不同 `sample_id`；
- 恰好 120 个不同 `spatial_group_id`；
- `days_between ∈ {-1, 0, 1}`、`correlation_bin ∈ {0, 1, 2, 3}`、
  `selection_round ∈ {1, ..., 10}`；
- 12 个 `(days_between, correlation_bin)` 分层各有 10 个 ROI；
- 每个分层的 `selection_round` 恰好覆盖 `1..10`。

命令不得提供任意样本 ID、任意 limit 或跳过分层的正式运行选项。A1 和 A2 使用不同、
固定的入口。

### 3.4 禁止泄漏

- 允许读取整份清单以验证固定 SHA-256，但只能加载 development GeoTIFF；
- 评分选择函数只接受带有 `split="development"` 的审计记录；
- 输出不得包含 calibration/internal-test 的 HR 摘要、预测、指标或分层表现；
- Phase 2B3-B 和 2B3-C 必须由新的命令、结果 schema 和规格实现；
- development 结果一旦冻结，不得因后续 calibration 或 internal-test 表现回改本阶段
  的评分名称、种子、窗口或选择规则。

## 4. 被评估的中心重建与风险标签

### 4.1 中心重建

只评估 LDSR-S2 的冻结中心输出：

```text
model = ldsr-s2-x4
seed = 3407
sampling_steps = 100
sampling_eta = 0.95
sampling_temperature = 1.0
histogram_matching = true
output_policy = clip_to_[0,1]
```

记该输出为 `P0 ∈ [0,1]^(4×512×512)`。同一个 `P0` 同时作为 LDSR 方差集合的第一个
成员，避免重复计算。所有模型输出仍需通过完整 provenance 和 LR 张量摘要绑定缓存身份。

### 4.2 主风险标签

本阶段复用 Phase 2A 的 `local_l1_risk`，不引入新的风险函数：

```text
pixel_error[p] = mean_c |P0[c,p] - HR[c,p]|
R9[p] = reflected_box_mean(pixel_error, window=9)
```

`window=9` 对应约 `22.5 m × 22.5 m` 的 HR 邻域，用于降低单像素配准差异对评分审计的
支配。`R9` 是唯一参与候选资格和评分选择的主风险图。

同时计算 `window=1` 的 `R1` 作为预注册敏感性描述，但不得用 `R1` 改变候选资格、排序
或冻结结果。两种风险都只在 development 分析阶段由 HR 产生；任何部署评分均不读取 HR。

## 5. 三种候选评分

所有评分方向统一为“值越大，预计风险越高”，输出有限、非负 `float64 (512, 512)`。
缓存预测先转移到 CPU 并转换为 `float64`，再完成下述评分运算；GPU 只负责生成
`float32` 预测，不能参与统计量计算。

### 5.1 LDSR 随机采样方差

固定 25 个连续且不同的种子：

```text
S25 = [3407, 3408, ..., 3431]
S5A = [3407, 3408, 3409, 3410, 3411]
S5B = [3412, 3413, 3414, 3415, 3416]
```

除 `seed` 外，所有 LDSR 参数与中心重建相同。对于种子集合 `S`：

```text
U_var_S[p] = mean_c Var_seed(P_seed[c,p], correction=0)
```

方差在已经完成 histogram matching、`[0,1]` 裁剪的最终重建上计算。它是工程上与现有
Phase 2A `ensemble_variance_score` 一致的基线，不宣称等同于传感器物理噪声模型。

- A1 对 4 个 ROI 计算 `S5A`、`S5B` 和 `S25`；
- A2 最多只计算 `S5A`，名称冻结为 `ldsr_variance_k5`；
- `S25` 只做 A1 敏感性参考，禁止由本规格扩展到 120 个 ROI；
- 如果 A1 判定 `K=5` 不稳定，则从 A2 候选中移除该评分，不自动提高 K。

### 5.2 LR 重投影一致性残差

对中心 LDSR 输出使用无参数、确定性的面积下采样：

```text
LR_hat = interpolate(P0, size=(128,128), mode="area")
e_lr[q] = mean_c |LR_hat[c,q] - LR[c,q]|
U_lr[p] = repeat_interleave(e_lr, 4, axis=height and width)[p]
```

使用精确的 `4×4` 常数块上采样，不使用可调平滑核。该分数名称为
`lr_reprojection_l1`。它只是模型无关的一致性代理；由于没有真实 Sentinel-2 PSF，结果
不得表述为严格的物理辐射一致性。

### 5.3 三模型分歧

复用相同 LR 上 Bicubic、SEN2SRLite 和中心 LDSR 的最终输出：

```text
M = stack([P_bicubic, P_sen2srlite, P0], dim=model)
U_disagree[p] = mean_c Var_model(M[model,c,p], correction=0)
```

名称为 `three_model_disagreement`。它不需要额外 LDSR 随机采样，但依赖三模型预测缓存。
模型顺序和 provenance 必须与 Phase 2B2-B 冻结顺序一致。

### 5.4 明确不加入的评分

本阶段不加入补丁平滑方差、加权融合分数、学习型误差预测器、测试时增强、SAM 或高频
残差。它们会增加 development 调参自由度，且局部平滑 conformal OpenSR 已有公开重叠。

## 6. A0：CPU 合同阶段

A0 在本地完成，不访问真实像素，不连接云服务器，不需要 GPU。交付内容包括：

- 三种评分的纯张量接口和严格输入验证；
- 全 development 记录选择与 12 分层验证；
- ROI 级诊断统计、固定 bootstrap 和确定性选择规则；
- 种子/模型/算子身份绑定的缓存元数据；
- 使用假模型和小张量的 CLI 编排测试；
- 云端分阶段脚本及其 shell/安全测试。

A0 通过条件：完整现有测试与新增测试通过、Ruff 通过、命令帮助可用、两个相同的合成
输入运行得到字节一致结果。A0 失败时不得启动 GPU。

## 7. A1：4 ROI GPU 稳定性冒烟

### 7.1 分级入口

A1 必须分为可单独回放的阶段：

1. `preflight`：验证挂载点、提交、依赖、模型、数据清单和空闲 GPU；
2. `single`：一个 ROI、种子 3407 做两次无缓存调用，验证输出字节重复性和显存；
3. `smoke`：4 个 ROI，生成三模型中心输出和 25 个 LDSR 种子缓存；
4. `replay`：禁止模型构造和推理，只从缓存重算评分、风险和结果；
5. 本地只拉取小型 JSON 审计，验证摘要后再决定是否进入 A2。

计算阶段可断点续跑；已存在的完整缓存必须验证后复用，部分、损坏或身份不符的缓存必须
失败关闭，不能静默覆盖。

### 7.2 K=5 稳定性诊断

每个 A1 ROI 分别计算：

- `Spearman(U_var_S5A, U_var_S5B)`；
- `Spearman(U_var_S5A, U_var_S25)`；
- `U_var_S5A` 与 `U_var_S25` 各自最高 10% 像素集合的 Jaccard；
- 三种评分对 `R9` 的描述性 Spearman 和风险—覆盖率，但不据 4 个 ROI 选择评分。

平均秩处理并列；若任一图为常数，相关系数按 0 记录并设置 `constant_score=true`。

`ldsr_variance_k5` 进入 A2 必须同时满足：

- 4 个 ROI 的 `S5A`/`S5B` Spearman 中位数不低于 `0.60`，且最小值不低于 `0.40`；
- 4 个 ROI 的 `S5A`/`S25` Spearman 中位数不低于 `0.80`，且最小值不低于 `0.60`；
- 4 个 ROI 的 `S5A`/`S25` 最高 10% Jaccard 中位数不低于 `0.50`；
- 所有预测和评分有限、范围有效、缓存回放字节一致；
- 单图峰值已分配显存不超过设备总显存的 80%；
- 长期盘在扣除现有文件后至少仍有 10 GiB 可用；
- 按 A1 每个未缓存预测的中位耗时线性外推并乘 `1.5` 安全系数后，A2 GPU 计算不超过
  2 小时。

前三项任一失败，只移除 `ldsr_variance_k5`；不阻止两个确定性候选进入 A2。数据、缓存、
重复性或资源完整性失败则停止整个阶段并先诊断。

## 8. A2：120 ROI development 评分审计

### 8.1 ROI 级统计单位

像素只用于在一个 ROI 内形成排序诊断；所有置信区间和候选比较均对 120 个 ROI 重采样。
不得把 `120×512×512` 个像素当成独立观测。

每个候选评分、每个 ROI 计算以下主指标：

1. `rho_i`：评分与 `R9` 的 Spearman 相关，越大越好；
2. 风险—覆盖率：按 `(score, row_major_pixel_index)` 稳定升序，在覆盖率
   `C={0.1,0.2,...,1.0}` 上保留前 `ceil(C×N)` 个像素，计算受信像素的平均 `R9`；
3. `AURC_i`：上述 10 个选择性平均风险的算术平均，越小越好；
4. `random_AURC_i`：完整 ROI 的平均 `R9`，即随机排序的解析期望；
5. `AURC_gain_i = random_AURC_i - AURC_i`，越大越好；
6. 描述性高风险漏检率：将 `R9` 最高 10% 定义为高风险，在 80% 受信覆盖率下计算仍被
   信任的高风险比例，越小越好。

风险和评分分位集合使用固定数量的像素、平均秩和 row-major 并列破坏规则，确保跨框架
回放一致。`R1` 重复计算同类描述性指标，但不进入选择。

### 8.2 聚合与不确定性

- 主汇总是 120 个 `rho_i` 的算术平均和 120 个 `AURC_gain_i` 的算术平均；
- 同时报告中位数、四分位数和 12 个 `(days_between, correlation_bin)` 分层均值；
- 置信区间使用 ROI 为单位的 10,000 次 percentile bootstrap；
- bootstrap 固定 `numpy.random.Generator(PCG64(23031))`；
- 报告双侧 95% 区间，即第 `2.5%` 和 `97.5%` 百分位；
- 候选差异使用同一 bootstrap 索引做配对重采样；
- 不输出像素级 p 值，也不把 12 个十样本分层的区间解释为正式保证。

### 8.3 候选资格

候选进入冻结选择必须同时满足：

- 至少 114/120 个 ROI 的评分图非常数；常数 ROI 的 `rho_i` 按 0 计，不能丢弃；
- 平均 `rho_i` 的 95% bootstrap 下界严格大于 0；
- 平均 `AURC_gain_i` 的 95% bootstrap 下界严格大于 0；
- 12 个分层中至少 9 个分层的平均 `rho_i` 大于 0；
- 没有任何分层的平均 `rho_i` 小于 `-0.10`；
- 所有 120 个 ROI 都具有完整输入、预测、评分和风险记录，无选择性缺失。

解析随机基线对应 `rho=0`、`AURC_gain=0`，不另外生成可偶然有利或不利的随机图。

### 8.4 确定性冻结规则

1. 从合格候选中选择平均 `rho_i` 最大者为统计领先者；
2. 对每个其他合格候选计算“领先者减候选者”的配对 bootstrap 95% 区间；
3. 若该差异区间下界大于 0，则候选显著落后；否则视为与领先者不可区分；
4. 在所有不可区分候选中选择推理成本最低者，固定成本顺序为：
   `lr_reprojection_l1`、`three_model_disagreement`、`ldsr_variance_k5`；
5. 将名称、完整参数、输入清单 SHA-256、代码提交和选择证据写入
   `frozen_score`，后续阶段不得修改。

`AURC_gain` 是资格和解释指标，不与 Spearman 加权成可调复合分数。若最终选中廉价 LR
残差，也必须如实报告扩散方差未带来稳定增益；这可以支持“真实跨传感器域中的负结果”
论点，但不能包装成新的不确定性算法。

### 8.5 A2 停止条件

- 若没有候选合格，停止，不进入 Phase 2B3-B；
- 若只有不足 120 个完整 ROI，不做完整案例分析后冒充预注册实验；
- 若不满足“至少 9/12 个正分层且没有分层低于 `-0.10`”的精确规则，候选不合格；
- 若需要改变窗口、覆盖率网格、种子或阈值才能通过，必须写新规格，不能原地调参；
- 若全部候选失败，保留失败结果并重新评估论文是否转为跨传感器失效研究。

## 9. 缓存、结果与可复现性

### 9.1 长期云端目录

大文件只保存在已挂载的长期盘：

```text
/root/rivermind-fs/trustsr/phase2b3a/
  predictions/<post-manifest-sha256>/
  scores/<post-manifest-sha256>/
  results/<post-manifest-sha256>/
```

不得要求下载到本地。若实际长期盘位置改变，只允许通过显式 `--storage-root` 指定并验证
它是挂载点；不得硬编码 SSH 主机或 GPU 名称。

### 9.2 身份绑定

预测缓存身份至少包含：模型 provenance、种子、LR 来源、`sample_id`、LR 形状/类型/SHA、
代码中的实现 schema。评分缓存身份至少包含：评分名称与版本、所有输入预测 SHA、LR SHA、
算子参数和输出张量 SHA。

种子不同必须得到不同缓存键。任何 tensor/JSON 单边缺失、符号链接、摘要不一致或非规范
路径均抛出完整性错误。

### 9.3 结果 schema

至少产生两个科学结果：

- `trustsr.phase2b3a-development-smoke.v1`：A1 四 ROI 稳定性和候选资格，并引用独立运行
  清单的 SHA-256；
- `trustsr.phase2b3a-development-score-audit.v1`：A2 120 ROI 主结果、分层、bootstrap、
  候选选择和 `frozen_score`。

运行时间、GPU 名称/UUID、驱动、显存和路径放入独立运行清单，不能进入要求字节一致的
科学结果。科学结果使用 canonical JSON；回放结果必须与计算阶段字节一致。

本地 Git 只接收验证后的小型 JSON 摘要和审计，不接收 GeoTIFF、预测、评分张量、模型、
绝对路径、主机名、用户名、SSH 信息或环境变量。提交前对 JSON 做大小和敏感字段检查。

## 10. 云端资源与执行边界

- A0 和规格/计划阶段不需要 GPU；用户可以保持云服务器关闭；
- 第一次需要 A1 GPU 时明确通知用户，由用户提供当次 SSH 连接；
- 云端直接复用镜像 base Python/PyTorch，不创建新的 conda 环境；
- 依赖安装必须拒绝替换现有 PyTorch/CUDA 栈；
- 不限制 GPU 商品型号，只记录实际型号；沿用 Phase 2B2-B 的硬件安全门槛：CUDA 可用、
  恰好一个可见设备、compute capability 至少 8.0、初始空闲显存至少 18 GiB，且没有
  外来计算进程；
- 资源阈值根据 A1 实测结果决定 A2 是否可在当前实例运行，不因换卡修改科学配置；
- 缓存允许跨重启复用，但每次运行重新记录代码、环境、模型和硬件 provenance；
- 远程计算完成并拉取小型审计验证后，明确通知用户可以暂停 GPU；脚本不执行云厂商关机。

## 11. 错误处理与安全

- 所有正式入口默认失败关闭，不自动换数据、模型、种子、设备或分割；
- 清单、输入审计、模型资产、缓存或提交摘要不匹配时，在模型构造前失败；
- 计算中断保留已经原子提交的缓存，临时文件不作为有效缓存；
- 回放阶段必须通过依赖注入或显式防线证明没有构造模型、没有 CUDA 推理；
- 所有 shell 调用使用固定 argv 和引号，不把外部文本拼接为命令；
- 密码、token、SSH 私钥、主机连接信息永不写入仓库、日志、结果或命令脚本；
- 不删除既有长期盘数据；清理缓存需要单独、明确的用户授权。

## 12. 测试策略

### 12.1 数学单元测试

- 手算小张量验证三种评分、方差 `correction=0`、面积下采样和 `4×4` 重复；
- 验证错误维度、通道、NaN/Inf、越界反射率、重复种子和常数分数；
- 手算 Spearman 并列秩、固定覆盖率选择、AURC、解析随机基线和高风险漏检率；
- 验证 bootstrap 固定种子字节一致、配对索引一致和区间百分位；
- 构造明确合格、不合格和成本平局案例验证冻结规则。

### 12.2 数据与防泄漏测试

- 验证 A1 精确选择 4 个 canonical development ROI；
- 验证 A2 精确选择 120 个 ROI 和 12×10 分层；
- 任一非 development 像素加载、缺失 ROI、重复空间组或分层破坏必须失败；
- 证明候选选择接口拒绝 calibration/internal-test 记录。

### 12.3 缓存与编排测试

- 假模型验证不同种子得到不同身份、中心预测只计算一次；
- 完整缓存命中不调用模型，损坏或半条目不静默重算；
- A1、A2 计算与 replay 产生相同 canonical scientific JSON；
- replay 使用会在构造时抛错的模型工厂证明零推理；
- shell 测试覆盖参数传递、挂载检查、base 环境、不含秘密和阶段顺序。

### 12.4 仓库门槛

```text
uv sync --dev
uv run pytest
uv run ruff check .
bash -n <所有新增 shell 脚本>
```

真实 A1/A2 运行不是本地单元测试的一部分；其小型审计必须记录确切代码提交并在回传后
由本地验证器检查。

## 13. 完成定义与后续门槛

Phase 2B3-A 只有在以下全部满足时完成：

- A0 本地质量门槛通过；
- A1 计算与无推理回放通过，并明确记录 K=5 是否进入 A2；
- A2 恰好覆盖 120 个 development ROI；
- 至少一个候选满足资格规则并由确定性规则冻结；
- A2 计算/回放科学 JSON 字节一致；
- 小型结果和审计通过本地摘要、大小、敏感字段与提交绑定验证；
- 失败分层和负结果没有被隐藏；
- 本阶段没有读取 calibration/internal-test 像素或拟合 conformal 阈值。

完成后先撰写 Phase 2B3-B 规格。后续 calibration 阶段只能消费 `frozen_score`，不能重新
选择评分；internal-test 在 Phase 2B3-C 之前继续保持未使用状态。

## 14. 依据

- Adame, Csillag, Goedert, *Image Super-Resolution with Guarantees via
  Conformalized Generative Models*, NeurIPS 2025,
  https://arxiv.org/abs/2502.09664
- 论文官方实现：
  https://github.com/adamesalles/experiments-conformal-superres
- Donike et al., *Trustworthy Super-Resolution of Multispectral Sentinel-2 Imagery
  With Latent Diffusion*, IEEE JSTARS 2025,
  https://doi.org/10.1109/JSTARS.2025.3542220
- LDSR-S2 官方 OpenSR 实现：
  https://github.com/ESAOpenSR/opensr-model
- Aybar et al., *A Radiometrically and Spatially Consistent Super-Resolution
  Framework for Sentinel-2*, Remote Sensing of Environment 334 (2026) 115222,
  https://doi.org/10.1016/j.rse.2025.115222
- SEN2SR 官方实现：
  https://github.com/ESAOpenSR/SEN2SR
- Aybar et al., *SEN2NAIP: A large-scale dataset for Sentinel-2 image
  super-resolution*, Scientific Data 11 (2024) 1389,
  https://doi.org/10.1038/s41597-024-04214-y
- 近期公开、非同行评议的 OpenSR conformal UQ 重叠实现：
  https://github.com/Haris-bin-shakeel/opensr-conformal-uq
