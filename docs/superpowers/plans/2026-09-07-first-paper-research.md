# TrustSR First-Paper Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to carry out this
> research plan task-by-task. Work sequentially on attached `main`; no concurrent repository
> writers. Steps use checkbox (`- [ ]`) syntax for tracking. This is a research-deliverable plan;
> software implementation specs are produced at the named gates before the corresponding code.

**Goal:** 完成一篇 Sentinel-2 RGBN ×4 超分辨率不确定性评分的成本、误差尺度与校准迁移
实证研究论文，形成可投稿前评审的完整初稿和可复现证据。

**Architecture:** 复用已经验收的数据／模型／缓存和 Git-safe 结果，先完成论文证据整理，
再设计最小评分基线与外部协议。development 选择完成并冻结全部配置后，一次完成 Spain
外部评估；最后统一分析与成稿。论文写作从第一阶段开始。

**Tech Stack:** 当前 Python 3.12、PyTorch、NumPy、SciPy、锁定的 OpenSR-Test、pytest、
Ruff；论文使用 Markdown／CSV 和按需选择的标准绘图工具，不为本轮规划增加依赖。

**Spec:** [已选择的第一篇论文路线](../../research-roadmap.md)，以及
[文献复核与研究约束](../../reports/2026-09-07-research-replan.md)。

**Decision:** 用户于 2026-09-07 选择推荐路线 B，授权新计划与 Git 发布。
本次仅完成计划文档；Task 1–8 都未执行。第一篇默认不含可选回退。

## Global Constraints

- 在当前 attached `main` 顺序工作；本地 Git 为权威，云端代码不得合回。
- Phase 2B3-C 保持 `empirically_met_but_inconclusive`，不重跑、不重开、不重新解释为确认。
- 已发布 Phase 2B3-A/B/C 科学结果及冻结参数均不可覆盖。
- M0 仅消费本地 Git-safe JSON；B/C 原始像素、缓存、ledger 不进入新研究访问范围。
- 输入 RGBN、L2A、×4；中心重建 LDSR seed 3407，K5 seeds 3407–3411。
- 本计划不训练新模型，不加入分组共形、SEN2NEON、下游分割或回退产品。
- 外部数据必须在任何像素／预测／指标访问前冻结协议；新数据和 GPU 访问须有明确执行授权。
- 数据资格与缺失处理只按访问前规则执行，不按观察到的模型效果筛选。
- 每个任务完成后更新本计划对应复选框，记录实际验证与提交；未执行项不能勾选。
- 新的软件实现只对真实风险设计必要测试；不扩大历史冻结模块，不复制整套阶段专用基础设施。
- 最终交付不要求正结果或显著优于所有基线；所有预设比较和失败均需记录。

## 时间、依赖与预算

| 任务 | 阶段 | 依赖 | 工作日时间盒 |
|---|---|---|---:|
| Task 1–2 | M0 证据与初稿地基 | 已发布 A/B/C JSON | 1–2 |
| Task 3–4 | M1 基线规格与元数据／统计草案 | M0 | 3–5 |
| Task 5–6 | M2 限定开发、实现与最终冻结 | M1 | 3–5 |
| Task 7 | M3 一次外部研究 | M2 完成及专门授权 | 2–4 加实际排队时间 |
| Task 8 | M4 分析与完整初稿 | M3；写作可提前 | 7–10 |

目标约 4–6 周；这是作者持续投入、数据与算力条件满足时的估算，不是硬截止日期或
录用承诺。若有学校要求、明确费用上限或硬截止日期，先写入预算文件再执行依赖步骤。

以官方名义 48 张 Spain 图像估计，共需 240 次 LDSR 图像调用、48 次 SEN2SRLite 调用，
其余评分复用这些预测。实际时长由开发冒烟实测；切块次数、加载、写盘和缓存复算
另计。默认一个 worker，不重复已有无收益的四 worker 优化。

## Task 1：形成可引用的证据清单与指标字典

**Files:**
- Create: `docs/reports/first-paper-evidence-inventory.md`
- Create: `paper/metric-dictionary.md`
- Read: `artifacts/phase2b3a/sen2naipv2-development-score-audit-v1.json`
- Read: 已提交的 `artifacts/phase2b3b/` 三份校准文件和 `artifacts/phase2b3c/` 三份结果文件。
- Read: `src/trustsr/evaluation/score_diagnostics.py`、`src/trustsr/evaluation/selective.py`。

**Interfaces:** 输入为只读的已提交文件；输出为“主张、数据角色、公式、文件、哈希、
可支持结论、不能支持结论”七列证据表和指标定义。该表是后续初稿的事实源。

- [ ] 核对实际 Git 提交、发布文件 SHA-256、canonical JSON 和交接中数值。
- [ ] 记录 development 的选择用途、calibration 的拟合用途、internal_test 的终局用途。
- [ ] 定义 R1、R9、十点离散 AURC、Spearman、ROI 最大局部误差、覆盖率及风险 UCB。
- [ ] 对历史 `high_risk_miss_rate_at_80` 写明公式和“高风险排除比例，越大越好”的标签；
      手算全部保留为 0、全部排除为 1，不更改原科学文件。
- [ ] 写明 `alpha=0.05` 是风险目标，置信误差是另一个参数；`accepted` 是流程验收。
- [ ] 逐项核对原检查点 B；只把当前证据作为继续有限实证研究的依据，不宣布跨域保证成立。

**Acceptance:** 每个已报告数字可在发布文件中定位或由明确公式复算；没有读取远端或
原始内部测试。指标字典能解释 AURC 与校准损失为何不同。

## Task 2：准备可复建图表与初稿骨架

**Files:**
- Create: `paper/manuscript.md`
- Create: `paper/related-work.md`
- Create: `paper/figure-table-plan.md`
- Create: `docs/superpowers/specs/first-paper-evidence-rendering-design.md`

**Interfaces:** 消费 Task 1 证据表与文献报告；输出论文结构和仅读取已发布 JSON 的
渲染器规格。规格定义 CSV 列、图注、数据角色、输出位置、确定性排序和旧文件保护。

- [ ] 写引言、三条研究问题、相关工作、当前方法和数据角色，不预写外部结果。
- [ ] 图表清单明确：数据与成本表、R9/R1 曲线、原 B/C 结果、外部迁移表、预设案例图。
- [ ] 将渲染器输入限定为精确 Git-safe 文件清单；输出目录为 `paper/tables/` 和
      `paper/figures/`，代码入口规划为 `scripts/paper/render_evidence.py`。
- [ ] 定义复建验收：已发表表格可从固定输入生成；CSV／数值确定一致，图像仅对科学
      内容和轴／图例检查，不要求带可变渲染元数据的 PDF 字节完全一致。
- [ ] 明确绘图依赖与范围，随后为该小型工具写独立实现步骤，不为制图重构评价流水线。

**Acceptance:** 已有部分可以开始成文；未来结果明确标记为“未执行”，不填入模拟的
正向效果。渲染器设计不依赖旧 B/C 缓存或云端状态。

## Task 3：明确最近邻评分基线的最小实现规格

**Files:**
- Create: `docs/superpowers/specs/first-paper-neighborhood-score-design.md`
- Read: `src/trustsr/risk/local.py`、`src/trustsr/risk/proxies.py`。

**Interfaces:** 消费文献和现有 K5 评分合同；输出邻域方差适配的公式、固定候选范围、
shape/dtype、边界处理、数值容差与合成例子的预期值。

- [ ] 对照论文二阶矩公式和作者代码，固定引用版本；记录两者有无实现差异。
- [ ] 定义 RGBN 反射率适配，说明与原 Lab 实验的区别，禁止把简单方差平滑冒充原公式。
- [ ] 在开发访问前规定一个小候选范围及选择规则，选择只使用 development。
- [ ] 指定合成 oracle：常数时为零；空间变化与跨种子变化可区分；窄边界处理确定；
      四波段 reduction、有限值和浮点舍入得到可手算或独立直接实现核验。
- [ ] 明确 HR 只作为风险标签，任何分数生成不得读取 HR。
- [ ] 明确五种评分共享中心预测，成本按独立部署所需调用核算，同时报告实验缓存总成本。

**Acceptance:** 规格足以据此写必要的数学测试与实现计划，不含新主干、SAR、训练型融合
或大型参数搜索。新模块的文件名与签名在此规格中确定后再写代码，避免先建通用框架。

## Task 4：完成 Spain 元数据审计和统计协议草案

**Files:**
- Create: `docs/reports/first-paper-spain-metadata-audit.md`
- Create: `docs/superpowers/specs/first-paper-spain-evaluation-design.md`

**Interfaces:** 消费官方版本化元数据及 Task 1、3；输出数据合同、成员／空间组摘要、
主次指标与缺失处理草案。外部图像、缩略图、预测和指标在此阶段不可访问。

- [ ] 核验 Crops／Urban 版本和实际成员数量；28+20 仅为当前官方名义数量。
- [ ] 记录 RGBN、L2A、×4、归一化、HR 协调、裁剪与 nodata 合同。
- [ ] 审计空间组、来源场景、日期及已知预训练重叠；没有元数据就记录未知，不假设独立。
- [ ] 主比较固定为每个 Spain 子集内 K5 对 LR 残差的配对 R9 AURC 差，负值为 K5 更低。
- [ ] 三模型、邻域基线、随机解析基线必须完整报告；R1、Spearman、覆盖、结构诊断和
      成本为次要结果。不能比较不同方法各自最优的覆盖点。
- [ ] 指定统计单位与加权目标；如报告区间，写明场景组重采样和配对方式、次数、种子、
      小组数限制及多重比较处理，标为近似区间，不能标为分布无关的风险证书。
- [ ] 若独立性不可支持，预先选择描述性报告分支；不实施像素级 bootstrap 来制造精度。
- [ ] K5 的原 B 阈值仅作迁移描述；其他分数按同覆盖预算比较，不共享数值阈值。
- [ ] 预设失败／缺失分母、停止规则及图像展示选择规则；外部图像不得充当调试数据。

**Acceptance:** 两子集均有明确可执行合同或具体无法验证项；统计结论范围匹配数据。
若数据根本不能满足 RGBN ×4 或必要来源要求，先解决来源，不下载像素“试试看”。

## Task 5：完成限定开发、合成验证和成本评估

**Files:**
- Create: `docs/reports/first-paper-development-readiness.md`
- Create: `docs/reports/first-paper-compute-budget.md`
- Create: Task 2–4 所定义的各自小型软件实现计划及相应实现／测试文件。

**Interfaces:** 输入为已审阅的 Task 2–4 规格；输出已验证的渲染、评分和独立外部入口，
一个选定的邻域配置以及以实测时间计算的费用预算。

- [ ] 顺序实现规格中确定的小模块；模型／缓存／统计基础复用现有代码，旧模块保护不放宽。
- [ ] 所有新外部 loader 和计算步骤先通过 synthetic/SPOT tripwire 检查，不读取 Spain。
- [ ] 对 development 复用范围确认后，仅在该范围内比较候选；不以 B/C 聚合值选择配置。
- [ ] 缓存不可用时先报告重建成本和所需权限；不自动重跑历史阶段。
- [ ] 核对当前锁定 OpenSR 版本的地图语义及坐标；结构诊断若不稳定，在冻结前连同
      对应论文主张一起移除，不能在看到外部效果后移除。
- [ ] 从开发或 SPOT 的固定运行记录时间／显存与后处理成本；核算外部完整矩阵费用。
- [ ] 生成已有证据的首批图表并更新初稿；记录实际测试命令和退出状态。

**Acceptance:** 科学输入／输出合同和必要测试通过，成本可计算；性能测量未使用外部
测试，且未承诺 48 张图像必然足够完成风险确认。

## Task 6：最终冻结与专门执行授权

**Files:**
- Finalize: `docs/superpowers/specs/first-paper-spain-evaluation-design.md`
- Create: `docs/reports/first-paper-freeze-record.md`

**Interfaces:** 汇合 Tasks 1–5；输出唯一最终协议版本及实施版本、数据摘要、预算与
访问范围。所有后续外部运行消费该版本，而不是草案。

- [ ] 开发结束后冻结模型身份、seed、评分配置、输入处理、ROI／组成员、主次指标、
      覆盖网格、统计区间、缺失规则、复现过程和展示规则。
- [ ] 核对原 C 终局结果完全未改，且新运行标识／产物目录与旧阶段分离。
- [ ] 在外部访问前审阅并提交冻结协议；计划中的方法选择至此结束。
- [ ] 提供具体数据范围、完整计算量和费用预算，取得尚未授予的外部访问／GPU 执行授权。

**Acceptance:** 可以向用户展示确切执行范围与费用依据。等待必要授权期间继续完善论文
已有章节；没有授权时不执行 Task 7，也不把等待视为数据许可。

## Task 7：一次完成外部研究并独立核验

**Files:**
- Create: `docs/reports/first-paper-spain-results.md`
- Create: 新协议精确允许的小型结果文件；云端像素、缓存、运行环境和凭据留在 Git 外。

**Interfaces:** 只消费冻结协议、已授权外部输入与模型；输出完整五评分结果、费用、失败
记录及缓存复算核验，不允许依据部分结果改变执行矩阵。

- [ ] 执行既定成员，身份核验后只补缺失预测；五种评分复用同一批预测。
- [ ] 按已冻结规则分别完成 Crops 和 Urban 的全部主次统计；不以中途效果扩样或换方法。
- [ ] 缓存复算核对统计、独立检查结果组成、成员和摘要；沿用通用校验工具。
- [ ] 只发布新协议精确允许的科学文件，保存全部失败和缺失说明。
- [ ] 内部 inconclusive 与外部迁移作为两个独立研究事实报告，不发出旧 C 的新决策。

**Acceptance:** 完整矩阵按协议执行并可复算；负结果亦完成该任务。协议错误或数据合同
问题不能以局部结果“补救”，需先记录偏离及其对可推断范围的影响。

## Task 8：完整初稿、质量审阅与研究结项

**Files:**
- Finalize: `paper/manuscript.md`、`paper/related-work.md`、`paper/metric-dictionary.md`
- Finalize: `paper/figure-table-plan.md` 及其实际主表／图。
- Create: `paper/reproducibility.md`
- Create: `docs/reports/first-paper-completion-review.md`

**Interfaces:** 消费所有阶段证据；输出完整论文、补充材料和复现说明，明确研究发现与
假设／解释的区别。

- [ ] 完成摘要、结果、讨论、局限与结论；外部结果填入实际数值，不删掉不利比较。
- [ ] 核验论文标题与内容一致：无结构证据就不主张幻觉检测，无下游就不主张任务收益。
- [ ] 检查所有数字、主张、图注、单位、置信范围、样本数和文献归属。
- [ ] 用固定输入复建主表／图，记录命令、依赖和科学输出哈希；不重开已消费内部测试。
- [ ] 依学校要求和论文实际贡献选择投稿类型，使用当前官方期刊信息核验范围与费用。
- [ ] 完成投稿前方法／统计审阅，记录剩余局限与返修方向；更新 handoff 和本计划。

**Acceptance:** 完整初稿、预设实验、全部必需图表与复现说明齐全。录用、显著优势、
普遍风险保证及后续扩展都不是本研究交付已经完成的同义词。

## 状态记录与默认下一步

每个任务的完成记录包含：交付文件、验证命令／人工复核点、Git 提交、偏离说明。
进度按 Task 1–8 的实际验收记录更新，不能把“计划提交”算成“研究执行完成”。

当前下一步：**Task 1，仅本地已发布证据清单与指标字典**。本次用户请求到计划 Git
发布为止，不在本轮启动 Task 1–8 的实施或任何真实数据／GPU 任务。
