# Phase 2B2-B：Development 三模型真实预测冒烟与可回放缓存设计

**日期：** 2026-08-31

**状态：** 已批准（用户授权按推荐路线直接实施）

**上位阶段：** `docs/superpowers/specs/2026-08-31-phase2b2a-crosssensor-input-contract.md`

**后续阶段：** Phase 2B3 在完整 calibration/development/internal_test 预注册范围内生成预测，并开展共形风险校准与选择性回退

## 1. 研究角色与阶段结论边界

Phase 2B2-B 的唯一目标，是证明冻结的真实 crosssensor 输入能够被 bicubic、SEN2SRLite 和 LDSR-S2 三个既有适配器一致消费，并形成完整、不可混淆、无需再次推理即可审计和计算 OpenSR 指标的预测缓存。

本阶段是 Checkpoint A 的开发集工程验收，不是论文主实验。它只允许回答：

- 三个模型能否在同一真实 RGBN 输入契约上产生合法 `4×512×512` 预测；
- 预测是否绑定数据版本、输入字节和模型配置；
- 缓存能否无模型构造、无推理地确定性回放；
- 四个 development 冒烟样本上的 OpenSR 七指标是否全部有限。

本阶段不能声称模型优于 bicubic、方法具有统计显著性、风险受到共形保证、能够泛化到全部区域，或已经达到论文主结果。即使学习模型的冒烟均值差于 bicubic，只要契约、缓存和回放全部成立，工程 Checkpoint A 仍可通过；该结果会作为后续风险控制研究的证据，而不会被隐藏或用于临时改变指标。

## 2. 方案选择与防泄漏约束

采用“一个 development 样本闸门，再四个 development 样本”的分阶段方案：

1. 本地用合成数据与假模型完成全部 TDD；
2. 云端 preflight 只验证硬件、模型资产和冻结输入审计；
3. `single` 对 canonical development 首样本运行三个模型；
4. `smoke` 对四个 development 样本运行三个模型，复用 `single` 已生成的三个缓存条目；
5. `replay` 不构造任何模型，不调用 `predict`，仅从缓存重建相同结果并核对字节。

Phase 2B2-A 的 12 对输入审计覆盖 calibration、development 和 internal_test，是数据契约证据；Phase 2B2-B 必须在任何模型构造、推理或指标计算前过滤为 `split == "development"`。不得加载 calibration 或 internal_test 的 HR 张量，不得计算或展示这两个 split 的任何指标。

不采用以下方案：

- 不对 Phase 2B2-A 全部 12 对计算指标，因为这会提前观察 calibration 与 internal_test 标签；
- 不直接对 360 对运行三个模型，因为当前尚未验证 crosssensor 的模型、显存、时间和缓存路径；
- 不新增训练、微调、TTA、LDSR 多次采样或阈值选择，因为这些会混淆本阶段的基础可复现性目标。

## 3. 冻结输入与信任链

Phase 2B2-B 只接受以下固定证据：

| 输入 | 固定值 |
|---|---|
| Phase 2B1B post-manifest SHA-256 | `c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a` |
| Phase 2B2-A input audit SHA-256 | `fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b` |
| Phase 2B1B audit SHA-256 | `d8964033958594a23ac7056519894d508977bfd2cc13da50a5833024274f3e90` |
| Phase 2B1A base manifest SHA-256 | `7487b0af2ebef86910e918d5d6b2fb927a6f5e46bac7c2e30be7ffb2ce994482` |
| Phase 2B2-A input audit schema | `trustsr.phase2b2a-input-audit.v1` |
| 波段顺序 | `B04, B03, B02, B08` |
| 输入/输出 | `(4,128,128)` / `(4,512,512)` |
| 比例 | ×4 |

云端 input audit 必须位于：

```text
<storage-root>/trustsr/phase2b2a/input-audits/<post-manifest-sha256>/
  phase2b2a-input-audit.json
```

其 canonical bytes 的 SHA-256 必须等于冻结值，且内容必须与仓库内 Git-safe 副本一致。post-manifest 仍由 Phase 2B2-A 严格加载器验证；不得提供任意 manifest、任意 input audit 或可变样本列表。

## 4. 固定 development 冒烟集

先调用 Phase 2B2-A 的 `select_input_smoke_records` 得到 canonical 12 格集合，然后只保留 `split == "development"`。选择结果必须满足：

- 恰好四个样本；
- correlation bin 按 `0,1,2,3` 排序且各出现一次；
- `selection_round == 1`、`days_between == -1`；
- 样本 ID 和空间组均唯一；
- 四个记录全部来自 development；
- 输入顺序变化不改变结果。

首样本固定为排序后 correlation bin 0 的 development 记录。CLI 不提供样本 ID、split、数量、bin、seed 或指标的覆盖参数。

## 5. 冻结模型政策

模型顺序固定为：

1. `bicubic-x4`：现有 `BicubicX4`，CPU；
2. `sen2srlite-x4`：现有 `SEN2SRLiteX4`，CPU，已冻结并校验资产摘要；
3. `ldsr-s2-x4`：现有 `LDSRS2X4`，CUDA，`seed=3407`、`sampling_steps=100`、`sampling_eta=0.95`、`sampling_temperature=1.0`、`histogram_matching=true`。

三个适配器均必须报告 `scale=4`，模型名称唯一，provenance 与适配器身份一致。不得按 GPU 型号改变采样参数；硬件只记录为运行环境，不进入科学配置。LDSR 仍执行现有单 GPU、CUDA、最低算力和空闲显存安全门槛，但不锁定具体显卡型号。

## 6. 缓存身份与持久化

复用 `PredictionCache` 的 `.safetensors + .json` 原子提交格式。每个缓存身份由以下内容共同决定：

- 模型适配器返回的完整 provenance；
- `experiment_schema=trustsr.phase2b2b-development-smoke.v1`；
- 固定 post-manifest SHA-256；
- 固定 Phase 2B2-A input audit SHA-256；
- `SRPair.source` 与 `sample_id`；
- LR shape、dtype 与连续 CPU 张量 SHA-256。

上述实验字段与模型 provenance 合并为 cache provenance，仅用于缓存键；结果中同时保留原始 `model_provenance` 和完整 `cache_provenance`，避免把数据版本误称为模型属性。任何模型配置、依赖版本、数据摘要或 LR 张量变化都必须产生不同 key。

每个预测必须是 detached、contiguous、CPU `float32`，形状 `(4,512,512)`，全部有限且位于 `[0,1]`。写入后立即重新读取并逐元素精确相等；指标只从重新读取的缓存张量计算，不从模型返回的临时对象计算。

长期缓存路径为：

```text
<storage-root>/trustsr/phase2b2b/predictions/<post-manifest-sha256>/
  <cache-key>.safetensors
  <cache-key>.json
```

Git 不跟踪预测像素、模型权重或缓存 sidecar。

## 7. 结果与缓存审计

确定性产物目录为：

```text
<storage-root>/trustsr/phase2b2b/results/<post-manifest-sha256>/
  single-result.json
  development-three-model-smoke.json
  development-cache-audit.json
```

`single-result.json` 包含一个样本、三个模型的身份、预测摘要与七项指标。它是早停闸门，不参与四样本均值。

`development-three-model-smoke.json` 的 schema 为 `trustsr.phase2b2b-development-smoke.v1`，包含：

- 四个上游固定摘要、输入审计 schema、波段顺序和 scale；
- `dataset_role=development_engineering_smoke_only`；
- 四个 development 样本的 sample ID、correlation bin、空间组、LR/HR 张量摘要；
- 三个模型的原始 provenance；
- 每模型四个样本的 cache key、预测 tensor SHA-256 和七项 OpenSR 指标；
- 每模型四样本算术均值。

`development-cache-audit.json` 的 schema 为 `trustsr.phase2b2b-cache-audit.v1`，按 `(model_name, correlation_bin, sample_id)` 排序记录恰好12个身份。每项包含 cache key、LR 摘要、预测摘要以及 `.json/.safetensors` 的文件名、字节数和文件 SHA-256。它不包含绝对路径、mtime、主机名、GPU 型号或时间戳。

运行时间、显存峰值、GPU/驱动环境只写入独立的 runtime 文件，不能进入上述三个确定性科学产物，也不进入 Git-safe 结果。

## 8. OpenSR 指标政策

只调用项目现有 `compute_opensr_metrics`，固定七项：

```text
reflectance, spectral, spatial, synthesis,
ha_metric, om_metric, im_metric
```

所有值必须可转换为有限 `float`，不得丢失指标、重命名、加权或在看到结果后选择子集。均值为四个 development 冒烟样本的简单算术平均。指标用于发现跨传感器失败模式和验证后续研究动机，不用于当前阶段模型排名或论文显著性结论。

## 9. CLI 阶段与恢复语义

新增命令 `trustsr-phase2b2b`，只有以下子命令：

```text
preflight  # 校验存储、manifest、input audit、CUDA 与模型资产；不推理
single     # canonical development 首样本 × 三模型
smoke      # 四个 development 样本 × 三模型
replay     # 禁止模型构造和 predict，仅从12个缓存重建并核对结果
```

所有子命令都要求显式 `--storage-root`、固定 manifest 路径与 SHA、固定 input-audit 路径与 SHA、两个模型目录及 `--confirm-cloud-storage`。shell runner 使用云镜像 base Python，不创建 conda 环境，不安装 PyTorch；缺失的轻量依赖可在模型下载/准备期间并行安装，但进入推理前必须锁定版本并验证模型摘要。

`single` 和 `smoke` 都先查询 cache；已存在且完整的条目直接复用，缺失条目才调用相应模型。损坏、身份不符或只有半个文件的条目 fail closed，不覆盖。已有 deterministic result 只有 canonical bytes 完全一致时才复用。

## 10. 无推理回放验收

`replay` 必须：

1. 在任何读取前记录12个已命名 cache 条目的文件名、大小、mtime_ns 和 SHA-256；
2. 从已提交的 cache audit 重建12个 `PredictionIdentity`；
3. 使用 `PredictionCache.get` 完整验证 sidecar、tensor 摘要、shape/range；
4. 重新加载四个 development HR，仅计算同样七项指标；
5. 在内存中重建 smoke result 与 cache audit 的 canonical bytes；
6. 要求重建 bytes 与已提交文件逐字节相同；
7. 再次记录缓存状态，要求文件集合、大小、mtime_ns 和 SHA-256 完全未变。

`replay` 不导入或构造三个模型，不调用任何 `predict`，也不改写缓存和确定性产物。它可以在 GPU 关闭后用 CPU 执行。

## 11. 路径与安全政策

- 继续复用 `require_cloud_confirmation` 的显式长期存储确认；
- storage root、manifest、input audit、模型目录、缓存目录和结果目录都禁止符号链接路径分量；
- 派生路径必须限制在 `<storage-root>/trustsr/phase2b2b/`；
- `preflight` 在创建结果目录前验证所有固定摘要；
- shell runner 拒绝 `/`、`/root`、HOME、相对路径、glob、含冒号仓库路径、非 mount 存储和不足容量/inode；
- 日志不得含 SSH 凭据、绝对输入资产路径或像素；
- 不复制真实 GeoTIFF 到 WSL，不修改 Phase 2B1B/2B2-A 的任何文件。

## 12. 测试策略

所有产品代码使用 red-green-refactor TDD。本地测试只使用合成 `LoadedCrosssensorPair`、假模型和临时缓存，不连接网络、不需要真实模型或 GPU。

### 12.1 选择与身份测试

- 从 canonical 12格集合只选四个 development 样本；
- 任一 split/bin/顺序/唯一性错误均失败；
- cache key 同时响应模型配置、post-manifest、input-audit 与 LR 摘要变化；
- 输出顺序与输入 mapping 顺序无关。

### 12.2 运行器测试

- single 恰好调用每模型一次，并只加载首个 development 样本；
- smoke 恰好形成12个身份，缓存命中不调用 predict；
- 指标只读取经 `PredictionCache.get` 验证的张量；
- 非法 shape、dtype、device、range、NaN、非有限指标或 provenance 立即失败；
- 不加载 calibration/internal_test pair，不产生其指标或缓存；
- 部分缓存可恢复，损坏缓存不被静默覆盖。

### 12.3 回放与产物测试

- replay 在模型 factory 被设置为抛错时仍成功；
- 缺失、额外、重复或损坏缓存条目失败；
- cache audit 必须恰好12项且与 smoke result 双向绑定；
- 重建 result/audit 必须 byte-identical；
- 回放前后缓存文件集合、大小、mtime_ns 和摘要不变；
- deterministic JSON 不含时间、运行时、主机、绝对路径或 GPU 型号。

### 12.4 仓库门槛

```text
uv run pytest -q
uv run ruff check .
uv lock --check
git diff --check
```

仓库不得新增真实 `.tif/.tiff/.taco/.safetensors`、模型权重、完整360行 sidecar 或超过1 MiB的跟踪文件。

## 13. Checkpoint A 验收与停止条件

Checkpoint A 通过必须同时满足：

- single 三模型均成功；
- smoke 恰好4个 development 样本、3个模型、12个唯一 cache key；
- 所有预测 shape、dtype、范围、模型 provenance 与数据摘要均有效；
- 84个样本级指标值与21个均值全部有限；
- replay 无模型、无推理成功且 deterministic bytes 相同；
- cache 文件在 replay 前后完全未变；
- runtime 与科学结果分离；
- 完整测试和仓库质量门槛通过。

以下任一情况停止，不自动扩大实验：

- 固定 manifest 或 input audit 摘要不匹配；
- 任一 development 记录/资产/张量与 Phase 2B2-A 不一致；
- GPU、模型资产或依赖版本不满足现有适配器门槛；
- single 出现 OOM、非有限输出、非法范围、缓存写后不等或 LDSR 运行时间不可接受；
- 需要读取 calibration/internal_test HR 才能继续；
- smoke/replay 的结果、缓存身份或文件状态不一致。

失败时保留已提交的有效缓存和独立 runtime 证据，不降低哈希、范围、确定性或防泄漏检查。

## 14. 明确延后

- 全部120个 development、120个 calibration 或120个 internal_test 的三模型推理；
- calibration 上的共形阈值拟合和 internal_test 风险保证；
- 局部风险图、选择性 bicubic 回退、风险—覆盖率曲线；
- 模型训练/微调、TTA、LDSR ensemble 或不确定性扩展；
- 外部数据集、消融、置信区间、显著性检验和论文图表。
