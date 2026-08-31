# Phase 2B1B：Crosssensor 360 对正式研究子集设计

**日期：** 2026-08-28

**状态：** 已批准（2026-08-28）

**上位阶段：** `docs/superpowers/specs/2026-08-28-phase2b1a-crosssensor-pilot.md`

**后续阶段：** Phase 2B2 构建超分模型推理缓存、资源基准与最小共形实验

## 1. 目标与阶段边界

Phase 2B1B 把已通过真实数据审计的 36 对 Phase 2B1A 试提取扩展为可供后续模型推理使用的
正式研究子集：

1. 从已冻结的 8,000 行空间防泄漏清单中确定性选择 360 对 LR/HR 影像；
2. 每个 `development`、`calibration`、`internal_test` 划分固定 120 对；
3. 保持时间差、相关性分层及划分内空间组唯一性；
4. 在云端已有 TACO 缓存中独立提取并审计 720 个 GeoTIFF；
5. 生成不含真实像素、可以提交 Git 的小型审计摘要。

本阶段不运行 SEN2SRLite 或 LDSR-S2，不生成模型预测缓存，不进行共形校准，不计算论文结果，
也不需要 GPU。它只冻结 Phase 2B2 将使用的数据协议和真实像素集合。

## 2. 方案选择

### 2.1 采用独立 sidecar 清单

Phase 2B1B 新建一个摘要寻址的 360 行 sidecar 清单，并引用已审计的 Phase 2B1A 完整
manifest。该方案保持 Phase 2B1A 的 schema、36 对选择和审计产物不可变，同时给 Phase 2B2
提供一个范围明确的输入接口。

未采用以下方案：

- 不重写 Phase 2B1A 的 8,000 行 manifest schema，因为这会带来无必要的迁移风险；
- 不把现有 `pilot` 字段原地扩展到 360 对，因为这会混淆“试提取”和“正式研究子集”的语义；
- 不让新清单直接依赖 Phase 2B1A 的 36 对像素路径，因为跨阶段资产引用会降低可迁移性。

### 2.2 独立重提取

全部 360 对都从已经校验并缓存在云端的同一个 TACO 对象重新提取到 Phase 2B1B 独立目录。
不会重新下载 9.7 GB 源对象。重复提取首轮 36 对的计算与存储开销相对于得到一个自包含、可迁移
的数据产物是可接受的。

## 3. 冻结输入与信任链

Phase 2B1B 只接受以下上游证据：

| 输入 | 固定值 |
|---|---|
| Phase 2B1A 审计后 manifest SHA-256 | `7487b0af2ebef86910e918d5d6b2fb927a6f5e46bac7c2e30be7ffb2ce994482` |
| 数据仓库修订 | `c370504201072fdb1dd388013ab8c0fc7d00a57e` |
| TACO 对象名 | `sen2naipv2-crosssensor.taco` |
| TACO 对象字节数 | `9,717,583,850` |
| TACO 对象 SHA-256 | `c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5` |
| 完整样本数 | `8,000` |
| 5 km 空间连通分量数 | `6,695` |

输入 manifest 必须位于 Phase 2B1A 的摘要寻址目录中，其目录名、实际字节哈希和记录内来源身份
必须一致。加载时继续使用 Phase 2B1A 的严格 canonical JSONL 校验，包括确定性 36 对选择、
完整资产状态、生产计数和 schema 验证。任何不一致都在写入 Phase 2B1B 文件之前停止。

`tacoreader==0.4.5` 的边界不变：它仅在云端基础环境中负责从 TACO v1 对象读取原始字节，
不进入项目依赖锁，也不参与选择算法、模型方法或统计结论。

## 4. 确定性 360 对选择协议

### 4.1 固定分层

沿用 Phase 2B1A 的三个时间差和四个全体数据相关性区间：

```text
days_between = {-1, 0, 1}
correlation cuts = {0.8842208864, 0.9041984739, 0.9265462586}
```

边界值归入较高区间。每个数据划分共有 `3 × 4 = 12` 个分层，每层固定选择 10 对，因此：

```text
10 对/层 × 12 层/划分 × 3 划分 = 360 对
```

### 4.2 十轮轮转选择

候选继续按下式升序排列：

```text
sha256("trustsr-pilot-v1\n" + sample_id)
```

选择器按划分独立运行。在每个划分中执行 10 轮；每一轮按 Phase 2B1A 的固定顺序遍历 12 个
分层，并从当前层选择排序最前且尚未被该划分使用过 `spatial_group_id` 的候选。

此顺序具有两个明确性质：

- 第一轮严格重现 Phase 2B1A 每个分层各一对的选择，因此 36 对试提取样本全部成为正式子集；
- 完成十轮后，每个划分使用 120 个互不相同的空间组，避免同一局部区域在一个划分内被过度重复。

选择结果按 `(split, round, days_between, correlation_bin)` canonical 排序。若任一层在任一轮无法
提供尚未使用的空间组，流程失败；不得减少样本量、重复空间组、改变分层或使用非确定性随机补齐。

Phase 2B1A 已冻结的 5 km 连通分量和 split 分配不重新计算、不重新分桶。选择器只消费已经
验证的分配，因此三个划分继续保持严格大于 5 km 的跨划分 centroid 最小距离。

## 5. Sidecar schema

预提取和提取后清单均使用 `trustsr.phase2b1b-selection.v1`，恰好包含 360 行。每行至少记录：

- `base_manifest_sha256` 和冻结 TACO 来源身份；
- `source_index`、`sample_id`、`split` 和 `spatial_group_id`；
- centroid、CRS、仿射变换、LR 栅格形状和 ×4 比例；
- LR/HR 获取时间、`days_between`、原始 `correlation` 和相关性区间；
- `selection_round`、选择哈希和固定波段顺序 `B04, B03, B02, B08`；
- `lr_asset` 与 `hr_asset`。

选择时从上游 manifest 逐字段复制元数据并做严格类型校验。预提取清单的两个资产字段都为
`null`；提取后清单的两个字段都包含相对路径、精确字节数、SHA-256、shape、dtype、CRS、
transform、nodata、像素最小值/最大值和获取时间。禁止一边为空、一边非空，也禁止部分样本
有资产而其他样本没有资产。

清单采用逐行 canonical JSON，按 `sample_id` 稳定排序后写入，并以完整文件 SHA-256 寻址。
记录不含绝对路径、主机名、SSH 信息、令牌、GPU 型号或运行时间戳。

## 6. 命令接口与数据流

新增独立控制台命令 `trustsr-phase2b1b`，包含三个阶段：

```text
trustsr-phase2b1b select \
  --source SOURCE \
  --storage-root ROOT \
  --base-manifest BASE_MANIFEST \
  --confirm-cloud-storage

trustsr-phase2b1b extract \
  --source SOURCE \
  --storage-root ROOT \
  --selection-manifest SELECTION_MANIFEST \
  --confirm-cloud-storage

trustsr-phase2b1b audit \
  --source SOURCE \
  --storage-root ROOT \
  --selection-manifest SELECTION_MANIFEST \
  --confirm-cloud-storage
```

数据流为：

1. `select` 验证源对象、Phase 2B1A 基础 manifest 及 36 对前缀，生成 360 行全空资产清单；
2. `extract` 重新验证所有输入，从已有 TACO 缓存提取 360 对到独立目录，并生成全资产清单；
3. `audit` 对 720 个文件重新流式计算大小和 SHA-256，复核选择及像素契约，写入审计摘要。

建议云端布局为：

```text
<storage-root>/trustsr/phase2b1b/
  selections/<manifest-sha256>/samples.jsonl
  subset-v1/<split>/<sample-id>/{lr,hr}.tif
  audits/<manifest-sha256>/phase2b1b-audit.json
```

云端包装脚本只按 `select → extract → audit` 顺序协调阶段，并把每一步 canonical JSON 结果写入
显式日志目录。脚本不硬编码 SSH 主机、端口、密码、云厂商、存储根目录、GPU 型号或 CUDA 版本。

## 7. 像素契约

每一对重新执行 Phase 2B1A 已验证的像素检查：

- 波段数为 4，顺序为 `B04, B03, B02, B08`；
- LR 为 `130×130 @ 10 m`，HR 为 `520×520 @ 2.5 m`；
- 比例固定为 ×4，LR/HR CRS 一致，空间覆盖范围在既定亚毫米级容差内相同；
- 原始 dtype、nodata、最小值和最大值被记录，不做归一化或重采样；
- LR/HR 获取时间必须与 selection manifest 中的时间一致；
- 文件路径必须被限制在 Phase 2B1B 的 `subset-v1` 目录内，且不能是符号链接。

本阶段不改变像素值。反射率归一化和模型输入转换属于 Phase 2B2。

## 8. 幂等、恢复与失败处理

每个命令先完成全部只读前置校验，再创建输出。候选 manifest 和审计使用临时文件与原子替换；
只有完整内容通过校验后才进入摘要寻址目录。

提取可按样本恢复：

- 样本目录不存在时正常提取；
- `lr.tif` 与 `hr.tif` 都存在、均为普通文件且重新校验通过时可以复用；
- 只存在一个文件、包含额外文件、存在符号链接、像素契约失败或哈希与已知记录不符时立即停止；
- 不覆盖或静默修复异常目录。

提取后 manifest 只能在 360 对全部就绪后生成。审计只接受全空的预提取清单或全满的提取后
清单所规定的对应阶段，不接受混合状态。若摘要目录已存在，其内容必须逐字节相同，否则停止。

## 9. 审计摘要

`trustsr.phase2b1b-audit.v1` 至少包含：

- 源对象、基础 manifest 和提取后 selection manifest 的 SHA-256；
- 总样本数 `360`、GeoTIFF 数 `720`；
- 每个划分的样本数、空间组数和每个分层的样本数；
- 各选择轮的计数以及第一轮与 Phase 2B1A 36 对集合一致的布尔证据；
- Phase 2B1A 冻结的三个跨划分最小距离；
- `real_pixels_local=false` 和 `gpu_used=false`。

仓库只提交该小型 canonical JSON 审计摘要及其来源说明。完整 360 行清单、TACO 和 GeoTIFF
继续留在用户指定的云端长期存储。

## 10. 实现边界

实现应复用而不破坏 Phase 2B1A 的读取、GeoTIFF 验证、canonical JSON、路径限制和原子写入
能力。新增能力保持独立：

- Phase 2B1B 十轮选择器及选择记录；
- sidecar 读写、严格 schema 校验和审计构建；
- 三阶段 CLI；
- 云端顺序执行包装脚本。

现有 `select_pilot()`、Phase 2B1A manifest schema、CLI 和审计 schema 均保持行为兼容。
Phase 2B1B 不修改 PyTorch、Torchvision、Triton、CUDA 或 `nvidia-*` 包，也不创建 Conda 环境。

## 11. 测试策略

所有实现遵循 red-green-refactor TDD。本地测试只使用运行时生成的合成元数据、微型 GeoTIFF
和测试替身，不连接网络、不访问云端、不导入真实 TACO。

### 11.1 选择器测试

- 每个划分得到 120 对、每层 10 对、每轮每层一对；
- 第一轮恰好等于现有 `select_pilot()` 的 36 对；
- 同一划分中 120 个空间组互不重复；
- 输入顺序变化不改变结果；
- 相关性边界仍归入较高区间；
- 任一后续轮容量不足时失败，而不是重复空间组。

### 11.2 Schema 与 CLI 测试

- canonical round-trip、摘要寻址、基础 manifest 哈希和来源身份校验；
- 360 行以外、错误轮次、错误分层、重复样本或重复空间组均被拒绝；
- 预提取/提取后全空或全满状态验证；
- 三阶段参数、云存储显式确认、路径逃逸和符号链接防护；
- 已完成样本恢复、部分目录拒绝和最终 manifest 原子提交；
- 720 个资产的路径、哈希、shape、时间及空间契约审计。

### 11.3 仓库回归门槛

- 全部现有和新增 pytest 测试通过；
- Ruff、`uv lock --check` 和 `git diff --check` 通过；
- 仓库不新增真实 `.taco`、GeoTIFF、完整 360 行清单或超过 1 MiB 的跟踪文件；
- 未安装 `tacoreader` 时，除显式云端提取命令外的所有导入与测试正常。

## 12. 云端验收与资源边界

只有本地实现和全量回归门槛通过后才请求用户启动云服务器。真实运行不需要 GPU，显卡型号、
CUDA 能力和显存不构成要求；需要的只是已有源对象、长期存储、足够磁盘和可用 CPU。

云端正式门槛为：

- 输入 manifest 和 TACO 信任链完全匹配；
- 恰好得到 360 对、720 个有效 GeoTIFF；
- 每个划分 120 对、每层 10 对、每个划分 120 个不同空间组；
- 第一轮 36 对与 Phase 2B1A 冻结选择完全相同；
- 所有像素、时间、空间、路径和哈希检查通过；
- 从头复算得到相同选择集合、selection manifest digest 和审计摘要；
- Git 只接收不含像素和主机信息的小型审计产物。

预计真实像素规模约为 1 GiB 量级，但以提取后实际文件字节数为准。开始提取前要求云端长期
存储至少有 5 GiB 可用空间；不因当前 GPU 型号变化而修改代码或规格。

## 13. 停止条件

遇到以下任一情况停止 Phase 2B1B：

- Phase 2B1A 基础 manifest、来源对象或生产计数不匹配；
- 第一轮不能重现已冻结 36 对；
- 任一分层不能提供第 10 个未使用空间组；
- 不能满足 360 对、每划分 120 对或划分内空间组唯一；
- 任意 LR/HR 对不满足像素、时间或空间契约；
- 输出目录部分存在、哈希冲突、含符号链接或逃逸长期存储根目录；
- 实现要求把真实像素复制到本地或下载其他 TACO 对象；
- 长期存储可用空间低于 5 GiB。

停止后保留可验证的已有缓存和日志，不使用降低标准的替代结果冒充通过。

## 14. 明确延后

- SEN2SRLite 与 LDSR-S2 在 360 对上的预测缓存；
- GPU/CPU 时间、显存和吞吐基准；
- 反射率归一化、局部误差、不确定性分数和 split-conformal 阈值；
- 风险—覆盖率曲线、域偏移分析、显著性检验和论文表格；
- 最新 `tacoreader.v1` 与 `0.4.5` 的可选双读一致性测试。
