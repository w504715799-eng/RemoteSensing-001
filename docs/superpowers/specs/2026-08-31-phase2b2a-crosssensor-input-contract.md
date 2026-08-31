# Phase 2B2-A：Crosssensor 模型输入契约与 12 对 CPU 冒烟设计

**日期：** 2026-08-31

**状态：** 已批准（2026-08-31）

**上位阶段：** `docs/superpowers/specs/2026-08-28-phase2b1b-crosssensor-research-subset.md`

**后续阶段：** Phase 2B2-B 在冻结输入契约上生成超分模型预测缓存与资源基准

## 1. 目标与阶段边界

Phase 2B2-A 把 Phase 2B1B 已审计的原始 `uint16` LR/HR GeoTIFF 转换为三个超分模型共同消费的、可复现的 RGBN 反射率张量。本阶段只完成以下工作：

1. 严格加载摘要寻址的 Phase 2B1B 提取后 sidecar；
2. 按 sidecar 重新验证被读取 GeoTIFF 的路径、字节哈希、栅格元数据和像素统计；
3. 把 LR `130×130` 与 HR `520×520` 做空间一致的中心裁剪，得到模型契约要求的 `128×128` 与 `512×512`；
4. 使用唯一固定规则把 `uint16` 数字量化值转换为 `[0,1]` 的 `torch.float32` 反射率；
5. 确定性选择每个 split 四对、合计 12 对样本，执行不依赖 GPU 的真实数据加载冒烟；
6. 写出不含像素、绝对路径、主机信息或运行时间戳的 Git-safe 审计摘要。

本阶段不运行 bicubic、SEN2SRLite 或 LDSR-S2，不下载或加载模型权重，不生成预测缓存，不计算模型指标，不进行共形校准，也不需要 GPU。它只冻结 Phase 2B2-B 及后续实验的输入语义。

## 2. 方案选择

### 2.1 采用按需严格加载

新增独立的 crosssensor 输入加载器。调用者提供持久存储根目录、提取后 sidecar 的显式路径和 SHA-256；加载器只读取请求的样本，并在构造张量前重新验证这些样本的两个 GeoTIFF。

该方案的优点是：

- 不复制约 1 GiB 原始数据，不产生第二份长期像素树；
- 加载、裁剪和归一化规则可以用微型真实 GeoTIFF 做单元测试；
- 模型推理与数据解释解耦，三个基线共享同一输入字节；
- 后续预测缓存身份可以绑定 sidecar、原始 LR 哈希和输入张量哈希。

不采用以下方案：

- 不预先把 360 对全部转换为 safetensors，因为这会重复存储像素并增加同步与审计负担；
- 不把读取、归一化和模型推理写进同一个 CLI 循环，因为这会让数据错误与模型错误难以区分；
- 不复用 OpenSR-Test 私有加载流程，因为 Phase 2B1B 的 GeoTIFF、裁剪和来源身份不同。

## 3. 冻结输入与信任链

Phase 2B2-A 只接受以下固定上游证据：

| 输入 | 固定值 |
|---|---|
| Phase 2B1B 提取后 manifest SHA-256 | `c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a` |
| Phase 2B1B audit SHA-256 | `d8964033958594a23ac7056519894d508977bfd2cc13da50a5833024274f3e90` |
| Phase 2B1A 基础 manifest SHA-256 | `7487b0af2ebef86910e918d5d6b2fb927a6f5e46bac7c2e30be7ffb2ce994482` |
| TACO 对象 SHA-256 | `c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5` |
| sidecar schema | `trustsr.phase2b1b-selection.v1` |
| 样本数与文件数 | 360 对、720 个 GeoTIFF |
| 波段顺序 | `B04, B03, B02, B08` |
| 比例 | ×4 |

sidecar 必须位于：

```text
<storage-root>/trustsr/phase2b1b/selections/
  c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a/
  samples.jsonl
```

路径、目录摘要和实际文件 SHA-256 必须一致。加载时复用 Phase 2B1B 的 canonical JSONL/schema 校验，不新建较弱的替代解析器。

## 4. 模型输入裁剪契约

Phase 2B1B 原始空间覆盖为：

```text
LR: 130×130 @ 10 m
HR: 520×520 @ 2.5 m
```

SEN2SRLite 与 LDSR-S2 已冻结的接口要求：

```text
input:  (4, 128, 128)
output: (4, 512, 512)
```

因此采用唯一的对称中心裁剪：

```python
lr_crop = lr_raw[:, 1:129, 1:129]
hr_crop = hr_raw[:, 4:516, 4:516]
```

这两个窗口删除相同的四周 10 m 地面边界，保持 LR/HR 覆盖范围一致。禁止 resize、插值、随机裁剪或为不同模型采用不同窗口。

裁剪后的仿射变换由原始变换平移得到：

```text
LR pixel offset = (column=1, row=1)
HR pixel offset = (column=4, row=4)
```

加载器必须验证两个裁剪窗口转换后的地理 bounds 在 `1e-3 m` 绝对容差内一致。裁剪窗口、输出形状和裁剪后 transform 写入加载审计；不把绝对文件路径写入审计。

## 5. 反射率归一化契约

LR 和 HR 必须同时满足：

- Rasterio 读取结果为四波段 `uint16`；
- nodata 精确为冻结数据集声明的 `65535`（`uint16` 最大值哨兵）；
- Rasterio mask 中不得存在无效像素，原始数组中也不得出现该哨兵值；
- 所有像素的原始最小值不小于 `0`；
- 所有像素的原始最大值不大于 `10000`；
- sidecar 中记录的 dtype、nodata、minimum 和 maximum 与重新读取结果精确一致；
- sidecar 中的 shape、CRS 和 transform 与重新读取元数据一致。

唯一允许的转换为：

```python
reflectance = torch.from_numpy(raw.copy()).to(torch.float32).div_(10_000.0)
```

转换后必须为 CPU 上连续的 `torch.float32`，形状分别为 `(4,128,128)` 和 `(4,512,512)`，全部有限并处于 `[0,1]`。不使用 `clamp` 静默修改越界值；遇到未知 dtype、nodata 或超过 `10000` 的真实数据时失败并重新评审归一化政策。

固定常量：

```text
REFLECTANCE_SCALE = 10000.0
RAW_DTYPE = uint16
RAW_NODATA = 65535.0
NODATA_POLICY = uint16_sentinel_65535_reject_invalid_v1
NORMALIZATION_POLICY = uint16_divide_10000_no_clip_v1
```

真实数据验收修订依据：冻结 post-manifest 的 360 个 LR 和 360 个 HR 资产均声明
`nodata=65535.0`；固定 12 对烟雾样本的 24 个 GeoTIFF 实测 nodata 与 sidecar
一致，完整影像及中心裁剪内的无效 mask 像素数均为 0，原始值范围为
`[20,9572]`。因此保留哨兵声明，同时对任何实际无效像素 fail closed。

## 6. 公共数据接口

新增 `trustsr.data.crosssensor_pairs`，公开以下不可变记录：

```python
@dataclass(frozen=True)
class CrosssensorPairMetadata:
    manifest_sha256: str
    sample_id: str
    split: str
    spatial_group_id: str
    days_between: int
    correlation_bin: int
    selection_round: int
    lr_asset_sha256: str
    hr_asset_sha256: str
    lr_crop_transform: tuple[float, float, float, float, float, float]
    hr_crop_transform: tuple[float, float, float, float, float, float]
    crop_bounds: tuple[float, float, float, float]
    crop_policy: str
    normalization_policy: str


@dataclass(frozen=True)
class LoadedCrosssensorPair:
    pair: SRPair
    metadata: CrosssensorPairMetadata
```

公共函数：

```python
def load_crosssensor_records(
    storage_root: Path,
    manifest_path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, object], ...]: ...


def load_crosssensor_pair(
    storage_root: Path,
    record: Mapping[str, object],
    *,
    manifest_sha256: str,
) -> LoadedCrosssensorPair: ...


def select_input_smoke_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]: ...
```

`SRPair.source` 固定为：

```text
sen2naipv2-crosssensor/c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a
```

现有 `SRPair`、模型适配器和 Phase 2B1B schema 不修改。

## 7. 12 对确定性冒烟选择

冒烟集不是论文评估集，也不用于选择阈值。它只验证所有 split、四个相关性区间及真实像素读取路径。

每个 split 从 `selection_round == 1` 且 `days_between == -1` 的四个 correlation bin 各取唯一一对，得到：

```text
4 对/split × 3 splits = 12 对
```

结果按 `(split, correlation_bin)` 排序。选择器必须验证每个 split 恰好覆盖 bin `0,1,2,3`，样本 ID 和空间组均唯一，并且输入顺序变化不影响结果。这里固定 `days_between == -1` 是为了让12对仍保持一个简单、完全可解释的二维覆盖；时间差泛化属于正式360对实验，不由冒烟集判断。

## 8. CLI 与审计产物

新增 CPU-only 命令：

```text
trustsr-phase2b2a audit-inputs \
  --storage-root ROOT \
  --selection-manifest MANIFEST \
  --selection-manifest-sha256 c7f8ffa... \
  --confirm-cloud-storage
```

执行顺序：

1. 在任何输出前验证长期存储确认、sidecar 路径和固定摘要；
2. 加载并严格校验全部360行 sidecar，但只读取12对真实像素；
3. 对12对的24个 GeoTIFF 重新计算字节数与 SHA-256；
4. 重新检查元数据、像素统计、裁剪 bounds 和归一化张量；
5. 对相同输入执行第二次加载，比较样本顺序、元数据和 LR/HR 张量 SHA-256；
6. 原子写入 canonical JSON 审计。

云端产物路径：

```text
<storage-root>/trustsr/phase2b2a/input-audits/<manifest-sha256>/
  phase2b2a-input-audit.json
```

Git-safe 副本路径：

```text
artifacts/datasets/sen2naipv2-phase2b2a-input-audit-v1.json
```

审计 schema 为 `trustsr.phase2b2a-input-audit.v1`，至少包含：

- 上游 manifest、audit、base manifest 和 TACO SHA-256；
- `smoke_pair_count=12`、`smoke_geotiff_count=24`；
- 每个 split 四对、每个 correlation bin 三对的计数；
- 原始与裁剪后的 shape、crop policy、reflectance scale、raw nodata、nodata policy 和 normalization policy；
- 12个 LR 和12个 HR 原始资产摘要；
- 12个 LR 和12个 HR 归一化张量摘要；
- 两次加载摘要一致的布尔证据；
- `model_inference_run=false`、`gpu_used=false`、`real_pixels_local=false`。

审计不记录绝对路径、SSH、主机名、GPU 型号、运行时间、原始像素或可逆像素统计以外的数据。

## 9. 路径、安全与恢复

- storage root 必须是显式确认的现有绝对目录，不能是 `/`、`/root`、当前 HOME、符号链接或包含符号链接的路径；
- sidecar 必须位于冻结的摘要寻址目录；
- 每个资产相对路径必须等于 sidecar 中的 canonical Phase 2B1B 路径，并限制在 `<storage-root>/trustsr/phase2b1b/` 内；
- GeoTIFF 不能是符号链接，且必须为普通文件；
- 审计写入使用临时文件与原子替换；已存在审计只有在 canonical bytes 完全一致时才复用；
- 任一文件、元数据、像素范围或重复加载不一致时停止，不覆盖 Phase 2B1B 数据。

## 10. 测试策略

所有产品代码遵循 red-green-refactor TDD。测试在本地生成微型 `uint16` GeoTIFF 和完整合成 sidecar，不连接网络、不需要 `tacoreader`、不读取云端数据。

### 10.1 加载器测试

- 正确重新验证文件大小、SHA-256、shape、dtype、CRS、transform、nodata 和像素范围；
- LR/HR 精确裁剪到 `(4,128,128)` 与 `(4,512,512)`；
- 使用手工可验证的边界像素证明裁剪索引是 `[1:129]` 与 `[4:516]`；
- `/10000` 后张量为连续 CPU float32 且不修改合法值；
- 裁剪后的 bounds 对齐；
- 文件哈希错误、symlink、路径逃逸、错误 dtype、nodata、越界值、错误 band count 或 transform 时失败。

### 10.2 冒烟选择测试

- 每个 split 四对且每个 correlation bin 一对；
- 总数12、样本和空间组唯一；
- 只接受第一轮和 `days_between=-1`；
- 输入顺序改变时结果完全相同；
- 任一格缺失或重复时失败，不使用替补规则。

### 10.3 CLI 与审计测试

- 命令必须显式提供固定 sidecar SHA 和云存储确认；
- 任何输入错误发生在创建输出目录之前；
- 恰好重新哈希24个文件并加载12对两次；
- canonical audit 重复运行产生相同 SHA-256 并报告复用；
- Git-safe 审计不含绝对路径、主机、凭据、像素或时间戳。

### 10.4 仓库质量门槛

```text
uv run pytest -q
uv run ruff check .
uv lock --check
git diff --check
```

仓库不得新增真实 `.tif/.tiff/.taco`、完整 sidecar、模型权重或超过1 MiB的跟踪文件。

## 11. 远程验收与资源边界

本地实现和完整测试通过后，才请求一个能够访问 Phase 2B1B 长期存储的云实例。真实验收只使用 CPU 和约24个 GeoTIFF 的顺序读取，不要求特定 GPU、CUDA 或显存。

云端验收条件：

- 长期存储 mount 与 inode 均可用；
- 冻结 sidecar 和720个 Phase 2B1B GeoTIFF 仍存在；
- 12对选择满足固定 split/bin 计数；
- 24个文件的字节与像素契约全部通过；
- 两次加载产生相同 tensor SHA-256；
- Git 只接收小型 canonical 审计；
- 验收完成且无远程进程后，明确通知用户可以暂停云服务器。

## 12. 停止条件

遇到以下任一情况停止 Phase 2B2-A：

- Phase 2B1B post-manifest SHA、路径或 schema 不匹配；
- 长期存储不是 mount、inode 耗尽或任一选中资产缺失；
- 真实 dtype 不是 `uint16`、nodata 不等于 `65535`、存在任一无效 mask/哨兵像素或真实反射率超出 `[0,10000]`；
- 中心裁剪后的 LR/HR bounds 不一致；
- 任一 sidecar 元数据或哈希与重新读取结果不同；
- 重复加载产生不同张量摘要；
- 实现需要把真实像素复制到 WSL、下载其他大数据或修改 Phase 2B1B 原始文件。

停止后保留只读证据，不用裁剪、clamp、忽略 nodata 或降低哈希检查来绕过失败。归一化契约若被真实数据否定，必须先形成带统计证据的新版本设计。

## 13. 明确延后

- bicubic、SEN2SRLite 与 LDSR-S2 的12对或360对模型推理；
- 模型权重下载、GPU 环境检查和显存/时间基准；
- 预测缓存 schema 与完整360对推理调度；
- LR 一致性残差、TTA、LDSR 多次采样方差；
- 局部风险、共形阈值、风险—覆盖率曲线与 Checkpoint B；
- 外部测试、消融、显著性检验和论文图表。
