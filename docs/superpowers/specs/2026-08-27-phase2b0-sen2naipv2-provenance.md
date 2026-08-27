# Phase 2B0：SEN2NAIPv2 来源冻结与本地禁下载设计

**日期：** 2026-08-27
**状态：** 已批准 Phase 2B 路线后的首个可执行子阶段
**上位路线：** `docs/superpowers/specs/2026-08-26-trustworthy-sentinel2-sr-roadmap.md`
**前置阶段：** `docs/superpowers/specs/2026-08-27-phase2a-conformal-core.md`

## 1. 目标

Phase 2B0 只建立可审计的数据来源和存储边界，不读取真实遥感影像。它把
SEN2NAIPv2 的仓库修订、许可声明、数据卡摘要、Git LFS 对象校验和与大小冻结成一个
小型 JSON 清单，并提供完全离线的校验接口。

本阶段回答：

> 在不向本地下载约 149 GB 像素数据的前提下，能否获得一个确定性、可复核、可供后续
> 云端小样本读取复用的数据来源契约？

## 2. 已核实的上游事实

截至 2026-08-27，官方 Hugging Face 数据仓库为
`tacofoundation/SEN2NAIPv2`，冻结 Git 修订：

```text
c370504201072fdb1dd388013ab8c0fc7d00a57e
```

该修订的数据卡：

- 声明许可证为 `cc0-1.0`；本项目只记录上游声明，不作法律判断；
- SHA-256 为
  `5897aed9410fef305953ff5b34e83697b466901583b880158af2902a8267a58d`；
- 声明 RGBN 波段顺序为 `B04, B03, B02, B08`；
- 声明 LR 为 `130×130 @ 10 m`，HR 为 `520×520 @ 2.5 m`，比例为 ×4；
- 声明 `unet` 62,242 对、`histmatch` 61,282 对、`crosssensor` 8,000 对；
- 只把划分方式描述为 `stratified`，没有给出可复现的地理防泄漏算法。

仓库中的 9 个 TACO Git LFS 对象合计 `149,356,128,592` 字节
（149.356 GB / 139.099 GiB）：

| 文件 | LFS SHA-256 | 字节数 |
|---|---|---:|
| `sen2naipv2-crosssensor.taco` | `c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5` | 9,717,583,850 |
| `sen2naipv2-histmatch.0000.part.taco` | `a276024df0f81ff53770cf1b415d0f86268bd2b090a467b80e2e8b3992d08acc` | 20,000,560,299 |
| `sen2naipv2-histmatch.0001.part.taco` | `c493107a10a488643346aba990717e2caa839f26a3cbde4e359c7f0f83158c4b` | 19,999,654,192 |
| `sen2naipv2-histmatch.0002.part.taco` | `f922d29d7701cbcad6e41da235805c26ad11200f50acc34e77af7de38abd66ae` | 19,999,987,735 |
| `sen2naipv2-histmatch.0003.part.taco` | `3faadd2e3b9b9a9611764e9e4f8c2d230667d6d75604eb26d82e5e8f1e65da26` | 9,361,318,942 |
| `sen2naipv2-unet.0000.part.taco` | `b8f7a8497328e62fcb53872d2d59220a0fc84fd05f6d205bbe54b4c2d32fa6c2` | 20,000,562,604 |
| `sen2naipv2-unet.0001.part.taco` | `c24c4500f779f0d9db6d7c2f568879dd3c40d26fd8e36cde1b235dfecb489e1f` | 20,000,506,401 |
| `sen2naipv2-unet.0002.part.taco` | `aba2f08677464e8359e5c98d7ebe77bfab623919a7cde3c24f8e0018dc4319ce` | 20,000,130,510 |
| `sen2naipv2-unet.0003.part.taco` | `8835cbf9d0d179190f495802e2b548e4cf1d45b7c18d8910b31a2449c9b49632` | 10,275,824,059 |

## 3. 强制存储边界

### 3.1 本地允许

- 源码、文档和小于 1 MiB 的 JSON/文本元数据；
- Git LFS 指针文本，不包括指针指向的对象；
- 单元测试现场生成的微型假 TACO/JSON 夹具；
- 不含真实遥感像素的确定性审计结果。

### 3.2 本地禁止

- 任意真实 `.taco` 数据对象；
- SEN2NAIPv2 的影像 patch、样本缓存、模型输出缓存；
- 任何会隐式执行 Git LFS smudge 或批量数据下载的默认命令；
- 把 OpenSR-Test 最终测试区域复制成校准或内部验证集。

### 3.3 云端允许

Phase 2B1 获得独立规格后，真实 TACO 分卷、索引、解码样本和模型缓存只放在用户指定的
云端长期存储目录。临时中间文件可放云端临时盘。本阶段不连接云服务器，也不要求 GPU。

项目代码不能硬编码 GPU 型号、SSH 地址、端口、用户名或云端根目录。

## 4. 范围

### 4.1 包含

- 一个版本化的 `trustsr.sen2naipv2-source.v1` JSON 来源清单；
- 对修订、SHA-256、文件名、非负大小、唯一文件名和合计大小的严格校验；
- 离线 Python 加载接口；
- 一个只读、离线、canonical JSON 审计 CLI；
- 防止误把像素数据或超大文件提交 Git 的仓库策略和测试；
- 记录 Phase 2B1 所需但当前尚未从真实 TACO 元数据验证的字段。

### 4.2 不包含

- 不安装 `tacoreader`，不修改 PyTorch 依赖；
- 不运行 `git lfs pull`、`huggingface-cli download` 或远程 range read；
- 不下载、解码或抽样任何真实像素；
- 不定义未经数据验证的经纬度、ROI 或 TACO 内部字段名；
- 不生成 train/calibration/internal-test 划分；
- 不运行 SEN2SRLite 或 LDSR-S2；
- 不产生论文指标，不宣称创新或统计保证。

## 5. 接口与清单契约

新增 `trustsr.data.provenance`：

```python
@dataclass(frozen=True)
class LfsObject:
    path: str
    sha256: str
    size_bytes: int

@dataclass(frozen=True)
class DatasetSource:
    schema: str
    repository: str
    revision: str
    license_claim: str
    card_sha256: str
    bands: tuple[str, ...]
    scale: int
    lr_shape: tuple[int, int]
    hr_shape: tuple[int, int]
    objects: tuple[LfsObject, ...]

    @property
    def total_bytes(self) -> int: ...

def load_dataset_source(path: Path) -> DatasetSource: ...
```

加载器必须拒绝：额外/缺失顶层字段、错误 schema、非 40 位小写 Git 修订、非 64 位小写
SHA-256、绝对路径或包含 `..` 的路径、重复文件名、零/负大小、非 RGBN 固定顺序、非 ×4
形状关系，以及总大小与清单声明不一致。不得访问网络。

来源清单放在：

```text
artifacts/datasets/sen2naipv2-source-v1.json
```

清单只包含来源元数据，不包含认证信息、绝对路径、时间戳或机器信息。

## 6. 离线审计 CLI

新增命令：

```text
trustsr-dataset-audit --source artifacts/datasets/sen2naipv2-source-v1.json
```

它只读取指定的小型 JSON，输出一行 canonical JSON，至少包含：

- `schema: "trustsr.dataset-audit.v1"`；
- 仓库、修订和上游许可声明；
- 对象数量、合计字节数和按变体统计；
- `metadata_only: true`、`network_accessed: false`、`pixel_data_downloaded: false`；
- `local_real_pixel_policy: "forbidden"`；
- `ready_for_phase2b1_schema_probe: true`。

相同输入连续运行两次必须字节一致。CLI 不提供下载开关，也不接受令牌、SSH 或数据根目录。

## 7. Phase 2B1 接口门槛

只有 Phase 2B0 通过后，才单独设计 Phase 2B1。Phase 2B1 的第一步是在云端长期存储上
安装/验证精确版本的 TACO 读取器，并只读取索引或极少量 range。只有确认真实元数据字段后，
才定义包含以下语义的样本清单：

- 稳定 `sample_id`；
- 可追溯 ROI/空间组；
- 传感器、LR/HR 获取日期；
- 地表组别或显式缺失标记；
- train/calibration/internal-test 划分；
- 数据对象 SHA-256 与源修订。

划分必须按空间组整体分配，不能按 patch 随机分割。同一空间组不能跨集合。若没有足够
坐标或 ROI 元数据，则 Phase 2B1 停止，不能用随机 patch 划分替代。

## 8. 验收条件

- 新增和既有测试全部通过，Ruff 通过；
- 固定清单严格匹配本规格列出的修订、数据卡 SHA-256、9 个对象和总大小；
- 审计 CLI 两次输出字节一致且明确声明没有网络和像素下载；
- 测试证明加载器不调用网络；
- 仓库中不存在大于 1 MiB 的新跟踪文件，也不存在真实 `.taco` 文件；
- 本阶段不需要 GPU，且不会连接云服务器。

## 9. 停止条件

- 若上游固定修订与清单不一致，停止并重新做来源审计，不能自动跟随 `main`；
- 若实现要求下载任意真实 TACO 对象，停止并移到 Phase 2B1 云端规格；
- 若无法从真实元数据证明空间组，停止，不以随机 patch 划分冒充防泄漏划分；
- 若任何命令默认执行网络或下载，Phase 2B0 不通过。

## 10. 依据

- SEN2NAIPv2 官方数据仓库与数据卡：
  https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2
- SEN2NAIP 数据论文：Aybar et al., *Scientific Data* 11, 1389 (2024),
  https://doi.org/10.1038/s41597-024-04214-y
- OpenSR-Test 官方实现：https://github.com/ESAOpenSR/opensr-test
