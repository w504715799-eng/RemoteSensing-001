# Phase 2B1A：Crosssensor 云端校验、空间清单与 36 样本试提取设计

**日期：** 2026-08-28

**状态：** 已批准（2026-08-28）

**上位阶段：** `docs/superpowers/specs/2026-08-27-phase2b0-sen2naipv2-provenance.md`

**后续阶段：** Phase 2B1B 扩展至每个划分 120 对；Phase 2B2 构建模型与保序校准缓存

## 1. 目标与本次最小闭环

Phase 2B1 拆成多个可独立验收的小阶段。本规格只定义 Phase 2B1A：

1. 在云端长期存储下载并校验唯一一个 `sen2naipv2-crosssensor.taco` 对象；
2. 从其 8,000 条元数据生成确定性的 5 km 空间防泄漏清单；
3. 按时间差与相关性分层，试提取 36 对真实 LR/HR GeoTIFF；
4. 生成不含像素、可提交 Git 的审计摘要。

本阶段回答：

> 在不污染本地磁盘、不依赖 GPU、且不按 patch 随机切分的前提下，能否建立一个可复现、
> 可审计并足以验证后续实验数据链路的小型 crosssensor 数据基座？

Phase 2B1A 不扩展到 360 对、不运行超分模型、不计算论文指标，也不对方法创新性作实验结论。

## 2. TACO v1 读取器边界及其研究影响

### 2.1 版本事实

本数据对象的元数据声明 TACO 格式版本为 `0.4.0`。当前验证可用的旧版读取路径为：

```text
tacoreader==0.4.5
```

TACO v2 规格明确说明其不向后兼容 v1；最新版 `tacoreader` 的默认 API 面向 v2。不过，
最新版项目仍在 `tacoreader.v1` 子包中保留 v1 兼容入口。因此，“最新版完全不能读取旧数据”
并不准确，但也不能把 v2 顶层 API 当成 `0.4.5` 的无差别替代。

### 2.2 对论文研究的影响

锁定 `0.4.5` 只影响数据摄取工程，不改变模型结构、训练损失、划分方法、评价指标或统计结论。
只要读取结果经过像素、空间、时间和哈希校验，它不会降低研究问题本身的学术价值。主要风险是：

- 旧依赖将来难以重装；
- 不同读取器可能对嵌套元数据或字节流作不同解释；
- 若下游代码直接依赖旧 API，会形成长期技术债。

本阶段通过以下边界消除这些风险：

- `tacoreader==0.4.5` 仅用于云端一次性解包，不加入项目主 `pyproject.toml`；
- 云端基础环境记录经过验证的依赖快照：`geopandas==1.1.4`、
  `pyarrow==25.0.1`、`shapely==2.1.2`、`pyproj==3.7.2`、
  `pyogrio==0.13.0`；实现时以独立约束文件保存，不创建新 Conda 环境、不改动 PyTorch；
- 下游只消费标准 GeoTIFF、canonical manifest 和 SHA-256，不导入 `tacoreader`；
- 后续可在隔离环境用最新版 `tacoreader.v1` 做非阻塞一致性复核，但 Phase 2B1A 不自动迁移。

因此，`0.4.5` 是可复现的数据转换工具，不是论文方法的一部分。论文方法和结果不得依赖其
私有 Python 对象或未记录的隐式行为。

## 3. 冻结来源与完整性

唯一允许下载的真实数据对象为：

| 字段 | 固定值 |
|---|---|
| 仓库 | `tacofoundation/SEN2NAIPv2` |
| Git 修订 | `c370504201072fdb1dd388013ab8c0fc7d00a57e` |
| 文件 | `sen2naipv2-crosssensor.taco` |
| 字节数 | `9,717,583,850` |
| SHA-256 | `c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5` |
| 上游许可声明 | `cc0-1.0`，仅记录声明，不作法律判断 |

官方端点不可达时，允许用镜像作为传输通道，但信任锚始终是 Phase 2B0 从固定官方 Git LFS
指针取得的字节数与 SHA-256。流程必须：

1. 只在用户显式传入的云端长期存储根目录写入；代码不得硬编码 SSH 地址、GPU 型号或根目录；
2. 先写入带 `.part` 后缀的文件，支持断点续传；
3. 下载完成后流式核对精确字节数和 SHA-256；
4. 仅校验通过才原子重命名为正式对象；失败对象进入 `quarantine/`，不得冒充缓存命中；
5. 下载器不得自动获取另外 8 个、合计约 140 GB 的 TACO 对象。

开始前云端可用空间须大于 15 GiB。真实 `.taco`、GeoTIFF、完整 8,000 行清单均不得进入本地
工作区或 Git。

## 4. 元数据契约

验证过的顶层表应为 8,000 行、26 列。至少必须含有：

```text
tortilla:id
stac:crs
stac:geotransform
stac:raster_shape
stac:time_start
stac:centroid
rai:admin0
rai:admin1
rai:admin2
days_between
correlation
scale_factor
```

上游 `split` 全部为 `train`，不得作为本项目划分使用。读取器适配层将每条记录规范化为
`trustsr.sen2naipv2-sample.v1`，至少包含：

- 稳定 `sample_id` 与来源对象/修订；
- centroid 经度、纬度，CRS、仿射变换与 LR/HR 栅格形状；
- LR/HR 获取时间；若顶层不足，则从该样本嵌套元数据读取；
- `days_between`、`correlation`、`scale_factor` 与 `admin0/1/2`；
- `spatial_group_id`、本项目 `split`、试提取分层与选择键；
- 对试提取样本，记录解包后 LR/HR 文件相对路径、字节数和 SHA-256；其余样本显式为
  `null`，不能伪装成已解包。

canonical manifest 不记录绝对路径、主机信息、访问令牌、SSH 配置或运行时间戳。完整清单留在
云端；仓库只接收 schema、代码、合成测试夹具和小于 1 MiB 的摘要/摘要哈希。

## 5. 5 km 空间防泄漏划分

### 5.1 空间连通分量

以 `stac:centroid` 为经纬度，使用半径 `6,371.0088 km` 的球面 haversine 距离。实现可将点
映射到单位球并用 `scipy.spatial.cKDTree` 搜索等价弦长，但最终判定必须为距离 `<= 5 km`。

把每个样本视为节点，任意两点距离 `<= 5 km` 时连边，再用 union-find 求传递闭包。一个连通
分量整体进入一个集合，不能拆开。`spatial_group_id` 定义为：

```text
sha256("\n".join(sorted(component_sample_ids)).encode("utf-8"))
```

### 5.2 确定性分配

直接取 `spatial_group_id` 的前 16 个十六进制字符，转换为无符号 64 位整数并除以
`2**64`：

- `[0.00, 0.50)` → `development`；
- `[0.50, 0.75)` → `calibration`；
- `[0.75, 1.00)` → `internal_test`。

在固定来源的元数据预检中，共得到 6,695 个分量，最大分量含 5 个样本。一次冻结的确定性
实现必须输出并版本化实际计数；若复算结果不等于下列预检值，则停止并审计算法/元数据，不得
静默接受：

| 划分 | 样本数 | 空间分量数 |
|---|---:|---:|
| `development` | 3,967 | 3,317 |
| `calibration` | 2,070 | 1,719 |
| `internal_test` | 1,963 | 1,659 |

同时必须满足：没有完全相同的 centroid 跨集合；任意两集合之间的最小 centroid 距离严格大于
5 km。预检最小值分别约为 `5.001489`、`5.001592` 和 `5.012058 km`；正式审计保存全精度
结果，不用这些四舍五入值参与判断。

## 6. 36 对试提取规则

试提取使用全体 8,000 条记录的固定相关性四分位点：

```text
q25 = 0.8842208864
q50 = 0.9041984739
q75 = 0.9265462586
```

边界值归入较高区间，由此得到 4 个相关性区间。每个划分内部建立
`days_between ∈ {-1, 0, 1}` × 4 个相关性区间，共 12 个分层。每层选择 1 条：

1. 用 `sha256("trustsr-pilot-v1\n" + sample_id)` 排序；
2. 贪心选择尚未在该划分试提取中使用过的 `spatial_group_id`；
3. 若某层没有候选，或无法保持每个划分 12 个不同空间组，立即停止而非改成随机抽样。

最终应为 3 个划分 × 12 层 = 36 个样本对、72 个 GeoTIFF。试提取不改变完整清单的划分，
也不把 `internal_test` 用于模型选择。

## 7. 解包后像素契约

`tacoreader==0.4.5` 只负责从已完整校验的本地云端 TACO 对象取出入选样本的原始 GeoTIFF
字节。每一对必须验证：

- 波段数为 4，语义顺序依据数据卡固定为 `B04, B03, B02, B08`；
- LR 为 `130×130 @ 10 m`，HR 为 `520×520 @ 2.5 m`，比例为 ×4；
- LR/HR centroid 与 CRS 相容，仿射覆盖范围符合容差；
- 原始 dtype、nodata、最小值和最大值被记录，不做隐式归一化或重采样；
- 每个文件的相对路径、精确字节数和 SHA-256 写回 manifest；
- 重复运行不得改写内容不同但文件名相同的缓存。

真实输出建议布局如下，其中 `<storage-root>` 必须由 CLI 参数或任务专用环境变量显式提供：

```text
<storage-root>/trustsr/phase2b1a/
  source/<object-sha256>/sen2naipv2-crosssensor.taco
  manifests/<manifest-sha256>/samples.parquet
  pilot-v1/<split>/<sample-id>/{lr,hr}.tif
  audits/<manifest-sha256>/phase2b1a-audit.json
```

## 8. 本地实现与接口边界

实现计划应以测试驱动方式增加三个相互隔离的能力：

- 纯 Python 元数据规范化、空间分组、确定性划分和试提取选择；
- 只在云端调用的下载/哈希/解包 CLI；
- canonical 审计摘要生成器。

本地测试只能使用运行时生成的微型合成元数据和假影像字节。网络、真实 TACO 和云目录均应
以显式参数注入；单元测试不得连接网络。主项目导入和现有 CPU 测试不得因未安装
`tacoreader` 而失败。

## 9. 验收条件

### 9.1 本地门槛

- 新增与既有测试全部通过，Ruff 通过；
- 合成测试证明 5 km 边界、传递连通、组件不跨集合、哈希确定性和四分位边界行为；
- 相同输入两次生成字节一致的 manifest 与审计摘要；
- 仓库中没有真实 `.taco`/GeoTIFF，也没有大于 1 MiB 的新跟踪文件；
- 未安装 `tacoreader` 时，除显式云端解包命令外的所有功能正常。

### 9.2 云端门槛

- 正式 TACO 对象大小和 SHA-256 与第 3 节完全相同；
- 8,000 行元数据、必需字段、6,695 个空间分量及划分计数通过审计；
- 三个划分之间没有共享空间分量，最小 centroid 距离均严格大于 5 km；
- 恰好得到 36 对、72 个通过像素契约的 GeoTIFF，并记录逐文件 SHA-256；
- 从头复算得到相同的 manifest digest 与相同试提取样本集合；
- 提交到 Git 的只是不含像素的小型摘要和云端产物哈希。

## 10. 停止条件与资源门槛

遇到以下任一情况即停止 Phase 2B1A，不用宽松替代方案掩盖问题：

- 来源字节数或 SHA-256 不匹配；
- TACO 版本、8,000 行/必需字段或预检空间计数不匹配；
- 任意空间组跨集合，或跨集合最小距离 `<= 5 km`；
- 36 个分层样本无法全部取得、LR/HR 结构不符合契约或读取器出现歧义；
- 实现要求把真实数据复制到本地，或下载 crosssensor 之外的对象；
- 云端剩余空间不足 15 GiB。

本阶段不需要 GPU 计算。只有在规格批准并完成本地实现后，执行 9.7 GB 下载和真实试提取时
才需要用户启动具有网络和长期挂载存储的云实例；显卡型号和 CUDA 能力不构成本阶段要求。

## 11. 明确延后

- Phase 2B1B：门槛通过后扩展到每个划分 120 对、合计 360 对；
- SEN2SRLite/LDSR-S2 推理缓存与资源基准；
- split-conformal 校准、风险覆盖率和空间敏感性分析；
- 与 `tacoreader.v1` 最新兼容入口的可选双读一致性测试；
- 论文表格、显著性检验、方法创新或 SCI 分区结论。

## 12. 依据

- SEN2NAIPv2 官方数据仓库与数据卡：
  https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2
- TACO v2 规格（明确 v2 不向后兼容 v1）：
  https://github.com/tacofoundation/specification
- 最新 Tacoreader（说明 v1 位于 `tacoreader.v1`）：
  https://github.com/tacofoundation/tacoreader
- Tacoreader 2.4.21 发布页：
  https://pypi.org/project/tacoreader/2.4.21/
- SEN2NAIP 数据论文：Aybar et al., *Scientific Data* 11, 1389 (2024),
  https://doi.org/10.1038/s41597-024-04214-y
