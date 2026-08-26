# Phase 1A：SEN2SRLite CPU 预训练基线规格

**日期：** 2026-08-26  
**状态：** 已批准实施  
**上位路线：** `2026-08-26-trustworthy-sentinel2-sr-roadmap.md` 的阶段 1  
**范围：** 只完成 SEN2SRLite 的 CPU 适配、预测缓存和固定 SPOT 开发集上的公平基线对比；LDSR-S2 留到阶段 1B 的云 GPU 环境。

## 1. 目标与边界

本阶段建立第一个真实预训练模型的可复现基线，使后续共形风险实验能够复用同一模型接口、同一数据清单和可校验的模型输出。

本阶段必须完成：

- 统一双三次插值与 SEN2SRLite 的最小 `SRModel` 接口；
- 固定使用 `SEN2SRLite_NonReference_RGBN_x4`，输入波段顺序为 B04、B03、B02、B08；
- 校验下载模型的全部资产哈希后才加载权重；
- 在 OpenSR-Test SPOT v3 的全部 9 个开发样本上比较双三次与 SEN2SRLite；
- 使用安全、带身份校验的预测缓存，使第二次运行不再执行模型推理；
- 生成不含运行时间等非确定字段的 JSON 结果，连续两次运行字节完全一致。

本阶段不做：

- 不训练或微调网络；
- 不接入 LDSR-S2、不使用 CUDA；
- 不使用 OpenSR-Test 的 NAIP、Spain Urban 或 Spain Crops 调参；
- 不以 SEN2SRLite 必须胜过双三次作为工程通过条件；
- 不把 SPOT 开发结果写成论文最终结论。

## 2. 固定模型与供应链约束

模型清单固定为 Hugging Face 上的：

```text
https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json
```

模型 ID 为 `SEN2SRLite_NonReference_RGBN_x4`。依赖版本固定为 `mlstac==0.4.9` 和 `sen2sr==0.8.5`，并由项目锁文件记录完整依赖图。

下载后必须校验以下 SHA-256；任何文件缺失或哈希不一致均立即失败，不能继续反序列化或推理：

| 文件 | SHA-256 |
|---|---|
| `example_data.safetensor` | `c895c7da8a8d48882b73a2a1955e4260714b97540eea290229a284d73f129985` |
| `hard_constraint.safetensor` | `fbad981519066387c413ead1d6af7ef3e0d2947c34147ba90163fc79ae539239` |
| `load.py` | `4b6c836b1f73078c62c84d4374b2d8daee5345f6239f64e0b6be29432383bac6` |
| `mlm.json` | `59caa5c6af96a6fbebdbd771d93c91cc2d3a770302cd2f262b5409e77a40e3f7` |
| `model.safetensor` | `479aa796d5068d0b1206118ccbca27bd3223df0214db1a9b31a1e18349ed1c7e` |

`mlm.json` 引用的 `load.py` 会执行动态 Python 加载，因此“先校验、后加载”是强制安全边界。单元测试不得访问网络，使用测试替身覆盖下载后的验证和模型推理接口。

## 3. 模型契约

两个模型均实现：

```python
class SRModel(Protocol):
    name: str
    scale: int

    def predict(self, lr: torch.Tensor) -> torch.Tensor: ...
    def provenance(self) -> dict[str, JsonScalar]: ...
```

输入契约：

- `torch.float32`；
- 形状严格为 `(4, H, W)`；
- 值全部有限且位于 `[0, 1]`；
- SEN2SRLite 当前固定接受 `(4, 128, 128)`，模型输入批次为 `(1, 4, 128, 128)`。

输出契约：

- CPU 上的连续 `torch.float32`；
- 形状为 `(4, 4H, 4W)`；
- 值全部有限并裁剪到 `[0, 1]`；
- 不保留梯度。

SEN2SRLite 的原始输出可能超过 1。裁剪是显式的“反射率输出策略”，必须写入 provenance；它不是模型隐藏行为。模型 provenance 至少记录模型 ID、清单 URL、资产哈希、依赖版本、设备和输出策略。双三次 provenance 记录插值模式、缩放倍数、`align_corners`/抗锯齿策略和输出策略。

## 4. 缓存设计

每个预测由以下身份字段唯一确定：

- 完整模型 provenance；
- 数据来源与 `sample_id`；
- LR 张量形状、数据类型和连续字节的 SHA-256。

身份对象使用排序键、紧凑分隔符的规范 JSON 编码，其 SHA-256 为缓存键。预测张量以 safetensors 保存，身份和校验信息写入 JSON sidecar；禁止 pickle。写入采用同目录临时文件后原子替换，避免中断产生半文件。

缓存命中时必须重新验证身份、张量形状、数据类型、数值范围和有限性。文件损坏、身份不匹配或元数据缺失不得被当作命中。缓存位于 `artifacts/cache/predictions/`，不提交 Git。

## 5. 固定基准命令与结果结构

新增 `trustsr-benchmark` 命令，默认：

- 数据：OpenSR-Test SPOT v3；
- 样本：固定全部 9 个，缺少或多出时失败；
- 模型：`BicubicX4` 与 `SEN2SRLiteX4`；
- 模型目录：`models/SEN2SRLite_RGBN/`；
- 缓存目录：`artifacts/cache/predictions/`；
- 输出：`artifacts/phase1/spot-v3-baselines.json`。

JSON 顶层为 `run` 和 `models`：

- `run` 记录 Git 提交、Python/PyTorch/OpenSR 版本、数据集、样本数和样本清单哈希；
- 每个模型记录 provenance、逐样本指标和有限值的平均指标；
- 两个模型必须引用同一清单哈希；
- 不记录耗时、时间戳、绝对路径或 `cache_hit`，以保证结果确定性。

指标沿用阶段 0 已验证的 OpenSR 指标管线。JSON 写入必须使用固定键排序和稳定浮点序列化；同一提交、同一依赖锁、同一模型资产和同一数据输入连续运行两次，输出文件字节完全一致。

## 6. 验证与通过条件

按以下顺序验证：

1. 模型接口、输入拒绝、输出裁剪、资产校验和缓存损坏行为的离线单元测试；
2. CLI 使用测试替身验证两个模型共享同一 9 样本清单及确定性 JSON；
3. 全量单元测试与 Ruff；
4. 本机 CPU 上真实下载经哈希校验的 SEN2SRLite，并运行全部 9 个 SPOT v3 样本；
5. 再运行一次，确认复用缓存且 JSON 字节完全一致；
6. 审查结果中所有指标均为有限值，输出形状与反射率范围满足契约。

阶段 1A 通过只表示预训练基线工程闭环可信，不要求 SEN2SRLite 在所有指标上优于双三次。通过后停止，向用户索取云 GPU SSH 信息，再为阶段 1B 单独制定 LDSR-S2 计划。

