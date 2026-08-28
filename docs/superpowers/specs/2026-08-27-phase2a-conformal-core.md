# Phase 2A：共形可信掩膜 CPU 核心设计

**日期：** 2026-08-27
**状态：** 文献审计后收窄的下一阶段规格
**上位路线：** `docs/superpowers/specs/2026-08-26-trustworthy-sentinel2-sr-roadmap.md`

## 1. 审计结论与本阶段定位

截至 2026-08-27，以下内容不能再作为本项目的论文首创点：

- LDSR-S2 已提供 RGBN ×4 潜扩散超分辨率和基于多次采样的像素级不确定性；
- NeurIPS 2025 的 *Image Super-Resolution with Guarantees via Conformalized
  Generative Models* 已覆盖“黑盒生成式超分辨率 + 不确定性分数 + 共形阈值 +
  二元可信掩膜”；
- SEN2SR 已使用 Fourier 低频硬约束实现 Sentinel-2 辐射和空间一致性；
- DiffFuSR 已覆盖 Sentinel-2 扩散超分辨率、退化感知和全波段融合；
- 固定 SAM/光谱损失、通用数据一致性投影和冻结扩散先验加轻量适配器均已有直接先例。

因此，Phase 2A 不宣称方法创新，而是复现并验证后续研究所需的最小统计核心。论文候选
贡献留到后续阶段验证：以 ROI 为统计单位的多光谱风险、跨传感器失效审计，以及对最终
回退产品重新校准的选择性超分辨率。

本阶段只回答：

> 能否在 Python 3.12、CPU 和小张量上，正确、确定性地实现通用共形可信掩膜基线，
> 并通过风险—覆盖率行为测试，为真实数据实验建立可信接口？

## 2. 范围

### 2.1 包含

- 四波段 `[0, 1]` SR/HR 张量的局部 L1 风险图；
- 多次随机 SR 输出的逐像素方差分数；
- 以完整 ROI/样本为独立校准单位的有限样本共形阈值；
- 可信掩膜、风险—覆盖率点和确定性 CPU 冒烟 CLI；
- 数值、形状、边界和失败关闭测试；
- 在输出中明确标记 `synthetic_smoke=true`，禁止把冒烟结果当论文证据。

### 2.2 不包含

- 不下载 SEN2NAIPv2、SEN2NEON 或新的 OpenSR-Test 数据；
- 不运行 LDSR-S2，不连接云 GPU，不生成真实多次采样缓存；
- 不实现 SAM、反射率和高频复合风险；
- 不实现 group/Mondrian、加权或跨域共形方法；
- 不实现 bicubic/SEN2SRLite 回退融合；
- 不训练、微调或修改任何 SR 主干；
- 不声称跨地区、跨传感器或逐像素风险保证。

这些内容分别进入后续独立规格，避免一个不可审查的大阶段。

## 3. 统计定义

设第 `i` 个校准 ROI 的不确定性分数图为 `s_i[p]`，局部保真风险图为
`r_i[p]`。每个 ROI 是一个独立校准样本；同一 ROI 内的像素不是独立样本。

### 3.1 局部 L1 风险

对于 `sr, hr ∈ [0, 1]^(C×H×W)`：

```text
pixel_error[p] = mean_c |sr[c,p] - hr[c,p]|
local_l1[p] = box_mean(pixel_error, odd_window)
```

边界使用反射填充，使输出仍为 `(H, W)`。`odd_window` 必须为正奇数且不大于
`min(H, W)`。在输入满足 `[0, 1]` 时，风险有固定上界 `B = 1`。

### 3.2 多次采样方差分数

对于 `samples ∈ [0, 1]^(K×C×H×W)`：

```text
score[p] = mean_c Var_k(samples[k,c,p], correction=0)
```

至少需要两个样本。使用总体方差而非无偏样本方差，以便同一输入在不同框架实现中保持
一致定义。分数必须为有限、非负 `(H, W)` 张量。

### 3.3 共形阈值

目标风险为 `alpha ∈ (0, 1]`。对候选阈值 `t`，第 `i` 个 ROI 的可信区域最坏风险为：

```text
worst_i(t) = max({r_i[p] : s_i[p] <= t} union {0})
```

有限样本上界为：

```text
bound(t) = (sum_i worst_i(t) + B) / (n + 1)
```

选择满足 `bound(t) <= alpha` 的最大有限观测分数作为阈值。若没有任何分数满足条件，
返回显式的“全拒绝”阈值 `-inf`。可信掩膜定义为 `score <= threshold`。

该实现复现通用共形 SR 基线的有限样本校准形式；它不自动赋予跨传感器保证。只有校准
ROI 与测试 ROI 可交换时，相关边际保证才有解释空间。

## 4. 接口边界

### 4.1 `trustsr.risk.local`

```python
def local_l1_risk(sr: torch.Tensor, hr: torch.Tensor, *, window: int) -> torch.Tensor
def ensemble_variance_score(samples: torch.Tensor) -> torch.Tensor
```

二者返回 CPU 或输入设备上的 `torch.float64` 二维张量，不静默裁剪输入，不接受 NaN、
Inf、越界反射率或错误维度。

### 4.2 `trustsr.calibration.conformal`

```python
@dataclass(frozen=True)
class ConformalCalibration:
    alpha: float
    threshold: float
    risk_bound: float
    calibration_size: int
    trusted_pixels: int
    total_pixels: int

def calibrate_fidelity_mask(
    scores: Sequence[torch.Tensor],
    risks: Sequence[torch.Tensor],
    *,
    alpha: float,
    risk_upper_bound: float = 1.0,
) -> ConformalCalibration

def trusted_mask(score: torch.Tensor, calibration: ConformalCalibration) -> torch.Tensor
```

阈值搜索必须确定性，且不得把像素摊平后伪装成独立校准样本。

### 4.3 `trustsr.evaluation.selective`

```python
@dataclass(frozen=True)
class SelectivePoint:
    threshold: float
    coverage: float
    roi_max_risk: float

def evaluate_selective_point(
    scores: Sequence[torch.Tensor],
    risks: Sequence[torch.Tensor],
    *,
    threshold: float,
) -> SelectivePoint
```

`coverage` 是所有像素中被信任的比例；`roi_max_risk` 是逐 ROI 可信区域最大风险（空区域
记 0）后在 ROI 间取平均。两者只用于经验诊断，不重新解释为理论保证。

## 5. CPU 冒烟协议

新增 `trustsr-conformal-smoke`，内部使用固定、无随机数的四波段小张量构造 3 个校准
ROI 和 2 个测试 ROI。CLI 输出 canonical JSON，至少包含：

- schema 名称与版本；
- `synthetic_smoke: true`；
- `alpha`、`threshold`、`risk_bound`、`calibration_size`；
- 校准和测试 `coverage`、`roi_max_risk`；
- 输入配置 `channels=4`、`scale=4`、`window`；
- 不包含时间戳、绝对路径、设备随机信息或耗时。

内部全拒绝阈值保持为 `-inf`；在 `trustsr.conformal-smoke.v1` JSON 载荷中，
`calibration.threshold` 使用 `null` 表示该哨兵值，以保持严格 JSON 兼容性。

相同命令连续运行两次必须字节一致。

## 6. 通过与停止条件

### 6.1 通过条件

- 全部新增单元测试和现有 307 个测试通过；
- Ruff 通过；
- 手算小例验证阈值选择、`+B/(n+1)` 校正和全拒绝分支；
- 打乱同一 ROI 内像素顺序不改变阈值，拆分 ROI 会改变统计样本量并被测试记录；
- 风险与分数严格正相关的构造中，降低覆盖率不会增加经验 ROI 最大风险；
- CLI 两次输出字节一致并标注合成冒烟。

### 6.2 停止条件

- 若无法在手算例上复现阈值公式，停止，不接真实数据；
- 若风险—覆盖率方向测试失败，先诊断分数/掩膜语义，不添加复合风险；
- 若必须修改 SR 模型接口才能完成本阶段，停止并重新审查边界；
- 不因目标 `alpha` 太小而放松有限样本修正，全拒绝是合法结果。

## 7. 后续阶段门槛

Phase 2A 通过后才撰写 Phase 2B 规格。Phase 2B 将只做 SEN2NAIPv2 小规模、按 ROI/地理
隔离的数据清单和真实模型缓存接入；GPU 仅在需要生成 LDSR 多次样本时启动。通用共形
基线在 Phase 2B 必须作为对照，后续遥感专用风险只有在风险—覆盖率或高风险排序上显著
优于该基线，才可能成为论文贡献。

## 8. 依据

- Adame, Csillag, Goedert, *Image Super-Resolution with Guarantees via
  Conformalized Generative Models*, NeurIPS 2025,
  https://arxiv.org/abs/2502.09664
- Donike et al., *Trustworthy Super-Resolution of Multispectral Sentinel-2 Imagery
  With Latent Diffusion*, JSTARS 2025,
  https://doi.org/10.1109/JSTARS.2025.3542220
- Aybar et al., *A Comprehensive Benchmark for Optical Remote Sensing Image
  Super-Resolution*, IEEE GRSL 2024,
  https://doi.org/10.1109/LGRS.2024.3401394
- Aybar et al., *A Radiometrically and Spatially Consistent Super-Resolution
  Framework for Sentinel-2*, Remote Sensing of Environment 2026,
  https://doi.org/10.1016/j.rse.2025.115222
