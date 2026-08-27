# Phase 1B：LDSR-S2 云 GPU 基线设计

**日期：** 2026-08-27  
**状态：** 用户已批准，待实施
**基础分支：** `feature/phase1-pretrained-baselines`  
**实施分支：** `feature/phase1b-ldsr-gpu`  
**上位路线：** `2026-08-26-trustworthy-sentinel2-sr-roadmap.md` 的阶段 1  

## 1. 目标

本阶段把 LDSR-S2 作为第三个黑盒 RGBN ×4 超分辨率基线接入 TrustSR，并在租用的单张 RTX 3090 上完成可复现推理。实施继续遵守“小步通过、失败即停”的原则：先验证环境，再运行单样本，再运行固定 9 样本；任何一步失败都不扩大规模。

阶段交付物：

- 一个遵守现有 `SRModel` 契约的 `LDSRS2X4` 适配器；
- 一个不依赖真实 GPU 的离线单元测试集；
- 一个记录固定模型、配置、权重、随机性和 GPU 环境的远程运行清单；
- 单个 SPOT v3 样本的两次强制重新推理一致性报告；
- 全部 9 个 SPOT v3 样本上双三次、SEN2SRLite 和 LDSR-S2 的公平比较；
- 从云端回传到本地并经 SHA-256 校验的预测缓存和结果 JSON。

阶段通过后仍不进入训练。多次扩散采样产生的不确定性图属于后续最小共形原型，不在本阶段批量执行。

## 2. 明确范围

### 本阶段做

- 输入固定为 Sentinel-2 L2A B04、B03、B02、B08，即 RGBN `[0,1]` 反射率；
- 输入严格为 `(4,128,128)`，输出严格为 `(4,512,512)`；
- 使用 LDSR-S2 官方 10 m 配置、100 个 DDIM 采样步骤和官方光谱校正默认行为；
- 固定随机种子并记录所有影响采样的参数；
- 在同一份 OpenSR-Test SPOT v3 九样本清单上比较三个模型；
- 记录单图耗时、峰值显存和远程运行环境，但把这些非确定字段放入独立运行报告，不写入确定性指标 JSON；
- 所有大文件只放在云端数据盘或本地忽略目录，不提交 Git。

### 本阶段不做

- 不训练或微调 LDSR-S2；
- 不运行完整的 10、15 或 25 次随机采样不确定性实验；
- 不处理完整 Sentinel-2 `.SAFE` 影像、拼接、重叠窗口或 GeoTIFF 地理参考；
- 不增加 `opensr-utils`；
- 不使用 NAIP、Spain Urban、Spain Crops、SEN2NAIP 或 SEN2NEON；
- 不以 SPOT 开发集结果宣称论文最终优势；
- 不合并阶段 0、阶段 1A 或阶段 1B 分支。

## 3. 上游版本与资产冻结

固定使用以下上游版本：

| 项目 | 固定值 |
|---|---|
| `opensr-model` | `1.1.1` |
| PyPI wheel SHA-256 | `6168336d800d24976751bba46dd6cb129906109608b8c6003354c89a7a5b72e0` |
| 上游 Git tag commit | `10f4c01cc8172586841ea9e78c6de9939da47337` |
| 配置文件 | `opensr_model/configs/config_10m.yaml` |
| 配置大小 | `1,487` bytes |
| 配置 SHA-256 | `ac76685d354bfec32e3e0641aef574bedd7d650402c97dbd0ade86304e69ca6f` |
| 检查点 | `opensr-ldsrs2_v1_0_0.ckpt` |
| 检查点 URL | `https://huggingface.co/simon-donike/RS-SR-LTDF/resolve/main/opensr-ldsrs2_v1_0_0.ckpt` |
| 检查点 SHA-256 | `e2621e3912eb7c14867c3d20c9029607ba941be8e166dc09621860fcac27dc3a` |
| 检查点大小 | `1,130,715,795` bytes（约 1.13 GB） |
| Python | `3.12.x` |
| 环境解析器 | 项目 `uv.lock`，`uv==0.12.5` |

版本依据来自 [ESAOpenSR/opensr-model v1.1.1](https://github.com/ESAOpenSR/opensr-model/tree/v1.1.1)、[PyPI opensr-model 1.1.1](https://pypi.org/project/opensr-model/1.1.1/) 和 [官方 LDSR-S2 checkpoint](https://huggingface.co/simon-donike/RS-SR-LTDF/blob/main/opensr-ldsrs2_v1_0_0.ckpt)。

官方配置固定包含：

- `ckpt_version: opensr-ldsrs2_v1_0_0.ckpt`；
- `sampling_steps: 100`；
- `sampling_eta: 0.95`；
- `sampling_temperature: 1.0`；
- `encode_conditioning: true`；
- `apply_normalization: false`。

项目把 `opensr-model==1.1.1` 放入可选 `gpu` 依赖，不让本地 CPU 默认环境承担 LDSR 依赖。锁文件必须包含 GPU extra 的完整解析结果。

## 4. 供应链与加载安全

上游 `.ckpt` 是 PyTorch pickle 容器，不能先加载再验证。适配器不得调用会自动下载并立即 `torch.load()` 的上游 `load_pretrained()`。

安全顺序固定为：

1. 从固定 HTTPS URL 下载到同目录唯一临时文件；
2. 下载过程中流式计算 SHA-256；
3. 校验文件大小和固定 SHA-256；
4. 原子替换为最终检查点路径；
5. 校验已安装包内 `config_10m.yaml` 的 SHA-256；
6. 使用 `torch.load(..., weights_only=True, map_location=device)` 读取已验证检查点；
7. 删除上游状态字典中名称包含 `loss` 的非推理权重；
8. 使用 `strict=True` 加载模型状态。

缺失、截断、哈希不一致、配置不一致、非预期 checkpoint 结构或严格权重加载失败均立即停止。任何失败都不能回退到未验证的上游自动下载逻辑。

下载函数必须限制最终路径位于用户指定的模型目录中，不接受来自配置文件的任意本地路径，不使用 shell 拼接 URL，不记录认证信息。

## 5. `LDSRS2X4` 模型契约

适配器继续实现现有结构协议：

```python
class SRModel(Protocol):
    name: str
    scale: int

    def predict(self, lr: torch.Tensor) -> torch.Tensor: ...
    def provenance(self) -> dict[str, JsonScalar]: ...
```

固定属性：

- `name = "ldsr-s2-x4"`；
- `scale = 4`；
- 构造配置包含 `device="cuda:0"`、`seed=3407`、`sampling_steps=100`、`sampling_eta=0.95`、`sampling_temperature=1.0` 和 `histogram_matching=True`；
- 真实构造必须拒绝 CPU 设备和不可用 CUDA，不静默回退；
- 离线测试允许注入假后端，不导入 `opensr_model`、不下载权重且不需要 CUDA。

输入必须是 CPU 或 CUDA 上的 `torch.float32`、形状 `(4,128,128)`、全部有限并处于 `[0,1]`。适配器产生批次 `(1,4,128,128)` 并移动到配置 GPU。

每次 `predict()` 在隔离的 RNG 上下文中设置 Python、NumPy、CPU PyTorch 和 CUDA 随机种子，并恢复调用前的 RNG 状态。CUDA 设置 `cudnn.benchmark=False`、`cudnn.deterministic=True`；运行清单记录 `torch.are_deterministic_algorithms_enabled()` 状态。不得污染调用者的全局随机流。

后端原始输出必须为 `(1,4,512,512)` 且全部有限。公共输出为无梯度、连续、CPU `float32` 的 `(4,512,512)`，并按现有反射率策略裁剪到 `[0,1]`。

provenance 使用 JSON 标量，至少记录：

- 模型名、缩放倍数、适配器 schema 版本；
- `opensr-model`、PyTorch、CUDA runtime 版本；
- checkpoint 名称、URL、大小和 SHA-256；
- 配置文件 SHA-256；
- seed、采样步数、eta、temperature、光谱校正和输出裁剪策略；
- 设备类型与 GPU 型号不进入模型 provenance，GPU 型号进入远程运行清单，避免同一算法因设备名称改变缓存键。

上述任何输出相关配置变化都必须改变预测缓存键。

## 6. 云端执行布局

云端使用持久数据盘，不占用 30 GB 系统盘：

```text
/root/rivermind-data/trustsr-phase1b/
├── repo/                  # Git 工作树
├── conda-env/             # Python 3.12 隔离环境
├── models/ldsr-s2/        # 1.13 GB 检查点
├── data/opensr/           # SPOT v3
├── artifacts/cache/       # 三模型预测缓存
└── artifacts/phase1b/     # 运行清单与结果
```

环境由 `conda` 创建 Python 3.12 前缀；在该前缀安装固定 `uv==0.12.5`，再从提交的 `uv.lock` 以 frozen 模式安装项目及 `gpu` extra。远端不得直接在 base Conda 环境安装项目依赖。

首次真实运行前记录：

- 主机 UTC 时间和 TrustSR Git 提交；
- GPU 名称、UUID、总显存、驱动版本和 compute capability；
- 容器 CPU/内存限制；
- Python、Conda、uv、PyTorch、CUDA runtime、CUDA toolkit 和 `opensr-model` 版本；
- `pip freeze`/环境锁摘要；
- 数据清单、配置、checkpoint 和代码提交 SHA-256。

运行清单不得包含 SSH 主机、端口、用户名、密码、私钥、GitHub 凭据或本地绝对路径。

## 7. 分段执行与关卡

### 关卡 1：本地离线实现

完成可选 GPU 依赖、下载验证、适配器、CLI 编排和离线假后端测试。本地 `uv sync --dev`、全量 pytest 和 Ruff 必须继续通过，且默认 CPU 环境不安装 `opensr-model`。

通过后才请求用户启动 GPU 实例并确认新的 SSH 连接参数。

### 关卡 2：远端环境冒烟

在数据盘创建隔离环境，验证锁文件安装、CUDA 可用、官方配置哈希和 checkpoint 下载哈希。只构造模型并报告静态显存，不加载 SPOT 全集。

失败时停止，不改变依赖版本碰碰运气；先在独立诊断分支记录兼容性问题。

### 关卡 3：单样本强制推理

固定使用 `spot-0000`、seed 3407 和 100 步：

1. 清除该样本当前 LDSR 缓存或使用隔离验收缓存；
2. 强制推理一次，记录输出摘要、耗时和峰值显存；
3. 恢复相同初始条件再次强制推理；
4. 首先要求两个输出 SHA-256 完全相同；若 CUDA 内核不能位级确定，则必须报告最大绝对差，并且只有 `max_abs_diff <= 1e-6` 才可继续；
5. 验证输出形状、有限性和 `[0,1]` 范围；
6. 从缓存重新计算指标并确认结果不触发推理。

单样本任一条件失败即停止，不运行其余 8 个样本。

### 关卡 4：九样本三模型基准

在阶段 1A 冻结的同一 SPOT v3 清单上运行：

- `bicubic-x4`；
- `sen2srlite-x4`；
- `ldsr-s2-x4`。

每个模型必须引用同一清单哈希。结果 JSON 沿用 `run`/`models` 结构，新增 LDSR 模型 provenance；不包含耗时、时间戳、绝对路径或缓存命中字段。连续两次 CLI 运行的结果 JSON 必须字节一致，第二次不得写入或修改预测缓存。

工程通过条件不要求 LDSR-S2 胜过其他模型。结果报告只描述指标差异，不把九个 SPOT 样本外推为论文结论。

### 关卡 5：产物回传与关机通知

回传到本地前对以下内容生成 SHA-256 清单：

- 环境与 GPU 运行清单；
- 单样本确定性报告；
- 三模型结果 JSON；
- 当前 LDSR 预测 safetensors/JSON sidecar。

本地接收后重新计算 SHA-256，只有全部匹配才认为回传成功。代码和小型文档通过 GitHub 分支管理；模型、数据和预测缓存通过 SSH 文件传输，不进入 Git。

产物回传并验证后明确通知用户可以在云平台控制台关机。SSH 内执行 `shutdown` 不代表云平台停止计费，因此项目不通过 SSH 自动关停实例。

## 8. SSH 与凭据策略

- 用户可在本地设计/编码阶段关闭 GPU 实例；实例重新启动后必须重新确认 SSH 地址、端口和认证方式；
- 获得用户批准后，为该实例建立独立 Ed25519 密钥，私钥只保存在本机 `~/.ssh` 且权限为 `0600`；
- 公钥只加入该实例的 `/root/.ssh/authorized_keys`；不修改其他主机条目；
- 密码不写入 shell 脚本、环境文件、Git、日志、实验清单或命令历史；
- 当前已在聊天中暴露的密码应在密钥验证后更换；
- 项目不保存云厂商控制台凭据，也不调用关机、释放实例或删除磁盘命令。

## 9. 失败与停止条件

出现以下任一情况立即停止当前关卡并报告：

- SSH 主机指纹变化且用户未确认；
- RTX 3090 不可见、可用显存不足 18 GB 或存在未知 GPU 进程；
- 锁定依赖与 CUDA 13/PyTorch 组合无法导入或运行；
- checkpoint/config/package 任一哈希不匹配；
- `weights_only=True` 无法安全加载固定 checkpoint；
- 单样本 OOM、非有限输出、形状错误或重复推理差异超过容差；
- SPOT 清单不再严格等于冻结的九样本；
- 本地与云端产物哈希不一致；
- 继续执行需要训练、额外付费服务或扩大数据范围。

停止后先诊断，不静默降低采样步数、修改 eta、关闭光谱校正、换 checkpoint 或缩小模型来制造“成功”。任何方法变化必须形成新的规格修订并由用户批准。

## 10. 完成定义

阶段 1B 只有同时满足以下条件才完成：

1. 本地默认 CPU 环境测试和 Ruff 通过；
2. GPU extra 在固定 Conda/uv 环境中可从锁文件重建；
3. 所有上游代码、配置和模型资产有固定版本与 SHA-256；
4. 单样本两次强制推理满足确定性门槛；
5. 九样本三模型结果可从预测缓存重建且连续运行字节一致；
6. 云端运行清单完整且不含凭据；
7. 回传产物 SHA-256 在本地复核通过；
8. 分支经过任务级和整分支代码审查；
9. 用户收到明确的云 GPU 可关机通知。
