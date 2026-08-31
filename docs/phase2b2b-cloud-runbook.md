# Phase 2B2-B 云端分阶段验收手册

## 当前状态与结论边界

Phase 2B2-B 只对四个冻结的 `development` 样本进行三模型工程冒烟。它验证真实 crosssensor 输入、预测缓存和无推理回放，不是论文主实验，不能用于模型优越性、统计显著性或共形风险保证结论。

Phase 2B2-A 的输入审计包含 calibration、development 和 internal_test 三个 split；本阶段会先验证完整审计，再只加载 development 的四个样本。不得查看或计算 calibration/internal_test 的 HR 指标。

## 1. 运行前只读检查

在云实例的已审核仓库 checkout 中设置变量。所有值仅保留在当前 shell，不写入仓库：

```bash
: "${PHASE2B2B_STORAGE_ROOT:?set this to the persistent filesystem mountpoint}"
: "${PHASE2B2B_REPOSITORY:?set this to the reviewed repository checkout}"
: "${PHASE2B2B_SEN2SRLITE_DIR:?set the verified SEN2SRLite model directory}"
: "${PHASE2B2B_LDSR_DIR:?set the verified LDSR-S2 model directory}"

PHASE2B2B_POST_MANIFEST_SHA256=c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a
PHASE2B2B_INPUT_AUDIT_SHA256=fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b
PHASE2B2B_POST_MANIFEST="${PHASE2B2B_STORAGE_ROOT%/}/trustsr/phase2b1b/selections/${PHASE2B2B_POST_MANIFEST_SHA256}/samples.jsonl"
PHASE2B2B_INPUT_AUDIT="${PHASE2B2B_STORAGE_ROOT%/}/trustsr/phase2b2a/input-audits/${PHASE2B2B_POST_MANIFEST_SHA256}/phase2b2a-input-audit.json"
```

执行只读检查：

```bash
mountpoint -q -- "$PHASE2B2B_STORAGE_ROOT"
df -h -- "$PHASE2B2B_STORAGE_ROOT"
df -ih -- "$PHASE2B2B_STORAGE_ROOT"
git -C "$PHASE2B2B_REPOSITORY" status --short --branch
git -C "$PHASE2B2B_REPOSITORY" rev-parse HEAD
test -f "$PHASE2B2B_POST_MANIFEST"
test -f "$PHASE2B2B_INPUT_AUDIT"
test -d "$PHASE2B2B_SEN2SRLITE_DIR"
test -d "$PHASE2B2B_LDSR_DIR"
/opt/conda/bin/python -c 'import torch, rasterio, opensr_test, safetensors; print(torch.cuda.is_available())'
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits
```

要求长期存储为真实 mount、空间严格大于 8 GiB、空闲 inode 严格大于 1024、仓库 commit 与已审核分支一致、base Python 可导入现有依赖、CUDA 可用且没有外来计算进程。不要创建 Conda 环境，也不要重新安装 PyTorch。

## 2. 公共参数

```bash
phase2b2b_run() {
  local stage="$1"
  scripts/phase2b2b/run_cloud.sh \
    "$PHASE2B2B_STORAGE_ROOT" "$PHASE2B2B_REPOSITORY" "$stage" \
    --selection-manifest "$PHASE2B2B_POST_MANIFEST" \
    --selection-manifest-sha256 "$PHASE2B2B_POST_MANIFEST_SHA256" \
    --input-audit "$PHASE2B2B_INPUT_AUDIT" \
    --input-audit-sha256 "$PHASE2B2B_INPUT_AUDIT_SHA256" \
    --sen2srlite-model-dir "$PHASE2B2B_SEN2SRLITE_DIR" \
    --ldsr-model-dir "$PHASE2B2B_LDSR_DIR" \
    --project-root "$PHASE2B2B_REPOSITORY" \
    --confirm-cloud-storage
}
```

runner 只调用云镜像的 `/opt/conda/bin/python`，不会创建环境、下载 crosssensor
数据或安装软件。既有模型适配器可能在模型目录缺少资产时获取固定模型文件，
但只能在下载后通过冻结的大小和 SHA-256 校验；轻量依赖缺失时应停止并修复
base 环境，不得在 runner 内临时升级 PyTorch 或其他依赖。

## 3. Preflight

```bash
phase2b2b_run preflight
```

确认 stdout 为单个 JSON 对象，并检查长期盘中的 `preflight-runtime.json`。此阶段应验证固定 manifest、Phase 2B2-A input audit、GPU 门槛和两个学习模型的资产/provenance，不得加载 LR/HR 像素或生成预测缓存。

任何摘要、模型资产、CUDA、空闲显存或外来进程错误都在这里停止。

## 4. 单样本闸门

```bash
phase2b2b_run single
```

该阶段只加载 development correlation bin 0 的首样本，并按固定顺序运行 bicubic、SEN2SRLite、LDSR-S2。验收：

- 三个预测均为 CPU contiguous `float32`、形状 `4×512×512`、有限且位于 `[0,1]`；
- 三个 cache key 都绑定模型 provenance、post-manifest、input-audit 和 LR tensor SHA-256；
- 写后重读张量逐元素相同；
- 21 个样本级 OpenSR 指标值全部有限；
- `single-runtime.json` 的耗时和峰值显存可接受。

发生 OOM、非法输出、非有限指标、缓存不一致或运行时间不可接受时停止。不要跳过 single 直接运行 smoke。

## 5. 四样本三模型冒烟

```bash
phase2b2b_run smoke
```

该阶段运行 development correlation bin 0–3 的四个样本，复用 single 的三个有效缓存，只补齐缺失项。验收：

- 恰好四个 development 样本、三个模型、12 个唯一 cache key；
- 84 个样本级指标和 21 个均值全部有限；
- `development-three-model-smoke.json` 与 `development-cache-audit.json` 不含主机、绝对路径、GPU 型号、运行时间或像素；
- 没有 calibration/internal_test 的样本、预测或指标。

学习模型不需要在这四个样本上胜过 bicubic；这是工程门槛，不是排名门槛。

## 6. 关闭模型进程并进行 CPU 回放

确认没有遗留模型进程后运行：

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits
phase2b2b_run replay
```

`replay` 不构造模型、不调用 `predict`，只读取四个 development 输入、12 个预测缓存和已提交结果。它必须报告 `byte_identical=true`，并证明缓存文件集合、大小、mtime_ns 和 SHA-256 在回放前后未变化。

回放可以在 GPU 计算空闲后执行；若云平台允许保留 CPU 实例而释放 GPU，可先释放 GPU 再回放。

## 7. Git-safe 证据复制

只有以下两个确定性 JSON 可以进入 Git：

```text
development-three-model-smoke.json
development-cache-audit.json
```

复制前后分别计算 SHA-256 并比较，随后扫描绝对路径、主机、SSH、凭据、GPU 型号、时间戳和 runtime 字段。不得复制 `.safetensors`、cache sidecar、模型权重、GeoTIFF、完整 360 行 manifest、runtime、日志或 SSH 配置。

## 8. 停机条件

完成以下检查后，才通知用户可以暂停 GPU：

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits
```

要求无计算 PID，四阶段均成功，两个 Git-safe 文件的远端/本地摘要一致，且本地完整测试通过。云服务器的暂停或关机由用户在云平台控制台完成，仓库脚本不执行关机操作。

本阶段通过后先按实测单样本耗时和缓存大小复盘 Phase 2B3 成本，不自动启动 360 对推理。
