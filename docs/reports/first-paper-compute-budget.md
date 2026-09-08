# 最新实测完成状态（2026-09-08）

用户明确费用自行管理，价格不再是前置条件。正式48 ROI／336预测研究已完成：
单次进程1234.032秒，含两轮科学重建；另行只读缓存核验378.192秒通过。
详见[外部结果](first-paper-spain-results.md)。下方保留历史规划依据，不再要求补价格或复测。

> 2026-09-08 最新补充：原云端合成CPU测量已完成，完整环境清单已核验。
> 48 ROI推理＋两轮CPU复算＋缓存代理名义约18.70分钟；规划余量另列，小时价待补。
> 详见 [云端CPU与预算草案](first-paper-cloud-cpu-budget.md)。下方保留历史测速规格，不是重跑指令。

# 第一篇论文计算预算（推理已测，完整流水线仍待核算）

2026-09-08 更新：限定 3 个 development ROI 的测速已完成，详见
[测速报告](first-paper-inference-timing.md)。RTX 4090 上 LDSR 均值 1.615943 秒／次；
SEN2SRLite 的旧 1 线程 CPU 为 0.209878 秒／次，bicubic CPU 为 0.006185 秒／次。
后续已定位线程数差异并采用云端 96 线程兼容策略，3/3 样本复现历史输出。
新策略三次调用均值 0.888265 秒（含首次冷调用）；48 ROI 推理部分更新为约
**7.179 分钟**，取代旧 1 线程的 6.637 分钟投影。不含解包、完整评分、缓存写盘、
加载等开销，也不等于整机计费时长或稳定吞吐保证。详见
[CPU 兼容核查](first-paper-sen2sr-reproducibility.md)。完整外部预算仍未完成。

下面保留已执行测速的访问前规格及预算公式，不是再次运行的指令。

2026-09-07。本地 CPU 已实现邻域评分和已发布证据渲染，不需要 GPU。
用户提供云端连接并授权执行后，已复用 development 的 600 个 K5 预测，未新增
真实图像推理；未读取 B/C 原始缓存。本地不保存原始数据，云端结果已持久化。

完整首轮 CPU 核验与计算 264.0334 秒，全量复算 265.3517 秒。纯评分单 ROI 均值：
3×3 为 0.269405 秒，9×9 为 0.352836 秒；各测 120 次，单线程。
这些计时不含原 K5 模型推理，不能替代下面的外部 GPU 预算。

## 已执行资源请求的原定范围

本次 development 身份和缓存已确认完整，评分与复算完成；服务器可关机，保留数据盘。
下次云端请求应服务于最终冻结前所需的 SPOT／development 推理成本测量或授权后的
外部评估，不重复本次无缺失的 K5 生成，不自动重新运行旧阶段。

## 外部预算公式

名义 48 ROI：240 次 LDSR + 48 次 SEN2SRLite + bicubic／CPU 后处理；实际成员
须待数据合同确认。邻域与 K5 共享预测，不重复计 240 次。
默认单 worker；不开展新的并行推理优化。

`LDSR inference wall-clock hours = 240*t_LDSR/3600`。

`serial instance hours = (240*t_LDSR + 48*t_SEN2SRLite_CPU + 48*t_bicubic_CPU
+ CPU_scoring_seconds + measured_load_and_IO_seconds)/3600`。

2026-09-08 复核 A 的固定预测审计：SEN2SRLite 的 `device` 为 `cpu`，LDSR 为
`cuda`。此前将两者均计为 GPU 推理时间的公式不准确。保持原设备身份，分别记录
LDSR 推理墙钟与租用整机的计费墙钟时间；`t_LDSR` 包含适配器 CPU 校验、RNG 隔离
及传输，不是 CUDA kernel 活跃时长，不能据此或 CPU 基线耗时推断 GPU 利用率。

需记录模型加载、预处理、推理、写盘、CPU 评分时间、峰值显存、缓存命中率和切块次数。
供应商、小时价、实例配置、时间上限尚未指定，因此当前不能给出可信费用或 GPU 时长。
SPOT／允许的 development 冒烟测量在最终外部冻结前完成，不拿 Spain 测速或调试。

## 限定测速入口规格（2026-09-08）

文件 `scripts/paper/benchmark_inference.py`；不是旧阶段的重跑入口，不创建预测缓存。
先用固定发布 A 结果的 SHA 和 post-manifest SHA 绑定 development 成员，再固定选择
ROI ID 字典序最前 3 项。仅核验及读取这 3 项 development 配对，校对归一化 tensor
哈希；HR 不参与推理、评分、选择或输出。拒绝旧实验树中的输出路径和已存在输出。
现有模型资产必须完整，禁止自动下载；加载仍用原适配器的哈希验证与固定配置。

运行矩阵：LDSR 第一个 ROI seed 3407 冷启动 1 次，之后 3 ROI × seeds 3407–3411
测量 15 次；SEN2SRLite CPU 冷启动 1 次 + 3 ROI 各 1 次；bicubic CPU 3 次。
冷启动单独列出并计入本次成本，不混入稳定态均值。单 worker，CPU 单线程。
每次记录同步后的墙钟秒数、输出 tensor SHA、CUDA allocated/reserved 峰值；模型加载
及输入准备时间另列。不输出误差指标，不保存预测，不改参数，失败即停止，不自动重试。

纯核心 `measure_prediction(model, lr, *, cuda_device=None)` 在 CPU 上用真实 bicubic
测试有限输出、输出形状与输入哈希绑定；硬件／样本选择逻辑另测。CLI 的 CUDA 检查
在任何数据或模型加载前执行，无 CUDA 时不创建输出。真实计时只能在云端完成。
输出新目录中的 runtime.json（含本次运行身份，不含绝对路径或凭据），运行外层使用
30 分钟硬超时限制，不承诺此上限足以成功。测完先汇报实测预算，不能自动启动 Spain。

本次请求只为补齐必要的硬件测量；安全解包、逐方法缺失汇总与协议冻结仍是独立 CPU
工作，不因有 GPU 而跳过，也不声称所有本地研究任务已经完成。

固定 3 个 development ID（由已发布 A 数值清单确定，未读影像）：

- `NA5120_E1193N0687__m_3612111_nw_10_060_20220519`
- `NA5120_E1207N0710__m_3812162_sw_10_060_20200617`
- `NA5120_E1209N0691__m_3712049_ne_10_060_20220614`

执行时使用已部署提交根目录和用户现有 base 解释器：

```sh
timeout --signal=TERM --kill-after=30s 30m python -m scripts.paper.benchmark_inference \
  --storage-root "$TRUSTSR_STORAGE_ROOT" \
  --ldsr-model-dir "$TRUSTSR_LDSR_MODELS" \
  --sen2sr-model-dir "$TRUSTSR_SEN2SR_MODELS" \
  --output-directory "$TRUSTSR_NEW_TIMING_OUTPUT"
```

变量由运行时经只读核验的绝对路径提供，不在 Git 中保存环境、凭据或服务器信息。
输出每次调用后更新，`status=running` 表示未完成，不能用于完整预算；仅末尾正常退出
且 `status=complete` 才能计算正式测量摘要。缓存写盘和外部解包开销不在本入口测量中，
必须另列，不能把 inference-only 时间报告为端到端运行总时间。

## 2026-09-08 本地完整合成串联

新增 `python -m scripts.paper.benchmark_external_cpu`，无需网络、数据路径或模型，
临时构造一个512网格样本，串联受限解码、七份合成预测缓存、五评分与隔离OpenSR。
两轮复算耗时4.185629／4.037788秒，科学JSON字节一致；详见
[集成报告](first-paper-external-local-integration.md)。本地PyTorch 2.13.0与云端冻结
2.12.1不同，因此不将此数直接混入正式成本。后续可在原云端环境只跑此合成CPU入口，
并补加载／下载／计费条件；无需再次运行真实GPU测速或SEN2SRLite诊断。
