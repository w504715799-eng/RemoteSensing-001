# Spain 外部执行入口与冻结前检查

2026-09-08。当前只有草案和本地验证，不是外部像素／GPU 执行授权。
入口：`python -m scripts.paper.run_spain {draft,preflight,run,replay}`。

## 当前可用内容

[机器可读草案](../paper/protocols/spain-external-draft-v1.json) 固定：
两个 100 数据包及其完整 28/20 成员、完整七槽模型身份、当前 Python 实现树哈希、
五评分、R1/R9、固定阈值、描述性统计、结构诊断、失败及展示规则。
该草案 `status=draft`，未知运行版本为 null，预算为 null；正式入口必定拒绝它。
主张边界和原始数据 nodata 未知项也在草案中，不能写成源数据语义已全部核实。

当前版本的验证和模型规则：

- 确认实际导入 `trustsr` 来自本仓库 `src/trustsr`，再核对协议 SHA、canonical 字节、
  科学合同和完整实现树；旧安装包不能冒用新仓库哈希。
- 预期 SHA 必须来自外部访问前独立审阅的最终协议，不从待运行文件临时自算当成授权。
- CPU 硬件、版本和 backend 与已有云端核验一致；LDSR 为原 RTX 4090／CUDA 配置。
- 模型只接收 LR；真实资产已经存在并经过旧适配器校验，不自动下载数据或缺失模型。
- 输出目录必须位于仓库、输入和模型目录之外，父目录须已存在；不允许旧阶段目录、
  软链接或无本研究标记的旧目录。缓存固定为新运行目录内 `cache/`。
- 所有现有缓存先完整核验，再按需加载模型；每次推理前把 started 状态同步到文件和
  目录。重启不自动重试失败／中断的槽；中断留下的完整有效缓存可直接恢复。
- 输入、资产、provenance 或缓存损坏终止整批；预测运行／输出合同失败保留对应方法
  的失败分母及静态原因。不得以模型效果筛样或换配置。

## 冻结前仍需补齐

1. 原云端环境的 numpy、pandas、scipy、safetensors、satalign 精确版本；固定 torch 等
   已知版本不能变。只需文本环境盘点，不读 Spain 或历史 B/C。
2. 原云端执行 `python -m scripts.paper.benchmark_external_cpu` 的合成 CPU 串联；
   结合现有推理测速、包下载／输入、加载／缓存开销与供应商小时价，形成预算证据。
   不重跑已完成 LDSR 测速、开发选择或 SEN2SRLite 诊断。
3. 审阅输入限制：零值保留；有限选择波段范围 [0,32767]；超过 10000 饱和并计数，
   除以 10000；不宣称已核实源数据 nodata。嵌入成员／实际 pickle 重建兼容性只在
   冻结授权后的受控加载中核验；不兼容则停止，不能回退普通 pickle。
4. 更新草案运行清单和完整预算，保持科学合同不变，生成单独的最终冻结文件并提交。
   核验代码哈希与审阅版本一致。冻结记录须包含实际预算证据 SHA；不能填占位值。
5. 展示确切数据／费用／执行范围，取得仍未授予的外部访问与必要 GPU 执行授权。

预算中的 maximum_wall_seconds 是单次调用硬上限；重启不自动获得额外付费额度。
state.json 保存成功和失败推理的耗时及模型工厂耗时，runtime.json 只描述当前成功调用
的输入／预测阶段／两轮复算耗时。它们不是供应商计费凭证，不能漏计失败启动／人工
等待／下载等时段后宣称实际费用。运行外层应另外保留整段墙钟和计费记录。

## 冻结后的命令模板（现在不要执行 run/replay）

在已经固定的现有云端环境内运行，不新建或更新环境。变量由执行时的已核验路径提供，
不把地址、凭据、原始影像或模型传回 Git。

```sh
python -m scripts.paper.run_spain preflight \
  --protocol "$TRUSTSR_EXTERNAL_PROTOCOL" \
  --protocol-sha256 "$TRUSTSR_REVIEWED_PROTOCOL_SHA"
```

preflight 只验证协议、代码、运行环境，不打开包／缓存／模型；草案将失败。
正式数据包提前按固定提交／路径取得并保留在独立数据目录，入口只认确切文件名及哈希。

```sh
python -m scripts.paper.run_spain run \
  --protocol "$TRUSTSR_EXTERNAL_PROTOCOL" \
  --protocol-sha256 "$TRUSTSR_REVIEWED_PROTOCOL_SHA" \
  --confirm-external-access \
  --package-directory "$TRUSTSR_EXTERNAL_PACKAGES" \
  --output "$TRUSTSR_EXTERNAL_RUN" \
  --allow-gpu \
  --ldsr-model-directory "$TRUSTSR_LDSR_MODELS" \
  --sen2sr-model-directory "$TRUSTSR_SEN2SR_MODELS"
```

完整缓存时可省略 GPU 标志和模型目录。存在缺失 LDSR 且没有 GPU 范围时，在模型加载前
停止；这不意味着可以先未经授权读取外部包。参数标志不能替代实际用户授权。
模型加载／输入／缓存身份问题须记录偏离后处理，不能看到结果再换版本。

```sh
python -m scripts.paper.run_spain replay \
  --protocol "$TRUSTSR_EXTERNAL_PROTOCOL" \
  --protocol-sha256 "$TRUSTSR_REVIEWED_PROTOCOL_SHA" \
  --confirm-external-access \
  --package-directory "$TRUSTSR_EXTERNAL_PACKAGES" \
  --output "$TRUSTSR_EXTERNAL_RUN"
```

replay 不接受 GPU 或模型路径，不补预测，不改已发布科学／状态文件，只重建并比较。
它仍会读取已授权的新外部包／缓存，不允许用来访问旧阶段。

## 产物与停止语义

- `run.json`：绑定协议的运行标记；`.lock`：单写入者锁。
- `state.json`：输入／模型绑定及逐预测尝试记录；`cache/`：独立的 safetensors 缓存。
- `science.json`：两个子集完整结果、输入处理回执、逐预测失败原因与所有分母。
- `verification.json`：第二轮无推理复算一致性；不声称独立验证了模型原始推理。
- `manifest.json`：最后原子发布，绑定科学／验证／执行状态摘要；缺它表示发布未完成。
- `runtime.json`：当前成功调用的开销，独立于确定性科学结果。

仅在两个子集完成两轮字节一致复算后整体发布，不先挑一个子集发布。不覆盖不同旧结果；
中断的相同部分文件允许完成原子发布。完整运行不能继续生成新预测。
这些文件先留在云端独立运行目录；后续按论文图表需求审阅精简科学文件后再发布 Git，
不会把运行目录或旧 B/C 状态自动复制进仓库。

## 本地验证检查点

本轮新增协议／执行／发布／CLI 定向测试36项，连同原评分及复算回归共70项通过
（18.98秒）；数据政策7项通过（0.97秒）。没有跑全量测试。合成集成覆盖真实OpenSR、
SEN2SRLite槽失败但主比较有效、两次重建、已完成恢复及只读复算；测试不加载真实模型。
只读审查指出的目录同步与旧安装包风险已经用先失败后通过的测试修复。
当前草案SHA-256为 `d0c6fa3e2672b03fe7af07f90bc6a7c6a625e6afae31d45a4dfbe9a7afa98e14`，
与当前实现树一致；此摘要不是最终外部执行的授权摘要。
