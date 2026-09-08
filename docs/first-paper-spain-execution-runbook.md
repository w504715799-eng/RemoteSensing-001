# Spain 外部执行入口与冻结前检查

2026-09-08。用户明确费用自行管理；价格和计费粒度不再是开发或冻结条件。
入口：`python -m scripts.paper.run_spain {draft,freeze,preflight,run,replay}`。
正式协议已完成一次真实Spain实验及独立缓存复算，见
[结果报告](reports/first-paper-spain-results.md)。下列命令用于理解复现流程，不是重复推理指令。

## 当前可用内容

[机器可读草案](../paper/protocols/spain-external-draft-v1.json) 固定：
两个 100 数据包及其完整 28/20 成员、完整七槽模型身份、当前 Python 实现树哈希、
五评分、R1/R9、固定阈值、描述性统计、结构诊断、失败及展示规则。
该草案 `status=draft`，运行版本已由原云端证据补齐，预算仍为 null；正式入口必定拒绝它。
主张边界和原始数据 nodata 未知项也在草案中，不能写成源数据语义已全部核实。

[正式协议](../paper/protocols/spain-external-frozen-v1.json) SHA-256：
`798200c5c8b2cd9102755202810e8161832064de686a81b41c02776ddaf85525`。
它使用 `budget.cost_management=user`，不要求小时价或货币字段；保留实测规划2100秒
及单次2700秒超时，并绑定原计时证据SHA。历史预算草案仅用作计时来源，其中待填
价格不是阻塞条件。旧显式金额协议格式仍可读取。

`freeze` 仅从本地固定证据与当前实现生成协议，不读取真实数据、加载模型或连接云端。
它校验生成文件并禁止覆盖不同内容；再次写入完全相同内容可安全返回。

```sh
python -m scripts.paper.run_spain freeze --output paper/protocols/spain-external-frozen-v1.json
```

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

## 执行检查

1. 原云端运行版本和CPU串联已核验，正式协议已绑定证据，不再补测或询价。
2. 输入限制已审阅：零值保留；有限选择波段范围 [0,32767]；超过 10000 饱和并计数，
   除以 10000；不宣称已核实源数据 nodata。嵌入成员／实际 pickle 重建兼容性只在
   正式受控加载中核验；不兼容则停止，不能回退普通 pickle。
3. 部署时核验实际代码与上述正式SHA一致；实现变化必须另行生成并审阅新版本。
4. 真实执行仍按已授予的数据和模型访问范围进行，费用不另设审批。

maximum_wall_seconds 是单次调用超时控制。state.json保存逐次推理和模型加载耗时，
runtime.json保存当前成功调用耗时；费用由用户管理，不据此推断供应商账单。

## 部署与执行命令模板

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

## 原云端CPU检查点补充

云端25项定向测试通过，合成串联两轮科学结果一致；全部运行版本已进入新版草案。
本地协议／CLI22项通过；预算建议35分钟、run单次45分钟上限仍待审阅，小时价待补，
不代表正式外部执行授权。旧草案SHA保留为上一个检查点的记录，当前SHA见handoff最新项。
