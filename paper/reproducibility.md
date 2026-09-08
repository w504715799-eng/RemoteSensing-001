# 当前可复现部分

## Spain 正式运行（2026-09-08）

正式实现提交 `2c851de9121277c51f0c838d57cbf179817862ce`，协议文件
`paper/protocols/spain-external-frozen-v1.json`，SHA-256
`798200c5c8b2cd9102755202810e8161832064de686a81b41c02776ddaf85525`。
云端按本地Git archive部署，377个源码／文本文件逐一匹配；原环境preflight通过。
原云端直连下载不可达，固定来源的两包经本地并行下载、中转至新持久化目录，
两端大小和SHA全部一致。真实受限解包及28/20全成员绑定通过，没有放宽allowlist，
原始源nodata语义仍未知。336份预测全部生成，未读取旧B/C原始数据或缓存。

科学运行入口、环境及失败恢复规则见
[执行手册](../docs/first-paper-spain-execution-runbook.md)。科学结果由两轮缓存复算
一致后原子发布，核验的是缓存科学重建，不声称独立重做了模型推理。
费用由用户管理，不能把耗时证据当作供应商账单。

论文CSV导出工具位于 `paper/tools/render_spain.py`，在冻结实验实现树之外，
只读已发布的小型科学JSON；不接触像素、模型或缓存，不影响运行协议。
使用最终发布的science摘要作为 `--science-sha256`，固定协议摘要作为
`--protocol-sha256`，并提供 `--science`、`--protocol`、`--output paper/tables`。
SHA、协议和完整成员校验先于输出；不同旧文件或软链接被拒绝，相同结果可重复导出。
五张表分别保留逐方法分母、等ROI曲线、主配对差、阈值迁移和结构诊断支持像素数。
随机rho及缺失值为空，不以零代替失败；全拒绝的零损失同时保留all_rejected标记。

下方为历史开发与准备检查点，不再构成重复访问或重新询价的指令。

最终结果SHA：`f09fc42555cb00deac5125e478b61817200de9a796bd668e5abcc15fbfb07fe4`。
独立进程只读复算378.192秒通过，发布的五个科学／状态／运行文件均未改变。
完整回执在 `paper/tables/spain-execution-v1.json`；不含原始影像或模型资产。

```sh
.venv/bin/python paper/tools/render_spain.py \
  --science paper/tables/spain-science-v1.json \
  --science-sha256 f09fc42555cb00deac5125e478b61817200de9a796bd668e5abcc15fbfb07fe4 \
  --protocol paper/protocols/spain-external-frozen-v1.json \
  --protocol-sha256 798200c5c8b2cd9102755202810e8161832064de686a81b41c02776ddaf85525 \
  --output paper/tables
.venv/bin/python paper/tools/plot_spain.py
```

四ROI面板通过 `paper/tools/render_spain_panels.py DEPLOYMENT_DIRECTORY NEW_FIGURE_DIRECTORY`
在原云端已完成的独立部署中生成，目录含code/、packages/、run/。只读固定包及缓存，
不加载模型；先核验正式协议、科学SHA和逐输入／预测哈希，不自动修补缺失缓存。
显示范围与确切ROI写入panel回执；gamma和颜色截断仅用于展示。不得据图重新选方法。
本轮只运行3项CSV导出定向回归，绘图检查采用实际重建、摘要核对和目视检查。

2026-09-08 补充：SEN2SRLite CPU 线程差异已核查，新
`src/trustsr/models/cloud_sen2srlite.py` 在独立串行 CPU 路径固定 96 线程并恢复调用者设置，
不修改历史适配器。路径无关证据是 `paper/tables/sen2sr-cpu-diagnostic-v1.json` 与
`paper/tables/sen2sr-cloud-policy-check-v1.json`；canonical SHA、范围、实际耗时和
重新验证条件见 [核查报告](../docs/reports/first-paper-sen2sr-reproducibility.md)。
权重无关测试：`.venv/bin/python -m pytest tests/models/test_cloud_sen2srlite.py -q`。
真实三个开发样本已完成验证，不重跑旧阶段或借此开放 Spain／B/C 数据。

2026-09-07；覆盖已发布证据渲染、评分合成验证与 development 候选比较，不代表外部研究完成。

```bash
.venv/bin/python scripts/paper/render_evidence.py
.venv/bin/python -m pytest tests/risk tests/paper tests/data/test_local_data_policy.py -q
.venv/bin/ruff check scripts/paper/render_evidence.py src/trustsr/risk/neighborhood.py tests/paper tests/risk/test_neighborhood.py
```

使用当前项目环境及 matplotlib。渲染器固定三份 Git-safe JSON 的路径和 SHA-256，
核验全部输入后才写入 `paper/tables/` 与 `paper/figures/`；不访问网络、模型或缓存。
CSV 科学数值确定一致；PDF 不要求元数据字节一致。曲线只展示 development，
不能当作独立外部确认。邻域模块是单独的 RGBN 适配，未覆盖历史 K5 模块。

合成验证包括：0／1 空间常量混合的总体方差 0.24、全常数零分数、空间变化、
独立逐点反射边界与高斯 oracle、四波段平均和非法输入拒绝。
这只能证明这些实现合同，不证明真实数据上排序更好。

development 运行入口为 `scripts/paper/evaluate_neighborhood.py`，需要云端原固定数据根目录，
参数 `--storage-root` 与全新 `--output-directory`。部署提交 `defa6fa4857d17d173915c80ebe8fdf780732f8f`。
不要为复建本论文已有表格重跑原数据；数值证据已保存在
`tables/neighborhood-development-v1.json`。两次 CPU 运行、输入哈希与发布验收见执行报告。

历史准备时外部复现记录尚缺；最终数据合同、成员映射、冻结协议和运行结果现已在本文件开头提供。
Spain 元数据审计中的官方质量字段有限暴露必须随最终实验报告披露。

## 外部入口的本地合同检查（2026-09-08）

`spain_inputs.py` 只处理已解码数组与 metadata 行对应；`external_scores.py` 只接收
内存张量，生成四张分数图及随机解析期望。分数构建不接收 HR，诊断共享 seed 3407
中心预测，邻域配置沿用 development 选中的 w3。随机 Spearman 为 null，不伪造一次
随机图；固定 K5 阈值不重新拟合。主比较汇总使用 ROI 等权，不合并 Crops／Urban。
这些测试不是 Spain 结果，也没有认证任何真实外部预测。

```sh
.venv/bin/python -m pytest tests/data/test_spain_inputs.py tests/paper tests/risk tests/data/test_local_data_policy.py --override-ini addopts='' -q
.venv/bin/python -m scripts.paper.benchmark_inference --help
```

以下为历史准备状态：真实推理成本需另行申请短时 GPU，在固定 3 个 development 样本上测量；SEN2SRLite
维持 CPU。测速只记录同步预测墙钟和内存，不生成新的质量结论，不重跑旧阶段。
完整访问范围和硬超时见 [计算预算](../docs/reports/first-paper-compute-budget.md)。
