# 当前可复现部分

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

外部复现记录尚缺：最终数据合同、成员映射、推理预算、冻结协议及运行结果。
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
