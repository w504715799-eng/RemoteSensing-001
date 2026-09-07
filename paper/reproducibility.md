# 当前可复现部分

2026-09-07；本说明只覆盖已发布证据渲染与新评分的合成验证，不代表外部研究完成。

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

外部复现记录尚缺：最终数据合同、成员映射、开发候选选择、预算、冻结协议及运行结果。
Spain 元数据审计中的官方质量字段有限暴露必须随最终实验报告披露。
