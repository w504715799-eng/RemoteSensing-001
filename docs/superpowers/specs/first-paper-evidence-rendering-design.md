# 已发布证据渲染器规格

2026-09-07。只读输入固定为 evidence-inventory 中 A、B、C 三份主结果及其 SHA-256。
不支持任意路径、云端访问、原始图像、B/C per-ROI 新计算；任一哈希不符时在写输出前失败。
入口 `scripts/paper/render_evidence.py`，不重构现有评价管线。

输出仅允许 `paper/tables/development_scores.csv`、`development_curves.csv`、
`calibration_evaluation.csv` 和 `paper/figures/development_risk_coverage.pdf`。
表列分别为：

- method, window, roi_count, mean_aurc, mean_rho；
- method, window, coverage, mean_selective_risk；
- role, roi_count, threshold, coverage, calibration_criterion, mean_roi_loss, risk_ucb, decision。

B 与 C 不适用的列留空，不填 0。确定排序为窗口 1,9、方法名升序、覆盖升序；
开发指标逐 ROI 等权平均；不把 candidate summary 的 gain 误当 AURC。
浮点 CSV 使用可往返表示。标准 matplotlib，无新基础框架；固定轴、图例与图注。
图的 PDF 元数据不要求哈希一致，但两次 CSV 字节一致且图数据与表一致。
所有旧 artifacts 字节必须不变。

实施步骤（尚未执行）：先针对手算两 ROI fixture 写均值／空列／同角色测试；
针对输入哈希错误写拒绝测试；实现最小只读解析与 CSV；再增加曲线绘制；
运行 pytest、Ruff、两次复建比较、历史 artifacts diff 和图形检查后提交。
