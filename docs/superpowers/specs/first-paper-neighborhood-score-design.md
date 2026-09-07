# 邻域二阶矩 RGBN 适配规格

2026-09-07；在任何新增 development 数据访问前定义。文献及固定作者代码版本见
[related work](../../../paper/related-work.md)。本方法称 paper-formula adaptation，
不称作者代码逐项复现，不改旧 K5 评分或任何 B/C 参数。

给定 K=5 个 RGBN 预测 Y，B_w 为反射填充的 w×w 均值：

`S = G_1(mean_band(mean_K(B_w(Y²)) - mean_K(B_w(Y))²))`。

G_1 是 sigma=1、半径 3、归一化的离散高斯核，反射边界。
先二阶矩后差，不能用简单平滑逐像素种子方差替代。
作者被检查函数一次迭代实现的是 mean_K(B(Y²)-B(Y)²)，
跨种子空间常量不同时会与上式不同。本文固定上式，不混用两者。

候选仅 w∈{3,9}，sigma 固定 1。只用允许的 120 development ROI 比较
等权 R9 AURC，较小者入选；差值绝对值 ≤1e-12 时选 w=3。
两候选结果都发布，不以 B/C 或外部结果选参。不调模型、种子或风险窗口。

拟定模块 `src/trustsr/risk/neighborhood.py`，函数
`neighborhood_variance_score(samples, *, window)`，输入 torch 浮点张量
(5,4,H,W)，有限 [0,1]；H,W 至少 max(window,4)。输出 CPU float64 (H,W)，
不修改输入、不读取 HR。中间仅截去浮点舍入导致的微小负方差，明显负值报错。
与独立直接求和 oracle 的最大绝对差容限 1e-12。

必须先写测试：全常数为零；五个空间常量 0,0,0,1,1 给 0.24；
五个完全相同但空间变化的样本产生正邻域分数，区别于像素种子方差；
四波段 reduction、边界反射与高斯用独立循环核验；非法 shape／NaN／范围拒绝。
评分生成无 HR 参数。HR 仅用于下游风险标签。

独立部署成本：LR 需要中心一次 LDSR；三模型需要中心 LDSR、SEN2SRLite、bicubic；
K5 和邻域各需五次 LDSR；随机解析参考的风险需要同一中心预测。
实验总成本只计共享 K5 加一次 SEN2SRLite 和 bicubic，不按评分数乘算。
