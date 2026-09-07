# OpenSR-Test 1.3.3 指标合同核验

2026-09-07，本地源码及合成张量检查，无 Spain 图像或云端访问。
官方源码参考提交 `b42b1cba8a04b32341044f1f29474e5448499158`，
[main.py](https://github.com/ESAOpenSR/opensr-test/blob/b42b1cba8a04b32341044f1f29474e5448499158/opensr_test/main.py)、
[correctness.py](https://github.com/ESAOpenSR/opensr-test/blob/b42b1cba8a04b32341044f1f29474e5448499158/opensr_test/correctness.py)。
实际行为以本地锁定安装版本为准，不直接升级依赖。

## 已核实语义

- 默认 `border_mask=16` 是真实裁剪，不是给边缘加零：HR/SR 每侧裁 16，×4 的 LR 每侧裁 4。
  64×64 合成 HR 的指标网格为 32×32；原始像素坐标不可直接复用。
- 默认先对 SR 做相对 HR 的光谱协调和空间配准。这个参考辅助过程是评估的一部分，
  不能放入不确定性分数生成，也不能把已配准地图与原始分数直接逐像素对应。
- `gradient_threshold='auto'` 实际按 `d_ref` 的约 75% 顺序统计量构造阈值，随后联合
  d_ref/d_im/d_om 筛选有效点；不是一个所有 ROI 共用的梯度物理阈值。
- `ha_metric`／`om_metric`／`im_metric` 默认是 softmin 三分量在有效点上的均值。
  合成三距离相同时三者均为 1/3；硬分类 `percent` 同分按 argmax 得 improvement=1，
  其他为 0。两种定义不能混称“像素比例”。
- 无有效像素时保留 NaN／不可用状态，不能改写为 0 并声称无幻觉。

## 验证与限制

`tests/paper/test_opensr_contract.py` 三个实际上游合同测试覆盖裁边、软硬定义和全缺失情形。
另在固定随机合成张量上调用真实 `correctness`，得到 32×32 网格及有限子集，三分量
总和近 1；出现配准平移过大的上游警告，因此不把该冒烟说成真实场景下已稳定。
pytest 中另有既有 PyTorch JIT 弃用警告；没有隐藏或修改旧适配器来消除警告。

当前外部草案只保留图像级探索性诊断方向，不做像素级幻觉排序结论。
冻结前仍需明确配准失败／警告、有效分母、全无效 ROI 的处理，并在新入口实现；
不以看过 Spain 指标后删掉不利诊断。原 A/B/C 模块及既有七标量适配器保持不变。
