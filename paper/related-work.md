# Related work and positioning

2026-09-07 核对；本研究定位为实证比较，不主张发明新的共形理论。

- [Adame et al., arXiv v3](https://arxiv.org/html/2502.09664v3)：讨论对生成式
  超分辨率的分数掩码校准及邻域评分。本文借鉴其二阶矩思想，适配 RGBN 反射率；
  不冒称复现其 Lab、模型、数据或全部理论条件。
- [Conformal Risk Control](https://research.google/pubs/conformal-risk-control/)：
  期望损失控制与有限样本置信上界不是同一命题；本文区分校准准则与终局证据。
- [SEN2NAIP](https://www.nature.com/articles/s41597-024-04214-y)：跨传感器
  协调参考适合开展超分辨率评估，但不能被描述为无误差真实地表。
- [OpenSR-Test 官方文档](https://esaopensr.github.io/opensr-test/)：区分一致性、
  合成与正确性诊断。本文不会用单一 L1 风险替代所有结构诊断。
- [OpenSR-Test 数据集](https://huggingface.co/datasets/isp-uv-es/opensr-test)：
  外部版本和来源必须绑定不可变提交，不能引用 mutable main 作为实验身份。

最接近方法的代码核对以作者提交
`819086a9e21077010bdcee233373a25c8d32bdd0` 为准。其
[generate_masks.py](https://github.com/adamesalles/experiments-conformal-superres/blob/819086a9e21077010bdcee233373a25c8d32bdd0/src/masks/generate_masks.py)
中的 `calculate_mask` 在一次迭代时先计算每个样本的空间方差，再跨样本平均，
与联合采样／邻域二阶矩并不相同；主入口另存简单像素方差。
同提交的 col、div2k 版本有相同函数结构。这是被检查函数的差异，不是对作者全部
实验实现的断言。本文必须明确写作“paper-formula RGBN adaptation”。
