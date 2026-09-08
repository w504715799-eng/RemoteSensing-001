# Related work and positioning

2026-09-08 按作者原文、官方软件与出版页面核对；本研究定位为实证比较，不主张发明新的共形理论。

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

## 与本文的具体区别

Adame 等的研究将生成式超分辨率掩码与共形风险控制结合，本文借鉴其邻域二阶矩
思想，但使用四波段反射率及固定 Sentinel-2 中心重建。其邻域误差先分别平滑图像
再计算差异；本文 R9 先取 RGBN 绝对误差均值再平滑，两者不能视为同一损失。
因此本文既不是其整套实验的复现，也不沿用其 Lab 数值上限。
[原文第 2 节](https://arxiv.org/html/2502.09664v3) 是方法归属依据。

LDSR-S2 是本文的固定生成式中心和随机样本来源；SEN2SRLite 与 bicubic 提供异构
分歧对照，未比较不同中心重建的最优图像质量。LDSR-S2 的作者引用信息来自
[官方模型仓库](https://github.com/ESAOpenSR/opensr-model#52--citation)。
SEN2SR 的模型背景见其[正式论文](https://doi.org/10.1016/j.rse.2025.115222)，
具体 Lite 权重身份仍以本研究冻结协议为准，不能由论文名称推定。

SEN2NAIP 和 OpenSR-Test 分别提供跨传感器数据与结构诊断背景。本文把参考误差
排序和辅助结构分量分开，既不以 R9 替代所有正确性诊断，也不把 softmin 平均值
解释为硬分类幻觉比例。数据论文不是本次 v2 包身份的证明；实际外部身份绑定
[不可变数据提交](https://huggingface.co/datasets/isp-uv-es/opensr-test/tree/e4600b9c74a621adeec047e5f6cc7a2d70a58134)
及已发布包哈希。

## 已核验参考文献

1. Adame, E., Csillag, D., and Goedert, G. T. (2025). *Image Super-Resolution with
   Guarantees via Conformalized Generative Models*. arXiv:2502.09664v3.
   [作者原文](https://arxiv.org/html/2502.09664v3)。
2. Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., and Schuster, T. (2024).
   *Conformal Risk Control*. ICLR.
   [作者机构出版记录](https://research.google/pubs/conformal-risk-control/)。
3. Aybar, C., Montero, D., Contreras, J., Donike, S., Kalaitzis, F., and
   Gómez-Chova, L. (2024). *SEN2NAIP: A large-scale dataset for Sentinel-2 Image
   Super-Resolution*. Scientific Data, 11, 1389.
   [出版原文](https://doi.org/10.1038/s41597-024-04214-y)。
4. Aybar, C., Montero, D., Donike, S., Kalaitzis, F., and Gómez-Chova, L. (2024).
   *A Comprehensive Benchmark for Optical Remote Sensing Image Super-Resolution*.
   IEEE Geoscience and Remote Sensing Letters, 21, 1–5.
   [DOI](https://doi.org/10.1109/LGRS.2024.3401394)；
   [官方引用条目](https://esaopensr.github.io/opensr-test/)。
5. Donike, S., Aybar, C., Gómez-Chova, L., and Kalaitzis, F. (2025).
   *Trustworthy Super-Resolution of Multispectral Sentinel-2 Imagery With Latent Diffusion*.
   IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing,
   18, 6940–6952. [DOI](https://doi.org/10.1109/JSTARS.2025.3542220)。
6. Aybar, C., Contreras, J., Donike, S., Portalés-Julià, E., Mateo-García, G., and
   Gómez-Chova, L. (2026). *A radiometrically and spatially consistent super-resolution
   framework for Sentinel-2*. Remote Sensing of Environment, 334, 115222.
   [出版原文](https://doi.org/10.1016/j.rse.2025.115222)。卷期年份为 2026，不能误用 DOI 中的 2025。

以上覆盖本研究直接使用的方法、模型、数据和指标，不声称是系统综述或穷尽所有
最新超分辨率方法。当前在线文档仅用于文献核对，未升级冻结的实验依赖。
