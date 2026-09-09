# im2im-UQ 补充基线：初步源码可行性核对

2026-09-09；只读作者仓库，没有运行其代码、安装依赖或读取实验图像。
正式论文：[ICML 2022](https://proceedings.mlr.press/v162/angelopoulos22a.html)。
作者仓库：[aangelopoulos/im2im-uq](https://github.com/aangelopoulos/im2im-uq)，本次浅克隆HEAD
`92124f5f5ac6954eb66f03e20735d8f12f47b797`。这是取回版本，不声称会议原始artifact版本。

## 已核验接口

- `core/models/finallayers/quantile_layer.py`：三个卷积头输出下分位数、中心和上分位数；
  输出形状为 `(N,3,C,H,W)`，损失是两项pinball加中心MSE。
- 同文件的nested-set函数以中心为基准扩张上下界，并用`1e-6`修正交叉；原函数会
  原地改写输出。适配实现必须说明这一行为并测试，不能把负区间宽度直接用于评分。
- `core/models/add_uncertainty.py`：wrapper包含主干、最后一层和校准参数`lhat`；
  支持quantiles、Gaussian、residual-magnitude等多种路线。本研究优先核验quantiles，
  不将整个仓库所有路线全部堆入主比较。
- `experiments/temca_test/config.yml`：示例为单通道UNet、×4、q=.05/.95、10epochs、
  学习率候选1e-4/1e-3、区间fraction-missed风险。不能将这些医学影像设定直接认为
  是已经验证的四通道卫星参数或直接使用其训练权重。
- `core/calibration/calibrate_model.py`：fraction-missed基于区间未包含参考值的比例，
  与本项目R9保留区最大误差不同。正式适配前仍需核验完整校准主路径及界实现，
  当前初审不等于端到端复现认证。

上述文件可在[固定提交](https://github.com/aangelopoulos/im2im-uq/tree/92124f5f5ac6954eb66f03e20735d8f12f47b797)查看。

## 实施决定与边界

保留该方法作为优先独立基线，新增训练角色和训练成本预算。原生区间实验与固定
LDSR中心的分位数评分适配分开命名、分开解释。前者可验证完整学习系统，后者用于
同中心的评分比较，但必须披露残差条件化和共同CRC的改动，不能继承原区间保证。
训练只使用公开来源且隔离于正式校准/测试的数据；wandb示例不是必须开通外部账号，
本地日志可完成实验记录，不启动在线sweep或上传数据。

目前没有新增GPU实验结果、适配权重或通过的跨框架数值测试。具体训练结构/预算与
独立数据清单绑定后冻结，不能先运行测试再选择一个较弱版本。
