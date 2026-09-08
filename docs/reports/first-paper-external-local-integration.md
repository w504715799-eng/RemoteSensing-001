# 外部入口本地集成检查点

2026-09-08；只用合成数据，没有连接云端、下载 Spain 或访问旧 B/C。
本检查点推进 Task 5，不是最终冻结或正式外部实验结果。

## 已交付

1. `spain_package.py` + 受限子进程：大小／SHA 校验先于解码，拒绝未知 globals、
   extension/persistent 引用、尾随 pickle、错误 shape／对象数组／成员不匹配。
   复用已有 band／归一化处理，内嵌 metadata 按行位置绑定；只输出六个投影字段。
   pair.source 绑定包 SHA，避免不同来源包共用预测缓存身份。
   资源限制与 allowlist 是固定可信来源的防御层，不是任意恶意 pickle 的 OS 沙箱。
   尚未证实真实 Spain 包与允许的 NumPy/Pandas 重建类型兼容；未知类型应停止，
   不得在外部结果可见后自动扩展 allowlist 或回退普通 pickle。
2. `external_replay.py`：七个缓存槽绑定实际 LR／来源／ROI／模型 provenance；校验
   seed 3407–3411 的一致性及新的云端 96 线程策略。缓存损坏是致命错误，缺失有明确
   依赖分母。SEN2SRLite 缺失只阻止三模型；非中心 LDSR 缺失不删除 LR／随机／三模型；
   中心缺失则五评分均缺失。K5 阈值迁移、主配对和结构诊断各自保留分母。
   子集入口绑定模型配置 SHA，穷尽成员，输出确定性科学 JSON；不推理、不重试、不写缓存。
   SHA 的预期值和成员仍须由最终协议入口认证；自洽缓存不能证明模型实际执行。
3. `external_opensr.py` + CPU 子进程：显式固定 OpenSR 1.3.3 参数；拦截实际 SR→HR
   配准返回的 NaN（上游协调函数原本丢弃该值），保留失败。三分量共用有限支持，报告
   裁后网格和有效像素数；无效结果为 null，绝不填零。独立进程隔离上游 CUDA RNG／
   cuDNN 副作用。每 ROI 120 秒上限，无重试或换配准器；仅图像级探索性 softmin 均值。

规格见 `first-paper-spain-package-design.md`、`first-paper-external-replay-design.md`、
`first-paper-opensr-failure-design.md`（均在 `docs/superpowers/specs/`）。

## 验证与耗时

按用户要求缩小验证范围，没有运行全仓测试：

```sh
.venv/bin/python -m pytest tests/data/test_spain_package.py tests/data/test_spain_inputs.py \
  tests/paper/test_external_replay.py tests/paper/test_external_scores.py \
  tests/paper/test_external_opensr.py -q --tb=short --override-ini addopts=''
.venv/bin/python -m pytest tests/data/test_local_data_policy.py -q --override-ini addopts=''
.venv/bin/python -m scripts.paper.benchmark_external_cpu
```

前两条分别 **77 passed（8.59 秒）**、**7 passed（0.84 秒）**；包含 38 个新增用例。
新功能测试先因缺失失败，实施后通过。变更 Python 文件 Ruff、`git diff --check` 通过。
只读审查发现的包来源绑定问题已经修复并用 protocol 4/5 合成包测试覆盖。

合成串联使用 1 个完整 512×512 ROI、7 份合成预测和真实上游 OpenSR 计算：

- 包身份核验、受限解码及归一化：0.248086 秒。
- 七份预测生成／写缓存段（包含合成构造和 bicubic）：0.128178 秒。
- 缓存读取＋五评分＋隔离 OpenSR＋汇总：两轮 4.185629／4.037788 秒。
- 两轮科学 JSON 字节一致；主比较有效对为 1，结构诊断有效。

该脚本不接受数据或模型路径，临时合成包／缓存退出时清理，不发布合成科学指标。
本地 x86_64、PyTorch 2.13.0+cu130、NumPy 2.5.2、单线程；**不是云端冻结环境**。
所以这些秒数只证明集成可运行，不能直接补进云端价格／端到端预算，也不能替代
已完成的真实 GPU 测速；无需重复后者。

## 下一步与尚未完成

- 把这些核心接入正式执行 CLI：从最终协议认证包身份／成员／完整模型 provenance，
  在独立外部缓存目录只生成缺失预测，记录运行失败原因、复算并整体发布。
- 最终输入来源／nodata 局限仍需明确披露；实际包的内嵌对应只在冻结授权后受控核验。
- 需要同一云端环境的合成 CPU 串联／输入与缓存开销、加载、下载及实际计费条件，
  才能完成端到端预算。新 CPU 计时无需新 GPU 推理。
- 最终协议和专门外部访问范围仍未冻结，Task 5 整项和 Task 6–8 不勾选。
  不重跑开发选择、SEN2SRLite 诊断或历史 A/B/C。
