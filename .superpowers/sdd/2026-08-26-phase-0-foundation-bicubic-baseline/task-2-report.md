# Task 2 报告：SRPair 数据契约

## 实现

- 新增 `src/trustsr/contracts.py`，定义 `@dataclass(frozen=True)` 的 `SRPair`：
  `sample_id`、`source`、`lr`、`hr`、`scale`。
- `SRPair.validate()` 验证 Phase 0 的固定 `scale=4`、CHW 三维布局、四个 RGBN 通道、HR 空间尺寸为 LR 的四倍、`torch.float32`、有限值，以及 `[0, 1]` 反射率范围。
- 新增 `tests/test_contracts.py`，覆盖有效 RGBN ×4 pair，以及通道数、尺寸和反射率范围错误。

## TDD 证据

### RED

在实现契约前运行：

```text
uv run pytest tests/test_contracts.py -v
...
collected 0 items / 1 error
ModuleNotFoundError: No module named 'trustsr.contracts'
```

### GREEN

实现后运行 focused 测试：

```text
uv run pytest tests/test_contracts.py -v
collected 4 items
tests/test_contracts.py ....                                             [100%]
4 passed in 1.31s
```

随后运行完整测试套件：

```text
uv run pytest
......                                                                   [100%]
6 passed in 1.51s
```

## 文件

- `src/trustsr/contracts.py`
- `tests/test_contracts.py`
- `.superpowers/sdd/2026-08-26-phase-0-foundation-bicubic-baseline/task-2-report.md`

## 自审

- 契约保持 RGBN、×4、`torch.float32`、`[0, 1]` 范围，未扩展到其他波段或倍率。
- 数据类冻结，满足不可变数据契约；张量本身仍遵循 PyTorch 的可变语义，这是张量对象的既有行为。
- 校验顺序先检查结构和尺寸，再检查 dtype、有限性与范围；错误信息与测试契约匹配。
- focused 与 full tests 均通过。

## 顾虑

无阻塞顾虑。当前范围没有增加额外导出、转换或数据加载逻辑。
