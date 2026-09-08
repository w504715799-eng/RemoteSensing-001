# 时间语义与 Sentinel-2 来源候选调查

2026-09-08。本轮完成了恢复清单的时间模式核查，以及三条预先固定样本的有界目录调查。
没有读取影像、旧预测缓存，也没有使用 GPU。结果记录在
`research/evidence/crosssensor-timestamp-source-probe-v1.json`，原始响应留在 Git 忽略目录。

## 时间模式与可能机制

恢复清单的 8,000 条 LR 时间、8,000 条 HR 时间，转换到 `Europe/Madrid` 后全部为
00:00:00；转换后的 HR 日期全部等于 NAIP 文件名末尾日期。LR 日期相对 NAIP 日期
为 -1 / 0 / +1 日的数量分别为 1,883 / 3,972 / 2,145。

这与“将日期午夜按欧洲中部时区转换为 Unix 时间戳”的机制完全相容。`Europe/Madrid`
只是本次检验采用的时区假设；共享同一历史偏移规则的其他时区也可能产生该模式。
这不能定位上游机器时区，也不能证明真实获取时刻恰好在午夜。

公开的 [Tortilla STAC 写入实现](https://github.com/tacofoundation/tortilla-python/blob/master/pytortilla/datamodel/main.py)
在 `check_times` 中直接对 datetime 对象调用 `.timestamp()`，该路径未先要求时区信息。
[Python 官方文档](https://docs.python.org/3/library/datetime.html#datetime.datetime.timestamp)
说明：不带时区的 datetime 会按本地时间处理。因此，若输入为不带时区的日期午夜，
就有产生上述偏移的实现路径。检查的源码快照为 13,954 字节，SHA-256：
`9947c5cf48ed30ac9233b8f2ea903932c21423f1278bc63816208231689aa30e`。

该快照来自本次读取的 master，未取得对应 Git commit，也没有证据证明它就是固定数据集
构建时使用的版本。它是机制线索，不是已查实的构建原因。原项目的 UTC 转换代码
`src/trustsr/data/taco_v1_adapter.py` 使用显式 UTC；三个已留存的原始嵌套目录数值也
与恢复值一致，故本轮没有修改冻结代码或历史时间字段。

## 固定目录查询

查询使用 [Earth Search 官方接口](https://element84.com/earth-search/examples/)，集合
`sentinel-2-l2a`。保留原有 `nested_probes` 的三个成员和顺序，每条使用中心点 ±0.02°
的发现范围、LR 原始 UTC 日期及次日的联合窗口。不加云量或质量筛选，不排序挑选来源，
不读取 assets。这个范围用于发现候选，不是完整源覆盖范围的证明。

| 成员坐标标识 | 查询 UTC 日期 | 返回的候选条目 | 差异 |
| --- | --- | --- | --- |
| E1183N0757 | 2022-07-10 至 11 | `S2B_10SDJ_20220711_0_L2A`、`S2B_10TDK_20220711_0_L2A` | 两个瓦片 |
| E1183N0791 | 2022-06-20 至 21 | `S2B_10TCL_20220621_0_L2A`、`S2B_10TDL_20220621_0_L2A` | 两个瓦片 |
| E1185N0724 | 2020-06-02 至 03 | `S2A_10SEH_20200603_0_L2A`、`S2A_10SEH_20200603_1_L2A` | 同瓦片，处理基线 02.14 / 05.00 |

六个返回候选都在时区假设还原的日期，而非原始 UTC 日期。完整产品 URI、粒度 ID、
时间、几何、响应摘要均留在证据文件；不根据这些结果指定某个产品为真实训练样本来源。
第三条的重处理版本尤其说明：相同日期与瓦片不足以辨别处理版本。

三个成功响应分别为 4,828 / 4,828 / 4,829 字节，共 14,485 字节；请求尝试次数为
1 / 1 / 2，第三条首次为 URLError。未跟随重定向、分页链接或资产链接。服务报告每条
matched=returned=2，但仍提供 next 链接。因此保留全部首屏候选并标记
`first_page_only_enumeration_not_certified`，不宣称已证明目录枚举完整。
这也不是整个 Sentinel-2 历史档案或原始 GEE 构建来源的完整性证明。

## 结论与下一步

本轮将“未知的异常时刻”推进为有全清单证据支持的日期序列化假设，并找到了可供上游
核对的具体产品候选。尚缺成员到实际产品/全部合成贡献者的生成记录，以及时间输入与
序列化约定。继续扩大同类近邻目录搜索不能自行补上这层证据。

下一项依赖是取得适用于固定 revision 的生成脚本或来源映射，并核对本次三个案例的
瓦片选择、处理版本和合成规则。已将这些具体问题加入本地未发送询问信
`docs/drafts/sen2naipv2-provenance-request.md`。若上游记录无法取得，须明确设计使用
可追溯来源的新数据路线，不能将本轮候选直接用于冻结独立分组或开始 GPU 实验。

离线复算（输出路径必须尚不存在）：

```bash
.venv/bin/python -m research.trustmask.timestamp_source_probe \
  --cache artifacts/progressive-timestamp-source \
  --output /tmp/timestamp-source-probe-replay.json
```

本轮测试覆盖夏冬时间、固定查询窗口、元数据白名单、分页/计数不完整、大小/编码/状态
拒绝与重定向拒绝。完整候选证据可从已保存响应离线重建；禁止把重放变成额外目录探测。
