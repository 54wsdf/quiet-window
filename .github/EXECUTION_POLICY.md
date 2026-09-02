# 执行策略

本文件定义本仓库中数据获取工作流的执行纪律。

## 1. 允许承担的任务

本仓库可以执行交通科研资产的公开数据获取、有效性校验、校验和计算、API 分页、解压/重打包、确定性分片以及向长期存储传输等任务。

优先资产大致按以下顺序处理：

- 铁路/城市轨道交通/公共交通的实际运行与历史归档；
- 时刻表、GTFS、NeTEx 等机器可读服务计划；
- AFC、OD、客流、断面流量、乘车量和占用率数据；
- 扰动、晚点、取消、事故和 service alert 数据；
- 基础设施、线路拓扑、能力、施工和封锁限制；
- 车辆编组、车底运用、车辆循环、基地、调车与维修数据；
- 高质量铁路排程、调度和调车 benchmark；
- 在科研上有价值的道路、轨迹和综合 mobility 数据。

S/A 级来源即使只覆盖一个时段或一个案例，也值得获取。同一系统、同一时期匹配度高时提高优先级，但不作为准入条件。

## 2. 来源与条款检查

获取前，在能够取得时记录提供方页面和 License/Terms URL。

默认允许：

- 政府或公共机构公开下载数据；
- 公共交通运营机构 Open Data；
- 文档明确允许自动获取的公开 API；
- GitHub、Zenodo、Figshare、UCI 等公开科研资产；
- 公开可下载的 benchmark、实例集与复现材料。

不得自动绕过需要登录、人工申请、签署协议、付费墙、实名验证或技术访问控制的来源。此类来源应记录为需要所有者操作。

## 3. 时间选择

来源必须选择年份或时段时：

- 默认优先 2019 年及以后；
- 尽量保留较新的完整年份以及有价值的当前未完整年份；
- benchmark、长期比较、特殊扰动案例或历史唯一资产可以保留更早时期；
- 对高频发布且内容高度重复的计划快照，不机械复制全部版本，除非版本演化本身具有科研价值；
- 对实际运行、扰动、需求等实现数据，尽量保留提供方原生的最细时间粒度。

## 4. 大数据处理

科研选择层面**不设置字节数上限**。

runner、API、GitHub Artifact 或中间桥接限制只能通过工程方法解决，常用方式包括：

- API 分页；
- 按日/月/年切片；
- 确定性字节分片；
- 来源支持时使用断点、Range 或 resumable download；
- 有界并行矩阵；
- 使用来源原生分区，例如按日 Parquet。

不得为了适应工程限制而静默截断数据。

若进行了字节分片，必须保存：

- 原始文件名；
- 原始总字节数；
- 原始 SHA-256；
- 有序分片名；
- 每片字节数与 SHA-256；
- 明确、可重复的重组方法。

## 5. 仓库内部结构

只在实际需要时创建目录，不建立空的装饰性目录。

```text
.github/workflows/        # 执行入口
scripts/lib/              # 通用下载、校验、分片、传输工具
scripts/fetchers/         # 提供方专用 resolver
configs/sources/          # 只允许公开源定义
schemas/                  # manifest/source schema
```

下载数据、临时分片和任何凭据必须留在 Git 历史之外。

## 6. Workflow 规范

- 工作流名称保持中性、简洁、准确描述技术任务。
- 优先一个逻辑 source family 对应一个 workflow，或通过可复用入口调用。
- 大规模 harvest 默认使用 `workflow_dispatch`。
- 网络任务设置明确的 `timeout-minutes`。
- 独立矩阵任务使用 `fail-fast: false`。
- 对瞬时网络故障采用 retry/backoff。
- HTTP 错误、零字节文件、HTML 错误页面都必须判定为失败，不能伪装成成功数据。
- 能验证时检查 Content-Type、压缩包可读性或文件结构。
- 宣布取得成功前生成 SHA-256。
- 不把数据正文写入 Actions 日志。
- 不打印包含 Secret 的 Header、签名 URL、Cookie 或凭据。

默认权限：

```yaml
permissions:
  contents: read
```

只有明确需要时才提升权限。

## 7. GitHub Artifact 纪律

GitHub Artifact 只是临时运输对象，不是长期存储。

- 不得把 Artifact 当成 canonical copy。
- 长期存储确认成功后，Artifact 应尽量保持较短 retention。
- 避免同一多 GB 数据反复生成重复 Artifact。
- 已经存在且校验通过的 Artifact 应优先复用、分片或转存，而不是重新从提供方下载。
- Artifact 配额耗尽时，停止制造新的大 Artifact，优先清理已永久化对象或改用 runner 直接写长期存储。

对于本仓库的新重型任务，**优先采用“来源 → runner → 长期存储”直传模式**；只有在直传不可用、需要人工桥接或调试时才使用大 Artifact。

## 8. 直接写长期存储

长期存储凭据必须通过 GitHub Actions Secret 或其他受保护运行时机制注入，绝不提交进公开仓库。

当前推荐接口为 rclone 配置 Secret，例如：

- `GDRIVE_RCLONE_CONFIG_B64`：仅存在于 Repository/Environment Secret；
- workflow 在 runner 临时目录恢复配置；
- 传输结束后删除临时配置；
- 日志不得打印配置正文。

若 Secret 不存在，直传 workflow 必须在真正下载重型数据前停止，并清楚说明“缺少运行时存储配置”，不能退化成自动制造超大 Artifact。

公共仓库中的 fork PR 不得获得长期存储 Secret。涉及 Secret 的传输工作流仅允许受控的 `workflow_dispatch`、受信任分支或其他明确安全触发方式。

## 9. 长期存储路径模型

持久化路径应尽量遵循：

```text
transport_data_lake/
  01_rail/{system}/
    00_source_archives/
    01_schedule/
    02_actual_operations/
    03_realtime/
    04_demand_od/
    05_disruptions_trackwork/
    06_infrastructure/
    07_rolling_stock_composition/
    08_performance_reliability/
    99_manifests/
```

其他模式可使用 `02_road`、`03_trajectory`、`04_mobility_demand` 与 `05_simulation_benchmarks`。

不得把私有存储凭据硬编码在仓库中。存储目标、路径或 ID 应优先在运行时注入；若只是非敏感逻辑路径，可以作为公开配置保存。

## 10. 完成门槛

一个逻辑资产只有同时满足以下条件，才能标记为 `ACQUIRED`：

1. 真正获取到了提供方数据字节；
2. 文件非零且内容合理；
3. 已有 SHA-256 与 manifest；
4. 如有分片，重组所需全部分片都存在；
5. 长期存储传输已确认成功。

中间状态统一使用：

- `DISCOVERED`
- `DOWNLOADED_TEMP`
- `ARTIFACT_ONLY`
- `TRANSFER_PARTIAL`
- `ACQUIRED`
- `FAILED`

不得因为 GitHub job 为绿色就直接写成 `ACQUIRED`。

## 11. 公共仓库卫生

这是 Public 仓库。默认假设所有 commit、文件和 Actions 日志都可被第三方看到。

尽量不要写入：未公开科研理由、论文假设、稿件、私有仓库名称/路径、内部凭据、个人存储标识以及其他用户特定敏感信息。

本仓库可以用中文向 Agent 说明技术执行纪律，但 README 继续保持泛化，不主动介绍这些内部用途。
