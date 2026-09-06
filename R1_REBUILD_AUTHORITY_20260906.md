# MPPD R1 重构科学权威（2026-09-06）

本文件自提交起作为新 R1 实现的科学权威。旧 README 中与本文件冲突的 R1B/R1C/R1D、固定候选列车集合、规划运行图软先验、A/E=5 s 下界以及一次性大跨度时钟平移表述均视为历史定义，待后续安全回写 README。

## 1. 唯一科学任务

R1 只有一个科学任务：**全网乘客—服务—站内移动联合反演**。

AFC 仅直接观测乘客进出系统的边界。R1 必须在同一个联合后验中恢复：

1. 全天实际服务轨迹数量 `N`；
2. 每条服务轨迹的线路/方向/交路身份及所有经停站实际到达、出发时刻；
3. 全部乘客的轨道路径、乘车链和换乘链；
4. 所有站的进站与出站移动时间分布；
5. 所有换乘关系下可辨识的多条站内路径及各自纯换乘移动时间分布。

这些对象不得通过“先恢复并冻结一块，再让其他块吸收残差”的科学逻辑分开定义。算法内部允许交替优化，但必须服务于同一个联合目标并允许各隐藏块反复更新。

## 2. 输入边界

新 R1 主推断只允许使用：

- AFC 进出闸记录；
- 轨道网络拓扑 `G_rail`；
- 可获得的站内拓扑 `G_station`；
- 若存在，可使用真实 ATS/运营观测作为直接观测项。

**规划运行图的绝对时刻与规划列次数不进入主推断。** 规划运行图只在 R1 完成后用于外部比较和科学验证。

因此：

- `train_count_is_input = false`；
- `planned_timetable_used_in_primary_inference = false`；
- 历史 `1673 candidate roots` 只允许作为 baseline；
- 当前 `1841 count-free trajectories` 只允许作为 warm start，不是最终列车数真值。

## 3. 完整乘客时间守恒

对乘客 `p`：

```text
t_out - t_in
= A + W0 + sum(V_l) + sum(K_j + W_j) + E
```

其中：

- `A`：纯进站移动；
- `W0`：首次等待；
- `V_l`：第 l 段实际乘车；
- `K_j`：第 j 次纯换乘移动，不含等待；
- `W_j`：完成换乘移动后的等待；
- `E`：纯出站移动。

物理硬约束更新为：

```text
A >= 15 s
E >= 15 s
K >= 5 s
W0 >= 0
Wj >= 0
```

进站与出站 15 s 是新的绝对下界；换乘保留 5 s，以容纳面对面或极短站内路径的极限情况。

## 4. 服务世界必须 count-free

实际服务世界定义为：

```text
S = {S_q}_{q=1..N}
```

其中 `N` 本身是未知变量。任何外部 authority 都不得预先给出完整 `service_event_keys` 或固定服务数量。

粗结构搜索必须允许：

- `birth`；
- `death`；
- `split`；
- `merge`；
- `path_reassignment`。

必须有复杂度控制，防止为提高 passenger coverage 无限造车。复杂度惩罚用于选择“最小充分服务集合”，不能解释为已知实际列次数。

## 5. 到达与出发必须分离

最终服务事件必须恢复：

```text
arrival_time(q,s)
departure_time(q,s)
```

并满足：

```text
arrival_time(q,s) <= departure_time(q,s)
arrival_time(q,s_next) > departure_time(q,s)
```

粗阶段 AFC 服务脊线只允许作为 `anchor_time`，不得直接声明为实际到达/发车真值。进入精阶段后必须显式拆分 arrival/departure。

## 6. 5 秒粗发现 -> 1 秒联合精修

多分辨率是同一个联合目标的数值策略，不是两个科学阶段。

### 粗阶段：5 s

目标是确定服务结构：数量、线路/方向、主支线身份和粗事件时刻。

允许的单轮事件更新时间：

```text
{-5, 0, +5} s
```

结构操作只允许在粗阶段正式发生。

### 精阶段：1 s

当服务数量、轨迹匹配、乘客 MAP 链、站内参数和目标函数变化同时进入稳定区间后，切换到 1 s 精修。

每个事件单轮仍满足：

```text
|delta tau(q,s)| <= 5 s
```

但搜索粒度为：

```text
{-5,-4,-3,-2,-1,0,1,2,3,4,5} s
```

精阶段不得静默 birth/death/split/merge。如果精阶段出现明显结构不稳定，必须返回 5 s 粗阶段重新进行结构反演。

历史一次性 `+60 s` 公共时钟平移只保留为诊断：它说明 AFC ridge 与 passenger world 存在系统性错位，但不得作为合法 R1 状态更新。

## 7. 站点进出站分布

每个站分别恢复 access/egress 分布，并至少输出：

```text
q05, median(q50), q95
```

最终区间采用后验可信区间而非简单 min/max。

每个站必须显式标注辨识状态：

- `DIRECTLY_IDENTIFIED`；
- `HIERARCHICALLY_INFERRED`；
- `UNRESOLVED`。

无直接证据的站不能包装为直接恢复成功。

## 8. 换乘路径数量也不得固定

对换乘关系 `m`：

```text
R_m = {r_1, ..., r_Rm}
```

其中 `R_m` 由数据和/或站内拓扑决定，可以是 1、2、3 或更多。当前每个关系固定两条 latent path 的结果只作为原型初始化。

每条路径均恢复独立纯换乘移动分布，并满足 `K >= 5 s`。模型必须允许路径成分 split/merge，并通过复杂度控制避免无限拆分。

## 9. 规划运行图的最终角色

规划运行图不参与主反演。R1 收敛后才比较：

```text
N_hat(line,direction,time) vs N_plan(line,direction,time)
headway_hat vs headway_plan
trajectory structure_hat vs plan
```

这样才能检验仅凭 AFC + 网络结构恢复实际服务世界的能力，而不是把计划答案提前塞入模型。

## 10. 最终资格化

R1 资格化不能只看 passenger coverage。必须同时通过：

- count-free 服务数量稳定性与参数敏感性；
- 所有服务轨迹物理连续性；
- arrival/departure 事件合法性；
- passenger 完整时间守恒；
- 全站 access/egress 分布及辨识标签；
- 全换乘关系的路径结构与移动分布；
- `A,E>=15 s`、`K>=5 s`、等待非负；
- unresolved passenger failure decomposition；
- 不同初始化/复杂度参数下的结构稳定性；
- 最后才进行规划运行图外部比较。

剩余未解析客流可以存在，但不能主要由候选截断、固定列车数、错误时间分辨率、错误物理下界或尚未实现的算法缺口造成。

## 11. 当前实现边界

新主线核心文件：

- `scripts/mppd_r1_joint_model.py`：联合状态与物理契约；
- `scripts/mppd_r1_joint_solver.py`：5 s -> 1 s 多分辨率控制和 5 s trust region；
- `scripts/mppd_r1_bootstrap_count_free.py`：把当前 count-free 服务发现结果接入新联合状态。

旧日期脚本继续保留用于实验资产和历史可追溯性，但不再自动拥有科学权威。
