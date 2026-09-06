# R1 Iteration 1 λ=0 压力测试科学裁决（2026-09-06）

本文件覆盖 workflow `MPPD R1 Joint Iteration One Local Update 2026-09-06` 生成的 `R1_COARSE_ITERATION1_WORKING_UPDATE_ACCEPTED` 工程标签。

该 workflow 的 acceptance 规则只要求：

1. full-day passenger resolved mass 高于 Iteration 0；
2. A/E/K 物理下界仍成立；
3. 不使用全局时钟平移。

它没有惩罚被修改的服务事件数量。因此该 acceptance 仅证明“这组修改在 passenger feasibility + 基础物理门下可行”，不能证明它是新的 R1 Iteration 1 科学状态。

## 实际 λ=0 压力测试结果

- Iteration 0 resolved mass: `598628`
- λ=0 工作态 resolved mass: `602512`
- resolved mass 增益: `+3884`
- resolved share: `0.49953074063388153`
- 修改事件数: `1096`
- 修改轨迹数: `1096`
- 单事件平均实际 resolved-mass 增益: 约 `3.54`
- 候选中 `1094` 个为 `-5 s`，`2` 个为 `+5 s`
- 候选 direct rescue mass（未去重求和）: `3239`
- 候选 direct damage mass（未去重求和）: `128`
- 服务轨迹总数仍为 `1841`
- 无全局时钟平移

站内 movement 重估仍满足：

- access lower bound = 15 s；
- egress lower bound = 15 s；
- transfer lower bound = 5 s；
- service_clock_shift_s = 0。

## 科学结论

该结果证明：**在当前 1841 条 count-free warm-start 服务世界上，存在事件级局部时刻改进空间；且末段 passenger residual 对大量事件产生向前移动的压力。**

但修改 1096/47999 个粗事件、涉及 1096/1841 条服务轨迹，只换取 3884 的 resolved-mass 增益，尚不足以证明这些事件应被真实移动。因此：

```text
λ=0 result = UNREGULARIZED EVENT-PRESSURE STRESS TEST
NOT = accepted R1 Iteration 1 state
```

后续必须通过 `lambda_event` 复杂度前沿比较“修改事件数量 ↔ 完整全天 passenger posterior 增益”，并进一步加入 AFC ridge likelihood / anchor degradation 后，才选择可接受的 Iteration 1 更新集。

任何后续代码不得把旧 `R1_COARSE_ITERATION1_WORKING_UPDATE_ACCEPTED` 标签当作新的科学 authority。
