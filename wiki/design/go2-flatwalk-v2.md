# Go2 FlatWalk v2 快速训练设计

## 目标

`go2-walk-flat-v2` 是 Go2 平地速度跟踪任务的 v2 正式版本。它在 v1 冻结配方的基础上降低训练规模，把完整训练控制在约 11 分钟预算内，同时不使用提前终止，也不牺牲真机可部署性。

v2 的验收不是“仿真曲线看起来正常”，而是同时满足：

- 训练完整执行 2000 iterations；
- TensorBoard 的 reward、episode length、value loss 与 policy noise 进入平台期；
- 16 个固定命令场景、220 个速度/摩擦组合和 60 秒零命令站立审计通过；
- 导出的 deployment artifact 能在 MuJoCo/MotrixSim 中运行，并保持与已实机验证参考策略相同的部署契约；
- 真机效果稳定。真机效果不达标时，即使训练时长缩短也不能视为成功。

## v1 → v2 差异

| 项目 | v1 冻结配方 | v2 配方 |
| --- | --- | --- |
| 并行环境数 | 4096 | 1024 |
| 训练迭代数 | 3000 | 2000 |
| minibatch 数 | 16 | 4 |
| 训练框架 | RSLRL PPO | RSLRL PPO |
| seed | 1 | 1 |
| `tracking_ang_vel` reward scale | 1.3 | 1.6 |
| hard cases 数量 | 24 | 28 |
| pure-yaw hard cases | 无 | `(0, 0, ±0.5)`、`(0, 0, ±0.8)` |
| hard-case 课程 | 600 iter 到 0.05，1000 iter 到 0.15，1500 iter 到 0.20 | 450 iter 到 0.08，700 iter 到 0.20，之后保持 |
| command 重采样间隔 | `(5, 10) s` | `(5, 10) s` |
| 机器人、观测、动作、其余 reward 与随机化 | v1 配方 | 保持不变 |

v2 只做训练配方收敛，不改变 actor/critic 网络、观测构造、动作尺度、episode 生命周期或部署输入输出契约。唯一 reward 变化是把 yaw 速度跟踪权重从 1.3 提高到 1.6，用于在较少采样量下保留转向能力。新增的 4 个 pure-yaw hard cases 覆盖无平移速度下的原地转向；课程提前让困难命令在 2000 iterations 内获得充分训练。

## 2026-11-26 验证记录

训练使用 `go2-walk-flat-v2/rslrl.ppo` 与 seed 1，实际执行 2000/2000 iterations、1024 个并行环境、4 个 minibatches，共 49.152M steps。训练墙钟 6 分 57 秒，低于 11 分钟预算；没有提前终止。Run 目录为 `runs/go2-walk-flat-v2/rslrl/torch/ppo/26-11-26_12-06-00-796973`。

TensorBoard 末段窗口显示策略已进入平台期：mean reward 在 1700–1799、1800–1899、1900–1999 iterations 分别为 102.19、102.57、102.54；episode length 为 983.47、982.65、978.77；value loss 为 0.0706、0.0714、0.0733；noise std 为 0.05723、0.05668、0.05586。末 300 iterations 的 reward 斜率为 +0.00252/iter，episode length 斜率为 -0.0207 step/iter。同一统计口径下，已实机验证参考策略末 100 iterations 的 mean reward 为 100.87、episode length 为 979.49、value loss 为 0.0777、noise std 为 0.0563；v2 分别为 102.54、978.77、0.0733、0.0559。

完整仿真审计全部通过：16 个固定命令场景 16/16 完成；220 个速度/摩擦组合 220/220 完成；60 秒零命令站立完成，min up z 0.9985、max roll 0.0246 rad、max pitch 0.0498 rad、四脚接触率 100%、步频 0。与已实机验证参考策略相比，220 矩阵整体 MAE 为 vx 0.162 vs 0.178、vy 0.332 vs 0.319、yaw 0.132 vs 0.133；max roll 均值为 0.118 vs 0.105 rad，max pitch 为 0.108 vs 0.117 rad，min up z 为 0.989 vs 0.990。v2 前向与偏航跟踪略好，横移误差和 roll 波动略高，但所有工况保持直立并完成。

本次 run 重新导出的 `artifacts/go2-walk-flat-v2.deploy` 通过 `mxlab deploy inspect`，ONNX 抽样最大误差 8.34e-7，输入/输出为 49/12 维。MuJoCo 部署冒烟 2 秒完成 100 步，`success=true`、`overrun_count=0`，推理均值 0.028 ms、p95 0.046 ms，实时因子 24.1。真机验收尚未执行；按本设计的验收标准，真机效果不达标时不能将 v2 认定为完成。

## 分支结构

分支从冻结 v1 提交创建，而不是携带实验分支的未提交状态。Go2 只保留一个任务和一条训练配置：

```text
go2-walk-flat-v2/rslrl.ppo
```

旧的 `go2-walk-flat` 与 `go2-walk-rough` 训练配置和部署 recipe 在该分支中删除。v2 的仿真与真机部署任务分别是：

```text
go2-walk-flat-v2/sim
go2-walk-flat-v2/hardware
```

Artifact 必须由 v2 分支重新训练并重新导出；不从旧任务或实验任务复制。默认输出路径为 `artifacts/go2-walk-flat-v2.deploy`。

## 验收

1. 配置验收：Hydra 解析结果必须为 `num_envs=1024`、`max_iterations=2000`、`num_mini_batches=4`、`seed=1`。
2. 收敛验收：读取 TensorBoard `Train/mean_reward`、`Train/mean_episode_length`、`Loss/value`、`Policy/mean_noise_std`；末段分窗口均值变化很小，且末 300 iterations 的 reward 线性拟合斜率接近 0。
3. 审计验收：16/16 个固定场景成功；220/220 个速度/摩擦组合成功，跟踪误差与姿态指标不劣于参考阈值；60 秒零命令站立保持稳定。
4. 部署验收：`export-deploy` 生成的 artifact 通过 `mxlab deploy inspect`；MuJoCo/MotrixSim 冒烟运行成功。
5. 真机验收：按部署安全流程在真机上确认行走、横移、转向和站立稳定性。真机不满足预期时，v2 不能被认定为完成。
