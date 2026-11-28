# 环境模拟器性能检查（2026-11-28）

基线代码：`5324ab0`。优化集中在环境实现，未改变观测、动作空间、奖励、战斗 tick、RNG 或游戏规则。
远端 `/data/exp/sp` 正在运行 60-worker PPO；本次在独立目录 `/tmp/stronghold-perf.TIVtuD` 测量，未更新或重启正在训练的进程。

## 已实现的改动

1. `simulator/js/headless.mjs`：RoomSession 默认禁用训练不消费的私有状态推送、观战快照和战斗展示事件。原来的 `send()` 空回调只丢弃最终消息，生成消息的开销仍然存在。现在在构造这些数据之前跳过。保留 `unitStats`、决策观测、公共消息节流及虚拟调度时序；可用 `new RoomSession({ headless: false })` 对照客户端输出路径。
2. `simulator/engine/server/sim/ai.js`：仍在攻击冷却中的单位，在维护阻挡和刷新攻击范围后直接返回，避免每 tick 合并技能攻击配置。攻击准备完毕时仍按原顺序选择目标并结算。
3. `simulator/engine/server/match/bandCommands.js`：按 effectId 查询时直接遍历指令记录，避免每次 `Object.values(...).flatMap(...)` 构建全表。保留首个匹配顺序，直接读取当前数据，不引入缓存失效问题。

这些改动在本地工作区生效。远端当前训练继续使用原代码；测试副本中的结果不能当作当前训练已经提速。

## 测量方法

新增 `scripts/benchmark_simulator.mjs`。先生成专家动作轨迹，再按相同 seed、动作内容和顺序驱动 RoomSession。
这样排除了模型策略变化造成的局长、胜负和动作数量差异。专家轨迹回放仅支持单人模式，因为联防专家示范和 RoomSession 的座位调度顺序不同。

- 性能样本：规程等级 4、8、14、20，各 seed `71000`；每组共 1,036 个决策步。
- 新旧版本各预热 2 轮，正式各测 4 轮；在同一进程中逐局交替新旧版本，并逐轮反转先后顺序。
- 默认使用训练的 compact 协议。每一帧包括终局的完整 JSON 都进入 SHA-256；新旧 hash 不一致即停止。
- `stepMs` 仅计 `RoomSession.step()`；`wallMs` 还包括 reset、序列化、基准脚本的动作匹配和哈希。
- `cpuMs` 是整个 Node 进程的用户态和内核态 CPU 用时，包含 V8 辅助线程，可能大于墙钟时间。
- 不包含 Python IPC、特征编码、GPU 推理或 PPO 更新；轨迹生成和预热不计入结果。
- 服务器同时存在训练负载，单局时长有波动。此结果是环境基准，不是 PPO 整体加速比例。

### 本地交替基准

Node `v25.9.0`，4 轮合计：

| 指标 | 修改前 | 修改后 | 速度比 |
|---|---:|---:|---:|
| 回放墙钟 | 13.77 s | 10.59 s | 1.30× |
| 环境 step | 13.53 s | 10.36 s | 1.31× |
| Node CPU 用时 | 16.42 s | 12.78 s | 1.28× |

### 远端交替基准

双路 Xeon Gold 5416S，32 物理核 / 64 逻辑 CPU，Node `v22.20.0`。4 轮合计：

| 指标 | 修改前 | 修改后 | 速度比 |
|---|---:|---:|---:|
| 回放墙钟 | 44.14 s | 33.58 s | **1.31×** |
| 环境 step | 43.18 s | 32.66 s | **1.32×** |
| Node CPU 用时 | 61.81 s | 49.69 s | 1.24× |

回放墙钟减少 **23.9%**，相同工作量吞吐增加 **31.5%**，Node CPU 用时减少 **19.6%**。
各等级单局墙钟中位数：

| 等级 | 决策步数 | 修改前 | 修改后 | 速度比 |
|---|---:|---:|---:|---:|
| 4 | 316 | 3.254 s | 2.333 s | 1.39× |
| 8 | 289 | 3.337 s | 2.546 s | 1.31× |
| 14 | 277 | 3.366 s | 2.643 s | 1.27× |
| 20 | 154 | 1.264 s | 0.946 s | 1.34× |

远端正式数据：`/tmp/stronghold-perf.TIVtuD/paired.json`，本地副本：`artifacts/simulator-perf/remote-paired.json`。
四条轨迹全部逐帧 hash 一致。远端初次分开运行新旧进程时只测得约 1.16×，故最终采用预热更充分且交替运行的结果；这也说明后台负载和预热对微基准有明显影响。

## 正确性验证

- 与修改前的 `5324ab0` 独立副本对照：等级 1、4、8、14、20，seeds `71000..71002`，共 **15 局、4,552 步**。逐帧 JSON 哈希、动作轨迹、胜负完全一致。
- 新增 `tests/test_headless.mjs`：单人完整专家阵容和联防合法策略，分别检查 full / compact 两种协议；逐步对比观测、完整有序候选动作、元状态 RNG 和完整战斗结算 receipt。
- Node：8 个原有测试 + 2 个新增测试通过。
- Python：13 passed，4 skipped。包括 compact/full 编码精确一致性、异步采样、BC 和 PPO smoke。跳过的是需要外部 `stronghold-training` 原项目目录的旧迁移对照测试；本次另做了上述修改前代码对照。

## 剩余热点和下一步

修改前的本地 V8 CPU 采样中：战斗推进约 54%，客户端快照约 8%，无用私有推送约 5%，`unitStats` 属性预览约 13%，候选动作约 4%。这些是特定轨迹的采样占比，不是全部关卡的固定比例。

当前有价值的后续方向：

1. **属性预览的精确复用。** `_unitStatsOf` 在命中现有缓存前仍构建 battle input、运行隔离 meta 回调和复制数据。下一步应建立覆盖棋盘、装备、手牌、资金、盟约层数、指令、队友状态的可靠失效条件，或优化预览的重复只读访问。不能简单认为“刷新商店不改面板”，因为资金和刷新效果会参与游戏规则。
2. **战斗内核继续按 profile 优化。** buff tick、敌方移动/阻挡、目标查询和技能 tick 仍占主要开销。优先减少临时对象和重复遍历，并保留相同 tick 顺序和数值运算顺序；修改 tick 大小或跳过真实战斗会改变 RL 环境。
3. **单独测端到端采样。** 固定 checkpoint 和评测种子，比较 32 / 40 / 48 / 60 workers 的每秒决策数、CPU 负载与推理批次。服务器是 32 个物理核，60 actors 还各带 Python/Node 进程；最佳 worker 数需要实测，不能由逻辑 CPU 数直接判断。本次未启动额外 GPU 评测。

## 复现

在仓库根目录执行（远端将 `node` 换成 `/data/exp/sp/.tools/node/bin/node`）：

```bash
mkdir -p artifacts/simulator-perf
node scripts/benchmark_simulator.mjs --generate \
  --modes 4,8,14,20 --seeds 1 --traces artifacts/simulator-perf/traces.json

# baseline-root 应指向包含旧 simulator/ 和 config/ 的独立副本。
node scripts/benchmark_simulator.mjs \
  --traces artifacts/simulator-perf/traces.json \
  --baseline-root /path/to/baseline --warmup 2 --repeats 4 \
  --output artifacts/simulator-perf/paired.json

# 分项计时是包含子调用的时间，不能相加；勿与正式对比混用。
node scripts/benchmark_simulator.mjs \
  --traces artifacts/simulator-perf/traces.json --timings

node --cpu-prof --cpu-prof-dir=artifacts/simulator-perf \
  scripts/benchmark_simulator.mjs --traces artifacts/simulator-perf/traces.json

node --test tests/test_js_simulator.mjs tests/test_headless.mjs
.venv/bin/python -m pytest -q
```

本次本地原始数据保存在 `artifacts/simulator-perf/`（已被 `.gitignore` 排除）。
