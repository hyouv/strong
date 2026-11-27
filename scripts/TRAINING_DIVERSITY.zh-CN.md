# 多阵营开局训练

## 从头训练，首轮启用随机掩码

```bash
cd /data/exp/sp
RUN_ID="$(date +%Y%m%d_%H%M%S)" \
DEVICE=cuda:0 bash scripts/train_server_diversity_scratch.sh all
```

完整执行 **全新 BC → NORMAL（等级 5–8）→ HARD（9–14）→ ABYSS（15–20）**。
BC 从随机初始化的模型开始，默认重新采集均衡开局示范，不加载任何旧模型权重。
同配置的新格式数据缓存存在时可以复用；旧的无元数据缓存会被拒绝。
BC 是监督学习，不应用 PPO 采样掩码；**第一次 PPO 采样起（包括 5 次 Critic 热身）**即使用：

- 50% 对局随机限制开局，另外 50% 对局自由选择。
- 受限局随机保留一个盟约组、35% 的合法策略（当前单人模式 60 个中保留 21 个）。
- `mask_p=0.500`、`band_keep=0.350` 全程固定，三个 PPO 阶段均不再从弱限制渐增。
- 回合内 `lambda=1.0`，评测不加训练掩码，保留盟约熵和策略熵的分别监控。

PPO 更新次数沿用原完整训练配置：NORMAL 360、HARD 240、ABYSS 500，
每阶段前 5 次只训练 Critic；其他学习率、熵奖励和难度混合配置沿用 `train_server_5m_v2.sh`。
这与下面的旧权重微调脚本不同。

输出独立保存在：

```text
logs/diversity_scratch_<RUN_ID>_all.log
checkpoints/diversity_scratch_<RUN_ID>_bc.pt
checkpoints/diversity_scratch_<RUN_ID>_{normal,hard,abyss}_{last,best}.pt
```

NORMAL 读取本次 BC，HARD 读取本次 NORMAL best，ABYSS 读取本次 HARD best；
脚本会覆盖环境中残留的 `NORMAL_INIT_CKPT` 等续训输入，且拒绝覆盖已存在的输出权重。
分阶段运行 `bc`、`ppo` 或 `normal|hard|abyss` 时，要显式传入**同一个 `RUN_ID`**，
才能找到本次训练的前置权重；这些入口开始新的阶段，不恢复中断阶段的优化器。
需要每局都随机受限时，可加 `PPO_OPENING_MASK_PROB_START=1.0`（END 默认跟随 START）；
默认保留一半自由选择对局，用于训练最终无掩码评测时的开局决策。

### BC 均衡开局数据

默认 `BC_OPENING_POLICY=balanced`、`BC_EPOCHS=10`，数据缓存改为：

```text
data/v2feat_solo_1_8_1200ep_balanced_v2.pt
```

采集器先查询每个种子的合法策略及其兼容盟约组，再优先补足出现次数较少的策略、
盟约组和组合，同分时用独立的固定种子随机打破平局。计数由主进程管理，
不随 Worker 数量变化，也不消耗游戏内 RNG。目标是尽量接近两组各半、各可用策略均匀覆盖，
具体计数以采集完成后日志的 `opening_coverage` 为准。
阵营专属策略只与支持它的盟约组配对；游戏模式本身禁用的策略不纳入采样。

选定开局后，专家实际执行该开局并游玩完整对局，再编码成 BC 数据；不修改旧轨迹的标签。
局中购买、合成、布阵等专家逻辑沿用原实现。均衡示范提供开局覆盖，本身不代表学会了按 Boss 选最优策略。
原始专家开局行为可通过 `BC_OPENING_POLICY=legacy` / `--opening-policy legacy` 显式使用。

缓存记录生成策略版本、等级、局数、种子和开局分布，加载均衡数据时校验这些配置。
配置不匹配时使用新的 `BC_DATASET_CACHE` 路径重新采集；不能把旧缓存改名当作均衡缓存。

v2 修复了遥二技能与吸血效果在同一次结算内递归触发的问题，并加入异常局隔离补采。
模拟器判为 `Invalid expert match` 的整局会丢弃，原始种子、开局和完整错误立即写入
`<缓存路径>.quarantine.jsonl`；成功的同批对局保留。最多换种子补采 3 次，难度、策略和
盟约组索引不变，新种子先检查开局兼容性（具体盟约内容随种子变化）。耗尽重试或发生
进程/协议错误时仍报错退出。
缓存元数据保存失败与替换种子；日志输出 `episodes_quarantined` 和 `episodes_replaced`。
这是对异常样本的容错，补采会排除出错局面，不能替代修复模拟器及检查异常比例。

## 从现有最佳权重渐进续训

服务器目录 `/data/exp/sp`。以下命令由操作者手动执行，前台输出同时保存到日志。

```bash
cd /data/exp/sp
DEVICE=cuda:0 bash scripts/train_server_diversity.sh normal
```

`normal` 默认加载 `checkpoints/v2feat_ppo_5m_solo_t2_best.pt`，训练等级 5–8，
保留 8% 等级 1–4 对局。共 240 次更新，其中前 5 次只校准 Critic。
这是从已有模型权重开始的新一轮微调，优化器和更新计数重新初始化。

指定其他输入权重、输出标签或更慢的限制课程：

```bash
INIT_CKPT=checkpoints/v2feat_ppo_5m_solo_t2_best.pt \
RUN_ID=opening_diversity_v1 \
PPO_OPENING_MASK_ANNEAL_UPDATES=180 \
DEVICE=cuda:0 bash scripts/train_server_diversity.sh normal
```

可选阶段为 `normal`、`hard`、`abyss`，分别训练等级 5–8、9–14、15–20。
后两者默认读取同代 `v2feat` 对应阶段的最佳权重，也可通过 `INIT_CKPT` 指定。
脚本只运行指定阶段；切换阶段不会自动继承上一轮随机掩码进度，需要通过环境变量设置起点。

## 渐进限制

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `PPO_OPENING_MASK_PROB_START` | 0.0 | 起始受限对局比例 |
| `PPO_OPENING_MASK_PROB_END` | 0.5 | 最终受限对局比例 |
| `PPO_OPENING_BAND_KEEP_START` | 1.0 | 受限对局起始保留的合法策略比例 |
| `PPO_OPENING_BAND_KEEP_END` | 0.35 | 受限对局最终保留的合法策略比例 |
| `PPO_OPENING_MASK_ANNEAL_UPDATES` | 120 | Critic 热身后，达到最终限制所用的 Actor 更新次数 |
| `PPO_GAE_LAMBDA_INTRA` | 1.0 | 回合内 GAE λ；跨回合 λ 仍为 0.95，γ 为 0.97 |

默认热身及第一次 Actor 更新没有额外限制。第 120 次 Actor 更新（总更新 125）达到终点，
其后保持不变。最终平均一半对局自由选择；另一半对局在两个合法盟约组中随机保留一个，
再从合法策略中均匀抽取 35%（向上取整，若可能至少保留两个），由模型在该子集中选择。

每局独立采样，整局固定掩码；不改游戏随机数，不修改商店或战斗规则。
掩码在采样时进入轨迹，PPO 更新复用原掩码。现有定向禁选仍可使用，随机掩码只会进一步缩小合法集合。
若第 0 组被禁用，会同时禁止未选组时的 `INFO_READY` 默认选择入口。
评测不启用训练掩码，即使复用训练 Worker 也会在 reset 时清除限制。

直接调用 `stronghold_rl.ppo` 的参数对应如下：

```text
--opening-mask-prob-start 0.0 --opening-mask-prob-end 0.5
--opening-band-keep-start 1.0 --opening-band-keep-end 0.35
--opening-mask-anneal-updates 120 --round-discount --gae-lambda-intra 1.0
```

通用 PPO CLI 和原 `train_server_5m_v2.sh` 默认关闭随机掩码；新续训脚本显式启用上述课程。

## 日志指标

- `bond_ent`：盟约组的实际选择分布熵（自然对数）。显式选择第 0 组与 `INFO_READY` 默认选择第 0 组合并，已选组后的确认动作不计入。
- `band_ent`：开局策略选择的条件分布熵（自然对数），不包含准备阶段动作熵。
- `bond_hn` / `band_hn`：在仍有至少两个选项的样本上统计 `H / log(K)`，用于排除强制选择的零熵和候选数量变化。越接近 1，当前候选上的分布越均匀。
- `bond_forced`：仅剩一个合法盟约组的样本比例。限制增强时 `bond_ent` 下降可能来自这一比例上升，应结合 `bond_hn` 查看。
- `bond_n` / `band_n`：熵统计的样本计数，包含 PPO 各 Epoch 的重复评估，不等于新对局数。若没有符合条件的样本，对应指标记为 0。
- `mask_p` / `band_keep`：本次更新使用的限制概率和策略保留比例。
- 原有 `o_ent` / `s_ent` 保留，定义不变。以上新指标仅监控，不额外修改熵奖励。

所有新熵均按样本数加权汇总，而不是按 minibatch 等权平均。
这些是训练分布上的指标；是否学会按 Boss 选择仍需看不受限评测的表现。

## 输出与后续使用

默认 `RUN_ID` 为启动时的时间戳，输出：

```text
logs/diversity_normal_<RUN_ID>.log
checkpoints/diversity_normal_<RUN_ID>_last.pt
checkpoints/diversity_normal_<RUN_ID>_best.pt
```

`last` 是最新更新（正常结束时是完成整个限制课程）的模型。
`best` 沿用原有不受限评测的胜率、隐秘核心通关率、平均回合数排序，包含初始评测；
若初始模型得分一直最高，`best` 可能仍是初始模型。需要继续训练已完成多样性课程的模型时，选择对应 `last`。
新检查点记录输入权重路径、完整启动参数和保存时的 PPO 配置。
程序拒绝输出路径覆盖输入或参考权重，也拒绝 `last` 与 `best` 使用同一个文件。

脚本支持覆盖原训练脚本的环境变量，例如 `NORMAL_UPDATES`、`NORMAL_LR`、`PPO_WORKERS`、
`EVAL_EVERY`、`SAVE_EVERY`。新微调默认 Actor 学习率 `1e-5 → 3e-6`，
KL 系数 `0.01 → 0`（60 次 Actor 更新），训练种子区间与上一轮分开。
