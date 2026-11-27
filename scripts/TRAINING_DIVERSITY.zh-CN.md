# 从现有最佳权重续训多阵营开局

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
