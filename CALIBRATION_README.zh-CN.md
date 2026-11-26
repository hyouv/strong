# 卫戍规程校准引擎

本包将 2026-11-26 本地 `reference_dev` 引擎整理成 hyouv/strong 的 `simulator/engine/` 目录结构。游戏版本为 `act3autochess@V3_1`，上游基础提交为 `55f528e9fc4872388360ee85e13eca8594856204`。包标识为 `stronghold-calibrated-engine-20261126`，记录在引擎的 `package.json` 中。

## 安装到 STRONG

在 STRONG 仓库根目录备份当前引擎，然后把 ZIP 解压到该根目录，覆盖同名文件。解压后的路径应为 `simulator/engine/server/data.js`。

Linux 示例，请把压缩包路径换成本机路径。

```bash
cp -a simulator/engine simulator/engine.before-calibration-20261126
unzip -o /path/to/stronghold-calibrated-engine-20261126.zip -d .
node verify_calibration.mjs
node --test tests/test_js_simulator.mjs
```

Windows 可使用资源管理器解压到 STRONG 根目录，再在该目录执行两条 Node 命令。Node.js 要求 22 或更新版本。引擎运行仅依赖 Node 内置模块。

验证脚本检查实测覆盖范围，以及 GameData 实际返回的血量倍率和攻速。它可以直接在本压缩包的解压目录执行。

## 校准内容

实测依据来自官方客户端只读采集的 `sceneStateData.globalBattleEffects`，生成器使用官方效果表解析每回合效果。

| 模式 | 采用实测倍率的回合 |
|---|---|
| 多人标准 `mode_multi_funny` | 1–14 |
| 多人绝境 `mode_multi_hard` | 1–15 |
| 多人终极 `mode_multi_abyss` | 1–14 |

合计 43 个模式／回合组合。多人险境与单人模式的倍率采用原研究表，终极第 15 回合也采用原表。

- 生成后的 `config.json` 提供实测攻击、血量、移速、攻速和补给线倍率。
- `gamedata.js`、`waves.js` 和 `spawns.js` 将攻速加值传到战斗实体。
- `regulation.js` 包含圆号手每第三次攻击额外攻击目标、鼓手每第三次攻击溅射的修正。
- 包内同时包含本地 `bot.js` 和 `rehearsal.js` 的已有改动，与刚完成的四席测试所用版本一致。行为克隆和机器人评估会使用这些改动。

绝境第 14 回合血量倍率为 9.52542。终极第 14 回合血量倍率为 12.280707，攻速加值为 30。这里是敌人回合属性倍率，Boss 共享血池另由引擎结算。

## 继续训练

模型目录和权重加载接口兼容此前检查的 STRONG 提交 `2d10e994b5cc00181d06f9f09f9107b67c5d342c`。模型使用的 11 份静态词表数据与该版本逐值一致，现有 508 万参数 checkpoint 可加载。

STRONG 当前 `stronghold_rl.ppo` 命令行通过 `parse_solo_modes` 选择单人模式。要训练本次评测的四席环境，训练侧需要支持 `mode_multi_hard` 和 `mode_multi_abyss`，并按席位维护备战步数、循环检测、轨迹、终止状态及优势计算。引擎的 reset/step 接口已支持四席轮流决策。

训练记录建议保存包标识 `stronghold-calibrated-engine-20261126`。引擎 `package.json` 中的 `upstream.revision` 记录基础提交，`calibration` 字段记录本地改动身份。原 STRONG hello.source 来自 `config/regulation.json`，可在该文件的 `source` 对象追加 `calibration_id` 以使现有 JS 和 Python 握手都带上此标识。

## 实测依据与重建

`calibration/manifest.json` 记录来源、改动文件、实测覆盖范围和六局评估摘要。`calibration/reference_dev.patch` 保存完整本地改动，包括生成器 `tools/build-data.mjs` 的修改。`calibration/docs/research/versions/act3autochess-v3_1/observed-enemy-effects.json` 保存逐回合效果清单。

若需在完整上游仓库重建数据，可切到上述基础提交，应用补丁，将效果清单放到相同的 `docs/research/versions/act3autochess-v3_1/` 路径，再按上游 `docs/DATA.md` 的流程执行 `node tools/build-data.mjs --version act3autochess@V3_1`。继续训练可直接使用包内已生成的数据。

## 验证与适用范围

本地校准版使用 `ppo_5m_solo_t2_best.pt` 的第 220 次更新权重控制四席，种子 1000000–1000002。绝境三局分别在第 13、14、13 回合失败，终极三局分别在第 12、11、9 回合失败，六局模拟器报错均为 0。评估采用按席位记录的循环处理，每档样本数为 3。

本地校准记录显示，终极 R10–R11 仍可能低估漏怪。效果清单的覆盖范围见上表，校准模拟结果用于研究和训练评估。

打包后的实际检查记录见 `calibration/VALIDATION.txt`。

代码许可和原作者声明见 `simulator/engine/LICENSE`、`NOTICE.md` 和 `THIRD-PARTY-NOTICES.md`。
