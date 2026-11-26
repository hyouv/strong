import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { getVersionData, setData } from './simulator/engine/server/data.js';

const metadata = JSON.parse(readFileSync(new URL('./simulator/engine/package.json', import.meta.url), 'utf8'));
assert.equal(metadata.calibration.id, 'stronghold-calibrated-engine-20261126');
const errors = [];
const data = getVersionData('act3autochess@V3_1', {
  log: { info() {}, warn(message) { errors.push(message); }, error(message) { errors.push(message); } },
});
assert.deepEqual(errors, []);
setData(data);
const { GameData } = await import('./simulator/engine/server/match/gamedata.js');
const observed = Object.fromEntries(Object.entries(data.config.modes)
  .map(([mode, entry]) => [mode, Object.entries(entry.enemyScale).filter(([, scale]) => scale.observed).map(([round]) => Number(round))])
  .filter(([, rounds]) => rounds.length));
assert.deepEqual(observed.mode_multi_hard, Array.from({ length: 15 }, (_, i) => i + 1));
assert.deepEqual(observed.mode_multi_abyss, Array.from({ length: 14 }, (_, i) => i + 1));
assert.equal(Object.values(observed).reduce((count, rounds) => count + rounds.length, 0), 43);
const hard = new GameData(data, 'mode_multi_hard');
const abyss = new GameData(data, 'mode_multi_abyss');
assert.equal(hard.enemyScale(14).hpMul, 9.52542);
assert.equal(abyss.enemyScale(14).hpMul, 12.280707);
assert.equal(abyss.enemyScale(14).aspdAdd, 30);
console.log(JSON.stringify({ ok: true, package: metadata.calibration.id, observed, hard_round_14: hard.enemyScale(14), abyss_round_14: abyss.enemyScale(14) }, null, 2));
