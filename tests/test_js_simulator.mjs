// SPDX-License-Identifier: GPL-3.0-or-later
import { data, source, COOP_MODES } from '../simulator/js/context.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { gunzipSync } from 'node:zlib';
import { strategyActions, strategyObservation, applyStrategyAction } from '../simulator/js/strategy_actions.mjs';
import { actionsFor } from '../simulator/js/candidates.mjs';
import { RoomSession } from '../simulator/js/session.mjs';
import { writeExamples } from '../simulator/js/examples.mjs';

const { Match } = await import('../simulator/engine/server/match/Match.js');
const { VirtualScheduler } = await import('../simulator/engine/server/match/scheduler.js');
const { beginCommandPrep, commandView, openSkillChangeDraft } = await import('../simulator/engine/server/match/bandCommands.js');
const { createRegistry } = await import('../simulator/engine/server/match/effectsMeta.js');

function makeSoloPrepMatch(t, seed = 137) {
  const sched = new VirtualScheduler({ instantCombat: true });
  const m = new Match({
    version: source.version,
    data,
    mode: 'solo',
    modeId: 'mode_single_1',
    difficulty: data.config.modes.mode_single_1.difficulty,
    roomCode: 'TEST',
    seed,
    seats: [{ seat: 0, playerId: 'p_0', name: 'Human', isBot: false, connected: true }],
    scheduler: sched,
    registry: createRegistry(),
    battleContent: 'full',
    clientCombat: false,
    botRehearsal: 0,
    send: () => true,
    broadcast() {},
    onEnd() {},
    log: { info() {}, debug() {}, warn() {}, error() {} },
  });
  t.after(() => m.dispose());
  m.start();
  return { m, sched, ps: () => m.players.get('p_0') };
}

function advanceToPhase(m, sched, ps, targetPhase, bandId = null) {
  for (let i = 0; i < 500 && m.phase !== targetPhase; i++) {
    if (m.phase === 'INFO_CHECK' && !ps.infoReady) {
      if (m.bondChoice?.picked == null) m.pickBondGroup(ps, 0, m.setupRevision);
      m.handle(ps.playerId, { t: 'g.infoReady', setupRevision: m.setupRevision });
      continue;
    }
    if (m.phase === 'BAND_DRAFT' && m.draftTurn() === ps.playerId) {
      const id = bandId ?? m.gd.bandIds()[0];
      m.handle(ps.playerId, { t: 'g.band', bandId: id });
      continue;
    }
    if (m.phase === 'CUSTOM_CHECK' && ps.diyBring == null) {
      m.handle(ps.playerId, { t: 'g.diyBring', slots: [] });
      continue;
    }
    if (!sched.runNext()) break;
  }
  assert.equal(m.phase, targetPhase);
}

test('worker CLI rejects actions from another policy version', () => {
  const commands = [
    { op: 'reset', modeId: 'mode_multi_funny', seed: 917, policy_version: 7 },
    { op: 'step', decision_id: 0, action_key: 0, policy_version: 8 },
    { op: 'step', decision_id: 0, action_key: 0, policy_version: 7 },
    { op: 'close' },
  ];
  const run = spawnSync(process.execPath, ['simulator/js/worker.mjs'], {
    input: commands.map((c) => JSON.stringify(c)).join('\n') + '\n',
    encoding: 'utf8',
    windowsHide: true,
  });
  assert.equal(run.status, 0, run.stderr);
  const [initial, rejected, next] = run.stdout.trim().split('\n').map((line) => JSON.parse(line));
  assert.equal(initial.policy_version, 7);
  assert.equal(rejected.ok, false);
  assert.match(rejected.error, /Policy version changed/);
  assert.equal(next.policy_version, 7);
  assert.equal(next.decision_id, 1);
});

test('RoomSession drives a complete four-seat match deterministically', () => {
  const session = new RoomSession();
  try {
    const info = session.hello();
    assert.equal(info.ok, true);
    assert.equal(info.seats, 4);
    assert.deepEqual(info.modes, [...COOP_MODES]);

    let frame = session.reset(1000, 'mode_multi_abyss', 1);
    const order = ['g.infoReady', 'g.band', 'g.diyBring', 'g.choice', 'g.commandChoice', 'g.ready', 'g.sell', 'g.destroy'];
    let steps = 0;
    while (!frame.terminal) {
      assert.ok(steps < 3000, 'Match exceeded step guard');
      let choice = 0;
      for (const t of order) {
        const idx = frame.actions.findIndex((a) => a.t === t);
        if (idx >= 0) { choice = idx; break; }
      }
      frame = session.step({ decision_id: frame.decision_id, action_key: choice, policy_version: 1 });
      assert.equal(frame.previous.accepted, true);
      steps++;
    }
    assert.equal(frame.terminal, true);
    assert.equal(frame.modeId, 'mode_multi_abyss');
    assert.ok(frame.fields >= 1);
    assert.ok(frame.actionsBySeat.every((n) => n > 0));
  } finally {
    session.dispose();
  }
});

test('hand moves include every empty slot and matching summon stack', (t) => {
  const { m, sched, ps } = makeSoloPrepMatch(t);
  const p = ps();
  advanceToPhase(m, sched, p, 'PREP');
  m.stageId = 'act2autochess_m04';
  p._deployMap = null;
  p.board.clear();
  p.hand.fill(null);
  p.recompute();

  const chessId = 'chess_char_3_19_a';
  const taken = m.pool.take(m.gd.baseIdOf(chessId), 1);
  const piece = p.newPiece('chess', chessId, { poolCopies: taken });
  const slot = p.hand.indexOf(null);
  p.hand[slot] = piece;
  p.recompute();

  const play = (action) => m.handle(p.playerId, action);
  const choices = () => actionsFor(m, p, m.publicView(), p.privateView());
  assert.ok(play({ t: 'g.move', uid: piece.uid, to: { area: 'board', row: 9, col: 3 } }).ok);
  const options = choices().filter((a) => a.t === 'g.move' && a.uid === piece.uid && a.to.area === 'hand');
  const empty = p.hand.flatMap((x, i) => (x == null ? [i] : []));
  assert.ok(empty.length > 1);
  for (const s of empty) assert.ok(options.some((a) => a.to.idx === s));

  const stackSlot = p.hand.findIndex((x) => x?.kind === 'token' && x.ownerUid === piece.uid);
  assert.ok(stackSlot >= 0);
  const stack = p.hand[stackSlot];
  const count = stack.count;
  assert.ok(play({ t: 'g.move', uid: stack.uid, to: { area: 'board', row: 9, col: 4 } }).ok);
  const token = p.board.get('9,4');
  const back = choices().find((a) => a.t === 'g.move' && a.uid === token.uid && a.to.area === 'hand' && a.to.idx === stackSlot);
  assert.ok(back && play(back).ok);
  assert.equal(p.hand[stackSlot].count, count);
});

test('opening strategy remains a decision without mutating RNG', (t) => {
  const { m, sched, ps } = makeSoloPrepMatch(t);
  const p = ps();
  advanceToPhase(m, sched, p, 'BAND_DRAFT');
  const choices = strategyActions(m, p);
  assert.ok(choices.length > 1);
  const before = m.rngMeta.state();
  const observation = strategyObservation(m, p);
  assert.equal(observation.version, source.version);
  assert.equal(m.rngMeta.state(), before);
  assert.equal(JSON.stringify(observation).includes('rng_state'), false);
  applyStrategyAction(m, p, choices[0]);
  assert.equal(p.bandId, choices[0].bandId);
  assert.throws(() => applyStrategyAction(m, p, choices[0]), /Stale or illegal/);
});

test('passives have no activation action; active commands obey upstream gates', (t) => {
  const { m, sched, ps } = makeSoloPrepMatch(t);
  const p = ps();
  advanceToPhase(m, sched, p, 'PREP');
  let active = 0;
  let passive = 0;
  for (const band of Object.values(data.bands).filter((b) => b.skills?.length)) {
    p.bandId = band.bandId;
    p.command = null;
    p.funds = 100;
    p.offers = [];
    p.commandDrafts = [];
    p.recompute();
    beginCommandPrep(p);
    const before = JSON.stringify(p.command);
    const rng = m.rngMeta.state();
    const view = commandView(p);
    const actions = strategyActions(m, p);
    assert.equal(JSON.stringify(p.command), before);
    assert.equal(m.rngMeta.state(), rng);
    if (view.type !== 'ACTIVE') {
      passive++;
      assert.equal(actions.length, 0);
    } else if (view.usable) {
      active++;
      assert.deepEqual(actions.map((a) => a.target), view.targets);
      p.funds = -1;
      assert.equal(strategyActions(m, p).length, 0);
    }
  }
  assert.ok(active > 5 && passive > 5);
});

test('cooldown, upgrade, replacement and passive carry-over match upstream rules', (t) => {
  const { m, sched, ps } = makeSoloPrepMatch(t);
  const p = ps();
  advanceToPhase(m, sched, p, 'PREP');

  p.bandId = 'band_jenga';
  p.command = null;
  p.funds = 100;
  p.recompute();
  beginCommandPrep(p);
  assert.equal(commandView(p).reason, 'locked');
  m.round = 4;
  beginCommandPrep(p);
  assert.notEqual(commandView(p).reason, 'locked');
  const effect = p.command.currentEffectId;
  p.command.records[effect].readyRound = 6;
  assert.equal(commandView(p).cooldown, 2);
  assert.equal(strategyActions(m, p).filter((a) => a.t === 'g.command').length, 0);
  assert.equal(openSkillChangeDraft(p, 'skill_change_2'), true);
  const choices = strategyActions(m, p).filter((a) => a.t === 'g.commandChoice');
  assert.equal(choices.length, 2);
  applyStrategyAction(m, p, choices[0]);
  assert.equal(p.commandDrafts.length, 0);
  assert.equal(p.command.records[effect].readyRound, 6);

  p.bandId = 'band_cannot';
  p.command = null;
  p.funds = 6;
  p.recompute();
  beginCommandPrep(p);
  assert.equal(commandView(p).type, 'PASSIVE');
  assert.equal(strategyActions(m, p).length, 0);
  p.endPrep();
  assert.equal(p.funds, 6);
  p.bandId = 'band_price';
  p.command = null;
  p.funds = 6;
  p.recompute();
  beginCommandPrep(p);
  p.endPrep();
  assert.equal(p.funds, 0);
});

test('writeExamples produces valid compressed replay in memory', () => {
  const res = writeExamples(null, 'mode_multi_funny', 42, 64 * 1024 * 1024, 'serial');
  assert.equal(res.ok, true);
  assert.equal(res.terminal, true);
  assert.ok(res.recorded > 20);
  assert.ok(res.replayBytes > 0);
  const lines = res.chunks.map((c) => gunzipSync(c).toString('utf8').trim()).map((s) => JSON.parse(s));
  assert.equal(lines[0].kind, 'reference_expert');
  assert.equal(lines[0].seed, 42);
  assert.equal(lines.at(-1).terminal, true);
  assert.equal(lines.length, res.recorded + 2);
});

test('RoomSession compact mode reports bossId/bossDmgRatio and supports valid bossId override', () => {
  const session = new RoomSession();
  try {
    let frame = session.reset(5555, 'mode_single_15', 0, true, 'boss_17');
    assert.equal(frame.pub.bossId, 'boss_17');
    assert.throws(() => session.reset(5555, 'mode_single_15', 0, true, 'boss_nonexistent'), /Invalid bossId/);

    frame = session.reset(5555, 'mode_single_15', 0, true, 'boss_17');
    const order = [15, 16, 18, 13, 12, 0, 2, 8]; // compact action type codes: infoReady, band, diyBring, choice, cmdChoice, ready, sell, destroy
    let steps = 0;
    while (!frame.terminal) {
      assert.ok(steps < 3000, 'Match exceeded step guard');
      let choice = 0;
      for (const code of order) {
        const idx = frame.cactions.findIndex((a) => a[0] === code);
        if (idx >= 0) { choice = idx; break; }
      }
      frame = session.step({ decision_id: frame.decision_id, action_key: choice, policy_version: 0 });
      steps++;
    }
    assert.equal(frame.bossId, 'boss_17');
    assert.equal(typeof frame.bossDmgRatio, 'number');
    assert.ok(frame.bossDmgRatio >= 0 && frame.bossDmgRatio <= 1);
  } finally {
    session.dispose();
  }
});
