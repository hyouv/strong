// SPDX-License-Identifier: GPL-3.0-or-later
// Offline expert supervision. Independent original Match, RNG and output, never a PPO rollout.
import { mkdirSync, openSync, closeSync, writeSync, renameSync, unlinkSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';
import { isDeepStrictEqual } from 'node:util';
import { pathToFileURL } from 'node:url';
import { config, data, source, ALL_MODES, isSoloMode } from './context.mjs';
import { Match } from '../engine/server/match/Match.js';
import { VirtualScheduler } from '../engine/server/match/scheduler.js';
import { botPrepBeginSteps, botPrepEndSteps, botPickBand, botPickCard } from '../engine/server/match/bot.js';
import { botPickBondGroup } from '../engine/server/match/botBondGroups.js';
import { actionsFor } from './candidates.mjs';

const methods = {
  buy: (slot) => ({ t: 'g.buy', slot }),
  sell: (uid) => ({ t: 'g.sell', uid }),
  refresh: () => ({ t: 'g.refresh' }),
  freeze: () => ({ t: 'g.freeze' }),
  levelUp: () => ({ t: 'g.levelUp' }),
  setReady: (ready) => ({ t: 'g.ready', ready }),
  move: (uid, to, dir) => ({ t: 'g.move', uid, to, ...(to.area === 'board' ? { dir: dir || to.dir || 'RIGHT' } : {}) }),
  equip: (itemUid, targetUid, replaceUid) => ({
    t: 'g.equip',
    itemUid,
    targetUid,
    ...(replaceUid == null ? {} : { replaceUid }),
  }),
  destroy: (uid) => ({ t: 'g.destroy', uid }),
  unequip: (itemUid) => ({ t: 'g.unequip', itemUid }),
  useArt: (itemUid, row, col, dir) => ({ t: 'g.art', itemUid, row, col, dir: dir || 'RIGHT' }),
  pickReward: (idx) => ({ t: 'g.reward', idx }),
  useCommand: (effectId, target) => ({ t: 'g.command', effectId, target }),
  pickCommand: (choiceId, idx) => ({ t: 'g.commandChoice', choiceId, idx }),
};

function* prepareSeat(match, ps, round) {
  const job = yield* botPrepBeginSteps(match, ps);
  if (job) job.run();
  yield* botPrepEndSteps(match, ps, job);
  if (match.phase === 'PREP' && match.round === round && !ps.ready) {
    ps.autoPickCommands();
    match.autoPickPersonalChoice(ps, 'random');
    ps.resolveTemp();
    ps.setReady(true);
  }
}

function prepareRound(match, seats, turnOrder) {
  const round = match.round;
  const jobs = seats.map((ps) => prepareSeat(match, ps, round));
  while (jobs.length && match.phase === 'PREP' && match.round === round) {
    const job = jobs[0];
    if (job.next().done) jobs.shift();
    else if (turnOrder === 'interleaved') jobs.push(jobs.shift());
  }
}

export function writeExamples(directory, modeId, seed, memoryLimit = 0, turnOrder = 'serial') {
  if (!['serial', 'interleaved'].includes(turnOrder)) throw Error('Unknown teacher turn order');
  if (!ALL_MODES.includes(modeId) || !Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw Error('Invalid demonstration request');
  }
  const solo = isSoloMode(modeId);
  const seatCount = solo ? 1 : 4;
  const inMemory = directory === null;
  if (inMemory && (!Number.isSafeInteger(memoryLimit) || memoryLimit <= 0)) {
    throw Error('Invalid replay memory lease');
  }
  if (!inMemory) mkdirSync(directory, { recursive: true });
  const suffix = turnOrder === 'serial' ? '' : '_' + turnOrder;
  const name = inMemory ? null : join(directory, modeId + '_' + seed + suffix + '.jsonl.gz');
  const tmp = inMemory ? null : name + '.' + process.pid + '.tmp';
  const fd = inMemory ? null : openSync(tmp, 'w');
  const chunks = [];
  let replayBytes = 0;
  let detail = null;
  let summary = null;
  let transaction = 0;
  let badField = false;
  let fields = 0;
  let busy = false;
  const errors = [];
  const schedule = new VirtualScheduler({ instantCombat: true });

  const writeRaw = (serializedLine) => {
    const compressed = gzipSync(serializedLine, { level: 1 });
    if (!inMemory) {
      writeSync(fd, compressed);
      return;
    }
    replayBytes += compressed.length;
    if (replayBytes > memoryLimit) throw Error('Compressed teacher replay exceeds its memory lease');
    chunks.push(compressed);
  };
  const write = (row) => writeRaw(JSON.stringify(row) + '\n');

  const match = new Match({
    version: config.version,
    data,
    mode: solo ? 'solo' : 'coop',
    modeId,
    difficulty: data.config.modes[modeId].difficulty,
    roomCode: 'EXPERT',
    seed,
    seats: Array.from({ length: seatCount }, (_, seat) => ({
      seat,
      playerId: 'p' + seat,
      name: 'Seat ' + seat,
      isBot: false,
      connected: true,
    })),
    scheduler: schedule,
    battleContent: 'full',
    clientCombat: false,
    botRehearsal: 0,
    botRehearsalDelegate: false,
    send: (_id, msg) => {
      if (msg.t === 'm.unitStats') detail = msg;
      return true;
    },
    broadcast() {},
    onEnd(s) { summary = s; },
    log: {
      info() {},
      debug() {},
      warn() {},
      error(...x) { errors.push(x.join(' ')); },
    },
  });

  const collect = match._collectSimErrors.bind(match);
  match._collectSimErrors = (field, result) => {
    fields++;
    if (!result?.perPlayer || result.synthetic || result.errors) badField = true;
    return collect(field, result);
  };

  function record(ps, action, doAction) {
    if (busy) return doAction();
    busy = true;
    try {
      const pub = match.publicView();
      const own = ps.privateView();
      const phase = match.phase;
      const actions = actionsFor(match, ps, pub, own);
      detail = null;
      if (phase === 'PREP' || phase === 'SP_DRAFT') match.unitStats(ps, transaction);
      const index = actions.findIndex((a) => a.t === action.t && isDeepStrictEqual(a, action));
      // Serialize the pre-action snapshot directly instead of structuredClone + stringify.
      const serialized =
        JSON.stringify({
          frame: { pub, own, phase, seat: ps.seat, actions, detail, decision_id: transaction },
          action,
          index,
        }) + '\n';
      const result = doAction();
      if (result?.ok) {
        writeRaw(serialized);
        transaction++;
      }
      return result;
    } finally {
      busy = false;
    }
  }

  for (const ps of match.order) {
    for (const [method, encode] of Object.entries(methods)) {
      const original = ps[method].bind(ps);
      ps[method] = (...args) => record(ps, encode(...args), () => original(...args));
    }
  }
  const pickPersonal = match.pickPersonalChoice.bind(match);
  match.pickPersonalChoice = (ps, idx, choiceId) =>
    record(ps, { t: 'g.choice', idx, choiceId }, () => pickPersonal(ps, idx, choiceId));

  function choose(ps, action) {
    const result = record(ps, action, () => match.handle(ps.playerId, action));
    if (!result.ok) throw Error('Expert selection refused: ' + JSON.stringify({ action, result }));
  }

  try {
    const provenance = turnOrder === 'interleaved' ? { turnOrder } : {};
    write({
      kind: 'reference_expert',
      source,
      modeId,
      seed,
      seats: seatCount,
      scope: 'full_match',
      candidateVersion: 2,
      ...provenance,
    });
    match.start();
    const prepared = new Map();
    for (let guard = 0; !match.ended && guard < 200000; guard++) {
      const phase = match.phase;
      if (phase === 'INFO_CHECK') {
        const ps = match.order.find((p) => !p.infoReady);
        if (ps) {
          if (match.bondChoice?.picked === null) {
            choose(ps, {
              t: 'g.bondGroup',
              idx: botPickBondGroup(match, ps),
              setupRevision: match.setupRevision,
            });
          }
          choose(ps, { t: 'g.infoReady', setupRevision: match.setupRevision });
          continue;
        }
      } else if (phase === 'BAND_DRAFT') {
        const ps = match.players.get(match.draftTurn());
        if (ps) {
          let id = botPickBand(match, ps);
          for (let i = 0; i < 8 && match.bandTaken(id, ps.playerId); i++) id = botPickBand(match, ps);
          if (match.bandTaken(id, ps.playerId)) {
            id = match.gd.bandIds().find((k) => match.gd.bandAllowed(k) && !match.bandTaken(k, ps.playerId));
          }
          choose(ps, { t: 'g.band', bandId: id });
          continue;
        }
      } else if (phase === 'CUSTOM_CHECK') {
        const ps = match.order.find((p) => p.diyBring == null);
        if (ps) {
          choose(ps, { t: 'g.diyBring', slots: [] });
          continue;
        }
      } else if (phase === 'SP_DRAFT') {
        const ps = match.players.get(match.spTurn());
        const available = match.sp.cards.filter((c) => match.spCardAvailable(c)).map((c) => c.idx);
        if (ps && available.length) {
          choose(ps, { t: 'g.choice', idx: botPickCard(match, ps, match.sp.cards, available) });
          continue;
        }
      } else if (phase === 'PREP') {
        const seats = match.order.filter(
          (p) => p.alive && !p.ready && prepared.get(p.playerId) !== match.round
        );
        if (seats.length) {
          for (const ps of seats) prepared.set(ps.playerId, match.round);
          prepareRound(match, seats, turnOrder);
          continue;
        }
      }
      if (!schedule.runNext()) throw Error('Expert scheduler stopped');
    }
    if (
      !summary ||
      !match.ended ||
      match.phase !== 'RESULT' ||
      errors.length ||
      match.errorCount ||
      match.simErrors ||
      schedule.errors.length ||
      badField ||
      !fields
    ) {
      throw Error('Invalid expert match: ' + JSON.stringify(errors));
    }
    const result = {
      terminal: true,
      round: match.round,
      roundsPassed: summary.roundsPassed,
      lastRound: match.gd.lastRound,
      victory: !!summary.victory,
      modeId,
      seed,
      fields,
      recorded: transaction,
      ...provenance,
    };
    write(result);
    if (inMemory) return { ok: true, replayBytes, chunks, ...result };
    closeSync(fd);
    renameSync(tmp, name);
    return { ok: true, path: name, ...result };
  } finally {
    match.dispose();
    if (!inMemory && existsSync(tmp)) {
      try { closeSync(fd); } catch {}
      unlinkSync(tmp);
    }
  }
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  console.log(JSON.stringify(writeExamples(process.argv[2], process.argv[3], Number(process.argv[4]))));
}
