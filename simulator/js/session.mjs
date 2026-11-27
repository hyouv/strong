// SPDX-License-Identifier: GPL-3.0-or-later
// Encapsulated four-seat Regulation co-op match session driven by external decisions.
import { config, data, source, COOP_MODES, ALL_MODES, isSoloMode } from './context.mjs';
import { Match } from '../engine/server/match/Match.js';
import { VirtualScheduler } from '../engine/server/match/scheduler.js';
import { actionsFor } from './candidates.mjs';

const MAX_SCHEDULER_STEPS = 200000;
const DIR_TO_IDX = { RIGHT: 0, UP: 1, LEFT: 2, DOWN: 3 };

function compactPiece(p) {
  if (!p) return null;
  const items = p.items;
  const out = {
    uid: p.uid,
    kind: p.kind,
    id: p.id,
    golden: !!p.golden,
    tier: p.tier || 0,
    count: p.count || 1,
  };
  if (p.row != null) out.row = p.row;
  if (p.col != null) out.col = p.col;
  if (p.dir) out.dir = p.dir;
  if (p.ownerUid != null) out.ownerUid = p.ownerUid;
  if (p.progress) out.progress = p.progress;
  if (p.showTag) out.showTag = p.showTag;
  if (p.garrisonTags) out.garrisonTags = p.garrisonTags;
  if (items && items.length) {
    out.items = items.map((it) => ({
      uid: it.uid,
      id: it.id,
      ...(it.progress ? { progress: it.progress } : {}),
      ...(it.showTag ? { showTag: it.showTag } : {}),
    }));
  }
  return out;
}

function compactShopSlot(s) {
  if (!s) return null;
  return {
    id: s.id,
    kind: s.kind,
    sold: !!s.sold,
    frozen: !!s.frozen,
    price: s.price || 0,
  };
}

function compactPub(pub) {
  const bc = pub.bondChoice;
  const sp = pub.sp;
  return {
    round: pub.round,
    lastRound: pub.lastRound,
    bossRound: pub.bossRound,
    hiddenRound: pub.hiddenRound,
    spRound: !!pub.spRound,
    modeId: pub.modeId,
    difficulty: pub.difficulty,
    stageId: pub.stageId,
    factions: pub.factions,
    bossId: pub.bossId,
    hiddenBossId: pub.hiddenBossId,
    disabledBonds: pub.disabledBonds,
    bondChoice: bc ? { picked: bc.picked, groups: bc.groups } : null,
    sp: sp
      ? {
          cards: (sp.cards || []).map((c) => ({
            idx: c.idx,
            id: c.id,
            soldOut: !!c.soldOut,
            tier: c.tier || 0,
          })),
        }
      : null,
  };
}

function compactOwn(own) {
  const shop = own.shop || {};
  const ro = shop.rewardOffer;
  const cmd = own.command;
  const pc = own.personalChoice;
  const cd = own.commandDraft;
  return {
    lp: own.lp,
    alive: own.alive,
    funds: own.funds,
    deployCount: own.deployCount,
    deployCap: own.deployCap,
    stashCount: own.stashCount,
    canReady: !!own.canReady,
    bandId: own.bandId,
    shop: {
      level: shop.level,
      maxLevel: shop.maxLevel,
      upgradePrice: shop.upgradePrice,
      refreshPrice: shop.refreshPrice,
      freeRefreshes: shop.freeRefreshes,
      frozen: !!shop.frozen,
      slots: (shop.slots || []).map(compactShopSlot),
      rewardOffer: ro ? { slots: (ro.slots || []).map(compactShopSlot) } : null,
    },
    command: cmd
      ? {
          effectId: cmd.effectId,
          type: cmd.type,
          usable: !!cmd.usable,
          currentCost: cmd.currentCost || 0,
          cooldown: cmd.cooldown || 0,
          usesLeft: cmd.usesLeft ?? null,
          targetType: cmd.targetType || null,
        }
      : null,
    effects: (own.effects || []).map((e) =>
      e
        ? {
            id: e.id,
            command: !!e.command,
            commandType: e.commandType || null,
            iconKind: e.iconKind || null,
            ...(e.counter != null ? { counter: e.counter } : {}),
            ...(e.progress ? { progress: e.progress } : {}),
            ...(e.roundProgress ? { roundProgress: e.roundProgress } : {}),
          }
        : null
    ),
    board: (own.board || []).map(compactPiece),
    hand: (own.hand || []).map(compactPiece),
    temp: (own.temp || []).map(compactPiece),
    bonds: (own.bonds || []).map((b) => ({
      bondId: b.bondId,
      active: !!b.active,
      count: b.count || 0,
      tier: b.tier || 0,
      layers: b.layers || 0,
      thresholds: b.thresholds,
    })),
    personalChoice: pc
      ? { cards: (pc.cards || []).slice(0, 8).map((c) => ({ id: c.id, tier: c.tier || 0 })) }
      : null,
    commandDraft: cd
      ? { cards: (cd.cards || []).slice(0, 8).map((c) => ({ effectId: c.effectId, cost: c.cost || 0 })) }
      : null,
    nextEnemies: (own.nextEnemies || []).map((e) => ({
      enemyKey: e.enemyKey,
      gate: e.gate,
      count: e.count || 1,
      t: e.t || 0,
      fly: !!e.fly,
      elite: !!e.elite,
      boss: !!e.boss,
      source: e.source || 'wave',
      ...(Array.isArray(e.start) ? { start: e.start } : {}),
    })),
    stats: own.stats
      ? {
          leaks: own.stats.leaks || 0,
          kills: own.stats.kills || 0,
          merges: own.stats.merges || 0,
        }
      : null,
  };
}

function compactDetail(detail) {
  if (!detail || !detail.units) return null;
  return {
    units: detail.units.map((u) => ({
      uid: u.uid,
      maxHp: u.maxHp,
      atk: u.atk,
      def: u.def,
      res: u.res,
      interval: u.interval,
      blockCnt: u.blockCnt,
      rangeLen: Array.isArray(u.range) ? u.range.length : 0,
    })),
  };
}

function compactActions(actions) {
  const n = actions.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const a = actions[i];
    const t = a.t;
    if (t === 'g.move') {
      const to = a.to;
      if (to.area === 'board') {
        out[i] = [6, a.uid, to.row, to.col, DIR_TO_IDX[a.dir || 'RIGHT'] ?? 0];
      } else {
        out[i] = [19, a.uid, to.idx];
      }
    } else if (t === 'g.ready') {
      out[i] = [0];
    } else if (t === 'g.buy') {
      out[i] = [1, a.slot];
    } else if (t === 'g.sell') {
      out[i] = [2, a.uid];
    } else if (t === 'g.refresh') {
      out[i] = [3];
    } else if (t === 'g.freeze') {
      out[i] = [4];
    } else if (t === 'g.levelUp') {
      out[i] = [5];
    } else if (t === 'g.equip') {
      out[i] = [7, a.itemUid, a.targetUid, a.replaceUid == null ? -1 : a.replaceUid];
    } else if (t === 'g.destroy') {
      out[i] = [8, a.uid];
    } else if (t === 'g.reward') {
      out[i] = [9, a.idx];
    } else if (t === 'g.command') {
      const tgt = a.target;
      if (!tgt || Object.keys(tgt).length === 0) out[i] = [10];
      else if ('uid' in tgt) out[i] = [11, 0, tgt.uid];
      else if ('slot' in tgt) out[i] = [11, 1, tgt.slot];
      else if ('row' in tgt && 'col' in tgt) out[i] = [11, 2, tgt.row, tgt.col];
      else throw Error('Unsupported compact g.command target: ' + JSON.stringify(tgt));
    } else if (t === 'g.commandChoice') {
      out[i] = [12, a.idx];
    } else if (t === 'g.choice') {
      out[i] = [13, a.idx];
    } else if (t === 'g.bondGroup') {
      out[i] = [14, a.idx];
    } else if (t === 'g.infoReady') {
      out[i] = [15];
    } else if (t === 'g.band') {
      out[i] = [16, a.bandId];
    } else if (t === 'g.bandSkip') {
      out[i] = [17];
    } else if (t === 'g.diyBring') {
      out[i] = [18];
    } else {
      throw Error('Unsupported compact action type: ' + t);
    }
  }
  return out;
}

export class RoomSession {
  constructor() {
    this.modeId = null;
    this.policyVersion = null;
    this.compact = false;
    this.match = null;
    this.scheduler = null;
    this.summary = null;
    this.frame = null;
    this.cursor = 0;
    this.decision = 0;
    this.failures = 0;
    this.rejected = new Set();
    this.collected = 0;
    this.actionsBySeat = [0, 0, 0, 0];
    this.errors = [];
    this.detail = null;
    this._cachedCompactDetail = null;
    this._cachedStatsRound = -1;
    this._cachedStatsSeat = -1;
  }

  hello() {
    return {
      ok: true,
      pid: process.pid,
      source,
      modes: [...COOP_MODES],
      seats: 4,
    };
  }

  _actor() {
    const match = this.match;
    const phase = match.phase;
    if (phase === 'BAND_DRAFT') return match.players.get(match.draftTurn());
    if (phase === 'SP_DRAFT') {
      return match.sp.cards.some((c) => match.spCardAvailable(c))
        ? match.players.get(match.spTurn())
        : null;
    }
    const seatCount = match.order.length;
    for (let i = 0; i < seatCount; i++) {
      const ps = match.order[(this.cursor + i) % seatCount];
      if (
        (phase === 'INFO_CHECK' && !ps.infoReady) ||
        (phase === 'CUSTOM_CHECK' && ps.diyBring == null) ||
        (phase === 'PREP' && ps.alive && !ps.ready)
      ) {
        return ps;
      }
    }
    return undefined;
  }

  _next(previous = null) {
    const match = this.match;
    const scheduler = this.scheduler;
    for (let i = 0; i < MAX_SCHEDULER_STEPS; i++) {
      if (
        match.errorCount ||
        match.simErrors ||
        scheduler.errors.length ||
        this.failures ||
        this.errors.length
      ) {
        throw Error(
          JSON.stringify({
            match: match.errors,
            sim: match.simErrors,
            scheduler: scheduler.errors,
            failures: this.failures,
            errors: this.errors,
          })
        );
      }
      if (match.ended) {
        if (
          !this.summary ||
          match.phase !== 'RESULT' ||
          !this.collected ||
          !this.actionsBySeat.every((n) => n > 0)
        ) {
          throw Error('Incomplete match');
        }
        this.frame = null;
        const out = {
          ok: true,
          terminal: true,
          modeId: this.modeId,
          difficulty: match.difficulty,
          victory: !!this.summary.victory,
          round: match.round,
          roundsPassed: this.summary.roundsPassed,
          lastRound: match.gd.lastRound,
          actionsBySeat: this.actionsBySeat,
          fields: this.collected,
          previous,
        };
        if (this.compact) {
          out.bossId = match.bossId ?? null;
          let bossDmgRatio = 0;
          if (out.victory) {
            bossDmgRatio = 1;
          } else if (
            match.round >= match.gd.bossRound &&
            match.bossPool &&
            match.bossPool.maxHp > 0
          ) {
            bossDmgRatio = Math.max(0, Math.min(1, 1 - match.bossPool.hp / match.bossPool.maxHp));
          }
          out.bossDmgRatio = bossDmgRatio;
        }
        return out;
      }
      const ps = this._actor();
      if (!ps) {
        if (!scheduler.runNext()) {
          throw Error('No pending decision or scheduler callback: ' + match.phase);
        }
        continue;
      }
      if (ps.botControlled) throw Error('Policy seat unexpectedly automated');
      const pub = match.publicView();
      const own = ps.privateView();
      const phase = match.phase;
      let actions = actionsFor(match, ps, pub, own);
      if (this.rejected.size > 0) {
        actions = actions.filter((a) => !this.rejected.has(JSON.stringify(a)));
      }
      if (!actions.length) throw Error('No action for ' + phase + ' ' + ps.playerId);
      this.detail = null;
      let wireDetail = null;
      if (phase === 'SP_DRAFT' || phase === 'PREP') {
        if (this.compact) {
          if (ps.board.size === 0) {
            wireDetail = { units: [] };
            this._cachedCompactDetail = wireDetail;
            this._cachedStatsRound = match.round;
            this._cachedStatsSeat = ps.seat;
          } else if (
            previous &&
            (!previous.accepted || previous.action?.t === 'g.freeze') &&
            this._cachedCompactDetail !== null &&
            this._cachedStatsRound === match.round &&
            this._cachedStatsSeat === ps.seat
          ) {
            wireDetail = this._cachedCompactDetail;
          } else {
            const r = match.unitStats(ps, this.decision);
            if (!r.ok || this.detail?.seq !== this.decision) {
              throw Error('Unit stats unavailable');
            }
            wireDetail = compactDetail(this.detail);
            this._cachedCompactDetail = wireDetail;
            this._cachedStatsRound = match.round;
            this._cachedStatsSeat = ps.seat;
          }
        } else {
          const r = match.unitStats(ps, this.decision);
          if (!r.ok || this.detail?.seq !== this.decision) {
            throw Error('Unit stats unavailable');
          }
          wireDetail = this.detail;
        }
      }
      this.frame = {
        ok: true,
        decision_id: this.decision,
        seat: ps.seat,
        phase,
        pub,
        own,
        detail: this.detail,
        actions,
      };
      if (this.compact) {
        return {
          ok: true,
          decision_id: this.decision,
          seat: ps.seat,
          phase,
          pub: compactPub(pub),
          own: compactOwn(own),
          detail: wireDetail,
          cactions: compactActions(actions),
        };
      }
      return { ...this.frame, previous };
    }
    throw Error('Scheduler failed to reach next decision');
  }

  reset(seed, requestedMode, version = null, compact = false, bossId = null) {
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw Error('Invalid seed');
    if (!ALL_MODES.includes(requestedMode)) throw Error('Invalid mode');
    if (version !== null && (!Number.isSafeInteger(version) || version < 0)) {
      throw Error('Invalid policy version');
    }
    const solo = isSoloMode(requestedMode);
    const seatCount = solo ? 1 : 4;
    this.policyVersion = version;
    this.compact = !!compact;
    this.modeId = requestedMode;
    this.match?.dispose();
    this.cursor = 0;
    this.decision = 0;
    this.failures = 0;
    this.collected = 0;
    this.errors = [];
    this.actionsBySeat = Array(seatCount).fill(0);
    this.summary = null;
    this.frame = null;
    this.detail = null;
    this._cachedCompactDetail = null;
    this._cachedStatsRound = -1;
    this._cachedStatsSeat = -1;
    this.rejected = new Set();

    this.scheduler = new VirtualScheduler({ instantCombat: true });
    this.match = new Match({
      version: config.version,
      data,
      mode: solo ? 'solo' : 'coop',
      modeId: this.modeId,
      difficulty: data.config.modes[this.modeId].difficulty,
      roomCode: 'POLICY',
      seed,
      seats: Array.from({ length: seatCount }, (_, seat) => ({
        seat,
        playerId: 'p' + seat,
        name: 'Seat ' + seat,
        isBot: false,
        connected: true,
      })),
      scheduler: this.scheduler,
      battleContent: 'full',
      clientCombat: false,
      botRehearsal: 0,
      botRehearsalDelegate: false,
      send: (_pid, msg) => {
        if (msg.t === 'm.unitStats') this.detail = msg;
        return true;
      },
      broadcast() {},
      onEnd: (s) => {
        this.summary = s;
      },
      log: {
        info() {},
        debug() {},
        warn() {},
        error: (...x) => {
          this.errors.push(x.map(String).join(' '));
        },
      },
    });
    const collect = this.match._collectSimErrors.bind(this.match);
    this.match._collectSimErrors = (f, r) => {
      this.collected++;
      if (!r?.perPlayer || r.synthetic || r.errors) this.failures++;
      return collect(f, r);
    };
    if (bossId != null && bossId !== '') {
      const validBosses = this.match.gd.bossWeights(false).map(([id]) => id);
      if (!validBosses.includes(bossId)) {
        throw Error(`Invalid bossId ${bossId} for mode ${this.modeId}`);
      }
      this.match.bossId = bossId;
    }
    this.match.start();
    return this._next();
  }

  step(msg) {
    if ((msg.policy_version ?? null) !== this.policyVersion) {
      throw Error('Policy version changed inside a match');
    }
    if (
      !this.frame ||
      msg.decision_id !== this.frame.decision_id ||
      !Number.isInteger(msg.action_key) ||
      !this.frame.actions[msg.action_key]
    ) {
      throw Error('Stale or invalid decision');
    }
    const { seat, actions } = this.frame;
    const action = actions[msg.action_key];
    const result = this.match.handle('p' + seat, action);
    const previous = { seat, action, accepted: !!result?.ok, code: result?.code ?? null };
    if (result?.ok) {
      this.actionsBySeat[seat]++;
      this.cursor = (seat + 1) % this.match.order.length;
      this.rejected.clear();
    } else {
      // Some content handlers validate an effect only upon use. The real refusal is an observed transition.
      this.cursor = seat;
      this.rejected.add(JSON.stringify(action));
    }
    this.decision++;
    return this._next(previous);
  }

  dispose() {
    this.match?.dispose();
    this.match = null;
    this.scheduler = null;
    this.frame = null;
  }
}
