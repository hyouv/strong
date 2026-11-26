// SPDX-License-Identifier: GPL-3.0-or-later
// Encapsulated four-seat Regulation co-op match session driven by external decisions.
import { config, data, source, COOP_MODES } from './context.mjs';
import { Match } from '../engine/server/match/Match.js';
import { VirtualScheduler } from '../engine/server/match/scheduler.js';
import { actionsFor } from './candidates.mjs';

const MAX_SCHEDULER_STEPS = 200000;

export class RoomSession {
  constructor() {
    this.modeId = null;
    this.policyVersion = null;
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
    for (let i = 0; i < 4; i++) {
      const ps = match.order[(this.cursor + i) % 4];
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
        return {
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
      if (phase === 'SP_DRAFT' || phase === 'PREP') {
        const r = match.unitStats(ps, this.decision);
        if (!r.ok || this.detail?.seq !== this.decision) {
          throw Error('Unit stats unavailable');
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
      return { ...this.frame, previous };
    }
    throw Error('Scheduler failed to reach next decision');
  }

  reset(seed, requestedMode, version = null) {
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw Error('Invalid seed');
    if (!COOP_MODES.includes(requestedMode)) throw Error('Invalid cooperative mode');
    if (version !== null && (!Number.isSafeInteger(version) || version < 0)) {
      throw Error('Invalid policy version');
    }
    this.policyVersion = version;
    this.modeId = requestedMode;
    this.match?.dispose();
    this.cursor = 0;
    this.decision = 0;
    this.failures = 0;
    this.collected = 0;
    this.errors = [];
    this.actionsBySeat = [0, 0, 0, 0];
    this.summary = null;
    this.frame = null;
    this.detail = null;
    this.rejected = new Set();

    this.scheduler = new VirtualScheduler({ instantCombat: true });
    this.match = new Match({
      version: config.version,
      data,
      mode: 'coop',
      modeId: this.modeId,
      difficulty: data.config.modes[this.modeId].difficulty,
      roomCode: 'POLICY',
      seed,
      seats: Array.from({ length: 4 }, (_, seat) => ({
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
      this.cursor = (seat + 1) % 4;
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
