// SPDX-License-Identifier: GPL-3.0-or-later
import test from 'node:test';
import assert from 'node:assert/strict';
import { gunzipSync } from 'node:zlib';
import { isDeepStrictEqual } from 'node:util';
import { RoomSession } from '../simulator/js/session.mjs';
import { writeExamples } from '../simulator/js/examples.mjs';

// Real expert formations exercise combat, summons, equipment and command changes.
// Compare every wire field, ordered action and full battle receipt, not just win rates.
for (const mode of ['mode_single_8', 'mode_multi_funny']) {
  test(`headless output preserves complete ${mode} trajectories`, () => {
    const seed = 71000;
    // Expert demonstrations act serially by seat; RoomSession rotates seats.
    // Solo can replay an expert trace, while co-op uses a legal frame-driven policy.
    let actions = null;
    if (mode.startsWith('mode_single_')) {
      const demo = writeExamples(null, mode, seed, 256 * 1024 * 1024);
      const rows = gunzipSync(Buffer.concat(demo.chunks)).toString().trim().split('\n').map(JSON.parse);
      actions = rows.filter(r => r.frame).map(r => r.action);
    }
    for (const compact of [false, true]) {
      const reference = new RoomSession({ headless: false });
      const fast = new RoomSession();
      try {
        let expected = reference.reset(seed, mode, null, compact);
        let actual = fast.reset(seed, mode, null, compact);
        const receipts = [[], []];
        for (const [i, session] of [reference, fast].entries()) {
          const collect = session.match._collectSimErrors;
          session.match._collectSimErrors = function (field, result) {
            receipts[i].push(structuredClone(result));
            return collect.call(this, field, result);
          };
        }
        let steps = 0;
        const prepCounts = new Map();
        while (true) {
          assert.equal(JSON.stringify(actual), JSON.stringify(expected), `wire frame ${steps}`);
          assert.deepEqual(fast.match.rngMeta.state(), reference.match.rngMeta.state());
          assert.deepEqual(fast.frame?.actions, reference.frame?.actions);
          if (expected.terminal) break;
          assert.ok(steps < 3000);
          const frame = reference.frame;
          let key;
          if (actions) {
            const action = actions[steps];
            key = frame.actions.findIndex(a => a.t === action?.t && isDeepStrictEqual(a, action));
            assert.ok(key >= 0, `expert action ${steps} remains legal`);
          } else {
            const turn = `${frame.pub.round}:${frame.seat}`;
            const count = (prepCounts.get(turn) || 0) + 1;
            prepCounts.set(turn, count);
            const board = new Set(frame.own.board.map(p => p.uid));
            const preferred = ['g.choice', 'g.commandChoice', 'g.infoReady', 'g.band', 'g.diyBring'];
            if (count >= 16) preferred.push('g.ready', 'g.sell', 'g.destroy');
            key = frame.actions.findIndex(a => preferred.includes(a.t));
            if (key < 0) key = frame.actions.findIndex(a => a.t === 'g.move' && a.to.area === 'board' && !board.has(a.uid));
            if (key < 0) key = frame.actions.findIndex(a => ['g.buy', 'g.equip', 'g.levelUp', 'g.ready', 'g.destroy'].includes(a.t));
            if (key < 0) key = 0;
          }
          const request = { decision_id: expected.decision_id, action_key: key };
          expected = reference.step(request);
          actual = fast.step(request);
          steps++;
        }
        if (actions) assert.equal(steps, actions.length);
        assert.ok(receipts[0].length > 0);
        assert.deepEqual(receipts[1], receipts[0]);
      } finally {
        reference.dispose();
        fast.dispose();
      }
    }
  });
}
