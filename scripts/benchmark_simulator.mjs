// SPDX-License-Identifier: GPL-3.0-or-later
// Fixed expert-action replays: measure the environment without policy inference.
import { readFileSync, writeFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import { performance } from 'node:perf_hooks';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { RoomSession } from '../simulator/js/session.mjs';
import { writeExamples } from '../simulator/js/examples.mjs';
import { Match } from '../simulator/engine/server/match/Match.js';
import { PlayerState } from '../simulator/engine/server/match/PlayerState.js';
import { VirtualScheduler } from '../simulator/engine/server/match/scheduler.js';

const args = process.argv.slice(2);
if (args.includes('--help')) {
  console.log(`Generate: node scripts/benchmark_simulator.mjs --generate --traces traces.json
Replay:   node scripts/benchmark_simulator.mjs --traces traces.json --output report.json
Compare:  node scripts/benchmark_simulator.mjs --traces traces.json --baseline-root /old/checkout
Options: --modes 4,8,14,20 --seeds 1 --seed-start 71000 --warmup 1 --repeats 3
         --full (default: compact) --client-output --timings (inclusive, overlapping)
Policy inference, Python IPC and trace generation are excluded from replay timings.
wallMs includes action lookup/hashing; stepMs measures RoomSession.step only.`);
  process.exit(0);
}
function option(name, fallback) {
  const i = args.indexOf(name);
  return i < 0 ? fallback : args[i + 1];
}
const tracePath = option('--traces', null);
const repeats = Number(option('--repeats', 3));
const warmup = Number(option('--warmup', 1));
const compact = !args.includes('--full');
const baselineRoot = option('--baseline-root', null);
if (baselineRoot && args.includes('--timings')) throw Error('Use --timings separately from paired comparisons');
const BaselineSession = baselineRoot
  ? (await import(pathToFileURL(resolve(baselineRoot, 'simulator/js/session.mjs')).href)).RoomSession
  : null;
if (!Number.isInteger(repeats) || repeats < 1 || !Number.isInteger(warmup) || warmup < 0) {
  throw Error('--repeats must be positive and --warmup nonnegative integers');
}
let traces;
if (args.includes('--generate')) {
  const modes = option('--modes', '4,8,14,20').split(',');
  const seeds = Number(option('--seeds', 1));
  traces = [];
  for (const mode of modes) for (let i = 0; i < seeds; i++) {
    const modeId = mode.startsWith('mode_') ? mode : `mode_single_${mode}`;
    if (!modeId.startsWith('mode_single_')) throw Error('Expert replay requires solo modes (co-op rotates seats differently)');
    const seed = Number(option('--seed-start', 71000)) + i;
    const demo = writeExamples(null, modeId, seed, 256 * 1024 * 1024);
    const rows = gunzipSync(Buffer.concat(demo.chunks)).toString().trim().split('\n').map(JSON.parse);
    traces.push({ modeId, seed, actions: rows.filter(r => r.frame).map(r => r.action) });
    console.error(`Generated ${modeId}/${seed}: ${traces.at(-1).actions.length} actions`);
  }
  if (tracePath) writeFileSync(tracePath, JSON.stringify(traces));
} else {
  if (!tracePath) throw Error('Use --generate --traces FILE first, then --traces FILE for replay');
  traces = JSON.parse(readFileSync(tracePath, 'utf8'));
}

const timings = {};
let measuring = false;
function instrument(proto, name, label = name) {
  const original = proto[name];
  proto[name] = function (...params) {
    if (!measuring) return original.apply(this, params);
    const t = performance.now();
    try { return original.apply(this, params); }
    finally {
      const stat = timings[label] ??= { calls: 0, ms: 0 };
      stat.calls++;
      stat.ms += performance.now() - t;
    }
  };
}
if (args.includes('--timings')) {
  for (const name of ['handle', 'flush', 'publicView', '_unitStatsOf']) instrument(Match.prototype, name);
  instrument(PlayerState.prototype, 'privateView');
  instrument(VirtualScheduler.prototype, 'runNext');
}

function replay(trace, measure, baseline = false) {
  const Session = baseline ? BaselineSession : RoomSession;
  const session = new Session({ headless: !baseline && !args.includes('--client-output') });
  const hash = createHash('sha256');
  let bytes = 0, serializeMs = 0, stepMs = 0, resetMs = 0;
  const totalStart = performance.now();
  const cpuStart = process.cpuUsage();
  measuring = measure;
  try {
    let t = performance.now();
    let frame = session.reset(trace.seed, trace.modeId, null, compact);
    resetMs = performance.now() - t;
    for (let i = 0; ; i++) {
      t = performance.now();
      const json = JSON.stringify(frame);
      serializeMs += performance.now() - t;
      bytes += Buffer.byteLength(json);
      hash.update(json);
      if (frame.terminal) {
        if (i !== trace.actions.length) throw Error(`Early terminal: ${i}/${trace.actions.length}`);
        const cpu = process.cpuUsage(cpuStart);
        return { variant: baseline ? 'baseline' : 'current', mode: trace.modeId, seed: trace.seed, steps: i, round: frame.round,
          victory: frame.victory, resetMs, stepMs, serializeMs, bytes,
          wallMs: performance.now() - totalStart, cpuMs: (cpu.user + cpu.system) / 1000,
          digest: hash.digest('hex') };
      }
      const action = trace.actions[i];
      const key = session.frame.actions.findIndex(a => a.t === action?.t && isDeepStrictEqual(a, action));
      if (key < 0) throw Error(`Replay divergence ${trace.modeId}/${trace.seed} step ${i}: ${JSON.stringify(action)}`);
      t = performance.now();
      frame = session.step({ decision_id: frame.decision_id, action_key: key });
      stepMs += performance.now() - t;
    }
  } finally { session.dispose(); measuring = false; }
}

for (let i = 0; i < warmup; i++) for (const trace of traces) {
  if (BaselineSession) replay(trace, false, true);
  replay(trace, false);
}
const results = [];
for (let i = 0; i < repeats; i++) for (const trace of traces) {
  if (BaselineSession) {
    // Alternate order to reduce bias from CPU load, frequency and JIT warmup.
    const pair = (i % 2 ? [false, true] : [true, false]).map(baseline => replay(trace, true, baseline));
    if (pair[0].digest !== pair[1].digest) throw Error(`Frame mismatch: ${trace.modeId}/${trace.seed}`);
    results.push(...pair);
  } else results.push(replay(trace, true));
}
const report = { node: process.version, compact, repeats, warmup, baselineRoot, timings, results };
const output = option('--output', null);
if (output) writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
