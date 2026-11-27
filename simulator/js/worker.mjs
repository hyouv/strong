// SPDX-License-Identifier: GPL-3.0-or-later
// NDJSON stdio worker process owning one independent four-seat Regulation Match at a time.
import readline from 'node:readline';
import { once } from 'node:events';
import { RoomSession } from './session.mjs';
import { writeExamples } from './examples.mjs';

const session = new RoomSession();
const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

try {
  for await (const line of input) {
    if (!line) continue;
    try {
      const q = JSON.parse(line);
      if (q.op === 'close') break;

      let reply = null;
      if (q.op === 'reset') {
        reply = session.reset(q.seed, q.modeId, q.policy_version, !!q.compact, q.bossId ?? null);
        reply.policy_version = session.policyVersion;
      } else if (q.op === 'step') {
        reply = session.step(q);
        reply.policy_version = session.policyVersion;
      } else if (q.op === 'examples') {
        reply = writeExamples(q.directory, q.modeId, q.seed, 0, q.turn_order);
      } else if (q.op === 'examples_memory') {
        reply = writeExamples(null, q.modeId, q.seed, q.memory_limit, q.turn_order);
      } else if (q.op === 'hello') {
        reply = session.hello();
      } else {
        throw Error('Unknown operation');
      }

      const chunks = reply?.chunks;
      if (chunks) delete reply.chunks;
      process.stdout.write(JSON.stringify(reply) + '\n');
      if (chunks) {
        for (let i = 0; i < chunks.length; i++) {
          if (!process.stdout.write(chunks[i])) await once(process.stdout, 'drain');
          chunks[i] = null;
        }
      }
    } catch (error) {
      process.stdout.write(JSON.stringify({ ok: false, error: String(error.stack || error) }) + '\n');
    }
  }
} finally {
  session.dispose();
}
