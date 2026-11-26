// SPDX-License-Identifier: GPL-3.0-or-later
// Must be imported before any upstream match/simulation modules so the pinned
// act3autochess@V3_1 dataset is installed as the process-wide singleton.
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { getData, getVersionData, setData } from '../engine/server/data.js';
import { rememberOfficialBounds } from '../engine/shared/plugins/officialBounds.js';

export const ROOT = path.resolve(fileURLToPath(new URL('../../', import.meta.url)));
export const ENGINE_ROOT = path.join(ROOT, 'simulator', 'engine');

export const config = JSON.parse(readFileSync(path.join(ROOT, 'config', 'regulation.json'), 'utf8'));
const engineMeta = JSON.parse(readFileSync(path.join(ENGINE_ROOT, 'package.json'), 'utf8'));

export const revision = engineMeta?.upstream?.revision ?? '';
if (revision !== config.source.revision) {
  throw new Error(`Engine revision mismatch: ${revision}`);
}
if (config.version !== 'act3autochess@V3_1') {
  throw new Error('This project targets Regulation V3_1 exclusively');
}

export const COOP_MODES = Object.freeze([
  'mode_multi_funny',
  'mode_multi_normal',
  'mode_multi_hard',
  'mode_multi_abyss',
]);

export const SOLO_MODES = Object.freeze([
  'mode_single_funny',
  ...Array.from({ length: 20 }, (_, i) => `mode_single_${i + 1}`),
]);

export const ALL_MODES = Object.freeze([...COOP_MODES, ...SOLO_MODES]);

export function isSoloMode(modeId) {
  return SOLO_MODES.includes(modeId);
}

const dataErrors = [];
export const data = getVersionData(config.version, {
  log: {
    info() {},
    warn(...args) { dataErrors.push(args.join(' ')); },
    error(...args) { dataErrors.push(args.join(' ')); },
  },
});

if (data.config?.version?.id !== config.version || dataErrors.length) {
  throw new Error(`Regulation data missing: ${dataErrors.join('; ')}`);
}
setData(data);
rememberOfficialBounds(data);

if (getData() !== data || ALL_MODES.some((id) => !data.config.modes[id])) {
  throw new Error('Wrong rules');
}

export const source = Object.freeze({
  ...config.source,
  version: config.version,
  ruleset: 'stronghold-regulation-v3_1',
});

export function writeJson(relative, value) {
  const destination = path.resolve(ROOT, relative);
  mkdirSync(path.dirname(destination), { recursive: true });
  writeFileSync(destination, JSON.stringify(value, null, 2) + '\n');
}
