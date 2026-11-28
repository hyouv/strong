// SPDX-License-Identifier: GPL-3.0-or-later
// These sessions consume decision frames and unitStats, never client render messages.
// Keep the scheduler, public-message throttling and all game hooks unchanged.
export function disableClientOutput(match) {
  match._sendPrivate = () => {};
  match._sendField = () => {};
  match.watchersOf = () => [];
  const newBattle = match.newBattle;
  match.newBattle = function (opts, ...args) {
    return newBattle.call(this, { ...opts, recordEvents: false }, ...args);
  };
}
