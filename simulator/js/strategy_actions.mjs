// SPDX-License-Identifier: GPL-3.0-or-later
// Upstream-visible candidate constraints for the strategy portion of the full action space.
// The reference transaction revalidates effect feasibility (e.g. empty eligible card pools).
// Passive effects are never manufactured as a player action.
import { source } from './context.mjs';
const { commandView, commandDraftView } = await import('../engine/server/match/bandCommands.js');
const { validateC2S } = await import('../engine/shared/protocol.js');

export function strategyActions(match, player) {
  const actions = [];
  if (
    match.phase === 'BAND_DRAFT' &&
    match.draft &&
    match.draftTurn() === player.playerId &&
    !match.draft.picks[player.playerId]
  ) {
    for (const bandId of match.gd.bandIds()) {
      if (match.gd.bandAllowed(bandId) && !match.bandTaken(bandId, player.playerId)) {
        actions.push({ t: 'g.band', bandId });
      }
    }
  }
  if (match.phase === 'PREP' && !player._gate()) {
    const draft = commandDraftView(player);
    if (draft) {
      for (let idx = 0; idx < draft.cards.length; idx++) {
        actions.push({ t: 'g.commandChoice', choiceId: draft.id, idx });
      }
    }
    const command = commandView(player);
    if (command?.type === 'ACTIVE' && command.usable) {
      for (const target of command.targets) {
        actions.push({ t: 'g.command', effectId: command.effectId, target: { ...target } });
      }
    }
  }
  for (const action of actions) {
    const invalid = validateC2S(action);
    if (invalid) {
      throw new Error(`Adapter constructed invalid protocol action: ${JSON.stringify(invalid)}`);
    }
  }
  return actions;
}

export function strategyObservation(match, player) {
  const pub = match.publicView();
  // Public/own views only: no RNG state, future shop, unopened offers or hidden inventories.
  return {
    version: source.version,
    phase: match.phase,
    round: match.round,
    modeId: match.modeId,
    stageId: pub.stageId,
    bossId: pub.bossId,
    factions: pub.factions,
    disabledBonds: pub.disabledBonds,
    bondChoice: pub.bondChoice ?? null,
    draft: pub.draft ?? null,
    self: player.privateView(),
    actions: strategyActions(match, player),
  };
}

export function applyStrategyAction(match, player, action) {
  const canonical = JSON.stringify(action);
  if (!strategyActions(match, player).some((candidate) => JSON.stringify(candidate) === canonical)) {
    throw new Error('Stale or illegal strategy action; observe again');
  }
  const result = match.handle(player.playerId, action);
  if (!result?.ok) {
    throw new Error(`Reference rejected strategy transaction: ${JSON.stringify(result)}`);
  }
  return result;
}
