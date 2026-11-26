// SPDX-License-Identifier: GPL-3.0-or-later
// Original-rule candidate projection used by live and demonstration rooms.
// Preserves exact action contents and deterministic enumeration order.
import { strategyActions } from './strategy_actions.mjs';
import { FIELD, tileKey, parseKey, pieceDir } from '../engine/server/match/board.js';
import { itemForbidsDiy, chessIsDiy } from '../engine/shared/itemRules.js';

const DIRECTIONS = Object.freeze(['RIGHT', 'UP', 'LEFT', 'DOWN']);

function moves(ps, piece, boardUids, out) {
  const loc = ps.find(piece.uid);
  if (piece.kind === 'item') return;
  const board = loc.area === 'board';
  let srcRow = 0;
  let srcCol = 0;
  if (board) {
    [srcRow, srcCol] = parseKey(loc.key);
  }

  if (board || loc.area === 'temp') {
    const hand = ps.hand;
    for (let idx = 0; idx < hand.length; idx++) {
      const occupant = hand[idx];
      // PlayerPlacement._moveToHand accepts any empty slot. liftTokens also requests
      // the matching stack by its exact slot, including when the hand is full.
      const ownStack =
        board &&
        occupant?.kind === 'token' &&
        (piece.kind === 'token'
          ? occupant.ownerUid === piece.ownerUid && occupant.id === piece.id
          : occupant.ownerUid === piece.uid);
      const swap =
        board &&
        piece.kind === 'chess' &&
        occupant?.kind === 'chess' &&
        ps._legal(occupant, srcRow, srcCol);
      if (occupant == null || ownStack || swap) {
        out.push({ t: 'g.move', uid: piece.uid, to: { area: 'hand', idx } });
      }
    }
  }

  if (!board && piece.kind === 'token' && !boardUids.has(piece.ownerUid)) return;

  const currentDir = board ? pieceDir(piece) : null;
  const atDeployCap = !board && piece.kind === 'chess' && ps.deployCount >= ps.deployCap;

  for (let row = FIELD.r0; row <= FIELD.r1; row++) {
    for (let col = FIELD.c0; col <= FIELD.c1; col++) {
      const key = tileKey(row, col);
      const occ = ps.board.get(key);
      const owner =
        board && occ && occ.uid === piece.ownerUid
          ? { key: loc.key, piece: occ, dir: pieceDir(occ) }
          : null;
      if (!ps._legal(piece, row, col, owner)) continue;
      if (
        board &&
        occ &&
        occ !== piece &&
        !(piece.kind === 'chess' && occ.ownerUid === piece.uid) &&
        !ps._legal(occ, srcRow, srcCol)
      ) {
        continue;
      }
      if (!board && ((piece.kind === 'token' && occ) || (atDeployCap && occ?.kind !== 'chess'))) {
        continue;
      }
      const sameTile = occ === piece;
      for (let d = 0; d < DIRECTIONS.length; d++) {
        const dir = DIRECTIONS[d];
        if (
          sameTile &&
          (dir === currentDir || (piece.kind === 'chess' && !ps._roomForOutOfRange(piece, key, dir)))
        ) {
          continue;
        }
        out.push({ t: 'g.move', uid: piece.uid, to: { area: 'board', row, col }, dir });
      }
    }
  }
}

function prep(match, ps, own) {
  const out = [...strategyActions(match, ps)];
  if (own.personalChoice) {
    for (let idx = 0; idx < own.personalChoice.cards.length; idx++) {
      out.push({ t: 'g.choice', idx, choiceId: own.personalChoice.id });
    }
  }
  if (own.canReady) out.push({ t: 'g.ready', ready: true });
  if (own.funds >= own.shop.refreshPrice || ps.shop.freeRefreshes > 0) out.push({ t: 'g.refresh' });
  out.push({ t: 'g.freeze' });
  if (ps.shop.level < ps.gd.maxShopLevel && ps.funds >= ps.shop.upgradePrice) out.push({ t: 'g.levelUp' });

  const hasFreeHand = !ps.equipHand ? ps.hand.some((x) => x == null) : false;
  const purchasable = (s) => {
    if (!s || s.sold || s.price > ps.funds) return false;
    if (ps.equipHand ? ps.handSlot(s.kind) < 0 : !hasFreeHand) return false;
    if (s.kind !== 'chess') return true;
    const rec = ps.gd.chess(s.id);
    return !match.chessExcluded(rec) && (!rec?.isDiy || !!rec?.diyFor);
  };

  for (let slot = 0; slot < own.shop.slots.length; slot++) {
    if (purchasable(own.shop.slots[slot])) out.push({ t: 'g.buy', slot });
  }
  const rewardSlots = own.shop.rewardOffer?.slots;
  if (rewardSlots) {
    for (let idx = 0; idx < rewardSlots.length; idx++) {
      if (purchasable(rewardSlots[idx])) out.push({ t: 'g.reward', idx });
    }
  }

  const pieces = [...ps.board.values(), ...ps.hand, ...ps.temp.slice(0, own.temp.length)].filter(Boolean);
  const boardUids = new Set();
  for (const p of ps.board.values()) boardUids.add(p.uid);
  const chessPieces = pieces.filter((x) => x.kind === 'chess');
  const benchEquipDisabled = ps.gd.config.version?.rules?.benchEquip === false;

  for (let i = 0; i < pieces.length; i++) {
    const piece = pieces[i];
    if (piece.kind !== 'item') {
      if (piece.kind === 'chess') out.push({ t: 'g.sell', uid: piece.uid });
      moves(ps, piece, boardUids, out);
      continue;
    }
    out.push({ t: 'g.destroy', uid: piece.uid });
    const rec = ps.gd.item(piece.id);
    if (rec?.itemType === 'EQUIP') {
      const policy = ps._slotPolicy();
      const forbidsDiy = itemForbidsDiy(rec);
      const occupies = policy.occupies(rec);
      for (let j = 0; j < chessPieces.length; j++) {
        const target = chessPieces[j];
        if (benchEquipDisabled && ps.find(target.uid).area !== 'board') continue;
        if (forbidsDiy && chessIsDiy(ps.gd.chess(target.id))) continue;
        const items = target.items || [];
        if (!policy.accepts(items, rec)) continue;
        const needReplace = occupies && policy.slotsUsed(items) >= policy.slotCap(ps.gd.equipSlots(target.id));
        if (needReplace) {
          for (let k = 0; k < items.length; k++) {
            out.push({ t: 'g.equip', itemUid: piece.uid, targetUid: target.uid, replaceUid: items[k].uid });
          }
        } else {
          out.push({ t: 'g.equip', itemUid: piece.uid, targetUid: target.uid });
        }
      }
    } else if (rec?.itemType === 'MAGIC' && ps.round.arts < ps.gd.maxArtsPerRound) {
      for (let row = FIELD.r0; row <= FIELD.r1; row++) {
        for (let col = FIELD.c0; col <= FIELD.c1; col++) {
          for (let d = 0; d < DIRECTIONS.length; d++) {
            out.push({ t: 'g.art', itemUid: piece.uid, row, col, dir: DIRECTIONS[d] });
          }
        }
      }
    }
  }
  return out;
}

export function actionsFor(match, ps, pub, own) {
  const phase = match.phase;
  if (phase === 'INFO_CHECK') {
    const actions =
      pub.bondChoice && pub.bondChoice.picked === null
        ? pub.bondChoice.groups.map((_, idx) => ({ t: 'g.bondGroup', idx, setupRevision: pub.setupRevision }))
        : [];
    actions.push({ t: 'g.infoReady', setupRevision: pub.setupRevision });
    return actions;
  }
  if (phase === 'BAND_DRAFT') {
    const actions = strategyActions(match, ps);
    if (match.draft.skipsLeft[ps.playerId] > 0) actions.push({ t: 'g.bandSkip' });
    return actions;
  }
  if (phase === 'CUSTOM_CHECK') {
    if (Object.keys(own.diy).length) throw Error('Unexpected custom lobby roster');
    return [{ t: 'g.diyBring', slots: [] }];
  }
  if (phase === 'SP_DRAFT') {
    return match.sp.cards.filter((c) => match.spCardAvailable(c)).map((c) => ({ t: 'g.choice', idx: c.idx }));
  }
  return prep(match, ps, own);
}
