"""Observation tokenizer (114 semantic tokens) and lossless hierarchical action codec.

Token layout (fixed N = 114 tokens):
  [0]        POLICY_CLS   (1)  Learnable Actor readout token (feeds type_head & Step 2/3/4 ctx)
  [1]        VALUE_CLS    (1)  Learnable Critic readout token (feeds value_head independently)
  [2]        MATCH_STAGE  (1)  Phase, round, difficulty, LP, stageId, 9-dim enemy factions
  [3]        BOSS         (1)  Boss & hiddenBoss entity IDs, boss-round flags
  [4]        ECONOMY_SHOP (1)  Funds, shop level, upgrade/refresh prices, frozen, deploy cap, stash
  [5]        BAND_COMMAND (1)  Opening bandId + active/passive command effectId, cost, CD, uses
  [6..11]    EFFECTS      (6)  Active in-match effects / SP buffs (`own.effects`)
  [12..47]   BOARD_TILES  (36) 4x9 battlefield grid (row 9..12, col 2..10) + path/terrain + unit
  [48..57]   HAND_SLOTS   (10) Bench / hand slots 0..9
  [58..62]   TEMP_SLOTS   (5)  Equipment / temp storage slots 0..4
  [63..68]   SHOP_SLOTS   (6)  Shop offer slots 0..5
  [69..71]   REWARD_SLOTS (3)  Free reward offer slots 0..2
  [72..79]   CHOICE_CARDS (8)  Bond group / SP draft / personal choice / command draft cards
  [80..105]  BONDS        (26) Bond synergy state tokens
  [106..113] ENEMIES      (8)  Next-wave enemy preview summary tokens

Isolated categorical ID fields per token (`NUM_CAT_FIELDS = 8`, 0 = PAD/NONE for all non-segment fields):
  col 0: segment_id  in [0, 15)
  col 1: entity_id   in [0, num_entities)  (chess / item / token / enemy / boss / effect / choice)
  col 2: equip0_id   in [0, num_items)     (equipped item slot 0 on chess; 0 elsewhere)
  col 3: equip1_id   in [0, num_items)     (equipped item slot 1 on chess; 0 elsewhere)
  col 4: band_id     in [0, num_bands+1)   (BAND_COMMAND token only)
  col 5: bond_id     in [0, num_bonds+1)   (BONDS tokens only)
  col 6: stage_id    in [0, num_stages)    (MATCH_STAGE token only)
  col 7: dir_id      in [0, 5)             (occupied BOARD_TILES only: 1=R, 2=U, 3=L, 4=D)
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch

from .catalog import (
    BOARD_COLS,
    BOARD_TILES,
    DIFF_TO_IDX,
    DIR_TO_IDX,
    FIELD_C0,
    FIELD_R0,
    PHASE_TO_IDX,
    GameCatalog,
    get_catalog,
)

# ---- Token layout constants ------------------------------------------------
POLICY_CLS_IDX = 0
VALUE_CLS_IDX = 1
MATCH_STAGE_IDX = 2
BOSS_IDX = 3
ECONOMY_SHOP_IDX = 4
BAND_COMMAND_IDX = 5
EFFECT_OFFSET = 6              # 6..11   (6)
BOARD_OFFSET = 12              # 12..47  (36)
HAND_OFFSET = 48               # 48..57  (10)
TEMP_OFFSET = 58               # 58..62  (5)
SHOP_OFFSET = 63               # 63..68  (6)
REWARD_OFFSET = 69             # 69..71  (3)
CHOICE_OFFSET = 72             # 72..79  (8)
BOND_OFFSET = 80               # 80..105 (26)
ENEMY_OFFSET = 106             # 106..113 (8)
NUM_TOKENS = 114

MAX_EFFECT_SLOTS = 6
MAX_CHOICE_SLOTS = 8
MAX_ENEMY_SLOTS = 8
NUM_SUB_OPTS = 4  # 4 directions for MOVE(board), up to 4 item replace slots for EQUIP

# Token segment types (for segment embedding)
SEG_POLICY_CLS = 0
SEG_VALUE_CLS = 1
SEG_MATCH_STAGE = 2
SEG_BOSS = 3
SEG_ECONOMY_SHOP = 4
SEG_BAND_COMMAND = 5
SEG_EFFECT = 6
SEG_BOARD = 7
SEG_HAND = 8
SEG_TEMP = 9
SEG_SHOP = 10
SEG_REWARD = 11
SEG_CHOICE = 12
SEG_BOND = 13
SEG_ENEMY = 14
NUM_SEGMENTS = 15

# Per-token isolated categorical fields:
#   [segment_id, entity_id, equip0_id, equip1_id, band_id, bond_id, stage_id, dir_id]
CAT_SEG = 0
CAT_ENTITY = 1
CAT_EQUIP0 = 2
CAT_EQUIP1 = 3
CAT_BAND = 4
CAT_BOND = 5
CAT_STAGE = 6
CAT_DIR = 7
NUM_CAT_FIELDS = 8

# Per-token continuous features
CONT_DIM = 28

# ---- Autoregressive action types -------------------------------------------
ACTION_TYPES: tuple[str, ...] = (
    "READY",           # 0: g.ready (0-arg)
    "BUY",             # 1: g.buy (src=shop)
    "SELL",            # 2: g.sell (src=piece)
    "REFRESH",         # 3: g.refresh (0-arg)
    "FREEZE",          # 4: g.freeze (0-arg)
    "LEVEL_UP",        # 5: g.levelUp (0-arg)
    "MOVE",            # 6: g.move (src=piece, tgt=board|hand, opt=dir if board)
    "EQUIP",           # 7: g.equip (src=item, tgt=chess, opt=replace_slot if full)
    "DESTROY",         # 8: g.destroy (src=item)
    "REWARD",          # 9: g.reward (src=reward_slot)
    "COMMAND_SELF",    # 10: g.command with target={} (0-arg)
    "COMMAND_TARGET",  # 11: g.command with uid/slot/tile (src=token)
    "COMMAND_CHOICE",  # 12: g.commandChoice (src=choice_slot)
    "CHOICE",          # 13: g.choice (src=choice_slot)
    "BOND_GROUP",      # 14: g.bondGroup (src=choice_slot)
    "INFO_READY",      # 15: g.infoReady (0-arg)
    "BAND",            # 16: g.band (src=band_idx in band space)
    "BAND_SKIP",       # 17: g.bandSkip (0-arg)
    "DIY_BRING",       # 18: g.diyBring (0-arg)
)
ACTION_TYPE_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(ACTION_TYPES)}
NUM_ACTION_TYPES = len(ACTION_TYPES)

TYPE_READY = ACTION_TYPE_TO_IDX["READY"]
TYPE_BUY = ACTION_TYPE_TO_IDX["BUY"]
TYPE_SELL = ACTION_TYPE_TO_IDX["SELL"]
TYPE_REFRESH = ACTION_TYPE_TO_IDX["REFRESH"]
TYPE_FREEZE = ACTION_TYPE_TO_IDX["FREEZE"]
TYPE_LEVEL_UP = ACTION_TYPE_TO_IDX["LEVEL_UP"]
TYPE_MOVE = ACTION_TYPE_TO_IDX["MOVE"]
TYPE_EQUIP = ACTION_TYPE_TO_IDX["EQUIP"]
TYPE_DESTROY = ACTION_TYPE_TO_IDX["DESTROY"]
TYPE_REWARD = ACTION_TYPE_TO_IDX["REWARD"]
TYPE_COMMAND_SELF = ACTION_TYPE_TO_IDX["COMMAND_SELF"]
TYPE_COMMAND_TARGET = ACTION_TYPE_TO_IDX["COMMAND_TARGET"]
TYPE_COMMAND_CHOICE = ACTION_TYPE_TO_IDX["COMMAND_CHOICE"]
TYPE_CHOICE = ACTION_TYPE_TO_IDX["CHOICE"]
TYPE_BOND_GROUP = ACTION_TYPE_TO_IDX["BOND_GROUP"]
TYPE_INFO_READY = ACTION_TYPE_TO_IDX["INFO_READY"]
TYPE_BAND = ACTION_TYPE_TO_IDX["BAND"]
TYPE_BAND_SKIP = ACTION_TYPE_TO_IDX["BAND_SKIP"]
TYPE_DIY_BRING = ACTION_TYPE_TO_IDX["DIY_BRING"]

NO_ARG = -1


def board_token_idx(row: int, col: int) -> int:
    return BOARD_OFFSET + (row - FIELD_R0) * BOARD_COLS + (col - FIELD_C0)


def _log_norm(val: float | int | None, scale: float = 1.0) -> float:
    if val is None or val <= 0:
        return 0.0
    return math.log1p(float(val)) / scale


@dataclass
class ActionTree:
    """Lossless prefix-tree index over `frame['actions']` for exact hierarchical masking."""

    type_mask: torch.Tensor
    type_to_key: dict[int, int]
    src_keys: dict[int, list[int]]
    src_to_key: dict[tuple[int, int], int]
    tgt_keys: dict[tuple[int, int], list[int]]
    tgt_to_key: dict[tuple[int, int, int], int]
    opt_keys: dict[tuple[int, int, int], list[int]]
    opt_to_key: dict[tuple[int, int, int, int], int]
    key_to_tuple: list[tuple[int, int, int, int]]
    num_bands: int

    def src_mask(self, a_type: int) -> torch.Tensor | None:
        srcs = self.src_keys.get(a_type)
        if not srcs:
            return None
        size = self.num_bands if a_type == TYPE_BAND else NUM_TOKENS
        mask = torch.zeros(size, dtype=torch.bool)
        mask[srcs] = True
        return mask

    def tgt_mask(self, a_type: int, src: int) -> torch.Tensor | None:
        tgts = self.tgt_keys.get((a_type, src))
        if not tgts:
            return None
        mask = torch.zeros(NUM_TOKENS, dtype=torch.bool)
        mask[tgts] = True
        return mask

    def opt_mask(self, a_type: int, src: int, tgt: int) -> torch.Tensor | None:
        opts = self.opt_keys.get((a_type, src, tgt))
        if not opts:
            return None
        mask = torch.zeros(NUM_SUB_OPTS, dtype=torch.bool)
        mask[opts] = True
        return mask

    def decode(self, a_type: int, src: int = NO_ARG, tgt: int = NO_ARG, opt: int = NO_ARG) -> int:
        if a_type in self.type_to_key:
            return self.type_to_key[a_type]
        if (a_type, src) in self.src_to_key:
            return self.src_to_key[(a_type, src)]
        if (a_type, src, tgt) in self.tgt_to_key:
            return self.tgt_to_key[(a_type, src, tgt)]
        if (a_type, src, tgt, opt) in self.opt_to_key:
            return self.opt_to_key[(a_type, src, tgt, opt)]
        raise KeyError(f"Illegal decoded action tuple: {(a_type, src, tgt, opt)}")


@dataclass
class EncodedFrame:
    """Tensorized state representation + exact hierarchical legal action tree."""

    cat_ids: torch.Tensor      # [NUM_TOKENS, NUM_CAT_FIELDS] long
    cont_feats: torch.Tensor   # [NUM_TOKENS, CONT_DIM] float32
    valid_mask: torch.Tensor   # [NUM_TOKENS] bool (True = valid token)
    action_tree: ActionTree


def _build_action_tree(
    actions: Sequence[Mapping[str, Any]],
    uid_to_token: Mapping[int, int],
    uid_to_items: Mapping[int, list[int]],
    catalog: GameCatalog,
) -> ActionTree:
    type_mask = torch.zeros(NUM_ACTION_TYPES, dtype=torch.bool)
    type_to_key: dict[int, int] = {}
    src_sets: dict[int, set[int]] = {}
    src_to_key: dict[tuple[int, int], int] = {}
    tgt_sets: dict[tuple[int, int], set[int]] = {}
    tgt_to_key: dict[tuple[int, int, int], int] = {}
    opt_sets: dict[tuple[int, int, int], set[int]] = {}
    opt_to_key: dict[tuple[int, int, int, int], int] = {}
    key_to_tuple: list[tuple[int, int, int, int]] = []

    def _register(action_key: int, a_type: int, src: int = NO_ARG, tgt: int = NO_ARG, opt: int = NO_ARG) -> None:
        type_mask[a_type] = True
        key_to_tuple.append((a_type, src, tgt, opt))
        if src == NO_ARG:
            if a_type in type_to_key:
                raise ValueError(f"Duplicate 0-arg action for type {ACTION_TYPES[a_type]}")
            type_to_key[a_type] = action_key
            return
        src_sets.setdefault(a_type, set()).add(src)
        if tgt == NO_ARG:
            if (a_type, src) in src_to_key:
                raise ValueError(f"Duplicate 1-arg action {(ACTION_TYPES[a_type], src)}")
            src_to_key[(a_type, src)] = action_key
            return
        tgt_sets.setdefault((a_type, src), set()).add(tgt)
        if opt == NO_ARG:
            if (a_type, src, tgt) in tgt_to_key:
                raise ValueError(f"Duplicate 2-arg action {(ACTION_TYPES[a_type], src, tgt)}")
            tgt_to_key[(a_type, src, tgt)] = action_key
            return
        opt_sets.setdefault((a_type, src, tgt), set()).add(opt)
        if (a_type, src, tgt, opt) in opt_to_key:
            raise ValueError(f"Duplicate 3-arg action {(ACTION_TYPES[a_type], src, tgt, opt)}")
        opt_to_key[(a_type, src, tgt, opt)] = action_key

    for key, act in enumerate(actions):
        t = act["t"]
        if t == "g.ready":
            _register(key, TYPE_READY)
        elif t == "g.refresh":
            _register(key, TYPE_REFRESH)
        elif t == "g.freeze":
            _register(key, TYPE_FREEZE)
        elif t == "g.levelUp":
            _register(key, TYPE_LEVEL_UP)
        elif t == "g.infoReady":
            _register(key, TYPE_INFO_READY)
        elif t == "g.bandSkip":
            _register(key, TYPE_BAND_SKIP)
        elif t == "g.diyBring":
            _register(key, TYPE_DIY_BRING)
        elif t == "g.buy":
            _register(key, TYPE_BUY, SHOP_OFFSET + int(act["slot"]))
        elif t == "g.reward":
            _register(key, TYPE_REWARD, REWARD_OFFSET + int(act["idx"]))
        elif t == "g.sell":
            _register(key, TYPE_SELL, uid_to_token[int(act["uid"])])
        elif t == "g.destroy":
            _register(key, TYPE_DESTROY, uid_to_token[int(act["uid"])])
        elif t == "g.bondGroup":
            _register(key, TYPE_BOND_GROUP, CHOICE_OFFSET + int(act["idx"]))
        elif t == "g.choice":
            idx = int(act["idx"])
            if not (0 <= idx < MAX_CHOICE_SLOTS):
                raise ValueError(f"Choice idx out of range: {idx}")
            _register(key, TYPE_CHOICE, CHOICE_OFFSET + idx)
        elif t == "g.commandChoice":
            idx = int(act["idx"])
            if not (0 <= idx < MAX_CHOICE_SLOTS):
                raise ValueError(f"Command choice idx out of range: {idx}")
            _register(key, TYPE_COMMAND_CHOICE, CHOICE_OFFSET + idx)
        elif t == "g.band":
            b_idx = catalog.band_to_idx[act["bandId"]]
            _register(key, TYPE_BAND, b_idx)
        elif t == "g.command":
            target = act.get("target") or {}
            if not target:
                _register(key, TYPE_COMMAND_SELF)
            elif "uid" in target:
                _register(key, TYPE_COMMAND_TARGET, uid_to_token[int(target["uid"])])
            elif "slot" in target:
                _register(key, TYPE_COMMAND_TARGET, SHOP_OFFSET + int(target["slot"]))
            elif "row" in target and "col" in target:
                _register(key, TYPE_COMMAND_TARGET, board_token_idx(int(target["row"]), int(target["col"])))
            else:
                raise ValueError(f"Unsupported g.command target: {target}")
        elif t == "g.move":
            src = uid_to_token[int(act["uid"])]
            to = act["to"]
            if to["area"] == "board":
                tgt = board_token_idx(int(to["row"]), int(to["col"]))
                d_idx = DIR_TO_IDX[act.get("dir", "RIGHT")]
                _register(key, TYPE_MOVE, src, tgt, d_idx)
            elif to["area"] == "hand":
                tgt = HAND_OFFSET + int(to["idx"])
                _register(key, TYPE_MOVE, src, tgt)
            else:
                raise ValueError(f"Unsupported g.move area: {to}")
        elif t == "g.equip":
            src = uid_to_token[int(act["itemUid"])]
            tgt_uid = int(act["targetUid"])
            tgt = uid_to_token[tgt_uid]
            rep_uid = act.get("replaceUid")
            if rep_uid is None:
                _register(key, TYPE_EQUIP, src, tgt)
            else:
                items = uid_to_items.get(tgt_uid, [])
                rep_slot = items.index(int(rep_uid))
                if not (0 <= rep_slot < NUM_SUB_OPTS):
                    raise ValueError(f"replace slot out of range: {rep_slot}")
                _register(key, TYPE_EQUIP, src, tgt, rep_slot)
        else:
            raise ValueError(f"Unsupported simulator action type: {t}")

    return ActionTree(
        type_mask=type_mask,
        type_to_key=type_to_key,
        src_keys={k: sorted(v) for k, v in src_sets.items()},
        src_to_key=src_to_key,
        tgt_keys={k: sorted(v) for k, v in tgt_sets.items()},
        tgt_to_key=tgt_to_key,
        opt_keys={k: sorted(v) for k, v in opt_sets.items()},
        opt_to_key=opt_to_key,
        key_to_tuple=key_to_tuple,
        num_bands=catalog.num_bands,
    )


def encode_frame(
    frame: Mapping[str, Any],
    catalog: GameCatalog | None = None,
    *,
    prep_steps: int = 0,
) -> EncodedFrame:
    """Convert one simulator decision frame into 114 semantic tokens + exact hierarchical masks."""
    cat = catalog if catalog is not None else get_catalog()
    pub = frame["pub"]
    own = frame["own"]
    phase = frame["phase"]
    detail = frame.get("detail")

    cat_ids = torch.zeros((NUM_TOKENS, NUM_CAT_FIELDS), dtype=torch.long)
    cont = torch.zeros((NUM_TOKENS, CONT_DIM), dtype=torch.float32)
    valid = torch.zeros(NUM_TOKENS, dtype=torch.bool)

    stats_by_uid: dict[int, Mapping[str, Any]] = {}
    if detail and detail.get("units"):
        for u in detail["units"]:
            uid = u.get("uid")
            if uid is not None:
                stats_by_uid[int(uid)] = u

    owned_base_counts: dict[int, int] = {}
    owned_item_counts: dict[int, int] = {}
    board_uids: set[int] = set()
    uid_to_token: dict[int, int] = {}
    uid_to_items: dict[int, list[int]] = {}

    def _tally_piece(p: Mapping[str, Any]) -> None:
        kind = p.get("kind")
        pid = p.get("id")
        if kind == "chess" and not p.get("golden"):
            cm = cat.chess_meta.get(pid)
            if cm:
                owned_base_counts[cm.base_idx] = owned_base_counts.get(cm.base_idx, 0) + 1
        elif kind == "item" and not p.get("golden"):
            im = cat.item_meta.get(pid)
            if im:
                owned_item_counts[im.entity_idx] = owned_item_counts.get(im.entity_idx, 0) + 1

    for p in own.get("board") or []:
        if p:
            board_uids.add(int(p["uid"]))
            _tally_piece(p)
    for p in own.get("hand") or []:
        if p:
            _tally_piece(p)
    for p in own.get("temp") or []:
        if p:
            _tally_piece(p)

    # ---- [0..1] POLICY_CLS & VALUE_CLS (pure learnable readout tokens) -----
    valid[POLICY_CLS_IDX] = True
    cat_ids[POLICY_CLS_IDX, CAT_SEG] = SEG_POLICY_CLS
    valid[VALUE_CLS_IDX] = True
    cat_ids[VALUE_CLS_IDX, CAT_SEG] = SEG_VALUE_CLS

    # ---- [2] MATCH_STAGE token ---------------------------------------------
    valid[MATCH_STAGE_IDX] = True
    phase_idx = PHASE_TO_IDX.get(phase, 0)
    rnd = int(pub.get("round") or 0)
    last_rnd = max(1, int(pub.get("lastRound") or 14))
    boss_round = int(pub.get("bossRound") or 99)
    hidden_round = int(pub.get("hiddenRound") or 99)
    is_boss = rnd >= boss_round
    is_hidden = rnd >= hidden_round
    is_sp = bool(pub.get("spRound"))
    diff_idx = DIFF_TO_IDX.get(pub.get("difficulty") or "FUNNY", 2)
    stage_id = pub.get("stageId")

    cat_ids[MATCH_STAGE_IDX, CAT_SEG] = SEG_MATCH_STAGE
    cat_ids[MATCH_STAGE_IDX, CAT_STAGE] = cat.stage_to_idx.get(stage_id, 0)

    ms = cont[MATCH_STAGE_IDX]
    # Phase one-hot [0..4]
    if 0 <= phase_idx < 5:
        ms[phase_idx] = 1.0
    ms[5] = rnd / float(last_rnd)
    ms[6] = max(0, boss_round - rnd) / float(last_rnd)
    ms[7] = 1.0 if is_boss else 0.0
    ms[8] = 1.0 if is_hidden else 0.0
    ms[9] = 1.0 if is_sp else 0.0
    ms[10] = float(diff_idx) / max(1, len(DIFF_TO_IDX) - 1)
    ms[11] = float(own.get("lp") or 0) / 30.0
    ms[12] = 1.0 if own.get("alive", True) else 0.0
    ms[13] = 1.0 if (pub.get("bondChoice") and pub["bondChoice"].get("picked") is None) else 0.0
    # Enemy faction multi-hot [14..22] (9 faction types)
    for fac in pub.get("factions") or []:
        fi = cat.faction_to_idx.get(fac)
        if fi is not None and 14 + fi < CONT_DIM:
            ms[14 + fi] = 1.0

    # ---- [3] BOSS token ----------------------------------------------------
    valid[BOSS_IDX] = True
    cat_ids[BOSS_IDX, CAT_SEG] = SEG_BOSS
    active_boss_id = pub.get("hiddenBossId") if is_hidden else pub.get("bossId")
    cat_ids[BOSS_IDX, CAT_ENTITY] = cat.entity_idx(active_boss_id)
    bf = cont[BOSS_IDX]
    bf[0] = 1.0 if active_boss_id else 0.0
    bf[1] = 1.0 if is_boss else 0.0
    bf[2] = 1.0 if is_hidden else 0.0
    bf[3] = 1.0 if pub.get("hiddenBossId") else 0.0
    bf[4] = max(0, boss_round - rnd) / float(last_rnd)

    # ---- [4] ECONOMY_SHOP token --------------------------------------------
    valid[ECONOMY_SHOP_IDX] = True
    cat_ids[ECONOMY_SHOP_IDX, CAT_SEG] = SEG_ECONOMY_SHOP
    shop = own.get("shop") or {}
    funds = int(own.get("funds") or 0)
    shop_lv = int(shop.get("level") or 1)
    max_lv = max(1, int(shop.get("maxLevel") or 6))
    upg_price = int(shop.get("upgradePrice") or 0)
    ref_price = int(shop.get("refreshPrice") or 0)
    free_ref = int(shop.get("freeRefreshes") or 0)
    dep_cnt = int(own.get("deployCount") or 0)
    dep_cap = max(1, int(own.get("deployCap") or 8))
    hand_list = own.get("hand") or []
    temp_list = own.get("temp") or []
    free_hand = sum(1 for x in hand_list if x is None)
    free_temp = sum(1 for x in temp_list if x is None)

    ec = cont[ECONOMY_SHOP_IDX]
    ec[0] = _log_norm(funds, 4.0)
    ec[1] = min(funds, 30) / 30.0
    ec[2] = float(shop_lv) / float(max_lv)
    ec[3] = 1.0 if shop_lv >= max_lv else 0.0
    ec[4] = float(upg_price) / 15.0
    ec[5] = 1.0 if (shop_lv < max_lv and funds >= upg_price) else 0.0
    ec[6] = float(ref_price) / 5.0
    ec[7] = float(free_ref) / 5.0
    ec[8] = 1.0 if (free_ref > 0 or funds >= ref_price) else 0.0
    ec[9] = 1.0 if shop.get("frozen") else 0.0
    ec[10] = float(dep_cnt) / float(dep_cap)
    ec[11] = float(dep_cap) / 12.0
    ec[12] = float(max(0, dep_cap - dep_cnt)) / 8.0
    ec[13] = float(free_hand) / 10.0
    ec[14] = float(free_temp) / 5.0
    ec[15] = float(own.get("stashCount") or 0) / 5.0
    ec[16] = 1.0 if own.get("canReady") else 0.0
    ec[17] = 1.0 if shop.get("rewardOffer") else 0.0
    ec[18] = 1.0 if own.get("personalChoice") else 0.0
    ec[19] = min(float(max(0, prep_steps)), 80.0) / 80.0
    ec[20] = 1.0 if prep_steps >= 40 else 0.0

    # ---- [5] BAND_COMMAND token --------------------------------------------
    valid[BAND_COMMAND_IDX] = True
    cat_ids[BAND_COMMAND_IDX, CAT_SEG] = SEG_BAND_COMMAND
    band_id = own.get("bandId")
    cmd = own.get("command") or {}
    if band_id and band_id in cat.band_to_idx:
        cat_ids[BAND_COMMAND_IDX, CAT_BAND] = cat.band_to_idx[band_id] + 1
    cat_ids[BAND_COMMAND_IDX, CAT_ENTITY] = cat.entity_idx(cmd.get("effectId"))

    bc = cont[BAND_COMMAND_IDX]
    cmd_type = cmd.get("type") or ""
    bc[0] = 1.0 if band_id else 0.0
    bc[1] = 1.0 if cmd_type == "ACTIVE" else 0.0
    bc[2] = 1.0 if cmd_type == "PASSIVE" else 0.0
    bc[3] = 1.0 if cmd.get("usable") else 0.0
    cmd_cost = int(cmd.get("currentCost") or 0)
    bc[4] = float(cmd_cost) / 10.0
    bc[5] = 1.0 if (cmd_type == "ACTIVE" and funds >= cmd_cost) else 0.0
    bc[6] = float(cmd.get("cooldown") or 0) / 5.0
    uses_left = cmd.get("usesLeft")
    bc[7] = float(uses_left) / 5.0 if uses_left is not None else 0.0
    bc[8] = 1.0 if uses_left is None else 0.0
    bc[9] = 1.0 if own.get("commandDraft") else 0.0
    tgt_type = cmd.get("targetType") or ""
    bc[10] = 1.0 if tgt_type == "NO_TARGET" else 0.0
    bc[11] = 1.0 if tgt_type == "COMMON_CHESS" else 0.0
    bc[12] = 1.0 if tgt_type == "SHOP_SLOT" else 0.0
    bc[13] = 1.0 if tgt_type == "MAP_TILE" else 0.0

    # ---- [6..11] EFFECTS tokens (up to 6 active in-match buffs/effects) ----
    effects = own.get("effects") or []
    for i in range(MAX_EFFECT_SLOTS):
        tok_idx = EFFECT_OFFSET + i
        cat_ids[tok_idx, CAT_SEG] = SEG_EFFECT
        if i < len(effects) and effects[i]:
            eff = effects[i]
            valid[tok_idx] = True
            cat_ids[tok_idx, CAT_ENTITY] = cat.entity_idx(eff.get("id"))
            ef = cont[tok_idx]
            ef[0] = 1.0
            ef[1] = 1.0 if eff.get("command") else 0.0
            ef[2] = 1.0 if eff.get("commandType") == "ACTIVE" else 0.0
            ef[3] = 1.0 if eff.get("iconKind") == "team" else 0.0
            ef[4] = 1.0 if eff.get("iconKind") == "band" else 0.0
            ef[5] = float(i) / max(1.0, float(MAX_EFFECT_SLOTS - 1))

    # ---- Helper to populate a piece (chess / item / token) -----------------
    def _fill_piece(tok_idx: int, p: Mapping[str, Any], on_board: bool) -> None:
        uid = int(p["uid"])
        uid_to_token[uid] = tok_idx
        items = p.get("items") or []
        uid_to_items[uid] = [int(it["uid"]) for it in items]

        kind = p.get("kind")
        pid = p.get("id")
        cat_ids[tok_idx, CAT_ENTITY] = cat.entity_idx(pid)
        if len(items) > 0:
            cat_ids[tok_idx, CAT_EQUIP0] = cat.item_idx(items[0].get("id"))
        if len(items) > 1:
            cat_ids[tok_idx, CAT_EQUIP1] = cat.item_idx(items[1].get("id"))
        if on_board:
            cat_ids[tok_idx, CAT_DIR] = DIR_TO_IDX.get(p.get("dir", "RIGHT"), 0) + 1

        f = cont[tok_idx]
        f[0] = 1.0  # occupied
        f[1] = 1.0 if kind == "chess" else 0.0
        f[2] = 1.0 if kind == "item" else 0.0
        f[3] = 1.0 if kind == "token" else 0.0
        f[4] = 1.0 if p.get("golden") else 0.0
        f[5] = float(p.get("tier") or 0) / 6.0
        f[6] = float(p.get("count") or 1) / 3.0
        f[7] = float(len(items)) / 2.0

        if kind == "chess":
            cm = cat.chess_meta.get(pid)
            if cm:
                f[8] = float(cm.pos_class) / 2.0
                f[9] = float(owned_base_counts.get(cm.base_idx, 0)) / 3.0
                f[10] = float(cm.sell_price) / 6.0
        elif kind == "item":
            im = cat.item_meta.get(pid)
            if im:
                f[9] = float(owned_item_counts.get(im.entity_idx, 0)) / 2.0
        elif kind == "token":
            f[8] = float(cat.token_pos_class.get(pid, 0)) / 2.0
            owner_uid = p.get("ownerUid")
            f[9] = 1.0 if (owner_uid is not None and int(owner_uid) in board_uids) else 0.0

        st = stats_by_uid.get(uid)
        if st:
            f[11] = _log_norm(st.get("maxHp"), 10.0)
            f[12] = _log_norm(st.get("atk"), 8.0)
            f[13] = _log_norm(st.get("def"), 8.0)
            f[14] = float(st.get("res") or 0) / 100.0
            iv = st.get("interval")
            f[15] = (1.0 / float(iv)) if (iv and iv > 0) else 0.0
            f[16] = float(st.get("blockCnt") or 0) / 4.0
            f[17] = float(len(st.get("range") or [])) / 20.0

    # ---- [12..47] BOARD_TILES (36) with spatial terrain & path features ----
    spatial_grid = cat.stage_spatial(stage_id, is_boss=is_boss)
    for idx in range(BOARD_TILES):
        tok_idx = BOARD_OFFSET + idx
        valid[tok_idx] = True
        cat_ids[tok_idx, CAT_SEG] = SEG_BOARD
        r = idx // BOARD_COLS
        c = idx % BOARD_COLS
        sp = spatial_grid[idx]
        f = cont[tok_idx]
        f[18] = float(r) / 3.0
        f[19] = float(c) / 8.0
        f[20] = 1.0 if sp.deploy_cls == 1 else 0.0   # melee tile
        f[21] = 1.0 if sp.deploy_cls == 2 else 0.0   # ranged tile
        f[22] = 1.0 if sp.is_ground_path else 0.0    # enemy ground route passes here
        f[23] = 1.0 if sp.on_upper_path else 0.0     # upper-gate route
        f[24] = 1.0 if sp.on_lower_path else 0.0     # lower-gate route
        f[25] = sp.path_progress                     # 0=spawn .. 1=blue gate
        f[26] = 1.0 if sp.is_special_tile else 0.0   # mire / infection / grassland / speedup / fence
        f[27] = 1.0 if sp.has_blower else 0.0        # active blower device

    for p in own.get("board") or []:
        if not p:
            continue
        tok_idx = board_token_idx(int(p["row"]), int(p["col"]))
        _fill_piece(tok_idx, p, on_board=True)

    # ---- [48..57] HAND_SLOTS (10) ------------------------------------------
    for i in range(10):
        tok_idx = HAND_OFFSET + i
        valid[tok_idx] = True
        cat_ids[tok_idx, CAT_SEG] = SEG_HAND
        cont[tok_idx, 18] = float(i) / 9.0
        if i < len(hand_list) and hand_list[i] is not None:
            _fill_piece(tok_idx, hand_list[i], on_board=False)

    # ---- [58..62] TEMP_SLOTS (5) -------------------------------------------
    for i in range(5):
        tok_idx = TEMP_OFFSET + i
        valid[tok_idx] = True
        cat_ids[tok_idx, CAT_SEG] = SEG_TEMP
        cont[tok_idx, 18] = float(i) / 4.0
        if i < len(temp_list) and temp_list[i] is not None:
            _fill_piece(tok_idx, temp_list[i], on_board=False)

    # ---- [63..68] SHOP_SLOTS (6) & [69..71] REWARD_SLOTS (3) ---------------
    shop_slots = shop.get("slots") or []
    for i in range(6):
        tok_idx = SHOP_OFFSET + i
        cat_ids[tok_idx, CAT_SEG] = SEG_SHOP
        cont[tok_idx, 18] = float(i) / 5.0
        if i < len(shop_slots) and shop_slots[i] is not None:
            valid[tok_idx] = True
            s = shop_slots[i]
            sid = s.get("id")
            kind = s.get("kind")
            cat_ids[tok_idx, CAT_ENTITY] = cat.entity_idx(sid)
            f = cont[tok_idx]
            f[0] = 1.0
            f[1] = 1.0 if kind == "chess" else 0.0
            f[2] = 1.0 if kind == "item" else 0.0
            f[3] = 1.0 if s.get("sold") else 0.0
            f[4] = 1.0 if s.get("frozen") else 0.0
            price = int(s.get("price") or 0)
            f[5] = float(price) / 6.0
            f[6] = 1.0 if (not s.get("sold") and price <= funds) else 0.0
            if kind == "chess" and sid in cat.chess_meta:
                cm = cat.chess_meta[sid]
                f[7] = float(cm.tier) / 6.0
                f[8] = float(cm.pos_class) / 2.0
                f[9] = float(owned_base_counts.get(cm.base_idx, 0)) / 3.0
            elif kind == "item" and sid in cat.item_meta:
                im = cat.item_meta[sid]
                f[7] = float(im.tier) / 6.0
                f[9] = float(owned_item_counts.get(im.entity_idx, 0)) / 2.0

    reward_offer = shop.get("rewardOffer") or {}
    reward_slots = reward_offer.get("slots") or []
    for i in range(3):
        tok_idx = REWARD_OFFSET + i
        cat_ids[tok_idx, CAT_SEG] = SEG_REWARD
        cont[tok_idx, 18] = float(i) / 2.0
        if i < len(reward_slots) and reward_slots[i] is not None:
            valid[tok_idx] = True
            s = reward_slots[i]
            sid = s.get("id")
            kind = s.get("kind")
            cat_ids[tok_idx, CAT_ENTITY] = cat.entity_idx(sid)
            f = cont[tok_idx]
            f[0] = 1.0
            f[1] = 1.0 if kind == "chess" else 0.0
            f[2] = 1.0 if kind == "item" else 0.0
            f[3] = 1.0 if s.get("sold") else 0.0
            if kind == "chess" and sid in cat.chess_meta:
                cm = cat.chess_meta[sid]
                f[7] = float(cm.tier) / 6.0
                f[8] = float(cm.pos_class) / 2.0
                f[9] = float(owned_base_counts.get(cm.base_idx, 0)) / 3.0

    # ---- [72..79] CHOICE_CARDS (8) -----------------------------------------
    bond_group_members: tuple[set[str], set[str]] = (set(), set())
    if phase == "INFO_CHECK" and pub.get("bondChoice") and pub["bondChoice"].get("picked") is None:
        groups = pub["bondChoice"].get("groups") or []
        for i, grp in enumerate(groups[:MAX_CHOICE_SLOTS]):
            tok_idx = CHOICE_OFFSET + i
            valid[tok_idx] = True
            cat_ids[tok_idx, CAT_SEG] = SEG_CHOICE
            cont[tok_idx, 0] = 1.0
            cont[tok_idx, 1] = 1.0  # bondGroup kind
            cont[tok_idx, 5] = float(len(grp)) / 6.0
            cont[tok_idx, 18] = float(i)
            if i < 2:
                bond_group_members[i].update(grp)
    elif phase == "SP_DRAFT" and pub.get("sp"):
        cards = pub["sp"].get("cards") or []
        for c in cards[:MAX_CHOICE_SLOTS]:
            idx = int(c.get("idx", 0))
            if not (0 <= idx < MAX_CHOICE_SLOTS):
                continue
            tok_idx = CHOICE_OFFSET + idx
            valid[tok_idx] = True
            cat_ids[tok_idx, CAT_SEG] = SEG_CHOICE
            cat_ids[tok_idx, CAT_ENTITY] = cat.entity_idx(c.get("id"))
            cont[tok_idx, 0] = 1.0
            cont[tok_idx, 2] = 1.0  # sp_draft kind
            cont[tok_idx, 3] = 1.0 if c.get("soldOut") else 0.0
            cont[tok_idx, 5] = float(c.get("tier") or 0) / 6.0
            cont[tok_idx, 18] = float(idx) / 7.0
    elif own.get("personalChoice"):
        cards = own["personalChoice"].get("cards") or []
        for i, c in enumerate(cards[:MAX_CHOICE_SLOTS]):
            tok_idx = CHOICE_OFFSET + i
            valid[tok_idx] = True
            cat_ids[tok_idx, CAT_SEG] = SEG_CHOICE
            cat_ids[tok_idx, CAT_ENTITY] = cat.entity_idx(c.get("id"))
            cont[tok_idx, 0] = 1.0
            cont[tok_idx, 2] = 1.0
            cont[tok_idx, 5] = float(c.get("tier") or 0) / 6.0
            cont[tok_idx, 18] = float(i) / 7.0
    elif own.get("commandDraft"):
        cards = own["commandDraft"].get("cards") or []
        for i, c in enumerate(cards[:MAX_CHOICE_SLOTS]):
            tok_idx = CHOICE_OFFSET + i
            valid[tok_idx] = True
            cat_ids[tok_idx, CAT_SEG] = SEG_CHOICE
            cat_ids[tok_idx, CAT_ENTITY] = cat.entity_idx(c.get("effectId"))
            cont[tok_idx, 0] = 1.0
            cont[tok_idx, 4] = 1.0  # commandChoice kind
            cont[tok_idx, 5] = float(c.get("cost") or 0) / 10.0
            cont[tok_idx, 18] = float(i) / 7.0

    # ---- [80..105] BONDS (26) ----------------------------------------------
    disabled_bonds = set(pub.get("disabledBonds") or [])
    own_bonds = {b["bondId"]: b for b in (own.get("bonds") or [])}
    for i, bid in enumerate(cat.bond_ids):
        tok_idx = BOND_OFFSET + i
        valid[tok_idx] = True
        cat_ids[tok_idx, CAT_SEG] = SEG_BOND
        cat_ids[tok_idx, CAT_BOND] = i + 1
        f = cont[tok_idx]
        f[0] = 1.0 if cat.core_bond_mask[i] else 0.0
        f[1] = 1.0 if bid in disabled_bonds else 0.0
        f[2] = 1.0 if bid in bond_group_members[0] else 0.0
        f[3] = 1.0 if bid in bond_group_members[1] else 0.0
        b = own_bonds.get(bid)
        if b:
            f[4] = 1.0 if b.get("active") else 0.0
            cnt = int(b.get("count") or 0)
            f[5] = float(cnt) / 8.0
            f[6] = float(b.get("tier") or 0) / 4.0
            f[7] = _log_norm(b.get("layers"), 7.0)
            ths = b.get("thresholds") or []
            next_th = next((th for th in ths if th > cnt), 0)
            f[8] = float(next_th - cnt) / 6.0 if next_th else 0.0

    # ---- [106..113] ENEMIES (up to 8) --------------------------------------
    enemies = own.get("nextEnemies") or []
    if enemies:
        agg: dict[tuple[str, str], dict[str, Any]] = {}
        for e in enemies:
            ek = e.get("enemyKey") or ""
            gt = e.get("gate") or "lower"
            rec = agg.get((ek, gt))
            if rec is None:
                agg[(ek, gt)] = {
                    "enemyKey": ek,
                    "gate": gt,
                    "count": int(e.get("count") or 1),
                    "t": float(e.get("t") or 0.0),
                    "fly": bool(e.get("fly")),
                    "elite": bool(e.get("elite")),
                    "boss": bool(e.get("boss")),
                }
            else:
                rec["count"] += int(e.get("count") or 1)
                rec["t"] = min(rec["t"], float(e.get("t") or 0.0))
        ranked = sorted(
            agg.values(),
            key=lambda x: (not x["boss"], not x["elite"], -x["count"], x["t"]),
        )[:MAX_ENEMY_SLOTS]
        for i, e in enumerate(ranked):
            tok_idx = ENEMY_OFFSET + i
            valid[tok_idx] = True
            cat_ids[tok_idx, CAT_SEG] = SEG_ENEMY
            cat_ids[tok_idx, CAT_ENTITY] = cat.entity_idx(e["enemyKey"])
            f = cont[tok_idx]
            f[0] = 1.0
            f[1] = _log_norm(e["count"], 4.0)
            f[2] = 1.0 if e["gate"] == "upper" else 0.0
            f[3] = min(e["t"], 120.0) / 120.0
            f[4] = 1.0 if e["fly"] else 0.0
            f[5] = 1.0 if e["elite"] else 0.0
            f[6] = 1.0 if e["boss"] else 0.0

    action_tree = _build_action_tree(frame["actions"], uid_to_token, uid_to_items, cat)
    return EncodedFrame(
        cat_ids=cat_ids,
        cont_feats=cont,
        valid_mask=valid,
        action_tree=action_tree,
    )
