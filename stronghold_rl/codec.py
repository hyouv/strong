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

import numpy as np
import torch

from .catalog import (
    BOARD_COLS,
    BOARD_TILES,
    BOSS_ROW_OFFSET,
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

# Compact wire action type codes (sent in `frame["cactions"]` when JS worker uses `compact: true`)
COMPACT_MOVE_HAND = 19

_BITS_TO_OPTS: tuple[list[int], ...] = tuple(
    [o for o in range(NUM_SUB_OPTS) if b & (1 << o)] for b in range(1 << NUM_SUB_OPTS)
)

# Precomputed per-(catalog_id, stage_id, is_boss) static NumPy token templates:
# (cat_ids_int32 [114, 8], cont_feats_f32 [114, 28], valid_mask_bool [114])
_STAGE_TEMPLATE_CACHE: dict[tuple[int, str | None, bool], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _get_stage_template(
    cat: GameCatalog,
    stage_id: str | None,
    is_boss: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_key = (id(cat), stage_id, is_boss)
    tpl = _STAGE_TEMPLATE_CACHE.get(cache_key)
    if tpl is not None:
        return tpl

    cat_base = np.zeros((NUM_TOKENS, NUM_CAT_FIELDS), dtype=np.int32)
    cont_base = np.zeros((NUM_TOKENS, CONT_DIM), dtype=np.float32)
    valid_base = np.zeros(NUM_TOKENS, dtype=np.bool_)

    valid_base[POLICY_CLS_IDX] = True
    cat_base[POLICY_CLS_IDX, CAT_SEG] = SEG_POLICY_CLS
    valid_base[VALUE_CLS_IDX] = True
    cat_base[VALUE_CLS_IDX, CAT_SEG] = SEG_VALUE_CLS
    valid_base[MATCH_STAGE_IDX] = True
    cat_base[MATCH_STAGE_IDX, CAT_SEG] = SEG_MATCH_STAGE
    cat_base[MATCH_STAGE_IDX, CAT_STAGE] = cat.stage_to_idx.get(stage_id, 0)
    valid_base[BOSS_IDX] = True
    cat_base[BOSS_IDX, CAT_SEG] = SEG_BOSS
    valid_base[ECONOMY_SHOP_IDX] = True
    cat_base[ECONOMY_SHOP_IDX, CAT_SEG] = SEG_ECONOMY_SHOP
    valid_base[BAND_COMMAND_IDX] = True
    cat_base[BAND_COMMAND_IDX, CAT_SEG] = SEG_BAND_COMMAND

    cat_base[EFFECT_OFFSET : EFFECT_OFFSET + MAX_EFFECT_SLOTS, CAT_SEG] = SEG_EFFECT

    spatial_grid = cat.stage_spatial(stage_id, is_boss=is_boss)
    for idx in range(BOARD_TILES):
        tok_idx = BOARD_OFFSET + idx
        valid_base[tok_idx] = True
        cat_base[tok_idx, CAT_SEG] = SEG_BOARD
        r = idx // BOARD_COLS
        c = idx % BOARD_COLS
        sp = spatial_grid[idx]
        cont_base[tok_idx, 18] = float(r) / 3.0
        cont_base[tok_idx, 19] = float(c) / 8.0
        cont_base[tok_idx, 20] = 1.0 if sp.deploy_cls == 1 else 0.0
        cont_base[tok_idx, 21] = 1.0 if sp.deploy_cls == 2 else 0.0
        cont_base[tok_idx, 22] = 1.0 if sp.is_ground_path else 0.0
        cont_base[tok_idx, 23] = 1.0 if sp.on_upper_path else 0.0
        cont_base[tok_idx, 24] = 1.0 if sp.on_lower_path else 0.0
        cont_base[tok_idx, 25] = sp.path_progress
        cont_base[tok_idx, 26] = 1.0 if sp.is_special_tile else 0.0
        cont_base[tok_idx, 27] = 1.0 if sp.has_blower else 0.0

    for i in range(10):
        tok_idx = HAND_OFFSET + i
        valid_base[tok_idx] = True
        cat_base[tok_idx, CAT_SEG] = SEG_HAND
        cont_base[tok_idx, 18] = float(i) / 9.0

    for i in range(5):
        tok_idx = TEMP_OFFSET + i
        valid_base[tok_idx] = True
        cat_base[tok_idx, CAT_SEG] = SEG_TEMP
        cont_base[tok_idx, 18] = float(i) / 4.0

    for i in range(6):
        tok_idx = SHOP_OFFSET + i
        cat_base[tok_idx, CAT_SEG] = SEG_SHOP
        cont_base[tok_idx, 18] = float(i) / 5.0

    for i in range(3):
        tok_idx = REWARD_OFFSET + i
        cat_base[tok_idx, CAT_SEG] = SEG_REWARD
        cont_base[tok_idx, 18] = float(i) / 2.0

    for i in range(len(cat.bond_ids)):
        tok_idx = BOND_OFFSET + i
        valid_base[tok_idx] = True
        cat_base[tok_idx, CAT_SEG] = SEG_BOND
        cat_base[tok_idx, CAT_BOND] = i + 1
        cont_base[tok_idx, 0] = 1.0 if cat.core_bond_mask[i] else 0.0

    tpl = (cat_base, cont_base, valid_base)
    _STAGE_TEMPLATE_CACHE[cache_key] = tpl
    return tpl


def board_token_idx(row: int, col: int) -> int:
    return BOARD_OFFSET + (row - FIELD_R0) * BOARD_COLS + (col - FIELD_C0)


def _log_norm(val: float | int | None, scale: float = 1.0) -> float:
    if val is None or val <= 0:
        return 0.0
    return math.log1p(float(val)) / scale


@dataclass(slots=True)
class ActionMaskSpec:
    """Lightweight, tensor-free legal mask index safe for cross-process IPC (zero PyTorch FDs)."""

    type_mask_bytes: bytes
    src_keys: dict[int, list[int]]
    tgt_keys: dict[tuple[int, int], list[int]]
    opt_bits: dict[tuple[int, int, int], int]
    num_bands: int


@dataclass(slots=True)
class FastActionDecoder:
    """Zero-tensor action decoder & mask spec for ultra-fast multiprocess rollout workers."""

    mask_spec: ActionMaskSpec
    type_to_key: dict[int, int]
    src_to_key: dict[tuple[int, int], int]
    tgt_to_key: dict[tuple[int, int, int], int]
    opt_to_key: dict[tuple[int, int, int, int], int]

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

    def pick_force_unblock_action(self) -> int:
        tm = self.mask_spec.type_mask_bytes
        if tm[TYPE_READY]:
            return self.type_to_key[TYPE_READY]
        src_keys = self.mask_spec.src_keys
        for t in (TYPE_COMMAND_CHOICE, TYPE_CHOICE):
            if tm[t]:
                return self.src_to_key[(t, src_keys[t][0])]
        for t in (TYPE_DESTROY, TYPE_SELL):
            if tm[t]:
                srcs = src_keys[t]
                for s in srcs:
                    if TEMP_OFFSET <= s < TEMP_OFFSET + 5:
                        return self.src_to_key[(t, s)]
                return self.src_to_key[(t, srcs[0])]
        return 0


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
    opt_bits: dict[tuple[int, int, int], int] | None = None
    _type_mask_bytes: bytes | None = None

    def to_mask_spec(self) -> ActionMaskSpec:
        """Convert to a compact pure-Python mask spec for multiprocess actor IPC."""
        if self.opt_bits is not None:
            ob = self.opt_bits
        else:
            ob = {}
            for k, opts in self.opt_keys.items():
                bits = 0
                for o in opts:
                    bits |= 1 << o
                ob[k] = bits
        tmb = (
            self._type_mask_bytes
            if self._type_mask_bytes is not None
            else self.type_mask.to(torch.uint8).numpy().tobytes()
        )
        return ActionMaskSpec(
            type_mask_bytes=tmb,
            src_keys=self.src_keys,
            tgt_keys=self.tgt_keys,
            opt_bits=ob,
            num_bands=self.num_bands,
        )

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


def _index_actions_dict(
    actions: Sequence[Mapping[str, Any]],
    uid_to_token: Mapping[int, int],
    uid_to_items: Mapping[int, list[int]],
    catalog: GameCatalog,
    *,
    build_key_to_tuple: bool,
) -> tuple[
    bytes,
    dict[int, int],
    dict[int, list[int]],
    dict[tuple[int, int], int],
    dict[tuple[int, int], list[int]],
    dict[tuple[int, int, int], int],
    dict[tuple[int, int, int], int],
    dict[tuple[int, int, int, int], int],
    list[tuple[int, int, int, int]],
]:
    tm = bytearray(NUM_ACTION_TYPES)
    type_to_key: dict[int, int] = {}
    src_sets: dict[int, set[int]] = {}
    src_to_key: dict[tuple[int, int], int] = {}
    tgt_sets: dict[tuple[int, int], set[int]] = {}
    tgt_to_key: dict[tuple[int, int, int], int] = {}
    opt_bits: dict[tuple[int, int, int], int] = {}
    opt_to_key: dict[tuple[int, int, int, int], int] = {}
    key_to_tuple: list[tuple[int, int, int, int]] = []
    dir_to_idx = DIR_TO_IDX
    band_to_idx = catalog.band_to_idx

    for key, act in enumerate(actions):
        t = act["t"]
        if t == "g.move":
            tm[TYPE_MOVE] = 1
            src = uid_to_token[int(act["uid"])]
            to = act["to"]
            area = to["area"]
            s_set = src_sets.get(TYPE_MOVE)
            if s_set is None:
                src_sets[TYPE_MOVE] = {src}
            else:
                s_set.add(src)
            k2 = (TYPE_MOVE, src)
            if area == "board":
                tgt = BOARD_OFFSET + (int(to["row"]) - FIELD_R0) * BOARD_COLS + (int(to["col"]) - FIELD_C0)
                d_idx = dir_to_idx[act.get("dir", "RIGHT")]
                t_set = tgt_sets.get(k2)
                if t_set is None:
                    tgt_sets[k2] = {tgt}
                else:
                    t_set.add(tgt)
                k3 = (TYPE_MOVE, src, tgt)
                opt_bits[k3] = opt_bits.get(k3, 0) | (1 << d_idx)
                k4 = (TYPE_MOVE, src, tgt, d_idx)
                if k4 in opt_to_key:
                    raise ValueError(f"Duplicate 3-arg action {('MOVE', src, tgt, d_idx)}")
                opt_to_key[k4] = key
                if build_key_to_tuple:
                    key_to_tuple.append((TYPE_MOVE, src, tgt, d_idx))
            elif area == "hand":
                tgt = HAND_OFFSET + int(to["idx"])
                t_set = tgt_sets.get(k2)
                if t_set is None:
                    tgt_sets[k2] = {tgt}
                else:
                    t_set.add(tgt)
                k3 = (TYPE_MOVE, src, tgt)
                if k3 in tgt_to_key:
                    raise ValueError(f"Duplicate 2-arg action {('MOVE', src, tgt)}")
                tgt_to_key[k3] = key
                if build_key_to_tuple:
                    key_to_tuple.append((TYPE_MOVE, src, tgt, NO_ARG))
            else:
                raise ValueError(f"Unsupported g.move area: {to}")
            continue

        a_type = -1
        src = NO_ARG
        tgt = NO_ARG
        opt = NO_ARG

        if t == "g.ready":
            a_type = TYPE_READY
        elif t == "g.refresh":
            a_type = TYPE_REFRESH
        elif t == "g.freeze":
            a_type = TYPE_FREEZE
        elif t == "g.levelUp":
            a_type = TYPE_LEVEL_UP
        elif t == "g.infoReady":
            a_type = TYPE_INFO_READY
        elif t == "g.bandSkip":
            a_type = TYPE_BAND_SKIP
        elif t == "g.diyBring":
            a_type = TYPE_DIY_BRING
        elif t == "g.buy":
            a_type = TYPE_BUY
            src = SHOP_OFFSET + int(act["slot"])
        elif t == "g.reward":
            a_type = TYPE_REWARD
            src = REWARD_OFFSET + int(act["idx"])
        elif t == "g.sell":
            a_type = TYPE_SELL
            src = uid_to_token[int(act["uid"])]
        elif t == "g.destroy":
            a_type = TYPE_DESTROY
            src = uid_to_token[int(act["uid"])]
        elif t == "g.bondGroup":
            a_type = TYPE_BOND_GROUP
            src = CHOICE_OFFSET + int(act["idx"])
        elif t == "g.choice":
            idx = int(act["idx"])
            if not (0 <= idx < MAX_CHOICE_SLOTS):
                raise ValueError(f"Choice idx out of range: {idx}")
            a_type = TYPE_CHOICE
            src = CHOICE_OFFSET + idx
        elif t == "g.commandChoice":
            idx = int(act["idx"])
            if not (0 <= idx < MAX_CHOICE_SLOTS):
                raise ValueError(f"Command choice idx out of range: {idx}")
            a_type = TYPE_COMMAND_CHOICE
            src = CHOICE_OFFSET + idx
        elif t == "g.band":
            a_type = TYPE_BAND
            src = band_to_idx[act["bandId"]]
        elif t == "g.command":
            target = act.get("target") or {}
            if not target:
                a_type = TYPE_COMMAND_SELF
            elif "uid" in target:
                a_type = TYPE_COMMAND_TARGET
                src = uid_to_token[int(target["uid"])]
            elif "slot" in target:
                a_type = TYPE_COMMAND_TARGET
                src = SHOP_OFFSET + int(target["slot"])
            elif "row" in target and "col" in target:
                a_type = TYPE_COMMAND_TARGET
                src = BOARD_OFFSET + (int(target["row"]) - FIELD_R0) * BOARD_COLS + (int(target["col"]) - FIELD_C0)
            else:
                raise ValueError(f"Unsupported g.command target: {target}")
        elif t == "g.equip":
            a_type = TYPE_EQUIP
            src = uid_to_token[int(act["itemUid"])]
            tgt_uid = int(act["targetUid"])
            tgt = uid_to_token[tgt_uid]
            rep_uid = act.get("replaceUid")
            if rep_uid is not None:
                items = uid_to_items.get(tgt_uid, [])
                rep_slot = items.index(int(rep_uid))
                if not (0 <= rep_slot < NUM_SUB_OPTS):
                    raise ValueError(f"replace slot out of range: {rep_slot}")
                opt = rep_slot
        else:
            raise ValueError(f"Unsupported simulator action type: {t}")

        tm[a_type] = 1
        if build_key_to_tuple:
            key_to_tuple.append((a_type, src, tgt, opt))
        if src == NO_ARG:
            if a_type in type_to_key:
                raise ValueError(f"Duplicate 0-arg action for type {ACTION_TYPES[a_type]}")
            type_to_key[a_type] = key
        elif tgt == NO_ARG:
            s_set = src_sets.get(a_type)
            if s_set is None:
                src_sets[a_type] = {src}
            else:
                s_set.add(src)
            k2 = (a_type, src)
            if k2 in src_to_key:
                raise ValueError(f"Duplicate 1-arg action {(ACTION_TYPES[a_type], src)}")
            src_to_key[k2] = key
        elif opt == NO_ARG:
            s_set = src_sets.get(a_type)
            if s_set is None:
                src_sets[a_type] = {src}
            else:
                s_set.add(src)
            k2 = (a_type, src)
            t_set = tgt_sets.get(k2)
            if t_set is None:
                tgt_sets[k2] = {tgt}
            else:
                t_set.add(tgt)
            k3 = (a_type, src, tgt)
            if k3 in tgt_to_key:
                raise ValueError(f"Duplicate 2-arg action {(ACTION_TYPES[a_type], src, tgt)}")
            tgt_to_key[k3] = key
        else:
            s_set = src_sets.get(a_type)
            if s_set is None:
                src_sets[a_type] = {src}
            else:
                s_set.add(src)
            k2 = (a_type, src)
            t_set = tgt_sets.get(k2)
            if t_set is None:
                tgt_sets[k2] = {tgt}
            else:
                t_set.add(tgt)
            k3 = (a_type, src, tgt)
            opt_bits[k3] = opt_bits.get(k3, 0) | (1 << opt)
            k4 = (a_type, src, tgt, opt)
            if k4 in opt_to_key:
                raise ValueError(f"Duplicate 3-arg action {(ACTION_TYPES[a_type], src, tgt, opt)}")
            opt_to_key[k4] = key

    return (
        bytes(tm),
        type_to_key,
        {k: sorted(v) for k, v in src_sets.items()},
        src_to_key,
        {k: sorted(v) for k, v in tgt_sets.items()},
        tgt_to_key,
        opt_bits,
        opt_to_key,
        key_to_tuple,
    )


def _index_actions_compact(
    cactions: Sequence[Sequence[Any]],
    uid_to_token: Mapping[int, int],
    uid_to_items: Mapping[int, list[int]],
    catalog: GameCatalog,
    *,
    build_key_to_tuple: bool,
) -> tuple[
    bytes,
    dict[int, int],
    dict[int, list[int]],
    dict[tuple[int, int], int],
    dict[tuple[int, int], list[int]],
    dict[tuple[int, int, int], int],
    dict[tuple[int, int, int], int],
    dict[tuple[int, int, int, int], int],
    list[tuple[int, int, int, int]],
]:
    """Ultra-fast indexer for flat integer-tuple `cactions` emitted in `compact: true` mode."""
    tm = bytearray(NUM_ACTION_TYPES)
    type_to_key: dict[int, int] = {}
    src_sets: dict[int, set[int]] = {}
    src_to_key: dict[tuple[int, int], int] = {}
    tgt_sets: dict[tuple[int, int], set[int]] = {}
    tgt_to_key: dict[tuple[int, int, int], int] = {}
    opt_bits: dict[tuple[int, int, int], int] = {}
    opt_to_key: dict[tuple[int, int, int, int], int] = {}
    key_to_tuple: list[tuple[int, int, int, int]] = []
    band_to_idx = catalog.band_to_idx

    for key, act in enumerate(cactions):
        code = act[0]
        if code == TYPE_MOVE:  # [6, uid, row, col, dir_idx]
            tm[6] = 1
            src = uid_to_token[act[1]]
            tgt = BOARD_OFFSET + (act[2] - FIELD_R0) * BOARD_COLS + (act[3] - FIELD_C0)
            opt = act[4]
            s_set = src_sets.get(6)
            if s_set is None:
                src_sets[6] = {src}
            else:
                s_set.add(src)
            k2 = (6, src)
            t_set = tgt_sets.get(k2)
            if t_set is None:
                tgt_sets[k2] = {tgt}
            else:
                t_set.add(tgt)
            k3 = (6, src, tgt)
            opt_bits[k3] = opt_bits.get(k3, 0) | (1 << opt)
            opt_to_key[(6, src, tgt, opt)] = key
            if build_key_to_tuple:
                key_to_tuple.append((6, src, tgt, opt))
            continue

        if code == COMPACT_MOVE_HAND:  # [19, uid, hand_idx]
            tm[6] = 1
            src = uid_to_token[act[1]]
            tgt = HAND_OFFSET + act[2]
            s_set = src_sets.get(6)
            if s_set is None:
                src_sets[6] = {src}
            else:
                s_set.add(src)
            k2 = (6, src)
            t_set = tgt_sets.get(k2)
            if t_set is None:
                tgt_sets[k2] = {tgt}
            else:
                t_set.add(tgt)
            tgt_to_key[(6, src, tgt)] = key
            if build_key_to_tuple:
                key_to_tuple.append((6, src, tgt, NO_ARG))
            continue

        if code == TYPE_EQUIP:  # [7, itemUid, targetUid, replaceUid_or_-1]
            tm[7] = 1
            src = uid_to_token[act[1]]
            tgt_uid = act[2]
            tgt = uid_to_token[tgt_uid]
            s_set = src_sets.get(7)
            if s_set is None:
                src_sets[7] = {src}
            else:
                s_set.add(src)
            k2 = (7, src)
            t_set = tgt_sets.get(k2)
            if t_set is None:
                tgt_sets[k2] = {tgt}
            else:
                t_set.add(tgt)
            rep_uid = act[3]
            if rep_uid < 0:
                tgt_to_key[(7, src, tgt)] = key
                if build_key_to_tuple:
                    key_to_tuple.append((7, src, tgt, NO_ARG))
            else:
                rep_slot = uid_to_items.get(tgt_uid, []).index(rep_uid)
                k3 = (7, src, tgt)
                opt_bits[k3] = opt_bits.get(k3, 0) | (1 << rep_slot)
                opt_to_key[(7, src, tgt, rep_slot)] = key
                if build_key_to_tuple:
                    key_to_tuple.append((7, src, tgt, rep_slot))
            continue

        # 0-arg actions: READY(0), REFRESH(3), FREEZE(4), LEVEL_UP(5), COMMAND_SELF(10), INFO_READY(15), BAND_SKIP(17), DIY_BRING(18)
        if len(act) == 1:
            tm[code] = 1
            type_to_key[code] = key
            if build_key_to_tuple:
                key_to_tuple.append((code, NO_ARG, NO_ARG, NO_ARG))
            continue

        # 1-arg actions
        if code == TYPE_BUY:
            src = SHOP_OFFSET + act[1]
        elif code == TYPE_SELL or code == TYPE_DESTROY:
            src = uid_to_token[act[1]]
        elif code == TYPE_REWARD:
            src = REWARD_OFFSET + act[1]
        elif code == TYPE_COMMAND_CHOICE or code == TYPE_CHOICE or code == TYPE_BOND_GROUP:
            idx = act[1]
            if not (0 <= idx < MAX_CHOICE_SLOTS):
                raise ValueError(f"Choice idx out of range: {idx}")
            src = CHOICE_OFFSET + idx
        elif code == TYPE_BAND:
            src = band_to_idx[act[1]]
        elif code == TYPE_COMMAND_TARGET:
            kind_tag = act[1]
            if kind_tag == 0:
                src = uid_to_token[act[2]]
            elif kind_tag == 1:
                src = SHOP_OFFSET + act[2]
            else:
                src = BOARD_OFFSET + (act[2] - FIELD_R0) * BOARD_COLS + (act[3] - FIELD_C0)
        else:
            raise ValueError(f"Unsupported compact action code: {act!r}")

        tm[code] = 1
        s_set = src_sets.get(code)
        if s_set is None:
            src_sets[code] = {src}
        else:
            s_set.add(src)
        src_to_key[(code, src)] = key
        if build_key_to_tuple:
            key_to_tuple.append((code, src, NO_ARG, NO_ARG))

    return (
        bytes(tm),
        type_to_key,
        {k: sorted(v) for k, v in src_sets.items()},
        src_to_key,
        {k: sorted(v) for k, v in tgt_sets.items()},
        tgt_to_key,
        opt_bits,
        opt_to_key,
        key_to_tuple,
    )


def _build_action_tree(
    actions: Sequence[Mapping[str, Any]],
    uid_to_token: Mapping[int, int],
    uid_to_items: Mapping[int, list[int]],
    catalog: GameCatalog,
) -> ActionTree:
    tm_bytes, type_to_key, src_keys, src_to_key, tgt_keys, tgt_to_key, opt_bits, opt_to_key, key_to_tuple = (
        _index_actions_dict(actions, uid_to_token, uid_to_items, catalog, build_key_to_tuple=True)
    )
    type_mask = torch.from_numpy(np.frombuffer(tm_bytes, dtype=np.bool_).copy())
    opt_keys = {k: _BITS_TO_OPTS[bits] for k, bits in opt_bits.items()}
    return ActionTree(
        type_mask=type_mask,
        type_to_key=type_to_key,
        src_keys=src_keys,
        src_to_key=src_to_key,
        tgt_keys=tgt_keys,
        tgt_to_key=tgt_to_key,
        opt_keys=opt_keys,
        opt_to_key=opt_to_key,
        key_to_tuple=key_to_tuple,
        num_bands=catalog.num_bands,
        opt_bits=opt_bits,
        _type_mask_bytes=tm_bytes,
    )


def _fill_features_numpy(
    frame: Mapping[str, Any],
    cat: GameCatalog,
    cat_np: np.ndarray,
    cont_np: np.ndarray,
    valid_np: np.ndarray,
    prep_steps: int,
) -> tuple[dict[int, int], dict[int, list[int]]]:
    """Populate `(cat_np [114,8] int32, cont_np [114,28] float32, valid_np [114] bool)` in place."""
    pub = frame["pub"]
    own = frame["own"]
    phase = frame["phase"]
    detail = frame.get("detail")

    rnd = int(pub.get("round") or 0)
    last_rnd = max(1, int(pub.get("lastRound") or 14))
    boss_round = int(pub.get("bossRound") or 99)
    hidden_round = int(pub.get("hiddenRound") or 99)
    is_boss = rnd >= boss_round
    is_hidden = rnd >= hidden_round
    stage_id = pub.get("stageId")
    mode_id = pub.get("modeId")
    difficulty = pub.get("difficulty") or "FUNNY"

    # Fast copy of precomputed static stage/board/slot/bond template
    cat_base, cont_base, valid_base = _get_stage_template(cat, stage_id, is_boss)
    np.copyto(cat_np, cat_base)
    np.copyto(cont_np, cont_base)
    np.copyto(valid_np, valid_base)

    log1p = math.log1p
    chess_meta = cat.chess_meta
    item_meta = cat.item_meta
    enemy_meta = cat.enemy_meta
    entity_idx_fn = cat.entity_idx
    item_idx_fn = cat.item_idx
    token_pos_class = cat.token_pos_class
    dir_to_idx = DIR_TO_IDX
    spatial_grid = cat.stage_spatial(stage_id, is_boss=is_boss)
    atk_scale, hp_scale, spd_scale = cat.enemy_scale(mode_id, max(1, rnd))

    stats_by_uid: dict[int, Mapping[str, Any]] = {}
    if detail:
        d_units = detail.get("units")
        if d_units:
            for u in d_units:
                uid = u.get("uid")
                if uid is not None:
                    stats_by_uid[int(uid)] = u

    owned_base_counts: dict[int, int] = {}
    owned_item_counts: dict[int, int] = {}
    owned_item_merge_counts: dict[int, int] = {}
    board_base_indices: set[int] = set()
    board_uids: set[int] = set()
    uid_to_token: dict[int, int] = {}
    uid_to_items: dict[int, list[int]] = {}

    board_list = own.get("board") or ()
    hand_list = own.get("hand") or ()
    temp_list = own.get("temp") or ()

    for p in board_list:
        if p:
            board_uids.add(int(p["uid"]))
            kind = p.get("kind")
            if kind == "chess":
                cm = chess_meta.get(p.get("id"))
                if cm:
                    board_base_indices.add(cm.base_idx)
                    if not p.get("golden"):
                        bi = cm.base_idx
                        owned_base_counts[bi] = owned_base_counts.get(bi, 0) + 1
                for eq_it in p.get("items") or ():
                    im = item_meta.get(eq_it.get("id"))
                    if im and not im.is_golden:
                        ei = im.entity_idx
                        owned_item_merge_counts[ei] = owned_item_merge_counts.get(ei, 0) + 1
            elif kind == "item" and not p.get("golden"):
                im = item_meta.get(p.get("id"))
                if im:
                    ei = im.entity_idx
                    owned_item_counts[ei] = owned_item_counts.get(ei, 0) + 1
                    owned_item_merge_counts[ei] = owned_item_merge_counts.get(ei, 0) + 1
    free_hand = 0
    unequipped_item_cnt = 0
    for p in hand_list:
        if p is None:
            free_hand += 1
        else:
            kind = p.get("kind")
            if kind == "item":
                unequipped_item_cnt += 1
                if not p.get("golden"):
                    im = item_meta.get(p.get("id"))
                    if im:
                        ei = im.entity_idx
                        owned_item_counts[ei] = owned_item_counts.get(ei, 0) + 1
                        owned_item_merge_counts[ei] = owned_item_merge_counts.get(ei, 0) + 1
            elif kind == "chess":
                if not p.get("golden"):
                    cm = chess_meta.get(p.get("id"))
                    if cm:
                        bi = cm.base_idx
                        owned_base_counts[bi] = owned_base_counts.get(bi, 0) + 1
                for eq_it in p.get("items") or ():
                    im = item_meta.get(eq_it.get("id"))
                    if im and not im.is_golden:
                        ei = im.entity_idx
                        owned_item_merge_counts[ei] = owned_item_merge_counts.get(ei, 0) + 1
    free_temp = 0
    for p in temp_list:
        if p is None:
            free_temp += 1
        else:
            kind = p.get("kind")
            if kind == "item":
                unequipped_item_cnt += 1
                if not p.get("golden"):
                    im = item_meta.get(p.get("id"))
                    if im:
                        ei = im.entity_idx
                        owned_item_counts[ei] = owned_item_counts.get(ei, 0) + 1
                        owned_item_merge_counts[ei] = owned_item_merge_counts.get(ei, 0) + 1
            elif kind == "chess":
                if not p.get("golden"):
                    cm = chess_meta.get(p.get("id"))
                    if cm:
                        bi = cm.base_idx
                        owned_base_counts[bi] = owned_base_counts.get(bi, 0) + 1
                for eq_it in p.get("items") or ():
                    im = item_meta.get(eq_it.get("id"))
                    if im and not im.is_golden:
                        ei = im.entity_idx
                        owned_item_merge_counts[ei] = owned_item_merge_counts.get(ei, 0) + 1

    # ---- Pre-pass [80..105] BONDS dynamic state & macro synergy summary ----
    bond_to_idx = cat.bond_to_idx
    core_bond_mask = cat.core_bond_mask
    total_activated_layers = 0
    active_core_cnt = 0
    max_core_tier = 0
    active_addon_cnt = 0
    active_core_bonds: set[int] = set()
    bond_counts: dict[int, int] = {}
    bond_next_delta: dict[int, int] = {}

    for bid in pub.get("disabledBonds") or ():
        bi = bond_to_idx.get(bid)
        if bi is not None:
            cont_np[BOND_OFFSET + bi, 1] = 1.0
    for b in own.get("bonds") or ():
        bi = bond_to_idx.get(b["bondId"])
        if bi is None:
            continue
        f = cont_np[BOND_OFFSET + bi]
        is_act = bool(b.get("active"))
        cnt = int(b.get("count") or 0)
        tier = int(b.get("tier") or 0)
        layers = int(b.get("layers") or 0)
        bond_counts[bi] = cnt
        if is_act:
            f[4] = 1.0
            if layers > 0:
                total_activated_layers += layers
            if core_bond_mask[bi]:
                active_core_cnt += 1
                active_core_bonds.add(bi)
                if tier > max_core_tier:
                    max_core_tier = tier
            else:
                active_addon_cnt += 1
        f[5] = float(cnt) / 8.0
        f[6] = float(tier) * 0.25
        if layers > 0:
            f[7] = log1p(layers) / 7.0
            f[10] = (700.0 if layers > 700 else float(layers)) / 350.0
        ths = b.get("thresholds")
        next_delta = 0
        if ths:
            for th in ths:
                if th > cnt:
                    next_delta = int(th - cnt)
                    f[8] = float(next_delta) / 6.0
                    break
            if next_delta == 1:
                f[9] = 1.0
            elif next_delta == 0 and cnt >= ths[-1]:
                f[11] = 1.0
        bond_next_delta[bi] = next_delta

    # ---- [2] MATCH_STAGE token ---------------------------------------------
    phase_idx = PHASE_TO_IDX.get(phase, 0)
    inv_last_rnd = 1.0 / float(last_rnd)
    diff_idx = DIFF_TO_IDX.get(difficulty, 2)
    bond_choice = pub.get("bondChoice")
    lp_val = int(own.get("lp") or 0)
    own_stats = own.get("stats") or {}
    leaks_val = int(own_stats.get("leaks") or 0)
    ms = cont_np[MATCH_STAGE_IDX]
    if 0 <= phase_idx < 5:
        ms[phase_idx] = 1.0
    ms[5] = rnd * inv_last_rnd
    ms[6] = max(0, boss_round - rnd) * inv_last_rnd
    if is_boss:
        ms[7] = 1.0
    if is_hidden:
        ms[8] = 1.0
    if pub.get("spRound"):
        ms[9] = 1.0
    ms[10] = float(diff_idx) / max(1, len(DIFF_TO_IDX) - 1)
    ms[11] = float(lp_val) / 30.0
    if own.get("alive", True):
        ms[12] = 1.0
    if bond_choice and bond_choice.get("picked") is None:
        ms[13] = 1.0
    fac_to_idx = cat.faction_to_idx
    for fac in pub.get("factions") or ():
        fi = fac_to_idx.get(fac)
        if fi is not None and 14 + fi < 23:
            ms[14 + fi] = 1.0
    ms[23] = (700.0 if total_activated_layers > 700 else float(total_activated_layers)) / 350.0
    if total_activated_layers > 350 and lp_val > 1:
        ms[24] = 1.0
    ms[25] = atk_scale / 3.0
    ms[26] = hp_scale / 4.0
    if leaks_val > 0:
        ms[27] = log1p(leaks_val) / 4.0

    # ---- [3] BOSS token ----------------------------------------------------
    hidden_boss_id = pub.get("hiddenBossId")
    active_boss_id = hidden_boss_id if is_hidden else pub.get("bossId")
    cat_np[BOSS_IDX, CAT_ENTITY] = entity_idx_fn(active_boss_id)
    bf = cont_np[BOSS_IDX]
    if active_boss_id:
        bf[0] = 1.0
    if is_boss:
        bf[1] = 1.0
    if is_hidden:
        bf[2] = 1.0
    if hidden_boss_id:
        bf[3] = 1.0
    bf[4] = max(0, boss_round - rnd) * inv_last_rnd
    boss_pool_hp = cat.boss_pool_hp(active_boss_id, mode_id, difficulty)
    if boss_pool_hp > 0.0:
        bf[5] = log1p(boss_pool_hp) / 16.0
        bf[16] = (2_500_000.0 if boss_pool_hp > 2_500_000.0 else boss_pool_hp) / 1_000_000.0
    bm = cat.boss_meta.get(active_boss_id) if active_boss_id else None
    if bm is not None:
        bf[10] = float(bm.lpr) / 30.0
        bem = enemy_meta.get(bm.enemy_key)
        if bem is not None:
            if bem.atk > 0.0:
                bf[6] = log1p(bem.atk) / 10.0
            if bem.def_ > 0.0:
                bf[7] = log1p(bem.def_) / 8.0
            bf[8] = bem.res / 100.0
            bf[9] = bem.move_speed / 2.0
            if bem.is_arts:
                bf[11] = 1.0
            if bem.is_ranged:
                bf[12] = 1.0

    # ---- [4] ECONOMY_SHOP token (board aggregates filled after board pass) -
    shop = own.get("shop") or {}
    funds = int(own.get("funds") or 0)
    shop_lv = int(shop.get("level") or 1)
    max_lv = max(1, int(shop.get("maxLevel") or 6))
    upg_price = int(shop.get("upgradePrice") or 0)
    ref_price = int(shop.get("refreshPrice") or 0)
    free_ref = int(shop.get("freeRefreshes") or 0)
    dep_cnt = int(own.get("deployCount") or 0)
    dep_cap = max(1, int(own.get("deployCap") or 8))

    ec = cont_np[ECONOMY_SHOP_IDX]
    ec[0] = (log1p(funds) / 4.0) if funds > 0 else 0.0
    ec[1] = (30.0 if funds > 30 else float(funds)) / 30.0
    ec[2] = float(shop_lv) / float(max_lv)
    if shop_lv >= max_lv:
        ec[3] = 1.0
    ec[4] = float(upg_price) / 15.0
    if shop_lv < max_lv and funds >= upg_price:
        ec[5] = 1.0
    ec[6] = float(ref_price) / 5.0
    ec[7] = float(free_ref) / 5.0
    if free_ref > 0 or funds >= ref_price:
        ec[8] = 1.0
    if shop.get("frozen"):
        ec[9] = 1.0
    ec[10] = float(dep_cnt) / float(dep_cap)
    ec[11] = float(dep_cap) / 12.0
    ec[12] = float(max(0, dep_cap - dep_cnt)) / 8.0
    ec[13] = float(free_hand) / 10.0
    ec[14] = float(free_temp) / 5.0
    ec[15] = float(own.get("stashCount") or 0) / 5.0
    if own.get("canReady"):
        ec[16] = 1.0
    reward_offer = shop.get("rewardOffer")
    if reward_offer:
        ec[17] = 1.0
    personal_choice = own.get("personalChoice")
    if personal_choice:
        ec[18] = 1.0
    ec[19] = min(float(max(0, prep_steps)), 80.0) / 80.0
    if prep_steps >= 40:
        ec[20] = 1.0

    # ---- [5] BAND_COMMAND token --------------------------------------------
    band_id = own.get("bandId")
    cmd = own.get("command") or {}
    if band_id:
        bidx = cat.band_to_idx.get(band_id)
        if bidx is not None:
            cat_np[BAND_COMMAND_IDX, CAT_BAND] = bidx + 1
    cat_np[BAND_COMMAND_IDX, CAT_ENTITY] = entity_idx_fn(cmd.get("effectId"))

    bc = cont_np[BAND_COMMAND_IDX]
    cmd_type = cmd.get("type") or ""
    if band_id:
        bc[0] = 1.0
    if cmd_type == "ACTIVE":
        bc[1] = 1.0
    elif cmd_type == "PASSIVE":
        bc[2] = 1.0
    if cmd.get("usable"):
        bc[3] = 1.0
    cmd_cost = int(cmd.get("currentCost") or 0)
    bc[4] = float(cmd_cost) / 10.0
    if cmd_type == "ACTIVE" and funds >= cmd_cost:
        bc[5] = 1.0
    bc[6] = float(cmd.get("cooldown") or 0) / 5.0
    uses_left = cmd.get("usesLeft")
    if uses_left is not None:
        bc[7] = float(uses_left) / 5.0
    else:
        bc[8] = 1.0
    cmd_draft = own.get("commandDraft")
    if cmd_draft:
        bc[9] = 1.0
    tgt_type = cmd.get("targetType") or ""
    if tgt_type == "NO_TARGET":
        bc[10] = 1.0
    elif tgt_type == "COMMON_CHESS":
        bc[11] = 1.0
    elif tgt_type == "SHOP_SLOT":
        bc[12] = 1.0
    elif tgt_type == "MAP_TILE":
        bc[13] = 1.0
    bc[17] = float(active_core_cnt) / 4.0
    bc[18] = float(max_core_tier) / 3.0
    bc[19] = float(active_addon_cnt) / 8.0

    # ---- [6..11] EFFECTS tokens (prioritize band/command & counted buffs) --
    raw_effects = own.get("effects")
    if raw_effects:
        valid_effects = [e for e in raw_effects if e]
        for eff in valid_effects:
            prog = eff.get("progress")
            rprog = eff.get("roundProgress")
            if prog or rprog:
                bc[16] = 1.0
                if isinstance(prog, Mapping):
                    tot = max(1, int(prog.get("total") or 1))
                    bc[14] = float(prog.get("current") or 0) / float(tot)
                if isinstance(rprog, Mapping):
                    rtot = max(1, int(rprog.get("total") or 1))
                    bc[15] = float(rprog.get("current") or 0) / float(rtot)
                break
        if len(valid_effects) > MAX_EFFECT_SLOTS:
            valid_effects.sort(
                key=lambda e: (
                    0
                    if (e.get("iconKind") == "band" or e.get("command"))
                    else (1 if (e.get("counter") is not None or e.get("progress") is not None) else 2)
                )
            )
            eff_overflow = 1.0
        else:
            eff_overflow = 0.0
        inv_eff = 1.0 / max(1.0, float(MAX_EFFECT_SLOTS - 1))
        for i, eff in enumerate(valid_effects[:MAX_EFFECT_SLOTS]):
            tok_idx = EFFECT_OFFSET + i
            valid_np[tok_idx] = True
            cat_np[tok_idx, CAT_ENTITY] = entity_idx_fn(eff.get("id"))
            ef = cont_np[tok_idx]
            ef[0] = 1.0
            if eff.get("command"):
                ef[1] = 1.0
            if eff.get("commandType") == "ACTIVE":
                ef[2] = 1.0
            ik = eff.get("iconKind")
            if ik == "team":
                ef[3] = 1.0
            elif ik == "band":
                ef[4] = 1.0
            ef[5] = float(i) * inv_eff
            cnt_eff = eff.get("counter")
            if cnt_eff is not None:
                ef[6] = 1.0
                cval = float(cnt_eff)
                if cval > 0.0:
                    ef[7] = (15.0 if cval > 15.0 else cval) / 15.0
            prog = eff.get("progress")
            if isinstance(prog, Mapping):
                tot = max(1, int(prog.get("total") or 1))
                ef[8] = float(prog.get("current") or 0) / float(tot)
            if eff_overflow > 0.0:
                ef[9] = 1.0

    # ---- Helper to populate a piece (chess / item / token) -----------------
    def _fill_piece(tok_idx: int, p: Mapping[str, Any], on_board: bool) -> None:
        uid = int(p["uid"])
        uid_to_token[uid] = tok_idx
        items = p.get("items") or ()
        if items:
            uid_to_items[uid] = [int(it["uid"]) for it in items]
            cat_np[tok_idx, CAT_EQUIP0] = item_idx_fn(items[0].get("id"))
            if len(items) > 1:
                cat_np[tok_idx, CAT_EQUIP1] = item_idx_fn(items[1].get("id"))
        else:
            uid_to_items[uid] = []

        kind = p.get("kind")
        pid = p.get("id")
        cat_np[tok_idx, CAT_ENTITY] = entity_idx_fn(pid)
        if on_board:
            cat_np[tok_idx, CAT_DIR] = dir_to_idx.get(p.get("dir", "RIGHT"), 0) + 1

        f = cont_np[tok_idx]
        f[0] = 1.0
        cm = None
        im = None
        if kind == "chess":
            f[1] = 1.0
            cm = chess_meta.get(pid)
            if cm:
                f[8] = float(cm.pos_class) * 0.5
                f[9] = float(owned_base_counts.get(cm.base_idx, 0)) / 3.0
                f[10] = float(cm.sell_price) / 6.0
        elif kind == "item":
            f[2] = 1.0
            im = item_meta.get(pid)
            if im:
                f[9] = float(owned_item_counts.get(im.entity_idx, 0)) * 0.5
        elif kind == "token":
            f[3] = 1.0
            f[8] = float(token_pos_class.get(pid, 0)) * 0.5
            owner_uid = p.get("ownerUid")
            if owner_uid is not None and int(owner_uid) in board_uids:
                f[9] = 1.0

        if p.get("golden"):
            f[4] = 1.0
        f[5] = float(p.get("tier") or 0) / 6.0
        f[6] = float(p.get("count") or 1) / 3.0
        f[7] = float(len(items)) * 0.5

        st = stats_by_uid.get(uid)
        if st:
            hp = st.get("maxHp")
            if hp and hp > 0:
                f[11] = log1p(hp) / 10.0
            atk = st.get("atk")
            if atk and atk > 0:
                f[12] = log1p(atk) / 8.0
            df = st.get("def")
            if df and df > 0:
                f[13] = log1p(df) / 8.0
            f[14] = float(st.get("res") or 0) / 100.0
            iv = st.get("interval")
            if iv and iv > 0:
                f[15] = 1.0 / float(iv)
            f[16] = float(st.get("blockCnt") or 0) * 0.25
            rlen = st.get("rangeLen")
            if rlen is None:
                rlen = len(st.get("range") or ())
            f[17] = float(rlen) / 20.0

        # Non-board pieces (HAND/TEMP) have free cols [19..27]; BOARD uses [18..27] for spatial terrain!
        if not on_board:
            prog = p.get("progress")
            if isinstance(prog, Mapping):
                tot = max(1, int(prog.get("total") or 1))
                f[19] = float(prog.get("current") or 0) / float(tot)
            elif items:
                best_pr = 0.0
                for it in items:
                    ipr = it.get("progress")
                    if isinstance(ipr, Mapping):
                        pr = float(ipr.get("current") or 0) / float(max(1, int(ipr.get("total") or 1)))
                        if pr > best_pr:
                            best_pr = pr
                if best_pr > 0.0:
                    f[19] = best_pr
            stag = p.get("showTag")
            if isinstance(stag, Sequence) and not isinstance(stag, (str, bytes)) and len(stag) > 0:
                sv = float(stag[0] or 0)
                if sv > 0.0:
                    f[20] = log1p(sv) / 4.0
            gtags = p.get("garrisonTags")
            if isinstance(gtags, Mapping) and gtags:
                gsum = 0.0
                for vals in gtags.values():
                    if isinstance(vals, Sequence) and not isinstance(vals, (str, bytes)) and len(vals) > 0:
                        v0 = float(vals[0] or 0)
                        if v0 > 0.0:
                            gsum += v0
                if gsum > 0.0:
                    f[21] = log1p(gsum) / 5.0
            if im is not None:
                if im.give_bond_idx >= 0 or im.can_give_bond:
                    f[22] = 1.0
                if im.give_bond_idx >= 0 and bond_next_delta.get(im.give_bond_idx, 0) == 1:
                    f[23] = 1.0
            elif cm is not None and cm.base_idx not in board_base_indices:
                for bi in cm.bond_indices:
                    if bond_next_delta.get(bi, 0) == 1:
                        f[23] = 1.0
                        break

    # ---- [12..47] BOARD pieces (terrain already copied from stage template) -
    board_melee = 0
    board_ranged = 0
    total_block = 0
    path_block = 0
    board_chess_cnt = 0
    equipped_items_cnt = 0
    golden_board_cnt = 0
    for p in board_list:
        if p:
            tile_idx = (int(p["row"]) - FIELD_R0) * BOARD_COLS + (int(p["col"]) - FIELD_C0)
            tok_idx = BOARD_OFFSET + tile_idx
            _fill_piece(tok_idx, p, on_board=True)
            kind = p.get("kind")
            pid = p.get("id")
            st = stats_by_uid.get(int(p["uid"]))
            if kind == "chess":
                board_chess_cnt += 1
                equipped_items_cnt += len(p.get("items") or ())
                if p.get("golden"):
                    golden_board_cnt += 1
                cm = chess_meta.get(pid)
                pos_cls = cm.pos_class if cm else 0
                blk = int(st.get("blockCnt") or 0) if st else (cm.block_cnt if cm else 0)
            elif kind == "token":
                pos_cls = token_pos_class.get(pid, 0)
                blk = int(st.get("blockCnt") or 0) if st else 0
            else:
                pos_cls = 0
                blk = 0
            if pos_cls == 1:
                board_melee += 1
            elif pos_cls == 2:
                board_ranged += 1
            if blk > 0:
                total_block += blk
                if 0 <= tile_idx < BOARD_TILES and spatial_grid[tile_idx].is_ground_path:
                    path_block += blk

    ec[21] = float(board_melee) / 8.0
    ec[22] = float(board_ranged) / 8.0
    ec[23] = (20.0 if total_block > 20 else float(total_block)) / 20.0
    ec[24] = (16.0 if path_block > 16 else float(path_block)) / 16.0
    ec[25] = float(equipped_items_cnt) / float(max(1, 2 * board_chess_cnt))
    if unequipped_item_cnt > 0:
        ec[26] = 1.0
    ec[27] = float(golden_board_cnt) / float(dep_cap)

    # ---- [48..57] HAND_SLOTS (10) & [58..62] TEMP_SLOTS (5) ----------------
    for i, p in enumerate(hand_list[:10]):
        if p is not None:
            _fill_piece(HAND_OFFSET + i, p, on_board=False)

    for i, p in enumerate(temp_list[:5]):
        if p is not None:
            _fill_piece(TEMP_OFFSET + i, p, on_board=False)

    # ---- [63..68] SHOP_SLOTS (6) & [69..71] REWARD_SLOTS (3) ---------------
    def _fill_offer_synergy(f: np.ndarray, kind: str | None, sid: str | None) -> None:
        if kind == "chess":
            f[1] = 1.0
            cm = chess_meta.get(sid)
            if cm:
                f[7] = float(cm.tier) / 6.0
                f[8] = float(cm.pos_class) * 0.5
                owned = owned_base_counts.get(cm.base_idx, 0)
                f[9] = float(owned) / 3.0
                if not cm.is_golden and owned + 1 >= cm.upgrade_num:
                    f[10] = 1.0
                new_base = cm.base_idx not in board_base_indices
                owned_matches = 0
                for bi in cm.bond_indices:
                    if new_base and bond_next_delta.get(bi, 0) == 1:
                        f[11] = 1.0
                    if bi in active_core_bonds:
                        f[12] = 1.0
                    if bond_counts.get(bi, 0) > 0:
                        owned_matches += 1
                if owned_matches > 0:
                    f[13] = (3.0 if owned_matches > 3 else float(owned_matches)) / 3.0
                if cm.block_cnt > 0:
                    f[15] = float(cm.block_cnt) / 4.0
                if cm.is_arts:
                    f[16] = 1.0
        elif kind == "item":
            f[2] = 1.0
            im = item_meta.get(sid)
            if im:
                f[7] = float(im.tier) / 6.0
                owned_it = owned_item_counts.get(im.entity_idx, 0)
                f[9] = float(owned_it) * 0.5
                if im.mergeable and not im.is_golden:
                    need = im.upgrade_num if im.upgrade_num > 1 else 2
                    if need < 100 and owned_item_merge_counts.get(im.entity_idx, 0) + 1 >= need:
                        f[10] = 1.0
                gb = im.give_bond_idx
                if gb >= 0:
                    if bond_next_delta.get(gb, 0) == 1:
                        f[11] = 1.0
                    if gb in active_core_bonds:
                        f[12] = 1.0
                    if bond_counts.get(gb, 0) > 0:
                        f[13] = 1.0 / 3.0
                if gb >= 0 or im.can_give_bond:
                    f[14] = 1.0

    shop_slots = shop.get("slots")
    if shop_slots:
        for i, s in enumerate(shop_slots[:6]):
            if s is None:
                continue
            tok_idx = SHOP_OFFSET + i
            valid_np[tok_idx] = True
            sid = s.get("id")
            kind = s.get("kind")
            cat_np[tok_idx, CAT_ENTITY] = entity_idx_fn(sid)
            f = cont_np[tok_idx]
            f[0] = 1.0
            sold = bool(s.get("sold"))
            if sold:
                f[3] = 1.0
            if s.get("frozen"):
                f[4] = 1.0
            price = int(s.get("price") or 0)
            f[5] = float(price) / 6.0
            if not sold and price <= funds:
                f[6] = 1.0
            _fill_offer_synergy(f, kind, sid)

    if reward_offer:
        reward_slots = reward_offer.get("slots")
        if reward_slots:
            for i, s in enumerate(reward_slots[:3]):
                if s is None:
                    continue
                tok_idx = REWARD_OFFSET + i
                valid_np[tok_idx] = True
                sid = s.get("id")
                kind = s.get("kind")
                cat_np[tok_idx, CAT_ENTITY] = entity_idx_fn(sid)
                f = cont_np[tok_idx]
                f[0] = 1.0
                if s.get("sold"):
                    f[3] = 1.0
                _fill_offer_synergy(f, kind, sid)

    # ---- [72..79] CHOICE_CARDS (8) -----------------------------------------
    bond_group_members0: Sequence[str] = ()
    bond_group_members1: Sequence[str] = ()
    if phase == "INFO_CHECK" and bond_choice and bond_choice.get("picked") is None:
        groups = bond_choice.get("groups") or ()
        core_pos = {bi: pos for pos, bi in enumerate(cat.core_bond_indices[:10])}
        for i, grp in enumerate(groups[:MAX_CHOICE_SLOTS]):
            tok_idx = CHOICE_OFFSET + i
            valid_np[tok_idx] = True
            cat_np[tok_idx, CAT_SEG] = SEG_CHOICE
            f = cont_np[tok_idx]
            f[0] = 1.0
            f[1] = 1.0
            f[5] = float(len(grp)) / 6.0
            f[18] = float(i)
            for bid in grp:
                bi = bond_to_idx.get(bid)
                if bi is not None:
                    cpos = core_pos.get(bi)
                    if cpos is not None:
                        f[6 + cpos] = 1.0
            if i == 0:
                bond_group_members0 = grp
            elif i == 1:
                bond_group_members1 = grp
    elif phase == "SP_DRAFT" and pub.get("sp"):
        cards = pub["sp"].get("cards") or ()
        for c in cards[:MAX_CHOICE_SLOTS]:
            idx = int(c.get("idx", 0))
            if not (0 <= idx < MAX_CHOICE_SLOTS):
                continue
            tok_idx = CHOICE_OFFSET + idx
            valid_np[tok_idx] = True
            cat_np[tok_idx, CAT_SEG] = SEG_CHOICE
            cat_np[tok_idx, CAT_ENTITY] = entity_idx_fn(c.get("id"))
            f = cont_np[tok_idx]
            f[0] = 1.0
            f[2] = 1.0
            if c.get("soldOut"):
                f[3] = 1.0
            f[5] = float(c.get("tier") or 0) / 6.0
            f[18] = float(idx) / 7.0
    elif personal_choice:
        cards = personal_choice.get("cards") or ()
        for i, c in enumerate(cards[:MAX_CHOICE_SLOTS]):
            tok_idx = CHOICE_OFFSET + i
            valid_np[tok_idx] = True
            cat_np[tok_idx, CAT_SEG] = SEG_CHOICE
            cat_np[tok_idx, CAT_ENTITY] = entity_idx_fn(c.get("id"))
            f = cont_np[tok_idx]
            f[0] = 1.0
            f[2] = 1.0
            f[5] = float(c.get("tier") or 0) / 6.0
            f[18] = float(i) / 7.0
    elif cmd_draft:
        cards = cmd_draft.get("cards") or ()
        for i, c in enumerate(cards[:MAX_CHOICE_SLOTS]):
            tok_idx = CHOICE_OFFSET + i
            valid_np[tok_idx] = True
            cat_np[tok_idx, CAT_SEG] = SEG_CHOICE
            cat_np[tok_idx, CAT_ENTITY] = entity_idx_fn(c.get("effectId"))
            f = cont_np[tok_idx]
            f[0] = 1.0
            f[4] = 1.0
            f[5] = float(c.get("cost") or 0) / 10.0
            f[18] = float(i) / 7.0

    # Link INFO_CHECK bond groups onto BONDS tokens [80..105] cols 2..3
    for bid in bond_group_members0:
        bi = bond_to_idx.get(bid)
        if bi is not None:
            cont_np[BOND_OFFSET + bi, 2] = 1.0
    for bid in bond_group_members1:
        bi = bond_to_idx.get(bid)
        if bi is not None:
            cont_np[BOND_OFFSET + bi, 3] = 1.0

    # ---- [106..113] ENEMIES (up to 8) --------------------------------------
    enemies = own.get("nextEnemies")
    if enemies:
        agg: dict[tuple[str, str], list[Any]] = {}
        boss_r0 = FIELD_R0 - BOSS_ROW_OFFSET
        for e in enemies:
            ek = e.get("enemyKey") or ""
            gt = e.get("gate") or "lower"
            k_eg = (ek, gt)
            rec = agg.get(k_eg)
            cnt_e = int(e.get("count") or 1)
            t_e = float(e.get("t") or 0.0)
            is_bounty = e.get("source") == "bounty"
            st_pos = e.get("start")
            valid_start = (
                st_pos
                if isinstance(st_pos, Sequence) and not isinstance(st_pos, (str, bytes)) and len(st_pos) >= 2
                else None
            )
            if rec is None:
                agg[k_eg] = [
                    ek,
                    gt,
                    cnt_e,
                    t_e,
                    bool(e.get("fly")),
                    bool(e.get("elite")),
                    bool(e.get("boss")),
                    is_bounty,
                    valid_start,
                ]
            else:
                rec[2] += cnt_e
                if t_e < rec[3]:
                    rec[3] = t_e
                if is_bounty:
                    rec[7] = True
                if rec[8] is None and valid_start is not None:
                    rec[8] = valid_start
        ranked = sorted(
            agg.values(),
            key=lambda x: (not x[6], not x[5], -x[2], x[3]),
        )[:MAX_ENEMY_SLOTS]
        boss_start_set = False
        for i, rec in enumerate(ranked):
            tok_idx = ENEMY_OFFSET + i
            valid_np[tok_idx] = True
            cat_np[tok_idx, CAT_SEG] = SEG_ENEMY
            cat_np[tok_idx, CAT_ENTITY] = entity_idx_fn(rec[0])
            f = cont_np[tok_idx]
            f[0] = 1.0
            f[1] = log1p(rec[2]) / 4.0 if rec[2] > 0 else 0.0
            if rec[1] == "upper":
                f[2] = 1.0
            f[3] = (120.0 if rec[3] > 120.0 else rec[3]) / 120.0
            if rec[4]:
                f[4] = 1.0
            if rec[5]:
                f[5] = 1.0
            is_e_boss = bool(rec[6])
            if is_e_boss:
                f[6] = 1.0
            em = enemy_meta.get(rec[0])
            if em is not None:
                if is_e_boss:
                    hp_eff = boss_pool_hp if boss_pool_hp > 0.0 else em.max_hp
                    atk_eff = em.atk
                    spd_eff = em.move_speed
                else:
                    hp_eff = em.max_hp * hp_scale
                    atk_eff = em.atk * atk_scale
                    spd_eff = em.move_speed * spd_scale
                if hp_eff > 0.0:
                    f[7] = log1p(hp_eff) / 14.0
                if atk_eff > 0.0:
                    f[8] = log1p(atk_eff) / 10.0
                if em.def_ > 0.0:
                    f[9] = log1p(em.def_) / 8.0
                f[10] = em.res / 100.0
                if spd_eff > 0.0:
                    f[11] = (3.0 if spd_eff > 3.0 else spd_eff) / 2.5
                f[12] = (10.0 if em.lpr > 10 else float(em.lpr)) / 5.0
                if em.is_arts:
                    f[13] = 1.0
                if em.is_ranged:
                    f[14] = 1.0
            if rec[7]:
                f[15] = 1.0
            st_pos = rec[8]
            if st_pos is not None:
                sr_norm = float(float(st_pos[0]) - boss_r0) / 3.0
                sc_norm = float(float(st_pos[1]) - FIELD_C0) / 8.0
                f[16] = sr_norm
                f[17] = sc_norm
                if is_e_boss and not boss_start_set:
                    bf[13] = 1.0
                    bf[14] = sr_norm
                    bf[15] = sc_norm
                    if rec[1] == "upper":
                        bf[17] = 1.0
                    boss_start_set = True
            elif is_e_boss and not boss_start_set:
                if rec[1] == "upper":
                    bf[17] = 1.0
                boss_start_set = True

    return uid_to_token, uid_to_items


def encode_frame_into(
    frame: Mapping[str, Any],
    catalog: GameCatalog,
    cat_out: np.ndarray,
    cont_out: np.ndarray,
    valid_out: np.ndarray,
    *,
    prep_steps: int = 0,
) -> FastActionDecoder:
    """Zero-tensor hot path for multiprocess actor workers: fills SharedMemory NumPy buffers directly."""
    uid_to_token, uid_to_items = _fill_features_numpy(frame, catalog, cat_out, cont_out, valid_out, prep_steps)
    cactions = frame.get("cactions")
    if cactions is not None:
        tm_bytes, type_to_key, src_keys, src_to_key, tgt_keys, tgt_to_key, opt_bits, opt_to_key, _ = (
            _index_actions_compact(cactions, uid_to_token, uid_to_items, catalog, build_key_to_tuple=False)
        )
    else:
        tm_bytes, type_to_key, src_keys, src_to_key, tgt_keys, tgt_to_key, opt_bits, opt_to_key, _ = (
            _index_actions_dict(frame["actions"], uid_to_token, uid_to_items, catalog, build_key_to_tuple=False)
        )
    spec = ActionMaskSpec(
        type_mask_bytes=tm_bytes,
        src_keys=src_keys,
        tgt_keys=tgt_keys,
        opt_bits=opt_bits,
        num_bands=catalog.num_bands,
    )
    return FastActionDecoder(
        mask_spec=spec,
        type_to_key=type_to_key,
        src_to_key=src_to_key,
        tgt_to_key=tgt_to_key,
        opt_to_key=opt_to_key,
    )


def encode_frame(
    frame: Mapping[str, Any],
    catalog: GameCatalog | None = None,
    *,
    prep_steps: int = 0,
) -> EncodedFrame:
    """Convert one simulator decision frame into 114 semantic tokens + exact hierarchical masks."""
    cat = catalog if catalog is not None else get_catalog()
    cat_np = np.empty((NUM_TOKENS, NUM_CAT_FIELDS), dtype=np.int32)
    cont_np = np.empty((NUM_TOKENS, CONT_DIM), dtype=np.float32)
    valid_np = np.empty(NUM_TOKENS, dtype=np.bool_)

    uid_to_token, uid_to_items = _fill_features_numpy(frame, cat, cat_np, cont_np, valid_np, prep_steps)

    cactions = frame.get("cactions")
    if cactions is not None:
        tm_bytes, type_to_key, src_keys, src_to_key, tgt_keys, tgt_to_key, opt_bits, opt_to_key, key_to_tuple = (
            _index_actions_compact(cactions, uid_to_token, uid_to_items, cat, build_key_to_tuple=True)
        )
    else:
        tm_bytes, type_to_key, src_keys, src_to_key, tgt_keys, tgt_to_key, opt_bits, opt_to_key, key_to_tuple = (
            _index_actions_dict(frame["actions"], uid_to_token, uid_to_items, cat, build_key_to_tuple=True)
        )

    type_mask = torch.from_numpy(np.frombuffer(tm_bytes, dtype=np.bool_).copy())
    opt_keys = {k: _BITS_TO_OPTS[bits] for k, bits in opt_bits.items()}
    action_tree = ActionTree(
        type_mask=type_mask,
        type_to_key=type_to_key,
        src_keys=src_keys,
        src_to_key=src_to_key,
        tgt_keys=tgt_keys,
        tgt_to_key=tgt_to_key,
        opt_keys=opt_keys,
        opt_to_key=opt_to_key,
        key_to_tuple=key_to_tuple,
        num_bands=cat.num_bands,
        opt_bits=opt_bits,
        _type_mask_bytes=tm_bytes,
    )
    return EncodedFrame(
        cat_ids=torch.from_numpy(cat_np.astype(np.int64)),
        cont_feats=torch.from_numpy(cont_np),
        valid_mask=torch.from_numpy(valid_np),
        action_tree=action_tree,
    )
