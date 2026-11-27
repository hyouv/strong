"""Static vocabulary and rule-metadata tables loaded from act3autochess@V3_1 data."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any

from stronghold_sim.protocol import ENGINE_ROOT

DATA_DIR = ENGINE_ROOT / "data" / "versions" / "act3autochess-v3_1"

FIELD_R0, FIELD_R1 = 9, 12
FIELD_C0, FIELD_C1 = 2, 10
BOARD_ROWS = FIELD_R1 - FIELD_R0 + 1  # 4
BOARD_COLS = FIELD_C1 - FIELD_C0 + 1  # 9
BOARD_TILES = BOARD_ROWS * BOARD_COLS  # 36
BOSS_ROW_OFFSET = 7

DIRECTIONS: tuple[str, ...] = ("RIGHT", "UP", "LEFT", "DOWN")
DIR_TO_IDX: dict[str, int] = {d: i for i, d in enumerate(DIRECTIONS)}

PHASES: tuple[str, ...] = (
    "INFO_CHECK",
    "BAND_DRAFT",
    "CUSTOM_CHECK",
    "SP_DRAFT",
    "PREP",
)
PHASE_TO_IDX: dict[str, int] = {p: i for i, p in enumerate(PHASES)}

DIFFICULTIES: tuple[str, ...] = ("TRAINING", "FAST", "FUNNY", "NORMAL", "HARD", "ABYSS")
DIFF_TO_IDX: dict[str, int] = {d: i for i, d in enumerate(DIFFICULTIES)}

# Special terrain tileKey flags of interest on deployable/playable tiles
SPECIAL_TILE_KEYS: frozenset[str] = frozenset(
    {
        "tile_infection",
        "tile_mire",
        "tile_duel_grassland",
        "tile_speedup",
        "tile_fence_bound",
    }
)

_EFFECT_STRIP_RE = re.compile(r"^(?:choice:|command:|bond:|band:|event:)")
_EFFECT_SUFFIX_RE = re.compile(r"#\d+$")


def normalize_effect_id(raw_id: str | None) -> str:
    """Normalize runtime effect IDs like 'choice:allybuff_select_18#1' -> 'allybuff_select_18'."""
    if not raw_id:
        return ""
    s = _EFFECT_STRIP_RE.sub("", raw_id)
    s = _EFFECT_SUFFIX_RE.sub("", s)
    return s


@dataclass(frozen=True)
class TileSpatialInfo:
    """Precomputed static spatial features for one board tile (row 9..12, col 2..10)."""

    deploy_cls: int       # 0=none, 1=melee, 2=ranged
    is_ground_path: bool  # True if any enemy ground route passes through this tile
    on_upper_path: bool   # True if upper-gate ground route passes through this tile
    on_lower_path: bool   # True if lower-gate ground route passes through this tile
    path_progress: float  # [0, 1] normalized progress along path toward blue gate (0 if off-path)
    is_special_tile: bool # True if infection/mire/grassland/speedup/fence tile
    has_blower: bool      # True if an active blower device sits on this tile


def _build_stage_spatial_grid(stage: dict[str, Any], boss_left: bool = False) -> tuple[TileSpatialInfo, ...]:
    """Compute per-tile deployability, ground path coverage, progress, and special terrain."""
    rows = stage.get("rows") or []
    legend = stage.get("tiles") or {}

    deploy_cls = [0] * BOARD_TILES
    special_flags = [False] * BOARD_TILES
    blower_flags = [False] * BOARD_TILES

    for r in range(FIELD_R0, FIELD_R1 + 1):
        for c in range(FIELD_C0, FIELD_C1 + 1):
            sr = r - BOSS_ROW_OFFSET if boss_left else r
            sc = c
            idx = (r - FIELD_R0) * BOARD_COLS + (c - FIELD_C0)
            if 0 <= sr < len(rows) and 0 <= sc < len(rows[sr]):
                glyph = rows[sr][sc]
                t = legend.get(glyph)
                if t:
                    h = t.get("height")
                    b = t.get("buildable")
                    if h == "LOW" and b in ("ALL", "MELEE"):
                        deploy_cls[idx] = 1
                    elif (h == "HIGH" and b in ("ALL", "RANGED")) or (h == "LOW" and b == "RANGED"):
                        deploy_cls[idx] = 2
                    if t.get("tileKey") in SPECIAL_TILE_KEYS:
                        special_flags[idx] = True

    for dev in stage.get("devices") or []:
        pos = dev.get("pos")
        if not isinstance(pos, list) or len(pos) < 2:
            continue
        r = pos[0] + BOSS_ROW_OFFSET if boss_left else pos[0]
        c = pos[1]
        if not (FIELD_R0 <= r <= FIELD_R1 and FIELD_C0 <= c <= FIELD_C1):
            continue
        active = dev.get("active") if isinstance(dev.get("active"), bool) else (not dev.get("hidden"))
        if not active:
            continue
        role = dev.get("role")
        idx = (r - FIELD_R0) * BOARD_COLS + (c - FIELD_C0)
        if role in ("crate", "mound"):
            deploy_cls[idx] = 0
        elif role == "platform":
            deploy_cls[idx] = 2
        elif role == "waterPlatform":
            deploy_cls[idx] = 1
        elif role == "blower":
            blower_flags[idx] = True

    # Trace enemy ground paths from stage.groundPathsWithDevices (fallback groundPaths)
    on_path = [False] * BOARD_TILES
    on_upper = [False] * BOARD_TILES
    on_lower = [False] * BOARD_TILES
    progress = [0.0] * BOARD_TILES

    paths = stage.get("groundPathsWithDevices") or stage.get("groundPaths") or {}
    for route_key, pts in paths.items():
        if not isinstance(pts, list) or not pts:
            continue
        start_r = pts[0][0] if isinstance(pts[0], list) and len(pts[0]) >= 2 else -1
        # Filter routes matching normal field (rows 9..12) vs boss field (rows <= 5)
        is_boss_route = start_r <= 8
        if is_boss_route != boss_left:
            continue
        # Gate classification matches shared/constants.js gateOf()
        target_r = start_r + (13 if boss_left else 6)
        is_upper_gate = target_r >= 18

        denom = max(1, len(pts) - 1)
        for step_i, pt in enumerate(pts):
            if not isinstance(pt, list) or len(pt) < 2:
                continue
            r = pt[0] + BOSS_ROW_OFFSET if boss_left else pt[0]
            c = pt[1]
            if not (FIELD_R0 <= r <= FIELD_R1 and FIELD_C0 <= c <= FIELD_C1):
                continue
            idx = (r - FIELD_R0) * BOARD_COLS + (c - FIELD_C0)
            on_path[idx] = True
            if is_upper_gate:
                on_upper[idx] = True
            else:
                on_lower[idx] = True
            prog = float(step_i) / float(denom)
            if prog > progress[idx]:
                progress[idx] = prog

    return tuple(
        TileSpatialInfo(
            deploy_cls=deploy_cls[i],
            is_ground_path=on_path[i],
            on_upper_path=on_upper[i],
            on_lower_path=on_lower[i],
            path_progress=progress[i],
            is_special_tile=special_flags[i],
            has_blower=blower_flags[i],
        )
        for i in range(BOARD_TILES)
    )


@dataclass(frozen=True)
class ChessMeta:
    entity_idx: int
    base_idx: int
    tier: int
    is_golden: bool
    price: int
    sell_price: int
    pos_class: int  # 0=all, 1=melee, 2=ranged
    bond_indices: tuple[int, ...]


@dataclass(frozen=True)
class ItemMeta:
    entity_idx: int
    item_idx: int  # 1..num_items in isolated item vocab for equip slots
    tier: int
    is_golden: bool
    price: int


@dataclass(frozen=True)
class GameCatalog:
    """Frozen lookup tables for all static entities in act3autochess@V3_1."""

    # Unified entity vocab (0 = PAD/EMPTY): chess, items, tokens, enemies, bosses, choices, effects
    entity_to_idx: dict[str, int]
    num_entities: int

    # Isolated equipment/item vocab (0 = NONE, 1..num_items) for chess equipped slots
    item_to_idx: dict[str, int]
    num_items: int

    bond_ids: tuple[str, ...]
    bond_to_idx: dict[str, int]
    core_bond_mask: tuple[bool, ...]

    band_ids: tuple[str, ...]
    band_to_idx: dict[str, int]

    faction_ids: tuple[str, ...]
    faction_to_idx: dict[str, int]

    stage_to_idx: dict[str, int]
    stage_normal_spatial: dict[str, tuple[TileSpatialInfo, ...]]
    stage_boss_spatial: dict[str, tuple[TileSpatialInfo, ...]]

    chess_meta: dict[str, ChessMeta]
    item_meta: dict[str, ItemMeta]
    token_pos_class: dict[str, int]
    mode_boss_weights: dict[str, tuple[tuple[str, float], ...]]

    @property
    def num_bonds(self) -> int:
        return len(self.bond_ids)

    @property
    def num_bands(self) -> int:
        return len(self.band_ids)

    @property
    def num_factions(self) -> int:
        return len(self.faction_ids)

    @property
    def num_stages(self) -> int:
        return len(self.stage_to_idx) + 1

    def entity_idx(self, key: str | None) -> int:
        if not key:
            return 0
        idx = self.entity_to_idx.get(key)
        if idx is not None:
            return idx
        norm = normalize_effect_id(key)
        return self.entity_to_idx.get(norm, 0)

    def item_idx(self, key: str | None) -> int:
        if not key:
            return 0
        return self.item_to_idx.get(key, 0)

    def stage_spatial(self, stage_id: str | None, is_boss: bool) -> tuple[TileSpatialInfo, ...]:
        if stage_id:
            tbl = self.stage_boss_spatial if is_boss else self.stage_normal_spatial
            if stage_id in tbl:
                return tbl[stage_id]
        default_tile = TileSpatialInfo(1, False, False, False, 0.0, False, False)
        return (default_tile,) * BOARD_TILES


def _pos_to_int(pos: str | None) -> int:
    p = (pos or "").upper()
    if p == "MELEE":
        return 1
    if p == "RANGED":
        return 2
    return 0


@lru_cache(maxsize=1)
def get_catalog(data_dir: Path | None = None) -> GameCatalog:
    root = Path(data_dir) if data_dir is not None else DATA_DIR

    bonds_raw: dict[str, Any] = json.loads((root / "bonds.json").read_text(encoding="utf-8"))
    bands_raw: dict[str, Any] = json.loads((root / "bands.json").read_text(encoding="utf-8"))
    chess_raw: dict[str, Any] = json.loads((root / "chess.json").read_text(encoding="utf-8"))
    items_raw: dict[str, Any] = json.loads((root / "items.json").read_text(encoding="utf-8"))
    tokens_raw: dict[str, Any] = json.loads((root / "tokens.json").read_text(encoding="utf-8"))
    stages_raw: dict[str, Any] = json.loads((root / "stages.json").read_text(encoding="utf-8"))
    enemies_raw: dict[str, Any] = json.loads((root / "enemies.json").read_text(encoding="utf-8"))
    bosses_raw: dict[str, Any] = json.loads((root / "bosses.json").read_text(encoding="utf-8"))
    choices_raw: Any = json.loads((root / "choices.json").read_text(encoding="utf-8"))
    effects_raw: dict[str, Any] = json.loads((root / "effects.json").read_text(encoding="utf-8"))
    factions_raw: dict[str, Any] = json.loads((root / "factions.json").read_text(encoding="utf-8"))
    config_raw: dict[str, Any] = json.loads((root / "config.json").read_text(encoding="utf-8"))

    bond_ids = tuple(bonds_raw.keys())
    bond_to_idx = {k: i for i, k in enumerate(bond_ids)}
    core_bond_mask = tuple(bool(bonds_raw[k].get("isCore")) for k in bond_ids)

    band_ids = tuple(bands_raw.keys())
    band_to_idx = {k: i for i, k in enumerate(band_ids)}

    faction_types = factions_raw.get("types") or {}
    faction_ids = tuple(faction_types.keys())
    faction_to_idx = {k: i for i, k in enumerate(faction_ids)}

    entity_to_idx: dict[str, int] = {}

    def _register(key: str | None) -> int:
        if not key:
            return 0
        idx = entity_to_idx.get(key)
        if idx is None:
            idx = len(entity_to_idx) + 1
            entity_to_idx[key] = idx
        return idx

    for cid in sorted(chess_raw.keys()):
        _register(cid)
    for iid in sorted(items_raw.keys()):
        _register(iid)
    for tid in sorted(tokens_raw.keys()):
        _register(tid)
    for eid in sorted(enemies_raw.keys()):
        _register(eid)
    for bid in sorted(bosses_raw.keys()):
        _register(bid)
        _register(bosses_raw[bid].get("enemyKey"))
    for eff_id in sorted(effects_raw.keys()):
        _register(eff_id)
    for b in bands_raw.values():
        for sk in b.get("skills") or []:
            _register(sk.get("effectId"))

    def _walk_choices(obj: Any) -> None:
        if isinstance(obj, dict):
            for k in ("id", "cardId", "effectId", "choiceId", "buffId"):
                v = obj.get(k)
                if isinstance(v, str) and v:
                    _register(v)
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    _walk_choices(v)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    _walk_choices(item)

    _walk_choices(choices_raw)

    item_to_idx = {iid: i + 1 for i, iid in enumerate(sorted(items_raw.keys()))}

    chess_meta: dict[str, ChessMeta] = {}
    for cid, c in chess_raw.items():
        base_id = c.get("baseId") or cid
        chess_meta[cid] = ChessMeta(
            entity_idx=_register(cid),
            base_idx=_register(base_id),
            tier=int(c.get("tier") or 0),
            is_golden=bool(c.get("isGolden")),
            price=int(c.get("price") or 0),
            sell_price=int(c.get("sellPrice") or 0),
            pos_class=_pos_to_int(c.get("position")),
            bond_indices=tuple(bond_to_idx[b] for b in (c.get("bonds") or []) if b in bond_to_idx),
        )

    item_meta: dict[str, ItemMeta] = {}
    for iid, it in items_raw.items():
        item_meta[iid] = ItemMeta(
            entity_idx=_register(iid),
            item_idx=item_to_idx[iid],
            tier=int(it.get("tier") or 0),
            is_golden=bool(it.get("isGolden")),
            price=int(it.get("price") or 0),
        )

    token_pos_class = {tid: _pos_to_int(t.get("position")) for tid, t in tokens_raw.items()}

    stage_to_idx = {sid: i + 1 for i, sid in enumerate(sorted(stages_raw.keys()))}
    stage_normal_spatial = {
        sid: _build_stage_spatial_grid(st, boss_left=False) for sid, st in stages_raw.items()
    }
    stage_boss_spatial = {
        sid: _build_stage_spatial_grid(st, boss_left=True) for sid, st in stages_raw.items()
    }

    mode_boss_weights: dict[str, tuple[tuple[str, float], ...]] = {}
    for mid, mcfg in (config_raw.get("modes") or {}).items():
        bw = mcfg.get("bossWeights") or {}
        pairs = tuple((str(bid), float(w)) for bid, w in bw.items() if float(w) > 0.0)
        if pairs:
            mode_boss_weights[str(mid)] = pairs

    return GameCatalog(
        entity_to_idx=entity_to_idx,
        num_entities=len(entity_to_idx) + 1,
        item_to_idx=item_to_idx,
        num_items=len(item_to_idx) + 1,
        bond_ids=bond_ids,
        bond_to_idx=bond_to_idx,
        core_bond_mask=core_bond_mask,
        band_ids=band_ids,
        band_to_idx=band_to_idx,
        faction_ids=faction_ids,
        faction_to_idx=faction_to_idx,
        stage_to_idx=stage_to_idx,
        stage_normal_spatial=stage_normal_spatial,
        stage_boss_spatial=stage_boss_spatial,
        chess_meta=chess_meta,
        item_meta=item_meta,
        token_pos_class=token_pos_class,
        mode_boss_weights=mode_boss_weights,
    )
