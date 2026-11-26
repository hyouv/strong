"""Protocol constants and pinned ruleset metadata for Stronghold Protocol (act3autochess@V3_1)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = ROOT / "simulator" / "engine"
WORKER_SCRIPT = ROOT / "simulator" / "js" / "worker.mjs"
CONFIG_PATH = ROOT / "config" / "regulation.json"

COOP_MODES: tuple[str, ...] = (
    "mode_multi_funny",
    "mode_multi_normal",
    "mode_multi_hard",
    "mode_multi_abyss",
)

SOLO_TIER1_MODES: tuple[str, ...] = (
    "mode_single_1",
    "mode_single_2",
    "mode_single_3",
    "mode_single_4",
)

SOLO_TIER2_MODES: tuple[str, ...] = (
    "mode_single_5",
    "mode_single_6",
    "mode_single_7",
    "mode_single_8",
)

SOLO_TIER3_MODES: tuple[str, ...] = tuple(f"mode_single_{i}" for i in range(9, 15))
SOLO_TIER4_MODES: tuple[str, ...] = tuple(f"mode_single_{i}" for i in range(15, 21))

SOLO_MODES: tuple[str, ...] = (
    "mode_single_funny",
    *(f"mode_single_{i}" for i in range(1, 21)),
)

SOLO_MODE_PRESETS: dict[str, tuple[str, ...]] = {
    "tier1": SOLO_TIER1_MODES,
    "1-4": SOLO_TIER1_MODES,
    "tier2": SOLO_TIER2_MODES,
    "5-8": SOLO_TIER2_MODES,
    "tier3": SOLO_TIER3_MODES,
    "9-14": SOLO_TIER3_MODES,
    "tier4": SOLO_TIER4_MODES,
    "15-20": SOLO_TIER4_MODES,
    "1-8": SOLO_TIER1_MODES + SOLO_TIER2_MODES,
}

ALL_MODES: tuple[str, ...] = COOP_MODES + SOLO_MODES

SEATS_PER_COOP_GAME = 4


def parse_solo_modes(spec: str) -> tuple[str, ...]:
    """Parse a solo mode preset ('tier1', 'tier2', '5-8'), numeric range ('5-8'), or comma list."""
    raw = spec.strip()
    if not raw:
        raise ValueError("Empty solo mode specification")
    if raw in SOLO_MODE_PRESETS:
        return SOLO_MODE_PRESETS[raw]
    out: list[str] = []
    for token in (p.strip() for p in raw.split(",") if p.strip()):
        if token in SOLO_MODE_PRESETS:
            out.extend(SOLO_MODE_PRESETS[token])
        elif "-" in token and not token.startswith("mode_"):
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if not (1 <= lo <= hi <= 20):
                raise ValueError(f"Invalid solo mode level range: {token!r}")
            out.extend(f"mode_single_{i}" for i in range(lo, hi + 1))
        elif token.isdigit():
            lv = int(token)
            if not (1 <= lv <= 20):
                raise ValueError(f"Invalid solo mode level: {token!r}")
            out.append(f"mode_single_{lv}")
        elif token in SOLO_MODES:
            out.append(token)
        else:
            raise ValueError(f"Unsupported solo mode token: {token!r}")
    # Preserve order while deduplicating
    return tuple(dict.fromkeys(out))


def load_regulation_config(config_path: Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else CONFIG_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def expected_source(config_path: Path | None = None) -> dict[str, Any]:
    cfg = load_regulation_config(config_path)
    return {
        **cfg["source"],
        "version": cfg["version"],
        "ruleset": "stronghold-regulation-v3_1",
    }
