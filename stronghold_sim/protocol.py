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

SOLO_MODES: tuple[str, ...] = (
    "mode_single_funny",
    *(f"mode_single_{i}" for i in range(1, 21)),
)

ALL_MODES: tuple[str, ...] = COOP_MODES + SOLO_MODES

SEATS_PER_COOP_GAME = 4


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
