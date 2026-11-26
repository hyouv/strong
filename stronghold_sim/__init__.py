"""Headless Stronghold Protocol: Regulation (act3autochess@V3_1) simulator package."""
from .protocol import (
    CONFIG_PATH,
    COOP_MODES,
    ENGINE_ROOT,
    ROOT,
    SEATS_PER_COOP_GAME,
    WORKER_SCRIPT,
    expected_source,
    load_regulation_config,
)
from .runtime import find_node, physical_first_cpus
from .worker import RoomClient, RoomPool, WorkerError

__all__ = [
    "CONFIG_PATH",
    "COOP_MODES",
    "ENGINE_ROOT",
    "ROOT",
    "SEATS_PER_COOP_GAME",
    "WORKER_SCRIPT",
    "RoomClient",
    "RoomPool",
    "WorkerError",
    "expected_source",
    "find_node",
    "load_regulation_config",
    "physical_first_cpus",
]
