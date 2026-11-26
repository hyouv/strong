"""Tests for the pure-Python Stronghold Protocol simulator client and pool."""
from __future__ import annotations

import gzip
import io
import json
import pytest

from stronghold_sim import COOP_MODES, RoomClient, RoomPool, WorkerError, expected_source


FAST_ORDER = (
    "g.infoReady",
    "g.band",
    "g.diyBring",
    "g.choice",
    "g.commandChoice",
    "g.ready",
    "g.sell",
    "g.destroy",
)


def pick_fast_action(actions: list[dict]) -> int:
    for action_type in FAST_ORDER:
        for idx, action in enumerate(actions):
            if action["t"] == action_type:
                return idx
    return 0


def test_room_client_full_match_and_version_guard() -> None:
    with RoomClient() as room:
        assert room.handshake is not None
        assert room.handshake["source"] == expected_source()
        assert tuple(room.handshake["modes"]) == COOP_MODES

        frame = room.reset(seed=1000, mode_id="mode_multi_abyss", policy_version=3)
        assert frame["decision_id"] == 0
        assert frame["policy_version"] == 3

        with pytest.raises(WorkerError, match="Policy version changed"):
            room.step(decision_id=0, action_key=0, policy_version=4)

        steps = 0
        while not frame.get("terminal"):
            assert steps < 3000
            choice = pick_fast_action(frame["actions"])
            frame = room.step(frame["decision_id"], choice, policy_version=3)
            assert frame["previous"]["accepted"] is True
            steps += 1

        assert frame["terminal"] is True
        assert frame["fields"] >= 1
        assert all(n > 0 for n in frame["actionsBySeat"])


def test_room_pool_parallel_exchange_and_examples() -> None:
    with RoomPool(num_workers=2) as pool:
        assert len(pool) == 2
        frames = pool.exchange(
            {
                0: {"op": "reset", "seed": 2026, "modeId": "mode_multi_funny", "policy_version": 0},
                1: {"op": "reset", "seed": 2027, "modeId": "mode_multi_normal", "policy_version": 0},
            }
        )
        assert frames[0]["pub"]["modeId"] == "mode_multi_funny"
        assert frames[1]["pub"]["modeId"] == "mode_multi_normal"

        # Step both rooms one decision concurrently
        next_frames = pool.exchange(
            {
                w: {
                    "op": "step",
                    "decision_id": frames[w]["decision_id"],
                    "action_key": 0,
                    "policy_version": 0,
                }
                for w in (0, 1)
            }
        )
        assert next_frames[0]["decision_id"] == 1
        assert next_frames[1]["decision_id"] == 1

        # Generate in-memory compressed teacher demonstration on worker 0
        demo = pool.request(
            0,
            {
                "op": "examples_memory",
                "modeId": "mode_multi_funny",
                "seed": 77,
                "memory_limit": 64 * 1024 * 1024,
                "turn_order": "interleaved",
            },
        )
        assert demo["terminal"] is True
        assert demo["turnOrder"] == "interleaved"
        assert len(demo["replay"]) == demo["replayBytes"]
        with gzip.GzipFile(fileobj=io.BytesIO(demo["replay"]), mode="rb") as gz:
            rows = [json.loads(line) for line in gz]
        assert rows[0]["kind"] == "reference_expert"
        assert rows[0]["turnOrder"] == "interleaved"
        assert rows[-1]["terminal"] is True
        assert len(rows) == demo["recorded"] + 2
