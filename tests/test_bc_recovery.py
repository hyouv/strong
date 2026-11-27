"""Regression coverage for invalid-match quarantine and the formerly failing real seed."""
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

import pytest

from stronghold_sim import RoomClient, RoomPool, WorkerError
from stronghold_rl.bc import collect_solo_expert_dataset
from stronghold_rl.bc_collection import ExpertReplayCollector
from stronghold_rl.codec import TYPE_BAND, TYPE_BOND_GROUP, CHOICE_OFFSET


INVALID = "JS worker error: Error: Invalid expert match: [injected simulation error]"


def test_quarantine_retains_successful_slots_and_preserves_opening(tmp_path):
    opening = {"bondGroup": 1, "bandId": "band_victoria"}
    calls = []

    def request(slot, cmd):
        calls.append((slot, cmd.copy()))
        if cmd["op"] == "expert_opening_options":
            # The first replacement seed is incompatible. It must not change the group or band to fit.
            return {"options": [{"bondGroup": 1, "bandIds": [] if cmd["seed"] == 102 else ["band_victoria"]}]}
        if cmd["seed"] == 100:
            raise WorkerError(INVALID)
        return {"replay": str(cmd["seed"]).encode()}

    log = tmp_path / "failures.jsonl"
    collector = ExpertReplayCollector(seed_start=100, num_episodes=2, max_retries=2, quarantine_path=log)
    commands = {i: {"op": "examples_memory", "seed": 100 + i, "modeId": "mode_single_5", "opening": opening} for i in range(2)}
    with ThreadPoolExecutor(2) as threads:
        results = collector.exchange(SimpleNamespace(threads=threads, request=request), commands)
    assert results == {0: {"replay": b"103"}, 1: {"replay": b"101"}}
    demos = [cmd for _, cmd in calls if cmd["op"] == "examples_memory"]
    assert sorted(c["seed"] for c in demos) == [100, 101, 103]
    assert all(c["opening"] == opening and c["modeId"] == "mode_single_5" for c in demos)
    assert commands[0]["seed"] == 100
    failure = json.loads(log.read_text())
    assert failure["original_seed"] == failure["seed"] == 100
    assert failure["error"] == INVALID
    assert collector.replacements == [{"mode": "mode_single_5", "original_seed": 100, "seed": 103, "opening": opening, "retries": 1}]


@pytest.mark.parametrize("error,expected_failures", [(INVALID, 3), ("JS worker exited unexpectedly (code=1)", 0)])
def test_retries_are_bounded_and_worker_crashes_are_not_hidden(tmp_path, error, expected_failures):
    def request(slot, cmd):
        raise WorkerError(error)

    log = tmp_path / "failures.jsonl"
    collector = ExpertReplayCollector(seed_start=100, num_episodes=1, max_retries=2, quarantine_path=log)
    with ThreadPoolExecutor(1) as threads:
        with pytest.raises(RuntimeError):
            collector.exchange(SimpleNamespace(threads=threads, request=request), {
                0: {"op": "examples_memory", "seed": 100, "modeId": "mode_single_5"},
            })
    assert len(collector.failures) == expected_failures
    if expected_failures:
        assert len(log.read_text().splitlines()) == expected_failures
    else:
        assert not log.exists()


def test_real_balanced_collection_replaces_one_invalid_match_without_losing_labels(monkeypatch, tmp_path):
    request = RoomPool.request
    failed = False

    def fail_once(self, slot, cmd):
        nonlocal failed
        if cmd["op"] == "examples_memory" and cmd["seed"] == 77000 and not failed:
            failed = True
            raise WorkerError(INVALID)
        return request(self, slot, cmd)

    monkeypatch.setattr(RoomPool, "request", fail_once)
    dataset, stats = collect_solo_expert_dataset(
        num_episodes=4, modes=("mode_single_5", "mode_single_6"), seed_start=77000,
        num_workers=2, encode_workers=1, quarantine_path=tmp_path / "failures.jsonl",
    )
    assert stats["episodes_kept"] == 4
    assert stats["episodes_quarantined"] == stats["episodes_replaced"] == 1
    groups = dataset.a_src[dataset.a_type == TYPE_BOND_GROUP] - CHOICE_OFFSET
    bands = dataset.a_src[dataset.a_type == TYPE_BAND]
    assert groups.tolist().count(0) == groups.tolist().count(1) == 2
    assert len(bands) == len(bands.unique()) == 4
    failed_opening = dataset._metadata["quarantined_episodes"][0]["opening"]
    assert dataset._metadata["replacement_episodes"][0]["opening"] == failed_opening


def test_seed_100980_haruka_lifesteal_full_expert_match():
    with RoomClient() as client:
        result = client.request({
            "op": "examples_memory", "modeId": "mode_single_5", "seed": 100980,
            "memory_limit": 64 * 1024 * 1024,
            "opening": {"bondGroup": 1, "bandId": "band_victoria"},
        })
    assert result["terminal"] and result["round"] >= 14 and result["fields"] >= 14
