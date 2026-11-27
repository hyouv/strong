"""Verify balanced BC coverage, real expert execution, and stale-cache rejection."""
from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import pytest
import torch

from stronghold_sim import RoomClient, RoomPool, WorkerError
from stronghold_rl.bc import BCDataset, collect_solo_expert_dataset
from stronghold_rl.bc_openings import BalancedOpeningSampler, bc_cache_metadata
from stronghold_rl.codec import TYPE_BAND, TYPE_BOND_GROUP, CHOICE_OFFSET


def test_balanced_openings_cover_real_1200_episode_curriculum() -> None:
    sampler = BalancedOpeningSampler(100000)
    replay = BalancedOpeningSampler(100000)
    with RoomPool(num_workers=4) as pool:
        for start in range(0, 1200, 4):
            commands = {w: {
                "op": "expert_opening_options", "seed": 100000 + start + w,
                "modeId": f"mode_single_{1 + (start + w) % 8}",
            } for w in range(4)}
            results = pool.exchange(commands)
            for w, cmd in commands.items():
                options = results[w]["options"]
                chosen = sampler.choose(options, cmd["modeId"])
                assert chosen == replay.choose(options, cmd["modeId"])
                matching = next(o for o in options if o["bondGroup"] == chosen["bondGroup"])
                assert chosen["bandId"] in matching["bandIds"]
                assert "band_protect" not in matching["bandIds"]
    counts = sampler.summary()
    assert len(counts["band_counts"]) == 60
    assert counts["band_count_min"] == counts["band_count_max"] == 20
    assert abs(sampler.groups[0] - sampler.groups[1]) <= 2


def test_forced_opening_is_executed_and_recorded_in_full_expert_match() -> None:
    with RoomClient() as client:
        options = client.request({"op": "expert_opening_options", "seed": 1234, "modeId": "mode_single_5"})["options"]
        second = next(o for o in options if o["bondGroup"] == 1)
        assert "band_equipment" in second["bandIds"]
        opening = {"bondGroup": 1, "bandId": "band_equipment"}
        result = client.request({
            "op": "examples_memory", "seed": 1234, "modeId": "mode_single_5",
            "memory_limit": 64 * 1024 * 1024, "opening": opening,
        })
        assert result["terminal"] and result["fields"] > 0
        rows = [json.loads(line) for line in gzip.GzipFile(fileobj=io.BytesIO(result["replay"]))]
        assert rows[0]["opening"] == rows[-1]["opening"] == opening
        groups = [r for r in rows[1:-1] if r["action"]["t"] == "g.bondGroup"]
        bands = [r for r in rows[1:-1] if r["action"]["t"] == "g.band"]
        assert len(groups) == len(bands) == 1
        assert groups[0]["action"]["idx"] == 1
        assert bands[0]["action"]["bandId"] == "band_equipment"
        assert bands[0]["frame"]["pub"]["bondChoice"]["picked"] == 1
        assert groups[0]["index"] >= 0 and bands[0]["index"] >= 0
        # An incompatible faction-specific strategy must fail, not silently fall back.
        bad = next(o for o in options if "band_ursus" not in o["bandIds"])
        with pytest.raises(WorkerError, match="incompatible"):
            client.request({
                "op": "examples_memory", "seed": 1234, "modeId": "mode_single_5",
                "memory_limit": 64 * 1024 * 1024,
                "opening": {"bondGroup": bad["bondGroup"], "bandId": "band_ursus"},
            })


def test_balanced_collection_labels_and_cache_provenance(tmp_path: Path) -> None:
    modes = ("mode_single_5", "mode_single_6")
    dataset, stats = collect_solo_expert_dataset(
        num_episodes=4, modes=modes, seed_start=77000, num_workers=2, encode_workers=1,
    )
    groups = dataset.a_src[dataset.a_type == TYPE_BOND_GROUP] - CHOICE_OFFSET
    bands = dataset.a_src[dataset.a_type == TYPE_BAND]
    assert groups.tolist().count(0) == groups.tolist().count(1) == 2
    assert len(bands) == len(bands.unique()) == 4
    assert stats["opening_policy"] == "balanced-v1"
    assert stats["opening_coverage"]["group_counts"] == {"0": 2, "1": 2}
    path = tmp_path / "balanced.pt"
    dataset.save(path)
    expected = bc_cache_metadata("balanced", modes, 4, 77000)
    restored = BCDataset.load(path, expected_metadata=expected)
    assert torch.equal(restored.a_src, dataset.a_src)
    with pytest.raises(ValueError, match="generation settings"):
        BCDataset.load(path, expected_metadata={**expected, "seed": 77001})
    legacy = tmp_path / "legacy.pt"
    torch.save({k: getattr(dataset, k) for k in dataset.__dataclass_fields__}, legacy)
    with pytest.raises(ValueError, match="generation settings"):
        BCDataset.load(legacy, expected_metadata=expected)
    assert len(BCDataset.load(legacy)) == len(dataset)
