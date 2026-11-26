"""Strict differential behavioral-parity tests against the original stronghold-training simulator.

Runs identical seeds/modes/actions on BOTH the original worker (`stronghold-training/workers/js_train/worker.mjs`)
and the migrated worker (`simulator/js/worker.mjs`) using the same Node.js binary, asserting exact step-by-step
equivalence of observations, legal action lists (including exact ordering), unitStats, terminal results,
and teacher demonstration streams.
"""
from __future__ import annotations

import gzip
import io
import json
import os
from pathlib import Path
import random
import pytest

from stronghold_sim import RoomClient

_ORIG_ENV = os.environ.get("STRONGHOLD_ORIGINAL_ROOT", "").strip()
ORIGINAL_ROOT = Path(_ORIG_ENV) if _ORIG_ENV else None
ORIGINAL_WORKER = (
    ORIGINAL_ROOT / "workers" / "js_train" / "worker.mjs"
    if ORIGINAL_ROOT is not None
    else None
)

pytestmark = pytest.mark.skipif(
    ORIGINAL_WORKER is None or not ORIGINAL_WORKER.is_file(),
    reason="Set STRONGHOLD_ORIGINAL_ROOT to the original stronghold-training directory to run parity tests",
)


def _strip_pid(hello_reply: dict) -> dict:
    return {k: v for k, v in hello_reply.items() if k != "pid"}


@pytest.mark.parametrize(
    ("mode_id", "seed"),
    [
        ("mode_multi_funny", 1000),
        ("mode_multi_abyss", 1001),
    ],
)
def test_step_by_step_match_parity(mode_id: str, seed: int) -> None:
    """Every single decision frame, legal action list, and terminal result must be 100% identical."""
    rng = random.Random(seed ^ 0x5A5A)

    with (
        RoomClient(worker_script=ORIGINAL_WORKER, cwd=ORIGINAL_ROOT) as orig,
        RoomClient() as migrated,
    ):
        assert _strip_pid(migrated.hello()) == _strip_pid(orig.hello())

        f_orig = orig.reset(seed=seed, mode_id=mode_id, policy_version=0)
        f_new = migrated.reset(seed=seed, mode_id=mode_id, policy_version=0)
        assert f_new == f_orig

        steps = 0
        while not f_orig.get("terminal"):
            assert steps < 3000
            actions = f_orig["actions"]
            # Exercise realistic action mix (buy/move/equip/command/ready) while ensuring match completes fast
            if f_orig["phase"] == "PREP":
                non_move = [i for i, a in enumerate(actions) if a["t"] != "g.move"]
                ready_idx = next((i for i, a in enumerate(actions) if a["t"] == "g.ready"), None)
                if ready_idx is not None and (steps % 7 == 0 or rng.random() < 0.35):
                    action_key = ready_idx
                elif non_move and rng.random() < 0.7:
                    action_key = rng.choice(non_move)
                else:
                    action_key = rng.randrange(len(actions))
            else:
                action_key = rng.randrange(len(actions))

            f_orig = orig.step(f_orig["decision_id"], action_key, policy_version=0)
            f_new = migrated.step(f_new["decision_id"], action_key, policy_version=0)
            assert f_new == f_orig, f"Mismatch at decision {steps} in {mode_id} seed={seed}"
            steps += 1

        assert f_new["terminal"] is True
        assert f_new == f_orig
        assert steps > 20


@pytest.mark.parametrize("turn_order", ["serial", "interleaved"])
def test_teacher_examples_byte_exact_parity(turn_order: str) -> None:
    """Offline bot demonstration streams must produce identical JSONL rows and metadata."""
    mode_id = "mode_multi_funny"
    seed = 1234 if turn_order == "serial" else 1235

    with (
        RoomClient(worker_script=ORIGINAL_WORKER, cwd=ORIGINAL_ROOT) as orig,
        RoomClient() as migrated,
    ):
        r_orig = orig.generate_examples(mode_id=mode_id, seed=seed, turn_order=turn_order)
        r_new = migrated.generate_examples(mode_id=mode_id, seed=seed, turn_order=turn_order)

        meta_orig = {k: v for k, v in r_orig.items() if k != "replay"}
        meta_new = {k: v for k, v in r_new.items() if k != "replay"}
        assert meta_new == meta_orig

        raw_orig = gzip.decompress(r_orig["replay"])
        raw_new = gzip.decompress(r_new["replay"])
        assert raw_new == raw_orig
        # Even the compressed gzip streams should match identically
        assert r_new["replay"] == r_orig["replay"]
