"""L1-4 bans follow each episode's mode and never leak into evaluation."""
from contextlib import nullcontext
import os
from pathlib import Path
import shlex
import subprocess

import pytest
import torch

from stronghold_rl.bc import evaluate_policy_solo
from stronghold_rl.catalog import get_catalog
from stronghold_rl.codec import ActionMaskSpec, NUM_ACTION_TYPES, TYPE_BAND
from stronghold_rl.model import PolicyConfig, StrongholdPolicyNet
from stronghold_rl.ppo import PPOConfig, collect_ppo_rollouts
from stronghold_rl.rollout import AsyncActorPool, mask_opening_diversity
from stronghold_sim import SOLO_TIER1_MODES


def test_mode_bans_preserve_hard_exclusions_when_optional_bans_exhaust_choices() -> None:
    tm = bytearray(NUM_ACTION_TYPES)
    tm[TYPE_BAND] = 1
    spec = ActionMaskSpec(bytes(tm), {TYPE_BAND: [0, 1, 2]}, {}, {}, 3)
    bans = {mode: (1,) for mode in SOLO_TIER1_MODES}
    for level in range(1, 21):
        frame = {"phase": "BAND_DRAFT", "pub": {"modeId": f"mode_single_{level}"}}
        result = mask_opening_diversity(frame, spec, mode_ban_band_idxs=bans)
        assert (1 in result.src_keys[TYPE_BAND]) == (level > 4)
    frame = {"phase": "BAND_DRAFT", "pub": {"modeId": "mode_single_1"}}
    result = mask_opening_diversity(frame, spec, mode_ban_band_idxs=bans, ban_band_idxs=frozenset({0, 2}))
    assert result.src_keys[TYPE_BAND] == [0, 2]
    with pytest.raises(ValueError, match="no legal band"):
        mask_opening_diversity(frame, spec, mode_ban_band_idxs={"mode_single_1": (0, 1, 2)})


@pytest.mark.parametrize("use_async", [False, True])
def test_mixed_rollout_masks_and_reused_pool_evaluation(use_async, monkeypatch) -> None:
    torch.manual_seed(8173)
    cat = get_catalog()
    weakness = cat.band_to_idx["band_weakness"]
    model = StrongholdPolicyNet(catalog=cat, config=PolicyConfig(d_model=32, n_heads=4, n_layers=1, dim_ff=64))
    model.eval()
    with torch.no_grad():
        model.band_head[-1].weight.zero_()
        model.band_head[-1].bias.zero_()
        model.band_head[-1].bias[weakness] = 30
        model.type_head[-1].bias[TYPE_BAND] = 30  # Never skip the draft.
    cfg = PPOConfig(max_prep_steps_per_round=8, kl_coef=0)
    assert cfg.tier1_ban_bands == ("band_weakness",)
    pool_context = AsyncActorPool(num_workers=2, min_batch=2) if use_async else nullcontext(None)
    with pool_context as pool:
        batch, stats = collect_ppo_rollouts(
            model, num_episodes=4, modes=("mode_single_15",),
            episode_modes=("mode_single_1", "mode_single_5", "mode_single_4", "mode_single_15"),
            seed_start=68100, num_workers=2, ppo_cfg=cfg, pool=pool,
            use_async_pool=use_async, catalog=cat,
        )
        assert stats["episodes"] == 4
        opening = batch["a_type"] == TYPE_BAND
        assert int(opening.sum()) == 4
        allowed = batch["band_mask"][opening, weakness]
        assert int(allowed.sum()) == 2  # Only L5 and L15 retain weakness.
        choices = batch["a_src"][opening]
        assert (choices[allowed] == weakness).all()
        assert (choices[~allowed] != weakness).all()
        with torch.no_grad():
            actual_lp = model.evaluate_batch(batch)["log_prob"]
        assert torch.allclose(actual_lp, batch["old_log_prob"], atol=1e-4, rtol=0)

        # The main training loop reuses its actor pool for eval. Check those
        # workers clear training restrictions, even on the same L1-4 modes.
        eval_masks = []
        original_act = model.act_batch_tensors

        def record_act(*args, **kwargs):
            out = original_act(*args, **kwargs)
            rows = out["has_band"]
            eval_masks.extend(out["band_mask"][rows, weakness].tolist())
            return out

        monkeypatch.setattr(model, "act_batch_tensors", record_act)
        ev = evaluate_policy_solo(
            model, modes=("mode_single_1", "mode_single_4"), seeds=(68200, 68201),
            num_workers=2, deterministic=True, break_cycles=False,
            max_prep_steps_per_round=8, pool=pool, use_async_pool=use_async, catalog=cat,
        )
        assert ev["episodes"] == 2
        assert eval_masks == [True, True]


@pytest.mark.parametrize("override", [None, ""])
def test_cold_launcher_default_and_explicit_empty_override(override) -> None:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, DRY_RUN="1", RUN_ID="tier1_mask_test", ROOT_DIR=str(root))
    env.pop("PPO_TIER1_BAN_BANDS", None)
    if override is not None:
        env["PPO_TIER1_BAN_BANDS"] = override
    result = subprocess.run(
        ["bash", str(root / "scripts/train_server_5m_v2_cold.sh"), "easy"],
        env=env, text=True, capture_output=True, check=True,
    )
    command = next(line for line in result.stdout.splitlines() if "--tier1-ban-bands" in line)
    args = shlex.split(command)
    assert args[args.index("--tier1-ban-bands") + 1] == ("band_weakness" if override is None else "")
