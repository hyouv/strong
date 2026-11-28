"""Random opening curricula, exact masked PPO ratios, and semantic entropy metrics."""
from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch

from stronghold_sim import RoomClient
from stronghold_rl.bc import evaluate_policy_solo
from stronghold_rl.catalog import get_catalog
from stronghold_rl.codec import (
    ActionMaskSpec, CHOICE_OFFSET, MAX_CHOICE_SLOTS, NUM_ACTION_TYPES,
    TYPE_BAND, TYPE_BOND_GROUP, TYPE_INFO_READY, encode_frame,
)
from stronghold_rl.model import PolicyConfig, StrongholdPolicyNet, extract_step_masks
from stronghold_rl.ppo import (
    PPOConfig, _OpeningEntropyStats, collect_ppo_rollouts, main,
    opening_mask_schedule, ppo_update,
)
from stronghold_rl.rollout import (
    AsyncActorPool, mask_opening_diversity, mask_random_opening, opening_bond_mask,
)


def _spec(kind: int, choices: list[int]) -> ActionMaskSpec:
    tm = bytearray(NUM_ACTION_TYPES)
    tm[kind] = 1
    if kind == TYPE_BOND_GROUP:
        tm[TYPE_INFO_READY] = 1
    return ActionMaskSpec(bytes(tm), {kind: choices}, {}, {}, 61)


INFO = {"phase": "INFO_CHECK", "pub": {"bondChoice": {
    "picked": None, "groups": [["ursusShip"], ["lethaniaShip"]],
}}}


def test_seeded_random_subsets_preserve_legality_and_episode_gate() -> None:
    bond = _spec(TYPE_BOND_GROUP, [CHOICE_OFFSET, CHOICE_OFFSET + 1])
    band = _spec(TYPE_BAND, list(range(12)))
    before_rng = random.getstate()
    selected_groups = set()
    masked_episodes = 0
    for seed in range(100):
        kwargs = dict(episode_seed=seed, probability=0.5, band_keep_ratio=0.35)
        bg = mask_random_opening(INFO, bond, **kwargs)
        bs = mask_random_opening({"phase": "BAND_DRAFT"}, band, **kwargs)
        assert bg == mask_random_opening(INFO, bond, **kwargs)
        assert bs == mask_random_opening({"phase": "BAND_DRAFT"}, band, **kwargs)
        assert (bg is bond) == (bs is band)  # one episode coin for both phases
        if bg is not bond:
            masked_episodes += 1
            picked = bg.src_keys[TYPE_BOND_GROUP]
            assert len(picked) == 1
            selected_groups.add(picked[0])
            assert bool(bg.type_mask_bytes[TYPE_INFO_READY]) == (CHOICE_OFFSET in picked)
            assert len(bs.src_keys[TYPE_BAND]) == 5  # ceil(12 * .35)
            assert set(bs.src_keys[TYPE_BAND]) <= set(band.src_keys[TYPE_BAND])
        # Increasing p cannot un-restrict the same seeded episode.
        stronger = mask_random_opening(INFO, bond, episode_seed=seed, probability=0.8)
        assert bg is bond or stronger == bg
    assert 25 < masked_episodes < 75
    assert selected_groups == {CHOICE_OFFSET, CHOICE_OFFSET + 1}
    assert random.getstate() == before_rng
    assert band.src_keys[TYPE_BAND] == list(range(12))
    assert bond.src_keys[TYPE_BOND_GROUP] == [CHOICE_OFFSET, CHOICE_OFFSET + 1]


def test_mask_defaults_edge_cases_and_existing_bans() -> None:
    bond = _spec(TYPE_BOND_GROUP, [CHOICE_OFFSET, CHOICE_OFFSET + 1])
    band = _spec(TYPE_BAND, list(range(8)))
    assert mask_random_opening(INFO, bond, episode_seed=7) is bond
    assert mask_random_opening({"phase": "BAND_DRAFT"}, band, episode_seed=7, probability=1) is band
    assert mask_random_opening({"phase": "PREP"}, bond, episode_seed=7, probability=1) is bond
    picked = {"phase": "INFO_CHECK", "pub": {"bondChoice": {"picked": 0}}}
    assert mask_random_opening(picked, bond, episode_seed=7, probability=1) is bond
    for size in (0, 1, 2):
        tiny = _spec(TYPE_BAND, list(range(size)))
        assert mask_random_opening({"phase": "BAND_DRAFT"}, tiny, episode_seed=7,
                                   probability=1, band_keep_ratio=0.01) is tiny
    banned = mask_opening_diversity(INFO, bond, ban_bond_ids=frozenset({"ursusShip"}))
    after = mask_random_opening(INFO, banned, episode_seed=7, probability=1)
    assert after == banned and after.type_mask_bytes[TYPE_INFO_READY] == 0
    banned_band = mask_opening_diversity({"phase": "BAND_DRAFT"}, band, ban_band_idxs=frozenset({0, 1}))
    after_band = mask_random_opening({"phase": "BAND_DRAFT"}, banned_band, episode_seed=7,
                                    probability=1, band_keep_ratio=0.35)
    assert set(after_band.src_keys[TYPE_BAND]) <= set(range(2, 8))
    with pytest.raises(ValueError):
        mask_random_opening(INFO, bond, episode_seed=7, probability=1.1)
    with pytest.raises(ValueError):
        PPOConfig(opening_band_keep_ratio=0)


def test_curriculum_warmup_endpoints_and_hold() -> None:
    kwargs = dict(warmup_updates=5, anneal_updates=120,
                  prob_start=0.0, prob_end=0.5, keep_start=1.0, keep_end=0.35)
    assert opening_mask_schedule(1, **kwargs) == (0.0, 1.0)
    assert opening_mask_schedule(5, **kwargs) == (0.0, 1.0)
    assert opening_mask_schedule(6, **kwargs) == (0.0, 1.0)
    assert opening_mask_schedule(125, **kwargs) == (0.5, 0.35)
    assert opening_mask_schedule(240, **kwargs) == (0.5, 0.35)
    previous = (0.0, 1.0)
    for upd in range(6, 126):
        current = opening_mask_schedule(upd, **kwargs)
        assert current[0] >= previous[0] and current[1] <= previous[1]
        previous = current


def _batch_for_action(enc, action_type: int, src: int) -> dict[str, torch.Tensor]:
    masks = extract_step_masks(enc, action_type, src, -1, -1)
    batch = {name: getattr(enc, name).unsqueeze(0) for name in ("cat_ids", "cont_feats", "valid_mask")}
    batch.update({name: torch.tensor([value]) for name, value in (
        ("a_type", action_type), ("a_src", src), ("a_tgt", -1), ("a_opt", -1),
    )})
    for name, value in vars(masks).items():
        batch[name] = value.unsqueeze(0) if isinstance(value, torch.Tensor) else torch.tensor([value])
    batch["bond_choice_mask"] = torch.tensor([opening_bond_mask(enc.action_tree.to_mask_spec())])
    return batch


def test_bond_entropy_merges_ready_alias_and_counts_forced_choices() -> None:
    cat = get_catalog()
    model = StrongholdPolicyNet(catalog=cat, config=PolicyConfig(d_model=32, n_heads=4, n_layers=1, dim_ff=64))
    for p in model.parameters():
        torch.nn.init.zeros_(p)
    model.eval()
    with RoomClient() as client:
        frame = client.reset(seed=19000, mode_id="mode_single_8")
    enc = encode_frame(frame, cat)
    # Uniform type and conditional group distributions: semantic group probs .75/.25.
    batch = _batch_for_action(enc, TYPE_INFO_READY, -1)
    out = model.evaluate_batch(batch)
    expected = -0.75 * math.log(0.75) - 0.25 * math.log(0.25)
    assert float(out["bond_choice_ent"][0]) == pytest.approx(expected, abs=1e-6)
    assert not out["bond_choice_ent"].requires_grad
    for keep in (CHOICE_OFFSET, CHOICE_OFFSET + 1):
        spec = enc.action_tree.to_mask_spec()
        tm = bytearray(spec.type_mask_bytes)
        if keep != CHOICE_OFFSET:
            tm[TYPE_INFO_READY] = 0
        tree = replace(enc.action_tree, type_mask=torch.tensor(list(tm), dtype=torch.bool),
                       src_keys={TYPE_BOND_GROUP: [keep]}, _type_mask_bytes=bytes(tm))
        restricted = replace(enc, action_tree=tree)
        forced_batch = _batch_for_action(restricted, TYPE_BOND_GROUP, keep)
        forced_out = model.evaluate_batch(forced_batch)
        assert float(forced_out["bond_choice_ent"][0]) == pytest.approx(0.0, abs=1e-6)
        stats = _OpeningEntropyStats()
        stats.add(out, batch)
        stats.add(forced_out, forced_batch)
        result = stats.result()
        assert result["bond_entropy"] == pytest.approx(expected / 2, abs=1e-6)
        assert result["bond_entropy_normalized"] == pytest.approx(expected / math.log(2), abs=1e-6)
        assert result["bond_forced_frac"] == 0.5
        assert result["bond_entropy_samples"] == 2


@pytest.mark.parametrize("use_async", [False, True])
def test_random_mask_rollout_ratios_and_unrestricted_eval(use_async: bool) -> None:
    cat = get_catalog()
    model = StrongholdPolicyNet(catalog=cat, config=PolicyConfig(d_model=32, n_heads=4, n_layers=1, dim_ff=64))
    cfg = PPOConfig(max_prep_steps_per_round=8, opening_mask_prob=1.0, opening_band_keep_ratio=0.35,
                    gamma_intra=1.0, gae_lambda_intra=1.0, kl_coef=0.0, ppo_epochs=1, minibatch_size=128)
    pool = AsyncActorPool(num_workers=2) if use_async else None
    try:
        batch, stats = collect_ppo_rollouts(model, num_episodes=4, modes=("mode_single_8",),
                                           num_workers=2, seed_start=19000, ppo_cfg=cfg,
                                           use_async_pool=use_async, pool=pool)
        assert stats["episodes"] == 4
        bonds = batch["bond_choice_mask"].sum(-1)
        assert int((bonds == 1).sum()) == 4
        assert int((bonds > 1).sum()) == 0
        bands = batch["a_type"] == TYPE_BAND
        assert int(bands.sum()) == 4
        counts = batch["band_mask"][bands].sum(-1)
        assert bool(((counts >= 2) & (counts <= math.ceil(cat.num_bands * 0.35))).all())
        with torch.no_grad():
            out = model.evaluate_batch(batch)
        assert torch.max(torch.abs(out["log_prob"] - batch["old_log_prob"])).item() < 1e-4
        update = ppo_update(model, torch.optim.AdamW(model.parameters(), lr=0.0), batch, cfg=cfg)
        assert update["bond_entropy"] == pytest.approx(0.0, abs=1e-5)
        assert update["bond_forced_frac"] == 1.0
        assert update["band_entropy"] > 0.0
        assert update["bond_entropy_samples"] == 4
        assert all(math.isfinite(v) for v in update.values())

        # Reuse the same workers: evaluation reset must clear every training mask.
        original_act = model.act_batch_tensors
        eval_bond_sizes, eval_band_sizes = [], []

        def capture(*args, **kwargs):
            for spec in args[3]:
                if TYPE_BOND_GROUP in spec.src_keys:
                    eval_bond_sizes.append(len(spec.src_keys[TYPE_BOND_GROUP]))
                if TYPE_BAND in spec.src_keys:
                    eval_band_sizes.append(len(spec.src_keys[TYPE_BAND]))
            return original_act(*args, **kwargs)

        model.act_batch_tensors = capture
        ev = evaluate_policy_solo(model, modes=("mode_single_8",), seeds=[19010, 19011],
                                  num_workers=2, max_prep_steps_per_round=8, deterministic=True,
                                  break_cycles=False, use_async_pool=use_async, pool=pool)
        assert ev["episodes"] == 2
        assert eval_bond_sizes == [2, 2]
        assert len(eval_band_sizes) == 2 and min(eval_band_sizes) > int(counts.max())
    finally:
        if pool is not None:
            pool.close()


def test_cli_rejects_bad_schedules_and_overwriting_input(tmp_path: Path) -> None:
    for args in (["--opening-mask-prob-end", "1.5"], ["--opening-band-keep-end", "0"],
                 ["--opening-mask-anneal-updates", "-1"],
                 ["--init-ckpt", str(tmp_path / "same.pt"), "--out", str(tmp_path / "same.pt")]):
        with pytest.raises(SystemExit) as error:
            main(args)
        assert error.value.code == 2


def test_finetune_launcher_builds_expected_command_without_training(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fake_python = tmp_path / "capture-python"
    capture = tmp_path / "args.json"
    fake_python.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "if '-m' in sys.argv:\n"
        " with open(os.environ['CAPTURE_ARGS'], 'w') as f: json.dump(sys.argv[1:], f)\n"
    )
    fake_python.chmod(0o755)
    checkpoint = tmp_path / "existing_best.pt"
    checkpoint.touch()
    env = dict(os.environ, ROOT_DIR=str(tmp_path), PYTHON_BIN=str(fake_python),
               CAPTURE_ARGS=str(capture), INIT_CKPT=str(checkpoint), RUN_ID="test")
    subprocess.run(["bash", str(root / "scripts/train_server_diversity.sh"), "normal"],
                   env=env, check=True, capture_output=True, text=True)
    args = json.loads(capture.read_text())
    def value(flag):
        return args[args.index(flag) + 1]
    assert value("--init-ckpt") == str(checkpoint)
    assert value("--gae-lambda-intra") == "1.0"
    assert value("--opening-mask-prob-start") == "0.0"
    assert value("--opening-mask-prob-end") == "0.5"
    assert value("--opening-band-keep-end") == "0.35"
    assert value("--opening-mask-anneal-updates") == "120"
    assert value("--out") == "checkpoints/diversity_normal_test_last.pt"
    assert value("--best-out") == "checkpoints/diversity_normal_test_best.pt"
    assert value("--seed") == "1210000"


def test_scratch_launcher_chains_fresh_weights_and_masks_from_warmup(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fake_python = tmp_path / "capture-python"
    capture = tmp_path / "args.jsonl"
    fake_python.write_text(
        f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\n"
        "if '-m' in sys.argv:\n"
        " with open(os.environ['CAPTURE_ARGS'], 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        " for flag in ('--out', '--best-out'):\n"
        "  if flag in sys.argv: Path(sys.argv[sys.argv.index(flag) + 1]).touch()\n"
    )
    fake_python.chmod(0o755)
    env = dict(os.environ, ROOT_DIR=str(tmp_path), PYTHON_BIN=str(fake_python),
               CAPTURE_ARGS=str(capture), RUN_ID="test", INIT_CKPT="old.pt",
               NORMAL_INIT_CKPT="old.pt", HARD_INIT_CKPT="old.pt", ABYSS_INIT_CKPT="old.pt")
    command = ["bash", str(root / "scripts/train_server_diversity_scratch.sh"), "all"]
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    commands = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(commands) == 4
    bc, *ppo_commands = commands
    assert bc[bc.index("-m") + 1] == "stronghold_rl.bc"
    assert "--init-ckpt" not in bc
    assert bc[bc.index("--opening-policy") + 1] == "balanced"
    assert bc[bc.index("--dataset-cache") + 1] == "data/v2feat_solo_1_8_1200ep_balanced_v2.pt"
    previous = bc[bc.index("--out") + 1]
    assert previous == "checkpoints/diversity_scratch_test_bc.pt"
    for stage, args in zip(("normal", "hard", "abyss"), ppo_commands):
        def value(flag):
            return args[args.index(flag) + 1]
        assert value("--init-ckpt") == previous
        assert value("--out") == f"checkpoints/diversity_scratch_test_{stage}_last.pt"
        assert value("--best-out") == f"checkpoints/diversity_scratch_test_{stage}_best.pt"
        assert value("--gae-lambda-intra") == "1.0"
        assert value("--opening-mask-prob-start") == value("--opening-mask-prob-end") == "0.5"
        assert value("--opening-band-keep-start") == value("--opening-band-keep-end") == "0.35"
        for update in (1, 5, 6, int(value("--updates"))):
            assert opening_mask_schedule(
                update, warmup_updates=int(value("--critic-warmup")),
                anneal_updates=int(value("--opening-mask-anneal-updates")),
                prob_start=float(value("--opening-mask-prob-start")),
                prob_end=float(value("--opening-mask-prob-end")),
                keep_start=float(value("--opening-band-keep-start")),
                keep_end=float(value("--opening-band-keep-end")),
            ) == (0.5, 0.35)
        previous = value("--best-out")

    # Reusing a run identifier must fail before any training process is invoked.
    repeated = subprocess.run(command, env=env, capture_output=True, text=True)
    assert repeated.returncode == 1
    assert "output already exists" in repeated.stderr
    assert len(capture.read_text().splitlines()) == 4
