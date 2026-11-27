"""End-to-end tests for solo simulator support, lossless codec, Transformer policy, BC, and Masked PPO."""
from __future__ import annotations

import copy
import gzip
import io
import json
from pathlib import Path

import numpy as np
import torch

from stronghold_sim import RoomClient, SOLO_TIER1_MODES, SOLO_TIER2_MODES, parse_solo_modes
from stronghold_rl.bc import (
    collect_solo_expert_dataset,
    evaluate_policy_solo,
    load_checkpoint,
    merge_bc_datasets,
    save_checkpoint,
    train_bc,
)
from stronghold_rl.catalog import get_catalog
from stronghold_rl.codec import (
    BAND_COMMAND_IDX,
    BOARD_OFFSET,
    BOSS_IDX,
    CAT_BAND,
    CAT_BOND,
    CAT_DIR,
    CAT_ENTITY,
    CAT_EQUIP0,
    CAT_EQUIP1,
    CAT_STAGE,
    CONT_DIM,
    ECONOMY_SHOP_IDX,
    EFFECT_OFFSET,
    MATCH_STAGE_IDX,
    NUM_CAT_FIELDS,
    NUM_TOKENS,
    encode_frame,
    encode_frame_into,
)
from stronghold_rl.model import PolicyConfig, StrongholdPolicyNet
from stronghold_rl.ppo import (
    PPOConfig,
    _finalize_ppo_batch,
    build_curriculum_episode_modes,
    collect_ppo_rollouts,
    ppo_update,
)
from stronghold_rl.rollout import AsyncActorPool, mask_cycle_action_types, prep_state_signature


def test_solo_mode_codec_bijection_and_logprob_consistency() -> None:
    """Every single legal action in a real solo expert match must round-trip losslessly,
    categorical fields must remain strictly isolated, and single-step sampling log_prob
    must match batched evaluate_batch exactly."""
    cat = get_catalog()
    model = StrongholdPolicyNet(catalog=cat, config=PolicyConfig(d_model=64, n_heads=4, n_layers=2, dim_ff=128))
    model.eval()

    with RoomClient() as client:
        demo = client.generate_examples("mode_single_1", seed=1234)
    assert demo["terminal"] is True
    assert demo["recorded"] > 50

    with gzip.GzipFile(fileobj=io.BytesIO(demo["replay"]), mode="rb") as gz:
        rows = [json.loads(line) for line in gz]

    checked_frames = 0
    checked_actions = 0
    saw_effect = False
    saw_path_tile = False
    for idx_row, row in enumerate(rows[1:-1]):
        fr = row["frame"]
        enc = encode_frame(fr, cat, prep_steps=45 if idx_row == 20 else 0)
        if idx_row == 20:
            assert abs(float(enc.cont_feats[ECONOMY_SHOP_IDX, 19]) - (45.0 / 80.0)) < 1e-5
            assert float(enc.cont_feats[ECONOMY_SHOP_IDX, 20]) == 1.0
        assert enc.cat_ids.shape == (NUM_TOKENS, NUM_CAT_FIELDS)
        assert enc.cont_feats.shape == (NUM_TOKENS, CONT_DIM)
        # Verify strict categorical isolation (no band/boss leaking into equip slots)
        assert int(enc.cat_ids[MATCH_STAGE_IDX, CAT_EQUIP0]) == 0
        assert int(enc.cat_ids[MATCH_STAGE_IDX, CAT_EQUIP1]) == 0
        assert int(enc.cat_ids[BOSS_IDX, CAT_EQUIP0]) == 0
        assert int(enc.cat_ids[BAND_COMMAND_IDX, CAT_EQUIP0]) == 0
        assert int(enc.cat_ids[MATCH_STAGE_IDX, CAT_STAGE]) > 0
        if fr["phase"] == "PREP":
            assert int(enc.cat_ids[BAND_COMMAND_IDX, CAT_BAND]) > 0
            if enc.valid_mask[EFFECT_OFFSET]:
                saw_effect = True
            if float(enc.cont_feats[BOARD_OFFSET : BOARD_OFFSET + 36, 22].sum()) > 0:
                saw_path_tile = True
        tree = enc.action_tree

        # 1. Every single legal action in frame['actions'] must round-trip 1-to-1
        assert len(tree.key_to_tuple) == len(fr["actions"])
        for action_key, tup in enumerate(tree.key_to_tuple):
            assert tree.decode(*tup) == action_key
            checked_actions += 1

        # 2. Check exact log_prob / value / entropy parity between act_single and evaluate_batch
        if idx_row % 25 == 0 or fr["phase"] != "PREP":
            act = model.act_single(enc, deterministic=False)
            a_type, a_src, a_tgt, a_opt = act["action_tuple"]
            m = act["masks"]
            batch = {
                "cat_ids": enc.cat_ids.unsqueeze(0),
                "cont_feats": enc.cont_feats.unsqueeze(0),
                "valid_mask": enc.valid_mask.unsqueeze(0),
                "a_type": torch.tensor([a_type], dtype=torch.long),
                "a_src": torch.tensor([a_src], dtype=torch.long),
                "a_tgt": torch.tensor([a_tgt], dtype=torch.long),
                "a_opt": torch.tensor([a_opt], dtype=torch.long),
                "type_mask": m.type_mask.unsqueeze(0),
                "src_mask": m.src_mask.unsqueeze(0),
                "has_src": torch.tensor([m.has_src], dtype=torch.bool),
                "band_mask": m.band_mask.unsqueeze(0),
                "has_band": torch.tensor([m.has_band], dtype=torch.bool),
                "tgt_mask": m.tgt_mask.unsqueeze(0),
                "has_tgt": torch.tensor([m.has_tgt], dtype=torch.bool),
                "opt_mask": m.opt_mask.unsqueeze(0),
                "has_opt": torch.tensor([m.has_opt], dtype=torch.bool),
            }
            with torch.no_grad():
                out = model.evaluate_batch(batch)
            assert abs(float(out["log_prob"].item()) - act["log_prob"]) < 1e-4
            assert abs(float(out["value"].item()) - act["value"]) < 1e-4
            assert abs(float(out["entropy"].item()) - act["entropy"]) < 1e-4
            sub_ent_sum = float(
                (out["type_ent"] + out["src_ent"] + out["band_ent"] + out["tgt_ent"] + out["opt_ent"]).item()
            )
            assert abs(sub_ent_sum - act["entropy"]) < 1e-5
            checked_frames += 1

    assert checked_frames >= 10
    assert checked_actions > 50_000
    assert saw_effect is True
    assert saw_path_tile is True


def test_bc_cold_start_and_masked_ppo_smoke(tmp_path: Path) -> None:
    """End-to-end smoke test: collect solo expert data -> BC cold-start -> live eval -> Masked PPO update."""
    cat = get_catalog()
    dataset, stats = collect_solo_expert_dataset(
        num_episodes=4,
        modes=SOLO_TIER1_MODES,
        seed_start=1100,
        num_workers=4,
        catalog=cat,
    )
    assert stats["episodes_kept"] == 4
    assert len(dataset) > 200

    cfg = PolicyConfig(d_model=64, n_heads=4, n_layers=2, dim_ff=128, dropout=0.1)
    model = StrongholdPolicyNet(catalog=cat, config=cfg)

    history = train_bc(
        model,
        dataset,
        epochs=4,
        batch_size=64,
        lr=2e-3,
        weight_decay=1e-2,
        label_smoothing=0.05,
        val_ratio=0.15,
        verbose=False,
    )
    assert history[-1]["policy_nll"] < history[0]["policy_nll"]
    assert history[-1]["type_acc"] > 0.55
    assert "val_nll" in history[-1] and "val_full_acc" in history[-1]

    ckpt = tmp_path / "bc_smoke.pt"
    save_checkpoint(model, ckpt, extra={"stats": stats})
    loaded, extra = load_checkpoint(ckpt, catalog=cat, dropout_override=0.0)
    assert extra["stats"]["episodes_kept"] == 4
    assert loaded.cfg.dropout == 0.0

    # Live solo evaluation after BC cold-start (with cycle-breaker)
    ev = evaluate_policy_solo(
        loaded,
        modes=("mode_single_1",),
        seeds=(9101,),
        deterministic=True,
        break_cycles=True,
        max_prep_steps_per_round=40,
        catalog=cat,
    )
    assert ev["episodes"] == 1
    assert ev["avg_steps"] > 10
    assert "cycle_breaks" in ev

    # Masked PPO rollout + critic-only warm-up + full update anchored to BC policy
    ref_model = copy.deepcopy(loaded)
    for p in ref_model.parameters():
        p.requires_grad_(False)

    ppo_cfg = PPOConfig(
        ppo_epochs=2,
        minibatch_size=64,
        max_prep_steps_per_round=40,
        kl_coef=0.05,
        target_kl=0.05,
    )
    rollout, r_stats = collect_ppo_rollouts(
        loaded,
        ref_model=ref_model,
        num_episodes=2,
        modes=("mode_single_1", "mode_single_2"),
        seed_start=21000,
        num_workers=2,
        ppo_cfg=ppo_cfg,
        catalog=cat,
    )
    assert r_stats["episodes"] == 2
    assert r_stats["transitions"] > 20

    # Verify critic_only leaves actor weights strictly untouched while updating value_head
    actor_before = loaded.type_head[0].weight.detach().clone()
    critic_before = loaded.value_head[0].weight.detach().clone()
    critic_opt = torch.optim.AdamW(loaded.value_head.parameters(), lr=3e-4)
    w_stats = ppo_update(loaded, critic_opt, rollout, cfg=ppo_cfg, critic_only=True)
    assert torch.equal(loaded.type_head[0].weight.detach(), actor_before)
    assert not torch.equal(loaded.value_head[0].weight.detach(), critic_before)
    assert w_stats["policy_loss"] == 0.0

    opt = torch.optim.AdamW(loaded.parameters(), lr=5e-5)
    u_stats = ppo_update(loaded, opt, rollout, cfg=ppo_cfg, critic_only=False)
    assert all(torch.isfinite(torch.tensor(v)) for v in u_stats.values())
    assert u_stats["kl_to_bc"] >= 0.0
    assert u_stats["approx_kl"] >= 0.0


def test_async_actor_pool_rollout_and_eval_parity() -> None:
    """Verify AsyncActorPool + vectorized act_batch_tensors produce exact log_prob parity with evaluate_batch."""
    cat = get_catalog()
    cfg = PolicyConfig(d_model=64, n_heads=4, n_layers=2, dim_ff=128, dropout=0.0)
    model = StrongholdPolicyNet(catalog=cat, config=cfg)
    model.eval()

    ppo_cfg = PPOConfig(max_prep_steps_per_round=25, kl_coef=0.0)
    with AsyncActorPool(num_workers=2, min_batch=2, batch_timeout_ms=2.0) as pool:
        rollout, r_stats = collect_ppo_rollouts(
            model,
            num_episodes=4,
            modes=("mode_single_1", "mode_single_2"),
            seed_start=23000,
            num_workers=2,
            ppo_cfg=ppo_cfg,
            catalog=cat,
            pool=pool,
        )
        assert r_stats["episodes"] == 4
        assert r_stats["transitions"] > 20
        assert r_stats["sim_errors"] == 0
        assert r_stats["avg_batch_size"] >= 1.0

        # Exact parity check between AsyncActorPool's old_log_prob and model.evaluate_batch
        with torch.no_grad():
            ev_out = model.evaluate_batch(rollout)
        max_lp_diff = float((ev_out["log_prob"] - rollout["old_log_prob"]).abs().max().item())
        assert max_lp_diff < 1e-4, f"AsyncActorPool log_prob mismatch vs evaluate_batch: {max_lp_diff}"

        # Strict deterministic eval parity between AsyncActorPool and legacy sync RoomPool on identical seeds
        ev_async = evaluate_policy_solo(
            model,
            modes=("mode_single_1", "mode_single_2"),
            seeds=(9200, 9201),
            num_workers=2,
            deterministic=True,
            break_cycles=False,
            max_prep_steps_per_round=25,
            catalog=cat,
            pool=pool,
        )
        ev_sync = evaluate_policy_solo(
            model,
            modes=("mode_single_1", "mode_single_2"),
            seeds=(9200, 9201),
            num_workers=2,
            deterministic=True,
            break_cycles=False,
            max_prep_steps_per_round=25,
            catalog=cat,
            use_async_pool=False,
        )
        assert ev_async["episodes"] == 2
        assert ev_async["win_rate"] == ev_sync["win_rate"]
        assert ev_async["avg_rounds_passed"] == ev_sync["avg_rounds_passed"]
        assert ev_async["avg_steps"] == ev_sync["avg_steps"]
        assert ev_async["forced_readies"] == ev_sync["forced_readies"]
        assert ev_async["cycle_breaks"] == 0
        assert set(ev_async["by_mode"].keys()) == {"mode_single_1", "mode_single_2"}
        assert ev_async["by_mode"]["mode_single_1"]["episodes"] == 1
        assert ev_async["by_mode"]["mode_single_2"]["episodes"] == 1

        # Also exercise cycle-breaking path in AsyncActorPool
        ev_cb = evaluate_policy_solo(
            model,
            modes=("mode_single_1", "mode_single_2"),
            seeds=(9200, 9201),
            num_workers=2,
            deterministic=True,
            break_cycles=True,
            max_prep_steps_per_round=25,
            catalog=cat,
            pool=pool,
        )
        assert ev_cb["episodes"] == 2
        assert ev_cb["sim_errors"] == 0


def test_curriculum_modes_round_discount_and_tier2_codec() -> None:
    """Verify mode parsing, curriculum mixing, round-boundary discounting, and mode_single_5..8 codec/rollout."""
    assert parse_solo_modes("tier1") == SOLO_TIER1_MODES
    assert parse_solo_modes("5-8") == SOLO_TIER2_MODES
    assert parse_solo_modes("tier2") == SOLO_TIER2_MODES
    assert parse_solo_modes("5,7") == ("mode_single_5", "mode_single_7")

    mixed = build_curriculum_episode_modes(10, SOLO_TIER2_MODES, mix_modes=SOLO_TIER1_MODES, mix_ratio=0.3)
    assert len(mixed) == 10
    assert sum(1 for m in mixed if m in SOLO_TIER1_MODES) == 3
    assert sum(1 for m in mixed if m in SOLO_TIER2_MODES) == 7

    # Rotating mix_offset guarantees uniform coverage across 1-14 even when n_mix < 14
    earlier = parse_solo_modes("1-14")
    abyss = parse_solo_modes("15-20")
    assert len(earlier) == 14 and len(abyss) == 6
    seen_mix: list[str] = []
    cur = 0
    for _ in range(2):
        ep_m = build_curriculum_episode_modes(120, abyss, mix_modes=earlier, mix_ratio=0.08, mix_offset=cur)
        m_items = [m for m in ep_m if m in earlier]
        assert len(m_items) == 10
        seen_mix.extend(m_items)
        cur += len(m_items)
    assert set(seen_mix) == set(earlier)

    # Verify round-boundary discounting: within-round steps get gamma_intra=1.0, round boundary gets gamma_round=0.9
    dummy_t = lambda r, v, done, rnd_end: {
        "cat_ids": torch.zeros((NUM_TOKENS, NUM_CAT_FIELDS), dtype=torch.long),
        "cont_feats": torch.zeros((NUM_TOKENS, CONT_DIM), dtype=torch.float32),
        "valid_mask": torch.ones(NUM_TOKENS, dtype=torch.bool),
        "a_type": 0,
        "a_src": -1,
        "a_tgt": -1,
        "a_opt": -1,
        "type_mask": torch.ones(19, dtype=torch.bool),
        "src_mask": torch.zeros(NUM_TOKENS, dtype=torch.bool),
        "has_src": False,
        "band_mask": torch.zeros(61, dtype=torch.bool),
        "has_band": False,
        "tgt_mask": torch.zeros(NUM_TOKENS, dtype=torch.bool),
        "has_tgt": False,
        "opt_mask": torch.zeros(4, dtype=torch.bool),
        "has_opt": False,
        "is_strategic": False,
        "old_log_prob": -0.5,
        "value": v,
        "reward": r,
        "done": done,
        "round_end": rnd_end,
    }
    ep = [
        dummy_t(0.0, 0.0, False, False),  # intra-round step -> gamma=1.0
        dummy_t(0.0, 0.0, False, True),   # round boundary   -> gamma=0.9
        dummy_t(1.0, 0.0, True, False),   # terminal reward  -> 1.0
    ]
    cfg = PPOConfig(gamma_intra=1.0, gamma_round=0.9, gae_lambda=1.0, kl_coef=0.0)
    batch, n_tr = _finalize_ppo_batch([ep], cfg=cfg, ref_model=None, device="cpu")
    assert n_tr == 3
    # Step 2 return = 1.0; Step 1 return = 0.9 * 1.0 = 0.9; Step 0 return = 1.0 * 0.9 = 0.9 (no intra-round decay!)
    assert abs(float(batch["returns"][2]) - 1.0) < 1e-5
    assert abs(float(batch["returns"][1]) - 0.9) < 1e-5
    assert abs(float(batch["returns"][0]) - 0.9) < 1e-5

    # Verify tier-2 (mode_single_5..8) codec + rollout + strategic entropy multiplier end-to-end
    cat = get_catalog()
    model = StrongholdPolicyNet(catalog=cat, config=PolicyConfig(d_model=64, n_heads=4, n_layers=2, dim_ff=128))
    ppo_cfg = PPOConfig(
        gamma_intra=1.0,
        gamma_round=0.97,
        strategic_ent_mult=2.5,
        hidden_clear_bonus=0.5,
        max_prep_steps_per_round=25,
        kl_coef=0.0,
    )
    rollout, r_stats = collect_ppo_rollouts(
        model,
        num_episodes=2,
        modes=SOLO_TIER2_MODES,
        seed_start=31000,
        num_workers=2,
        ppo_cfg=ppo_cfg,
        catalog=cat,
    )
    assert r_stats["episodes"] == 2
    assert r_stats["transitions"] > 20
    assert bool(rollout["is_strategic"].any())
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    u_stats = ppo_update(model, opt, rollout, cfg=ppo_cfg)
    assert u_stats["strategic_entropy"] >= 0.0
    assert 0.0 <= u_stats["explore_entropy"] <= u_stats["entropy"] * ppo_cfg.strategic_ent_mult + 1e-4
    assert all(torch.isfinite(torch.tensor(v)) for v in u_stats.values())

    # Verify merge_bc_datasets keeps primary tail intact
    ds_primary, _ = collect_solo_expert_dataset(num_episodes=1, modes=("mode_single_5",), seed_start=32000, num_workers=1, catalog=cat)
    ds_aux, _ = collect_solo_expert_dataset(num_episodes=1, modes=("mode_single_1",), seed_start=32001, num_workers=1, catalog=cat)
    merged = merge_bc_datasets(ds_primary, ds_aux, aux_ratio=0.2, seed=0)
    assert len(merged) > len(ds_primary)
    assert torch.equal(merged.a_type[-len(ds_primary):], ds_primary.a_type)


def test_compact_protocol_and_fast_codec_exact_parity() -> None:
    """Compact JS protocol (`compact=True` + `cactions` + lossless unitStats cache) and zero-tensor
    `encode_frame_into` must produce 100% bit-exact tensors and action masks vs full protocol."""
    cat = get_catalog()
    cat_buf = np.empty((NUM_TOKENS, NUM_CAT_FIELDS), dtype=np.int32)
    cont_buf = np.empty((NUM_TOKENS, CONT_DIM), dtype=np.float32)
    valid_buf = np.empty(NUM_TOKENS, dtype=np.bool_)

    with RoomClient() as rc_full, RoomClient() as rc_comp:
        f_full = rc_full.reset(seed=41234, mode_id="mode_single_5", policy_version=0, compact=False)
        f_comp = rc_comp.reset(seed=41234, mode_id="mode_single_5", policy_version=0, compact=True)
        steps = 0
        prep_steps = 0
        last_rnd = -1

        while not f_full.get("terminal"):
            assert not f_comp.get("terminal")
            assert "cactions" in f_comp and "actions" not in f_comp
            rnd = int(f_full["pub"].get("round") or 0)
            if rnd != last_rnd:
                last_rnd = rnd
                prep_steps = 0

            enc_full = encode_frame(f_full, cat, prep_steps=prep_steps)
            enc_comp = encode_frame(f_comp, cat, prep_steps=prep_steps)
            dec_into = encode_frame_into(f_comp, cat, cat_buf, cont_buf, valid_buf, prep_steps=prep_steps)

            assert torch.equal(enc_full.cat_ids, enc_comp.cat_ids), f"cat_ids mismatch at step {steps}"
            assert torch.equal(enc_full.cont_feats, enc_comp.cont_feats), f"cont_feats mismatch at step {steps}"
            assert torch.equal(enc_full.valid_mask, enc_comp.valid_mask), f"valid_mask mismatch at step {steps}"
            assert enc_full.action_tree.to_mask_spec() == enc_comp.action_tree.to_mask_spec()
            assert enc_full.action_tree.key_to_tuple == enc_comp.action_tree.key_to_tuple
            assert enc_full.action_tree.to_mask_spec() == dec_into.mask_spec
            assert torch.equal(enc_full.cat_ids, torch.from_numpy(cat_buf.astype(np.int64)))
            assert torch.equal(enc_full.cont_feats, torch.from_numpy(cont_buf))
            assert torch.equal(enc_full.valid_mask, torch.from_numpy(valid_buf))

            # Exercise freeze, buy, levelUp, move, equip, ready across a full match
            acts = f_full["actions"]
            if f_full["phase"] == "PREP":
                prep_steps += 1
                if prep_steps == 1:
                    ak = next((i for i, a in enumerate(acts) if a["t"] == "g.freeze"), 0)
                elif prep_steps >= 18:
                    ak = next((i for i, a in enumerate(acts) if a["t"] == "g.ready"), 0)
                else:
                    ak = (
                        next((i for i, a in enumerate(acts) if a["t"] == "g.equip"), None)
                        or next((i for i, a in enumerate(acts) if a["t"] == "g.buy"), None)
                        or next((i for i, a in enumerate(acts) if a["t"] == "g.levelUp"), None)
                        or (steps * 7) % len(acts)
                    )
            else:
                ak = steps % len(acts)

            # Verify FastActionDecoder decodes the exact same action_key
            tup = enc_full.action_tree.key_to_tuple[ak]
            assert dec_into.decode(*tup) == ak

            f_full = rc_full.step(f_full["decision_id"], ak, policy_version=0)
            f_comp = rc_comp.step(f_comp["decision_id"], ak, policy_version=0)
            steps += 1

        assert f_comp.get("terminal") is True
        assert f_comp.get("victory") == f_full.get("victory")
        assert f_comp.get("roundsPassed") == f_full.get("roundsPassed")
        assert steps > 50


def test_prep_state_signature_and_cycle_masking() -> None:
    """Free refreshes, shop slot changes, item swaps must change PREP signature; cycle escape masks MOVE/FREEZE."""
    from stronghold_rl.codec import ActionMaskSpec, NUM_ACTION_TYPES, TYPE_BUY, TYPE_FREEZE, TYPE_MOVE, TYPE_READY

    base_frame = {
        "own": {
            "funds": 8,
            "shop": {
                "level": 5,
                "freeRefreshes": 2,
                "frozen": False,
                "slots": [
                    {"id": "chess_a", "sold": False, "frozen": False},
                    {"id": "chess_b", "sold": False, "frozen": False},
                ],
                "rewardOffer": None,
            },
            "board": [{"uid": 1, "row": 9, "col": 3, "dir": "RIGHT", "items": [{"id": "item_a"}]}],
            "hand": [None, {"uid": 2}],
            "temp": [None],
        }
    }
    sig0 = prep_state_signature(base_frame)

    # Free refresh changing shop entity IDs & freeRefreshes must produce a distinct signature
    refreshed = copy.deepcopy(base_frame)
    refreshed["own"]["shop"]["freeRefreshes"] = 1
    refreshed["own"]["shop"]["slots"][0]["id"] = "chess_c"
    assert prep_state_signature(refreshed) != sig0

    # Replacing an equipped item with a different item of same count must change signature
    swapped_item = copy.deepcopy(base_frame)
    swapped_item["own"]["board"][0]["items"] = [{"id": "item_b"}]
    assert prep_state_signature(swapped_item) != sig0

    # Slot-level freeze must change signature
    slot_frozen = copy.deepcopy(base_frame)
    slot_frozen["own"]["shop"]["slots"][0]["frozen"] = True
    assert prep_state_signature(slot_frozen) != sig0

    # Cycle escape must mask MOVE & FREEZE while preserving productive BUY/READY actions
    tm = bytearray(NUM_ACTION_TYPES)
    tm[TYPE_READY] = 1
    tm[TYPE_BUY] = 1
    tm[TYPE_FREEZE] = 1
    tm[TYPE_MOVE] = 1
    spec = ActionMaskSpec(type_mask_bytes=bytes(tm), src_keys={}, tgt_keys={}, opt_bits={}, num_bands=61)
    masked_spec, did_mask = mask_cycle_action_types(spec)
    assert did_mask is True
    assert masked_spec.type_mask_bytes[TYPE_MOVE] == 0
    assert masked_spec.type_mask_bytes[TYPE_FREEZE] == 0
    assert masked_spec.type_mask_bytes[TYPE_BUY] == 1
    assert masked_spec.type_mask_bytes[TYPE_READY] == 1

    # If ONLY MOVE is legal, masking must refuse to produce an all-zero mask
    tm_only_move = bytearray(NUM_ACTION_TYPES)
    tm_only_move[TYPE_MOVE] = 1
    spec_only = ActionMaskSpec(type_mask_bytes=bytes(tm_only_move), src_keys={}, tgt_keys={}, opt_bits={}, num_bands=61)
    same_spec, did_mask2 = mask_cycle_action_types(spec_only)
    assert did_mask2 is False
    assert same_spec.type_mask_bytes[TYPE_MOVE] == 1


def test_prep_timeout_and_repeat_state_penalties() -> None:
    """Verify prep_timeout_penalty and repeat_state_penalty wire through AsyncActorPool and sync rollout."""
    cat = get_catalog()
    model = StrongholdPolicyNet(catalog=cat, config=PolicyConfig(d_model=64, n_heads=4, n_layers=2, dim_ff=128))
    cfg = PPOConfig(
        max_prep_steps_per_round=12,
        prep_timeout_penalty=-0.10,
        repeat_state_penalty=-0.005,
        kl_coef=0.0,
    )
    batch_async, stats_async = collect_ppo_rollouts(
        model,
        num_episodes=2,
        modes=("mode_single_1",),
        seed_start=25000,
        num_workers=2,
        ppo_cfg=cfg,
        catalog=cat,
    )
    assert stats_async["episodes"] == 2
    assert "repeat_states" in stats_async and stats_async["repeat_states"] >= 0
    assert torch.isfinite(batch_async["returns"]).all()

    batch_sync, stats_sync = collect_ppo_rollouts(
        model,
        num_episodes=1,
        modes=("mode_single_1",),
        seed_start=25010,
        num_workers=1,
        ppo_cfg=cfg,
        catalog=cat,
        use_async_pool=False,
    )
    assert stats_sync["episodes"] == 1
    assert "repeat_states" in stats_sync and stats_sync["repeat_states"] >= 0
    assert torch.isfinite(batch_sync["returns"]).all()


def test_round_boundary_gae_lambda_preserves_intra_round_credit() -> None:
    """With gae_lambda_intra=1.0, long intra-round PREP chains suffer zero lambda decay across steps."""
    dummy_t = lambda r, v, done, rnd_end: {
        "cat_ids": torch.zeros((NUM_TOKENS, NUM_CAT_FIELDS), dtype=torch.long),
        "cont_feats": torch.zeros((NUM_TOKENS, CONT_DIM), dtype=torch.float32),
        "valid_mask": torch.ones(NUM_TOKENS, dtype=torch.bool),
        "a_type": 0,
        "a_src": -1,
        "a_tgt": -1,
        "a_opt": -1,
        "type_mask": torch.ones(19, dtype=torch.bool),
        "src_mask": torch.zeros(NUM_TOKENS, dtype=torch.bool),
        "has_src": False,
        "band_mask": torch.zeros(61, dtype=torch.bool),
        "has_band": False,
        "tgt_mask": torch.zeros(NUM_TOKENS, dtype=torch.bool),
        "has_tgt": False,
        "opt_mask": torch.zeros(4, dtype=torch.bool),
        "has_opt": False,
        "is_strategic": False,
        "old_log_prob": -0.5,
        "value": v,
        "reward": r,
        "done": done,
        "round_end": rnd_end,
    }
    # 20 intra-round steps in round 1 (v=0.2), 1 round-end transition, 20 intra-round steps in round 2 (v=0.2), terminal +1.5
    ep = (
        [dummy_t(0.0, 0.2, False, False) for _ in range(20)]
        + [dummy_t(0.0, 0.2, False, True)]
        + [dummy_t(0.0, 0.2, False, False) for _ in range(20)]
        + [dummy_t(1.5, 0.2, True, False)]
    )
    cfg_fixed = PPOConfig(gamma_intra=1.0, gamma_round=0.97, gae_lambda=0.95, gae_lambda_intra=1.0, kl_coef=0.0)
    b_fixed, _ = _finalize_ppo_batch([ep], cfg=cfg_fixed, ref_model=None, device="cpu")

    # Round 2 steps: no intra-round discount or lambda decay -> exact return 1.5 for every step in round 2
    assert abs(float(b_fixed["returns"][-1]) - 1.5) < 1e-5
    assert abs(float(b_fixed["returns"][21]) - 1.5) < 1e-5
    # Round 1 boundary step (idx 20): delta = 0.97*0.2 - 0.2 = -0.006, gae_next = 1.3 -> ret = 0.2 + delta + 0.97*0.95*1.3
    expected_r1 = 0.97 * (0.2 + 0.95 * (1.5 - 0.2))
    assert abs(float(b_fixed["returns"][20]) - expected_r1) < 1e-5
    # Step 0 (20 PREP steps earlier in round 1) gets IDENTICAL return (zero intra-round decay!)
    assert abs(float(b_fixed["returns"][0]) - expected_r1) < 1e-5

    # Contrast with legacy per-step lambda=0.95: credit at step 0 decays severely toward Critic value 0.2
    cfg_legacy = PPOConfig(gamma_intra=1.0, gamma_round=0.97, gae_lambda=0.95, gae_lambda_intra=None, kl_coef=0.0)
    b_legacy, _ = _finalize_ppo_batch([ep], cfg=cfg_legacy, ref_model=None, device="cpu")
    assert float(b_fixed["returns"][0]) > float(b_legacy["returns"][0]) + 0.8
