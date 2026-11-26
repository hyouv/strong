"""End-to-end tests for solo simulator support, lossless codec, Transformer policy, BC, and Masked PPO."""
from __future__ import annotations

import copy
import gzip
import io
import json
from pathlib import Path

import torch

from stronghold_sim import RoomClient, SOLO_TIER1_MODES
from stronghold_rl.bc import (
    collect_solo_expert_dataset,
    evaluate_policy_solo,
    load_checkpoint,
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
)
from stronghold_rl.model import PolicyConfig, StrongholdPolicyNet
from stronghold_rl.ppo import PPOConfig, collect_ppo_rollouts, ppo_update


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
