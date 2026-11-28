"""Batch diversity must survive saturated policies and respect opening legality."""
from __future__ import annotations

import copy

import pytest
import torch

from stronghold_rl.codec import CONT_DIM, NUM_ACTION_TYPES, NUM_CAT_FIELDS, NUM_SUB_OPTS, NUM_TOKENS, TYPE_BAND, TYPE_READY
from stronghold_rl.model import PolicyConfig, StrongholdPolicyNet
from stronghold_rl.ppo import PPOConfig, band_diversity_loss, ppo_update


def test_collapsed_batch_has_restoring_gradient_even_for_near_zero_probabilities() -> None:
    logits = torch.tensor([[80.0, -80.0, -80.0]] * 4, requires_grad=True)
    mask = torch.ones_like(logits, dtype=torch.bool)
    loss, _ = band_diversity_loss(logits.log_softmax(-1), mask)
    loss.backward()
    assert loss.item() > 100
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[:, 0] > 0.1).all()
    assert (logits.grad[:, 1:] < -0.05).all()
    uniform_loss, _ = band_diversity_loss(torch.zeros_like(logits).log_softmax(-1), mask)
    assert abs(uniform_loss.item()) < 1e-6


def test_different_decisive_choices_are_balanced_without_per_state_entropy() -> None:
    logits = torch.tensor([[40.0, -40.0], [-40.0, 40.0]], requires_grad=True)
    lp = logits.log_softmax(-1)
    loss, marginal_lp = band_diversity_loss(lp, torch.ones_like(logits, dtype=torch.bool))
    assert abs(loss.item()) < 1e-6
    assert -(lp.exp() * lp).sum().item() < 1e-6
    assert torch.allclose(marginal_lp.exp(), torch.tensor([0.5, 0.5]))


def test_unequal_availability_and_forced_choices_have_no_false_penalty() -> None:
    mask = torch.tensor([[True, True, False, False], [False, True, True, False], [False, True, False, False]])
    logits = torch.zeros(3, 4, requires_grad=True)
    lp = logits.masked_fill(~mask, -1e9).log_softmax(-1)
    loss, marginal_lp = band_diversity_loss(lp, mask)
    assert abs(loss.item()) < 1e-6
    assert torch.allclose(marginal_lp.exp(), torch.tensor([1 / 6, 2 / 3, 1 / 6]))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.allclose(logits.grad, torch.zeros_like(logits), atol=1e-7)


def test_invalid_empty_opening_is_rejected() -> None:
    with pytest.raises(ValueError, match="legal band"):
        band_diversity_loss(torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.bool))


def _small_model_and_batch() -> tuple[StrongholdPolicyNet, dict[str, torch.Tensor]]:
    torch.manual_seed(4301)
    model = StrongholdPolicyNet(config=PolicyConfig(d_model=32, n_heads=4, n_layers=1, dim_ff=64))
    model.eval()
    # Two openings plus six PREP transitions: the loss must not rely on a random
    # minibatch containing opening actions, or on nonzero task advantages.
    n = 8
    opening = torch.arange(n) < 2
    types = torch.where(opening, TYPE_BAND, TYPE_READY)
    type_mask = torch.zeros(n, NUM_ACTION_TYPES, dtype=torch.bool)
    type_mask[torch.arange(n), types] = True
    band_mask = opening[:, None].expand(n, model.num_bands).clone()
    batch = {
        "cat_ids": torch.zeros(n, NUM_TOKENS, NUM_CAT_FIELDS, dtype=torch.long),
        "cont_feats": torch.zeros(n, NUM_TOKENS, CONT_DIM),
        "valid_mask": (torch.arange(NUM_TOKENS) < 2).expand(n, NUM_TOKENS).clone(),
        "a_type": types,
        "a_src": torch.where(opening, 0, -1),
        "a_tgt": torch.full((n,), -1, dtype=torch.long),
        "a_opt": torch.full((n,), -1, dtype=torch.long),
        "type_mask": type_mask,
        "src_mask": torch.zeros(n, NUM_TOKENS, dtype=torch.bool),
        "has_src": torch.zeros(n, dtype=torch.bool),
        "band_mask": band_mask,
        "has_band": opening,
        "tgt_mask": torch.zeros(n, NUM_TOKENS, dtype=torch.bool),
        "has_tgt": torch.zeros(n, dtype=torch.bool),
        "opt_mask": torch.zeros(n, NUM_SUB_OPTS, dtype=torch.bool),
        "has_opt": torch.zeros(n, dtype=torch.bool),
        "advantages": torch.zeros(n),
    }
    with torch.no_grad():
        # Reproduce a strongly collapsed opening while retaining real policy code.
        model.band_head[-1].weight.zero_()
        model.band_head[-1].bias.zero_()
        model.band_head[-1].bias[0] = 12
        out = model.evaluate_batch(batch)
        batch["old_log_prob"] = out["log_prob"].clone()
        batch["returns"] = out["value"].clone()
    return model, batch


def test_band_forward_matches_action_probability_and_ppo_recovers_with_zero_advantage() -> None:
    model, batch = _small_model_and_batch()
    openings = {k: v[:2] for k, v in batch.items()}
    lp = model.band_log_probs(openings)
    assert torch.allclose(lp[:, 0], model.evaluate_batch(openings)["log_prob"], atol=1e-6)
    before = lp[:, 0].exp().mean().item()
    cfg = PPOConfig(ppo_epochs=1, minibatch_size=2, kl_coef=0, ent_coef=0, vf_coef=0,
                    band_diversity_coef=0.01, target_kl=0)
    stats = ppo_update(model, torch.optim.SGD(model.parameters(), lr=0.1), batch, cfg=cfg)
    after = model.band_log_probs(openings)[:, 0].exp().mean().item()
    assert after < before
    assert stats["band_diversity_loss"] > 1
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())


@pytest.mark.parametrize("critic_only,coefficient,has_openings", [(False, 0.0, True), (True, 0.01, True), (False, 0.01, False)])
def test_disabled_warmup_and_no_openings_skip_extra_forward(critic_only, coefficient, has_openings) -> None:
    model, batch = _small_model_and_batch()
    if not has_openings:
        batch = {k: v[2:] for k, v in batch.items()}
    before = copy.deepcopy(model.state_dict())

    def unexpected_forward(_batch):
        raise AssertionError("Diversity forward should be skipped")

    model.band_log_probs = unexpected_forward
    cfg = PPOConfig(ppo_epochs=1, minibatch_size=4, kl_coef=0, ent_coef=0, vf_coef=0,
                    band_diversity_coef=coefficient, target_kl=0)
    stats = ppo_update(model, torch.optim.SGD(model.parameters(), lr=0.1), batch, cfg=cfg, critic_only=critic_only)
    assert stats["band_diversity_loss"] == 0
    assert torch.equal(model.band_head[-1].bias, before["band_head.2.bias"])

