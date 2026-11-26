"""Masked PPO fine-tuning loop for solo Stronghold Protocol (act3autochess@V3_1).

Features:
- Batched GPU rollout inference across parallel RoomPool workers (`act_batch`)
- Post-rollout batched BC reference log-prob evaluation (eliminates per-step ref forward calls)
- Optional Critic warm-up phase (calibrates `value_head` to shaped rewards while keeping BC policy frozen)
- Non-negative BC KL anchor + `target_kl` early stopping to prevent 4-step autoregressive policy drift
- Periodic held-out evaluation and best/periodic checkpoint saving
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Sequence

import torch
import torch.nn as nn

from stronghold_sim import RoomPool, SOLO_TIER1_MODES

from .bc import (
    describe_device,
    evaluate_policy_solo,
    load_checkpoint,
    pick_force_unblock_action,
    resolve_device,
    save_checkpoint,
)
from .catalog import GameCatalog, get_catalog
from .codec import encode_frame
from .model import PolicyConfig, StepMasks, StrongholdPolicyNet


@dataclass
class PPOConfig:
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    kl_coef: float = 0.05           # anchor penalty to frozen BC reference policy (0 disables)
    target_kl: float = 0.05         # early-stop remaining PPO epochs when approx_kl exceeds this (0 disables)
    lr: float = 3e-5                # gentle actor+backbone LR for 4-step autoregressive fine-tuning
    critic_lr: float = 2e-4         # faster LR during critic-only warm-up
    critic_warmup_updates: int = 3  # calibrate value_head before updating actor from BC
    grad_clip: float = 0.5
    ppo_epochs: int = 2
    minibatch_size: int = 512
    ref_eval_batch_size: int = 1024
    max_prep_steps_per_round: int = 80
    # Reward shaping (solo curriculum)
    step_penalty: float = -0.0005
    round_survive_bonus: float = 0.06
    lp_loss_penalty: float = -0.025
    victory_reward: float = 1.0
    defeat_reward: float = -1.0


@dataclass
class _Transition:
    cat_ids: torch.Tensor
    cont_feats: torch.Tensor
    valid_mask: torch.Tensor
    a_type: int
    a_src: int
    a_tgt: int
    a_opt: int
    masks: StepMasks
    old_log_prob: float
    value: float
    reward: float = 0.0
    done: bool = False


@torch.no_grad()
def _compute_ref_log_probs_batched(
    ref_model: StrongholdPolicyNet,
    batch: dict[str, torch.Tensor],
    *,
    chunk_size: int = 1024,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Evaluate frozen BC reference policy log_probs in large GPU minibatches."""
    ref_model.eval()
    n = int(batch["a_type"].shape[0])
    out_chunks: list[torch.Tensor] = []
    keys = (
        "cat_ids",
        "cont_feats",
        "valid_mask",
        "a_type",
        "a_src",
        "a_tgt",
        "a_opt",
        "type_mask",
        "src_mask",
        "has_src",
        "band_mask",
        "has_band",
        "tgt_mask",
        "has_tgt",
        "opt_mask",
        "has_opt",
    )
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        mb = {k: batch[k][start:end].to(device, non_blocking=True) for k in keys}
        lp = ref_model.evaluate_batch(mb)["log_prob"].detach().cpu()
        out_chunks.append(lp)
    return torch.cat(out_chunks, dim=0)


def collect_ppo_rollouts(
    model: StrongholdPolicyNet,
    *,
    ref_model: StrongholdPolicyNet | None = None,
    num_episodes: int = 8,
    modes: Sequence[str] = SOLO_TIER1_MODES,
    seed_start: int = 20000,
    num_workers: int = 8,
    ppo_cfg: PPOConfig | None = None,
    device: torch.device | str = "cpu",
    catalog: GameCatalog | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Run parallel solo rollouts with batched GPU inference and batched BC ref log-prob evaluation."""
    cfg = ppo_cfg if ppo_cfg is not None else PPOConfig()
    cat = catalog if catalog is not None else get_catalog()
    model.eval()
    if ref_model is not None:
        ref_model.eval()

    t0 = time.perf_counter()
    all_episodes: list[list[_Transition]] = []
    wins = 0
    rounds_passed: list[int] = []
    forced_readies = 0

    n_workers = max(1, min(num_workers, num_episodes))
    with RoomPool(num_workers=n_workers) as pool:
        next_ep = 0
        active_frames: dict[int, dict[str, Any]] = {}
        ep_buf: dict[int, list[_Transition]] = {}
        ep_prev_round: dict[int, int] = {}
        ep_prev_lp: dict[int, int] = {}
        ep_prep_steps: dict[int, int] = {}

        def _spawn_cmd(ep_idx: int) -> dict[str, Any]:
            return {
                "op": "reset",
                "seed": seed_start + ep_idx,
                "modeId": modes[ep_idx % len(modes)],
                "policy_version": 0,
            }

        init_cmds = {}
        for w in range(n_workers):
            init_cmds[w] = _spawn_cmd(next_ep)
            ep_buf[w] = []
            ep_prev_round[w] = -1
            ep_prev_lp[w] = -1
            ep_prep_steps[w] = 0
            next_ep += 1
        active_frames = pool.exchange(init_cmds)

        while active_frames:
            step_cmds: dict[int, dict[str, Any]] = {}
            batch_workers: list[int] = []
            batch_frames: list[dict[str, Any]] = []
            batch_encs: list[Any] = []

            for w, frame in active_frames.items():
                rnd = int(frame["pub"].get("round") or 0)
                lp = int(frame["own"].get("lp") or 0)

                if ep_buf[w] and ep_prev_round[w] >= 0 and rnd > ep_prev_round[w]:
                    r_delta = rnd - ep_prev_round[w]
                    lp_lost = max(0, ep_prev_lp[w] - lp)
                    ep_buf[w][-1].reward += cfg.round_survive_bonus * r_delta + cfg.lp_loss_penalty * lp_lost
                    ep_prep_steps[w] = 0
                elif rnd != ep_prev_round[w]:
                    ep_prep_steps[w] = 0

                ep_prev_round[w] = rnd
                ep_prev_lp[w] = lp

                enc = encode_frame(frame, cat, prep_steps=ep_prep_steps[w])
                if frame["phase"] == "PREP" and ep_prep_steps[w] >= cfg.max_prep_steps_per_round:
                    forced_readies += 1
                    action_key = pick_force_unblock_action(enc)
                    step_cmds[w] = {
                        "op": "step",
                        "decision_id": frame["decision_id"],
                        "action_key": action_key,
                        "policy_version": 0,
                    }
                else:
                    batch_workers.append(w)
                    batch_frames.append(frame)
                    batch_encs.append(enc)

            if batch_encs:
                acts = model.act_batch(batch_encs, deterministic=False, device=device)
                for w, frame, enc, act in zip(batch_workers, batch_frames, batch_encs, acts):
                    a_type, a_src, a_tgt, a_opt = act["action_tuple"]
                    is_prep = frame["phase"] == "PREP"
                    if is_prep:
                        ep_prep_steps[w] += 1
                    ep_buf[w].append(
                        _Transition(
                            cat_ids=enc.cat_ids,
                            cont_feats=enc.cont_feats,
                            valid_mask=enc.valid_mask,
                            a_type=a_type,
                            a_src=a_src,
                            a_tgt=a_tgt,
                            a_opt=a_opt,
                            masks=act["masks"],
                            old_log_prob=act["log_prob"],
                            value=act["value"],
                            reward=cfg.step_penalty if is_prep else 0.0,
                        )
                    )
                    step_cmds[w] = {
                        "op": "step",
                        "decision_id": frame["decision_id"],
                        "action_key": act["action_key"],
                        "policy_version": 0,
                    }

            replies = pool.exchange(step_cmds)
            reset_cmds: dict[int, dict[str, Any]] = {}
            next_active: dict[int, dict[str, Any]] = {}

            for w, rep in replies.items():
                if rep.get("terminal"):
                    win = bool(rep.get("victory"))
                    if win:
                        wins += 1
                    rounds_passed.append(int(rep.get("roundsPassed", 0)))
                    if ep_buf[w]:
                        ep_buf[w][-1].reward += cfg.victory_reward if win else cfg.defeat_reward
                        ep_buf[w][-1].done = True
                        all_episodes.append(ep_buf[w])
                    ep_buf[w] = []
                    ep_prev_round[w] = -1
                    ep_prev_lp[w] = -1
                    ep_prep_steps[w] = 0
                    if next_ep < num_episodes:
                        reset_cmds[w] = _spawn_cmd(next_ep)
                        next_ep += 1
                else:
                    next_active[w] = rep

            if reset_cmds:
                next_active.update(pool.exchange(reset_cmds))
            active_frames = next_active

    flat: list[_Transition] = []
    adv_list: list[float] = []
    ret_list: list[float] = []

    for ep in all_episodes:
        gae = 0.0
        ep_adv = [0.0] * len(ep)
        ep_ret = [0.0] * len(ep)
        for i in range(len(ep) - 1, -1, -1):
            tr = ep[i]
            next_val = 0.0 if (i == len(ep) - 1 or tr.done) else ep[i + 1].value
            non_term = 0.0 if tr.done else 1.0
            delta = tr.reward + cfg.gamma * next_val * non_term - tr.value
            gae = delta + cfg.gamma * cfg.gae_lambda * non_term * gae
            ep_adv[i] = gae
            ep_ret[i] = gae + tr.value
        flat.extend(ep)
        adv_list.extend(ep_adv)
        ret_list.extend(ep_ret)

    advantages = torch.tensor(adv_list, dtype=torch.float32)
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    old_log_prob = torch.tensor([t.old_log_prob for t in flat], dtype=torch.float32)
    batch: dict[str, torch.Tensor] = {
        "cat_ids": torch.stack([t.cat_ids for t in flat]),
        "cont_feats": torch.stack([t.cont_feats for t in flat]),
        "valid_mask": torch.stack([t.valid_mask for t in flat]),
        "a_type": torch.tensor([t.a_type for t in flat], dtype=torch.long),
        "a_src": torch.tensor([t.a_src for t in flat], dtype=torch.long),
        "a_tgt": torch.tensor([t.a_tgt for t in flat], dtype=torch.long),
        "a_opt": torch.tensor([t.a_opt for t in flat], dtype=torch.long),
        "type_mask": torch.stack([t.masks.type_mask for t in flat]),
        "src_mask": torch.stack([t.masks.src_mask for t in flat]),
        "has_src": torch.tensor([t.masks.has_src for t in flat], dtype=torch.bool),
        "band_mask": torch.stack([t.masks.band_mask for t in flat]),
        "has_band": torch.tensor([t.masks.has_band for t in flat], dtype=torch.bool),
        "tgt_mask": torch.stack([t.masks.tgt_mask for t in flat]),
        "has_tgt": torch.tensor([t.masks.has_tgt for t in flat], dtype=torch.bool),
        "opt_mask": torch.stack([t.masks.opt_mask for t in flat]),
        "has_opt": torch.tensor([t.masks.has_opt for t in flat], dtype=torch.bool),
        "old_log_prob": old_log_prob,
        "advantages": advantages,
        "returns": torch.tensor(ret_list, dtype=torch.float32),
    }

    if ref_model is not None and cfg.kl_coef > 0:
        batch["ref_log_prob"] = _compute_ref_log_probs_batched(
            ref_model,
            batch,
            chunk_size=cfg.ref_eval_batch_size,
            device=device,
        )
    else:
        batch["ref_log_prob"] = old_log_prob.clone()

    stats = {
        "episodes": len(all_episodes),
        "transitions": len(flat),
        "win_rate": wins / max(1, len(all_episodes)),
        "avg_rounds_passed": sum(rounds_passed) / max(1, len(rounds_passed)),
        "forced_readies": forced_readies,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    return batch, stats


def ppo_update(
    model: StrongholdPolicyNet,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, torch.Tensor],
    *,
    cfg: PPOConfig | None = None,
    critic_only: bool = False,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Run clipped Masked PPO epochs with optional critic-only warm-up and target-KL early stopping."""
    c = cfg if cfg is not None else PPOConfig()
    if str(device).startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    model.train()
    n = int(rollout["a_type"].shape[0])

    saved_req_grad: list[tuple[nn.Parameter, bool]] = []
    if critic_only:
        critic_ids = {id(p) for p in model.value_head.parameters()}
        for p in model.parameters():
            saved_req_grad.append((p, p.requires_grad))
            p.requires_grad_(id(p) in critic_ids)

    sum_ploss = 0.0
    sum_vloss = 0.0
    sum_ent = 0.0
    sum_kl_bc = 0.0
    sum_approx_kl = 0.0
    sum_clip = 0.0
    n_batches = 0
    epochs_ran = 0
    early_stopped = 0.0

    try:
        for _ in range(c.ppo_epochs):
            perm = torch.randperm(n)
            ep_kl_sum = 0.0
            ep_batches = 0
            epochs_ran += 1

            for start in range(0, n, c.minibatch_size):
                idx = perm[start : start + c.minibatch_size]
                mb = {k: v[idx].to(device, non_blocking=True) for k, v in rollout.items()}

                out = model.evaluate_batch(mb)
                new_lp = out["log_prob"]
                old_lp = mb["old_log_prob"]
                adv = mb["advantages"]

                value_loss = nn.functional.mse_loss(out["value"], mb["returns"])
                entropy = out["entropy"].mean()

                if critic_only:
                    policy_loss = torch.zeros((), device=device)
                    kl_bc = torch.zeros((), device=device)
                    approx_kl = torch.zeros((), device=device)
                    clip_frac = torch.zeros((), device=device)
                    loss = value_loss
                else:
                    log_ratio = (new_lp - old_lp).clamp(-20.0, 20.0)
                    ratio = log_ratio.exp()
                    surr1 = ratio * adv
                    surr2 = ratio.clamp(1.0 - c.clip_eps, 1.0 + c.clip_eps) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    kl_bc = (
                        0.5 * ((new_lp - mb["ref_log_prob"]) ** 2).mean()
                        if c.kl_coef > 0
                        else torch.zeros((), device=device)
                    )
                    approx_kl = 0.5 * (log_ratio ** 2).mean()
                    clip_frac = ((ratio - 1.0).abs() > c.clip_eps).float().mean()
                    loss = policy_loss + c.vf_coef * value_loss - c.ent_coef * entropy + c.kl_coef * kl_bc

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                params_to_clip = [p for p in model.parameters() if p.requires_grad]
                nn.utils.clip_grad_norm_(params_to_clip, c.grad_clip)
                optimizer.step()

                sum_ploss += float(policy_loss.item())
                sum_vloss += float(value_loss.item())
                sum_ent += float(entropy.item())
                sum_kl_bc += float(kl_bc.item())
                kl_val = float(approx_kl.item())
                sum_approx_kl += kl_val
                sum_clip += float(clip_frac.item())
                ep_kl_sum += kl_val
                ep_batches += 1
                n_batches += 1

            if (not critic_only) and c.target_kl > 0.0 and (ep_kl_sum / max(1, ep_batches)) > c.target_kl:
                early_stopped = 1.0
                break
    finally:
        if saved_req_grad:
            for p, req in saved_req_grad:
                p.requires_grad_(req)

    denom = max(1, n_batches)
    return {
        "policy_loss": sum_ploss / denom,
        "value_loss": sum_vloss / denom,
        "entropy": sum_ent / denom,
        "kl_to_bc": sum_kl_bc / denom,
        "approx_kl": sum_approx_kl / denom,
        "clip_frac": sum_clip / denom,
        "epochs_ran": float(epochs_ran),
        "early_stopped": early_stopped,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Masked PPO fine-tuning from a BC checkpoint (solo modes 1-4).")
    parser.add_argument("--init-ckpt", type=str, default="", help="Optional BC checkpoint path")
    parser.add_argument("--ref-ckpt", type=str, default="", help="Optional frozen BC reference ckpt (defaults to --init-ckpt)")
    parser.add_argument("--updates", type=int, default=60, help="Total PPO rollout+update iterations")
    parser.add_argument("--episodes-per-update", type=int, default=32)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5, help="Actor + backbone LR")
    parser.add_argument("--critic-lr", type=float, default=2e-4, help="Critic warm-up LR")
    parser.add_argument("--critic-warmup", type=int, default=3, help="Initial updates training value_head only")
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--target-kl", type=float, default=0.05)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:<idx>")
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=5, help="Run held-out evaluation every N updates (0=end only)")
    parser.add_argument("--eval-seed-start", type=int, default=9000)
    parser.add_argument("--eval-temp", type=float, default=0.0)
    parser.add_argument("--no-break-cycles", action="store_true")
    parser.add_argument("--save-every", type=int, default=10, help="Save latest checkpoint every N updates")
    parser.add_argument("--out", type=str, default="checkpoints/ppo_solo_t1_last.pt")
    parser.add_argument("--best-out", type=str, default="", help="Optional path to save highest-win-rate eval checkpoint")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    print(f"Using device: {describe_device(device)}", flush=True)

    cat = get_catalog()
    if args.init_ckpt:
        model, _ = load_checkpoint(args.init_ckpt, catalog=cat, device=device)
        ref_path = args.ref_ckpt or args.init_ckpt
        ref_model, _ = load_checkpoint(ref_path, catalog=cat, device=device)
        for p in ref_model.parameters():
            p.requires_grad_(False)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded init checkpoint {args.init_ckpt} ({n_params:,} params, cfg={model.cfg})", flush=True)
    else:
        model = StrongholdPolicyNet(catalog=cat, config=PolicyConfig()).to(device)
        ref_model = None

    warmup = args.critic_warmup if args.init_ckpt else 0
    ppo_cfg = PPOConfig(
        lr=args.lr,
        critic_lr=args.critic_lr,
        critic_warmup_updates=warmup,
        ppo_epochs=args.ppo_epochs,
        kl_coef=args.kl_coef if ref_model is not None else 0.0,
        target_kl=args.target_kl,
        minibatch_size=args.minibatch_size,
    )
    critic_opt = torch.optim.AdamW(model.value_head.parameters(), lr=ppo_cfg.critic_lr, weight_decay=1e-4)
    full_opt = torch.optim.AdamW(model.parameters(), lr=ppo_cfg.lr, weight_decay=1e-4)

    eval_seeds = [args.eval_seed_start + i for i in range(args.eval_episodes)]
    best_score = (-1.0, -1.0)  # (win_rate, avg_rounds_passed)
    best_out = args.best_out or str(Path(args.out).with_name(Path(args.out).stem + "_best.pt"))

    def _run_eval(label: str, upd: int) -> None:
        nonlocal best_score
        if args.eval_episodes <= 0:
            return
        ev = evaluate_policy_solo(
            model,
            modes=SOLO_TIER1_MODES,
            seeds=eval_seeds,
            num_workers=args.workers,
            deterministic=(args.eval_temp <= 0.0),
            temperature=args.eval_temp,
            break_cycles=not args.no_break_cycles,
            device=device,
            catalog=cat,
        )
        print(f"[{label}] live eval: {json.dumps(ev, ensure_ascii=False)}", flush=True)
        score = (float(ev["win_rate"]), float(ev["avg_rounds_passed"]))
        if score > best_score:
            best_score = score
            save_checkpoint(model, best_out, extra={"update": upd, "eval": ev, "device": device})
            print(f"[{label}] new best checkpoint -> {best_out} (win={score[0]*100:.1f}%, rounds={score[1]:.2f})", flush=True)

    # Baseline eval before PPO updates
    _run_eval("INIT", 0)

    for upd in range(1, args.updates + 1):
        is_warmup = upd <= warmup
        rollout, r_stats = collect_ppo_rollouts(
            model,
            ref_model=ref_model,
            num_episodes=args.episodes_per_update,
            modes=SOLO_TIER1_MODES,
            seed_start=args.seed + (upd - 1) * args.episodes_per_update,
            num_workers=args.workers,
            ppo_cfg=ppo_cfg,
            device=device,
            catalog=cat,
        )
        u_stats = ppo_update(
            model,
            critic_opt if is_warmup else full_opt,
            rollout,
            cfg=ppo_cfg,
            critic_only=is_warmup,
            device=device,
        )
        tag = f"WARMUP {upd:02d}/{warmup:02d}" if is_warmup else f"PPO {upd:02d}/{args.updates:02d}"
        stop_flag = " [KL-STOP]" if u_stats["early_stopped"] > 0 else ""
        print(
            f"[{tag}] "
            f"win={r_stats['win_rate']*100:.1f}% rounds={r_stats['avg_rounds_passed']:.2f} "
            f"steps={r_stats['transitions']} forced_rdy={r_stats['forced_readies']} ({r_stats['elapsed_sec']:.1f}s) | "
            f"ploss={u_stats['policy_loss']:.4f} vloss={u_stats['value_loss']:.4f} "
            f"ent={u_stats['entropy']:.3f} akl={u_stats['approx_kl']:.4f} "
            f"kl_bc={u_stats['kl_to_bc']:.4f} clip={u_stats['clip_frac']*100:.1f}%{stop_flag}",
            flush=True,
        )

        if args.save_every > 0 and upd % args.save_every == 0:
            save_checkpoint(model, args.out, extra={"update": upd, "device": device})
        if args.eval_every > 0 and upd % args.eval_every == 0:
            _run_eval(f"EVAL@{upd:02d}", upd)

    save_checkpoint(model, args.out, extra={"update": args.updates, "device": device})
    if args.eval_every <= 0 or args.updates % args.eval_every != 0:
        _run_eval("FINAL", args.updates)


if __name__ == "__main__":
    main()
