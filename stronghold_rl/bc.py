"""Solo expert trajectory generator and Behavior Cloning (BC) cold-start trainer."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import dataclass, replace
import gzip
import io
import json
import multiprocessing
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from stronghold_sim import RoomPool, SOLO_TIER1_MODES

from .catalog import GameCatalog, get_catalog
from .codec import (
    NO_ARG,
    TEMP_OFFSET,
    TYPE_CHOICE,
    TYPE_COMMAND_CHOICE,
    TYPE_DESTROY,
    TYPE_READY,
    TYPE_SELL,
    EncodedFrame,
    encode_frame,
)
from .model import PolicyConfig, StrongholdPolicyNet, extract_step_masks
from .rollout import AsyncActorPool


def resolve_device(device_arg: str) -> str:
    """Resolve 'auto', 'cpu', 'cuda', or 'cuda:<idx>' safely."""
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device_arg}, but CUDA is not available.")
        dev = torch.device(device_arg)
        idx = 0 if dev.index is None else dev.index
        if idx >= torch.cuda.device_count():
            raise RuntimeError(f"Requested {device_arg}, but only {torch.cuda.device_count()} CUDA device(s) exist.")
        return device_arg
    if device_arg == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported --device value: {device_arg!r} (expected auto, cpu, cuda, or cuda:<idx>)")


def describe_device(device: str) -> str:
    if device.startswith("cuda"):
        dev = torch.device(device)
        idx = 0 if dev.index is None else dev.index
        return f"{device} ({torch.cuda.get_device_name(idx)})"
    return device


def pick_force_unblock_action(enc: EncodedFrame) -> int:
    """Guaranteed progress action when prep step limit is reached (handles overflow/choice gates)."""
    tree = enc.action_tree
    if tree.type_mask[TYPE_READY]:
        return tree.decode(TYPE_READY)
    for t in (TYPE_COMMAND_CHOICE, TYPE_CHOICE):
        if tree.type_mask[t]:
            src = tree.src_keys[t][0]
            return tree.decode(t, src)
    for t in (TYPE_DESTROY, TYPE_SELL):
        if tree.type_mask[t]:
            srcs = tree.src_keys[t]
            temp_srcs = [s for s in srcs if TEMP_OFFSET <= s < TEMP_OFFSET + 5]
            return tree.decode(t, temp_srcs[0] if temp_srcs else srcs[0])
    return 0


def _prep_state_signature(frame: Mapping[str, Any]) -> tuple[Any, ...]:
    """Compact signature of observable PREP state for detecting deterministic move/freeze cycles."""
    own = frame["own"]
    shop = own.get("shop") or {}
    board = tuple(
        sorted(
            (int(p["uid"]), int(p["row"]), int(p["col"]), p.get("dir", "RIGHT"), len(p.get("items") or []))
            for p in (own.get("board") or [])
            if p
        )
    )
    hand = tuple(int(p["uid"]) if p else None for p in (own.get("hand") or []))
    temp = tuple(int(p["uid"]) if p else None for p in (own.get("temp") or []))
    shop_sold = tuple(bool(s.get("sold")) if s else None for s in (shop.get("slots") or []))
    return (
        int(own.get("funds") or 0),
        int(shop.get("level") or 1),
        bool(shop.get("frozen")),
        shop_sold,
        board,
        hand,
        temp,
    )


@dataclass
class BCDataset:
    """In-memory tensorized dataset of expert transitions for vectorized BC training."""

    cat_ids: torch.Tensor
    cont_feats: torch.Tensor
    valid_mask: torch.Tensor
    a_type: torch.Tensor
    a_src: torch.Tensor
    a_tgt: torch.Tensor
    a_opt: torch.Tensor
    type_mask: torch.Tensor
    src_mask: torch.Tensor
    has_src: torch.Tensor
    band_mask: torch.Tensor
    has_band: torch.Tensor
    tgt_mask: torch.Tensor
    has_tgt: torch.Tensor
    opt_mask: torch.Tensor
    has_opt: torch.Tensor
    returns: torch.Tensor

    def __len__(self) -> int:
        return int(self.a_type.shape[0])

    def slice(self, indices: torch.Tensor, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        return {
            "cat_ids": self.cat_ids[indices].to(device, non_blocking=True),
            "cont_feats": self.cont_feats[indices].to(device, non_blocking=True),
            "valid_mask": self.valid_mask[indices].to(device, non_blocking=True),
            "a_type": self.a_type[indices].to(device, non_blocking=True),
            "a_src": self.a_src[indices].to(device, non_blocking=True),
            "a_tgt": self.a_tgt[indices].to(device, non_blocking=True),
            "a_opt": self.a_opt[indices].to(device, non_blocking=True),
            "type_mask": self.type_mask[indices].to(device, non_blocking=True),
            "src_mask": self.src_mask[indices].to(device, non_blocking=True),
            "has_src": self.has_src[indices].to(device, non_blocking=True),
            "band_mask": self.band_mask[indices].to(device, non_blocking=True),
            "has_band": self.has_band[indices].to(device, non_blocking=True),
            "tgt_mask": self.tgt_mask[indices].to(device, non_blocking=True),
            "has_tgt": self.has_tgt[indices].to(device, non_blocking=True),
            "opt_mask": self.opt_mask[indices].to(device, non_blocking=True),
            "has_opt": self.has_opt[indices].to(device, non_blocking=True),
            "returns": self.returns[indices].to(device, non_blocking=True),
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.__dict__, p)

    @classmethod
    def load(cls, path: str | Path) -> BCDataset:
        raw = torch.load(path, map_location="cpu", weights_only=True)
        return cls(**raw)


def _encode_replay_bytes(payload: tuple[bytes, float]) -> bytes | None:
    """Worker-process helper: decompress & encode one compressed expert replay into serialized bytes.

    Returning plain `bytes` avoids PyTorch multiprocessing `file_descriptor` shared-memory FD passing,
    which otherwise exhausts `ulimit -n` (e.g. 1024) on Linux with 32+ parallel workers.
    """
    replay_bytes, ep_return = payload
    torch.set_num_threads(1)
    cat = get_catalog()
    with gzip.GzipFile(fileobj=io.BytesIO(replay_bytes), mode="rb") as gz:
        rows = [json.loads(line) for line in gz]

    cat_ids_list: list[torch.Tensor] = []
    cont_list: list[torch.Tensor] = []
    valid_list: list[torch.Tensor] = []
    a_type_list: list[int] = []
    a_src_list: list[int] = []
    a_tgt_list: list[int] = []
    a_opt_list: list[int] = []
    type_mask_list: list[torch.Tensor] = []
    src_mask_list: list[torch.Tensor] = []
    has_src_list: list[bool] = []
    band_mask_list: list[torch.Tensor] = []
    has_band_list: list[bool] = []
    tgt_mask_list: list[torch.Tensor] = []
    has_tgt_list: list[bool] = []
    opt_mask_list: list[torch.Tensor] = []
    has_opt_list: list[bool] = []

    last_rnd = -1
    prep_steps = 0
    for row in rows[1:-1]:
        fr = row["frame"]
        rnd = int(fr["pub"].get("round") or 0)
        if rnd != last_rnd:
            last_rnd = rnd
            prep_steps = 0

        action_key = int(row["index"])
        if action_key < 0:
            if fr["phase"] == "PREP":
                prep_steps += 1
            continue

        enc = encode_frame(fr, cat, prep_steps=prep_steps)
        if fr["phase"] == "PREP":
            prep_steps += 1

        a_type, a_src, a_tgt, a_opt = enc.action_tree.key_to_tuple[action_key]
        decoded = enc.action_tree.decode(a_type, a_src, a_tgt, a_opt)
        if decoded != action_key:
            raise RuntimeError(f"Codec round-trip mismatch: {decoded} != {action_key}")
        masks = extract_step_masks(enc, a_type, a_src, a_tgt, a_opt)

        cat_ids_list.append(enc.cat_ids)
        cont_list.append(enc.cont_feats)
        valid_list.append(enc.valid_mask)
        a_type_list.append(a_type)
        a_src_list.append(a_src)
        a_tgt_list.append(a_tgt)
        a_opt_list.append(a_opt)
        type_mask_list.append(masks.type_mask)
        src_mask_list.append(masks.src_mask)
        has_src_list.append(masks.has_src)
        band_mask_list.append(masks.band_mask)
        has_band_list.append(masks.has_band)
        tgt_mask_list.append(masks.tgt_mask)
        has_tgt_list.append(masks.has_tgt)
        opt_mask_list.append(masks.opt_mask)
        has_opt_list.append(masks.has_opt)

    if not a_type_list:
        return None
    n = len(a_type_list)
    tensors = {
        "cat_ids": torch.stack(cat_ids_list),
        "cont_feats": torch.stack(cont_list),
        "valid_mask": torch.stack(valid_list),
        "a_type": torch.tensor(a_type_list, dtype=torch.long),
        "a_src": torch.tensor(a_src_list, dtype=torch.long),
        "a_tgt": torch.tensor(a_tgt_list, dtype=torch.long),
        "a_opt": torch.tensor(a_opt_list, dtype=torch.long),
        "type_mask": torch.stack(type_mask_list),
        "src_mask": torch.stack(src_mask_list),
        "has_src": torch.tensor(has_src_list, dtype=torch.bool),
        "band_mask": torch.stack(band_mask_list),
        "has_band": torch.tensor(has_band_list, dtype=torch.bool),
        "tgt_mask": torch.stack(tgt_mask_list),
        "has_tgt": torch.tensor(has_tgt_list, dtype=torch.bool),
        "opt_mask": torch.stack(opt_mask_list),
        "has_opt": torch.tensor(has_opt_list, dtype=torch.bool),
        "returns": torch.full((n,), ep_return, dtype=torch.float32),
    }
    buf = io.BytesIO()
    torch.save(tensors, buf)
    return buf.getvalue()


def collect_solo_expert_dataset(
    *,
    num_episodes: int = 16,
    modes: Sequence[str] = SOLO_TIER1_MODES,
    seed_start: int = 1000,
    num_workers: int = 4,
    encode_workers: int | None = None,
    only_victories: bool = False,
    verbose: bool = False,
    catalog: GameCatalog | None = None,
) -> tuple[BCDataset, dict[str, Any]]:
    """Generate solo expert replays in parallel via RoomPool and encode them across CPU processes."""
    _ = catalog if catalog is not None else get_catalog()
    t0 = time.perf_counter()

    n_sim = max(1, min(num_workers, num_episodes))
    n_enc = n_sim if encode_workers is None else max(1, encode_workers)
    # Keep tiny unit tests single-process to avoid Windows spawn overhead
    use_mp = n_enc > 1 and num_episodes >= 16

    victories = 0
    kept_episodes = 0
    total_episodes = 0
    chunks: list[dict[str, torch.Tensor]] = []

    enc_pool = (
        ProcessPoolExecutor(max_workers=n_enc, mp_context=multiprocessing.get_context("spawn"))
        if use_mp
        else None
    )
    try:
        with RoomPool(num_workers=n_sim) as pool:
            ep = 0
            while ep < num_episodes:
                batch_n = min(len(pool), num_episodes - ep)
                cmds = {}
                for w in range(batch_n):
                    idx = ep + w
                    mode_id = modes[idx % len(modes)]
                    cmds[w] = {
                        "op": "examples_memory",
                        "modeId": mode_id,
                        "seed": seed_start + idx,
                        "memory_limit": 64 * 1024 * 1024,
                        "turn_order": "serial",
                    }
                replies = pool.exchange(cmds)
                ep += batch_n

                payloads: list[tuple[bytes, float]] = []
                for w in range(batch_n):
                    res = replies[w]
                    total_episodes += 1
                    win = bool(res["victory"])
                    if win:
                        victories += 1
                    if only_victories and not win:
                        continue
                    kept_episodes += 1
                    ep_return = 1.0 if win else (float(res["roundsPassed"]) / max(1, int(res["lastRound"])) - 1.0)
                    payloads.append((res["replay"], ep_return))

                if enc_pool is not None:
                    encoded_blobs = list(enc_pool.map(_encode_replay_bytes, payloads))
                else:
                    encoded_blobs = [_encode_replay_bytes(p) for p in payloads]

                for blob in encoded_blobs:
                    if blob is not None:
                        ch = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
                        chunks.append(ch)

                if verbose:
                    tr_so_far = sum(int(c["a_type"].shape[0]) for c in chunks)
                    print(
                        f"[Collect] {total_episodes}/{num_episodes} eps "
                        f"(win={victories/max(1,total_episodes)*100:.1f}%, "
                        f"transitions={tr_so_far}, elapsed={time.perf_counter()-t0:.1f}s)",
                        flush=True,
                    )
    finally:
        if enc_pool is not None:
            enc_pool.shutdown(wait=True)

    fields = tuple(chunks[0].keys())
    merged = {k: torch.cat([c[k] for c in chunks], dim=0) for k in fields}
    dataset = BCDataset(**merged)
    stats = {
        "episodes_total": total_episodes,
        "episodes_kept": kept_episodes,
        "expert_win_rate": victories / max(1, total_episodes),
        "transitions": len(dataset),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    return dataset, stats


@torch.no_grad()
def _eval_bc_split(
    model: StrongholdPolicyNet,
    dataset: BCDataset,
    indices: torch.Tensor,
    *,
    batch_size: int,
    value_coef: float,
    device: torch.device | str,
) -> dict[str, float]:
    model.eval()
    sum_loss = 0.0
    sum_ploss = 0.0
    sum_vloss = 0.0
    correct_type = 0
    correct_src = 0
    total_src = 0
    correct_tgt = 0
    total_tgt = 0
    correct_opt = 0
    total_opt = 0
    correct_full = 0
    total = 0

    for start in range(0, int(indices.shape[0]), batch_size):
        idx = indices[start : start + batch_size]
        batch = dataset.slice(idx, device=device)
        out = model.evaluate_batch(batch)
        policy_loss = -out["log_prob"].mean()
        value_loss = nn.functional.mse_loss(out["value"], batch["returns"])
        loss = policy_loss + value_coef * value_loss

        bs = int(idx.shape[0])
        sum_loss += float(loss.item()) * bs
        sum_ploss += float(policy_loss.item()) * bs
        sum_vloss += float(value_loss.item()) * bs

        ok_type = out["pred_type"] == batch["a_type"]
        ok_src = (~batch["has_src"]) | (out["pred_src"] == batch["a_src"])
        ok_band = (~batch["has_band"]) | (out["pred_band"] == batch["a_src"])
        ok_tgt = (~batch["has_tgt"]) | (out["pred_tgt"] == batch["a_tgt"])
        ok_opt = (~batch["has_opt"]) | (out["pred_opt"] == batch["a_opt"])

        correct_type += int(ok_type.sum().item())
        if batch["has_src"].any():
            correct_src += int(((out["pred_src"] == batch["a_src"]) & batch["has_src"]).sum().item())
            total_src += int(batch["has_src"].sum().item())
        if batch["has_tgt"].any():
            correct_tgt += int(((out["pred_tgt"] == batch["a_tgt"]) & batch["has_tgt"]).sum().item())
            total_tgt += int(batch["has_tgt"].sum().item())
        if batch["has_opt"].any():
            correct_opt += int(((out["pred_opt"] == batch["a_opt"]) & batch["has_opt"]).sum().item())
            total_opt += int(batch["has_opt"].sum().item())
        correct_full += int((ok_type & ok_src & ok_band & ok_tgt & ok_opt).sum().item())
        total += bs

    return {
        "loss": sum_loss / max(1, total),
        "policy_nll": sum_ploss / max(1, total),
        "value_mse": sum_vloss / max(1, total),
        "type_acc": correct_type / max(1, total),
        "src_acc": correct_src / max(1, total_src),
        "tgt_acc": correct_tgt / max(1, total_tgt),
        "opt_acc": correct_opt / max(1, total_opt),
        "full_acc": correct_full / max(1, total),
    }


def _build_adamw_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Apply AdamW weight decay to 2D+ weight matrices while excluding biases and LayerNorm/1D params."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def train_bc(
    model: StrongholdPolicyNet,
    dataset: BCDataset,
    *,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.05,
    value_coef: float = 0.25,
    grad_clip: float = 1.0,
    val_ratio: float = 0.0,
    restore_best: bool = True,
    cosine_lr: bool = True,
    device: torch.device | str = "cpu",
    seed: int = 0,
    verbose: bool = True,
) -> list[dict[str, float]]:
    """Train policy + value critic via Behavior Cloning with masked label smoothing & decoupled weight decay."""
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    model.to(device)
    opt = torch.optim.AdamW(_build_adamw_param_groups(model, weight_decay), lr=lr)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.05)
        if cosine_lr and epochs > 1
        else None
    )

    n = len(dataset)
    use_val = val_ratio > 0.0 and n >= 20
    if use_val:
        val_n = max(1, int(n * val_ratio))
        train_n = n - val_n
        train_indices = torch.arange(0, train_n, dtype=torch.long)
        val_indices = torch.arange(train_n, n, dtype=torch.long)
    else:
        train_n = n
        train_indices = torch.arange(0, n, dtype=torch.long)
        val_indices = None

    best_val_nll = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        perm = train_indices[torch.randperm(train_n)]
        sum_loss = 0.0
        sum_ploss = 0.0
        sum_vloss = 0.0
        correct_type = 0
        correct_src = 0
        total_src = 0
        correct_tgt = 0
        total_tgt = 0
        correct_opt = 0
        total_opt = 0
        correct_full = 0
        total = 0

        for start in range(0, train_n, batch_size):
            idx = perm[start : start + batch_size]
            batch = dataset.slice(idx, device=device)
            out = model.evaluate_batch(batch, label_smoothing=label_smoothing)

            policy_loss = out["bc_loss"].mean()
            value_loss = nn.functional.mse_loss(out["value"], batch["returns"])
            loss = policy_loss + value_coef * value_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            bs = int(idx.shape[0])
            sum_loss += float(loss.item()) * bs
            sum_ploss += float((-out["log_prob"].mean()).item()) * bs
            sum_vloss += float(value_loss.item()) * bs

            ok_type = out["pred_type"] == batch["a_type"]
            ok_src = (~batch["has_src"]) | (out["pred_src"] == batch["a_src"])
            ok_band = (~batch["has_band"]) | (out["pred_band"] == batch["a_src"])
            ok_tgt = (~batch["has_tgt"]) | (out["pred_tgt"] == batch["a_tgt"])
            ok_opt = (~batch["has_opt"]) | (out["pred_opt"] == batch["a_opt"])

            correct_type += int(ok_type.sum().item())
            if batch["has_src"].any():
                correct_src += int(((out["pred_src"] == batch["a_src"]) & batch["has_src"]).sum().item())
                total_src += int(batch["has_src"].sum().item())
            if batch["has_tgt"].any():
                correct_tgt += int(((out["pred_tgt"] == batch["a_tgt"]) & batch["has_tgt"]).sum().item())
                total_tgt += int(batch["has_tgt"].sum().item())
            if batch["has_opt"].any():
                correct_opt += int(((out["pred_opt"] == batch["a_opt"]) & batch["has_opt"]).sum().item())
                total_opt += int(batch["has_opt"].sum().item())
            correct_full += int((ok_type & ok_src & ok_band & ok_tgt & ok_opt).sum().item())
            total += bs

        cur_lr = opt.param_groups[0]["lr"]
        if sched is not None:
            sched.step()

        metrics: dict[str, float] = {
            "epoch": float(epoch),
            "lr": float(cur_lr),
            "loss": sum_loss / max(1, total),
            "policy_nll": sum_ploss / max(1, total),
            "value_mse": sum_vloss / max(1, total),
            "type_acc": correct_type / max(1, total),
            "src_acc": correct_src / max(1, total_src),
            "tgt_acc": correct_tgt / max(1, total_tgt),
            "opt_acc": correct_opt / max(1, total_opt),
            "full_acc": correct_full / max(1, total),
        }

        val_str = ""
        if val_indices is not None:
            vm = _eval_bc_split(
                model,
                dataset,
                val_indices,
                batch_size=batch_size,
                value_coef=value_coef,
                device=device,
            )
            metrics["val_nll"] = vm["policy_nll"]
            metrics["val_type_acc"] = vm["type_acc"]
            metrics["val_full_acc"] = vm["full_acc"]
            marker = ""
            if vm["policy_nll"] < best_val_nll:
                best_val_nll = vm["policy_nll"]
                marker = " *"
                if restore_best:
                    best_state = copy.deepcopy(model.state_dict())
            val_str = (
                f" | val_nll={vm['policy_nll']:.4f} "
                f"val_type={vm['type_acc']*100:.1f}% val_full={vm['full_acc']*100:.1f}%{marker}"
            )

        history.append(metrics)
        if verbose:
            print(
                f"[BC epoch {epoch:02d}/{epochs:02d}] ({time.perf_counter()-t_ep:.1f}s, lr={cur_lr:.2e}) "
                f"nll={metrics['policy_nll']:.4f} v_mse={metrics['value_mse']:.4f} "
                f"type={metrics['type_acc']*100:.1f}% src={metrics['src_acc']*100:.1f}% "
                f"tgt={metrics['tgt_acc']*100:.1f}% opt={metrics['opt_acc']*100:.1f}% "
                f"full={metrics['full_acc']*100:.1f}%{val_str}",
                flush=True,
            )

    if best_state is not None and restore_best:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def evaluate_policy_solo(
    model: StrongholdPolicyNet,
    *,
    modes: Sequence[str] = SOLO_TIER1_MODES,
    seeds: Sequence[int] = (9000, 9001, 9002, 9003),
    num_workers: int = 8,
    deterministic: bool = True,
    temperature: float = 0.0,
    break_cycles: bool = True,
    max_prep_steps_per_round: int = 80,
    device: torch.device | str = "cpu",
    catalog: GameCatalog | None = None,
    pool: AsyncActorPool | None = None,
    use_async_pool: bool = True,
    min_batch: int = 0,
    batch_timeout_ms: float = 2.0,
) -> dict[str, Any]:
    """Run live solo evaluation in parallel via AsyncActorPool (or RoomPool fallback) + batched GPU inference."""
    cat = catalog if catalog is not None else get_catalog()
    model.eval()
    t0 = time.perf_counter()
    use_det = deterministic and temperature <= 0.0

    total_eps = len(seeds)
    if total_eps == 0:
        return {"episodes": 0, "win_rate": 0.0, "avg_rounds_passed": 0.0, "avg_steps": 0.0, "forced_readies": 0, "cycle_breaks": 0}

    n_workers = max(1, min(num_workers, total_eps))
    if pool is not None:
        return pool.evaluate_episodes(
            model,
            modes=modes,
            seeds=seeds,
            deterministic=deterministic,
            temperature=temperature,
            break_cycles=break_cycles,
            max_prep_steps_per_round=max_prep_steps_per_round,
            min_batch=min_batch if min_batch > 0 else None,
            batch_timeout_ms=batch_timeout_ms,
            device=device,
        )
    if use_async_pool and n_workers >= 2:
        with AsyncActorPool(
            num_workers=n_workers,
            min_batch=min_batch,
            batch_timeout_ms=batch_timeout_ms,
        ) as async_pool:
            return async_pool.evaluate_episodes(
                model,
                modes=modes,
                seeds=seeds,
                deterministic=deterministic,
                temperature=temperature,
                break_cycles=break_cycles,
                max_prep_steps_per_round=max_prep_steps_per_round,
                device=device,
            )

    wins = 0
    rounds_list: list[int] = []
    steps_list: list[int] = []
    forced_readies = 0
    cycle_breaks = 0

    with RoomPool(num_workers=n_workers) as sync_pool:
        pool_ref = sync_pool
        next_ep = 0
        ep_steps: dict[int, int] = {}
        ep_prep_steps: dict[int, int] = {}
        ep_last_round: dict[int, int] = {}
        ep_seen_sigs: dict[int, dict[tuple[Any, ...], int]] = {}

        def _spawn_cmd(idx: int) -> dict[str, Any]:
            return {
                "op": "reset",
                "seed": int(seeds[idx]),
                "modeId": modes[idx % len(modes)],
                "policy_version": 0,
            }

        init_cmds = {}
        for w in range(n_workers):
            init_cmds[w] = _spawn_cmd(next_ep)
            ep_steps[w] = 0
            ep_prep_steps[w] = 0
            ep_last_round[w] = -1
            ep_seen_sigs[w] = {}
            next_ep += 1
        active_frames = pool_ref.exchange(init_cmds)

        while active_frames:
            step_cmds: dict[int, dict[str, Any]] = {}
            det_workers: list[int] = []
            det_frames: list[dict[str, Any]] = []
            det_encs: list[EncodedFrame] = []
            stoch_items: list[tuple[int, dict[str, Any], EncodedFrame, float]] = []

            for w, frame in active_frames.items():
                rnd = int(frame["pub"].get("round") or 0)
                if rnd != ep_last_round[w]:
                    ep_last_round[w] = rnd
                    ep_prep_steps[w] = 0
                    ep_seen_sigs[w].clear()

                enc = encode_frame(frame, cat, prep_steps=ep_prep_steps[w])
                if frame["phase"] == "PREP" and ep_prep_steps[w] >= max_prep_steps_per_round:
                    forced_readies += 1
                    action_key = pick_force_unblock_action(enc)
                    step_cmds[w] = {
                        "op": "step",
                        "decision_id": frame["decision_id"],
                        "action_key": action_key,
                        "policy_version": 0,
                    }
                    if frame["phase"] == "PREP":
                        ep_prep_steps[w] += 1
                    ep_steps[w] += 1
                    continue

                if break_cycles and frame["phase"] == "PREP":
                    sig = _prep_state_signature(frame)
                    visits = ep_seen_sigs[w].get(sig, 0)
                    ep_seen_sigs[w][sig] = visits + 1
                    if visits >= 2 and bool(enc.action_tree.type_mask[TYPE_READY]):
                        cycle_breaks += 1
                        action_key = enc.action_tree.decode(TYPE_READY)
                        step_cmds[w] = {
                            "op": "step",
                            "decision_id": frame["decision_id"],
                            "action_key": action_key,
                            "policy_version": 0,
                        }
                        ep_prep_steps[w] += 1
                        ep_steps[w] += 1
                        continue
                    if visits >= 1:
                        cycle_breaks += 1
                        escape_temp = max(temperature, 0.6) if temperature > 0.0 else 0.6
                        stoch_items.append((w, frame, enc, escape_temp))
                        continue

                if use_det:
                    det_workers.append(w)
                    det_frames.append(frame)
                    det_encs.append(enc)
                else:
                    stoch_items.append((w, frame, enc, temperature if temperature > 0.0 else 1.0))

            if det_encs:
                acts = model.act_batch(det_encs, deterministic=True, device=device)
                for w, frame, act in zip(det_workers, det_frames, acts):
                    if frame["phase"] == "PREP":
                        ep_prep_steps[w] += 1
                    ep_steps[w] += 1
                    step_cmds[w] = {
                        "op": "step",
                        "decision_id": frame["decision_id"],
                        "action_key": act["action_key"],
                        "policy_version": 0,
                    }

            if stoch_items:
                # Group stochastic items by temperature (usually 1 temperature)
                by_temp: dict[float, list[tuple[int, dict[str, Any], EncodedFrame]]] = {}
                for w, frame, enc, temp in stoch_items:
                    by_temp.setdefault(temp, []).append((w, frame, enc))
                for temp, group in by_temp.items():
                    acts = model.act_batch(
                        [item[2] for item in group],
                        deterministic=False,
                        temperature=temp,
                        device=device,
                    )
                    for (w, frame, _), act in zip(group, acts):
                        if frame["phase"] == "PREP":
                            ep_prep_steps[w] += 1
                        ep_steps[w] += 1
                        step_cmds[w] = {
                            "op": "step",
                            "decision_id": frame["decision_id"],
                            "action_key": act["action_key"],
                            "policy_version": 0,
                        }

            replies = pool_ref.exchange(step_cmds)
            reset_cmds: dict[int, dict[str, Any]] = {}
            next_active: dict[int, dict[str, Any]] = {}

            for w, rep in replies.items():
                if rep.get("terminal") or ep_steps[w] > 3000:
                    if rep.get("victory"):
                        wins += 1
                    rounds_list.append(int(rep.get("roundsPassed", 0)))
                    steps_list.append(ep_steps[w])
                    ep_steps[w] = 0
                    ep_prep_steps[w] = 0
                    ep_last_round[w] = -1
                    ep_seen_sigs[w].clear()
                    if next_ep < total_eps:
                        reset_cmds[w] = _spawn_cmd(next_ep)
                        next_ep += 1
                else:
                    next_active[w] = rep

            if reset_cmds:
                next_active.update(pool_ref.exchange(reset_cmds))
            active_frames = next_active

    return {
        "episodes": total_eps,
        "win_rate": wins / max(1, total_eps),
        "avg_rounds_passed": sum(rounds_list) / max(1, len(rounds_list)),
        "avg_steps": sum(steps_list) / max(1, len(steps_list)),
        "forced_readies": forced_readies,
        "cycle_breaks": cycle_breaks,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def save_checkpoint(model: StrongholdPolicyNet, path: str | Path, extra: dict[str, Any] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": model.cfg.to_dict(),
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, p)


def load_checkpoint(
    path: str | Path,
    *,
    catalog: GameCatalog | None = None,
    device: torch.device | str = "cpu",
    dropout_override: float | None = None,
) -> tuple[StrongholdPolicyNet, dict[str, Any]]:
    raw = torch.load(path, map_location=device, weights_only=True)
    cfg = PolicyConfig.from_dict(raw["config"])
    if dropout_override is not None:
        cfg = replace(cfg, dropout=float(dropout_override))
    model = StrongholdPolicyNet(catalog=catalog, config=cfg).to(device)
    model.load_state_dict(raw["state_dict"])
    return model, raw.get("extra", {})


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate solo expert data and run BC cold-start training.")
    parser.add_argument("--episodes", type=int, default=160, help="Number of solo expert matches to generate")
    parser.add_argument("--workers", type=int, default=8, help="Parallel JS simulator workers")
    parser.add_argument("--encode-workers", type=int, default=0, help="Parallel CPU encoders (0 = match --workers)")
    parser.add_argument("--epochs", type=int, default=25, help="BC training epochs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Contiguous tail validation split ratio")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dim-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:<idx>")
    parser.add_argument("--out", type=str, default="checkpoints/bc_solo_t1.pt")
    parser.add_argument("--dataset-cache", type=str, default="", help="Optional .pt path to save/load dataset")
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=0, help="Parallel eval workers (0 = match --workers)")
    parser.add_argument("--eval-seed-start", type=int, default=9000)
    parser.add_argument("--eval-temp", type=float, default=0.0, help="0=greedy+cycle-breaker, >0=low-T sampling")
    parser.add_argument("--no-break-cycles", action="store_true")
    parser.add_argument("--eval-only", type=str, default="", help="Skip training and evaluate this checkpoint path")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    print(f"Using device: {describe_device(device)}", flush=True)
    cat = get_catalog()
    eval_workers = args.eval_workers if args.eval_workers > 0 else args.workers

    if args.eval_only:
        model, _ = load_checkpoint(args.eval_only, catalog=cat, device=device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded {args.eval_only} ({n_params:,} params, cfg={model.cfg})", flush=True)
        ev = evaluate_policy_solo(
            model,
            modes=SOLO_TIER1_MODES,
            seeds=[args.eval_seed_start + i for i in range(args.eval_episodes)],
            num_workers=eval_workers,
            deterministic=(args.eval_temp <= 0.0),
            temperature=args.eval_temp,
            break_cycles=not args.no_break_cycles,
            device=device,
            catalog=cat,
        )
        print("Live solo evaluation:", json.dumps(ev, ensure_ascii=False), flush=True)
        return

    cache_path = Path(args.dataset_cache) if args.dataset_cache else None
    if cache_path and cache_path.is_file():
        dataset = BCDataset.load(cache_path)
        print(f"Loaded cached dataset: {len(dataset)} transitions from {cache_path}", flush=True)
    else:
        enc_w = args.encode_workers if args.encode_workers > 0 else args.workers
        dataset, stats = collect_solo_expert_dataset(
            num_episodes=args.episodes,
            modes=SOLO_TIER1_MODES,
            seed_start=args.seed,
            num_workers=args.workers,
            encode_workers=enc_w,
            verbose=True,
            catalog=cat,
        )
        print("Expert dataset collected:", json.dumps(stats, ensure_ascii=False), flush=True)
        if cache_path:
            dataset.save(cache_path)
            print(f"Saved dataset cache to {cache_path}", flush=True)

    cfg = PolicyConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
    )
    model = StrongholdPolicyNet(catalog=cat, config=cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model config: {cfg} -> {n_params:,} params ({n_params/1e6:.2f}M)", flush=True)

    t_train = time.perf_counter()
    history = train_bc(
        model,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        val_ratio=args.val_ratio,
        restore_best=args.val_ratio > 0.0,
        device=device,
    )
    print(f"BC training finished in {time.perf_counter() - t_train:.2f}s on {device}", flush=True)
    save_checkpoint(
        model,
        args.out,
        extra={"bc_last": history[-1] if history else {}, "device": device, "n_params": n_params},
    )
    print(f"Saved BC checkpoint to {args.out}", flush=True)

    if args.eval_episodes > 0:
        ev = evaluate_policy_solo(
            model,
            modes=SOLO_TIER1_MODES,
            seeds=[args.eval_seed_start + i for i in range(args.eval_episodes)],
            num_workers=eval_workers,
            deterministic=(args.eval_temp <= 0.0),
            temperature=args.eval_temp,
            break_cycles=not args.no_break_cycles,
            device=device,
            catalog=cat,
        )
        print("Live solo evaluation:", json.dumps(ev, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
