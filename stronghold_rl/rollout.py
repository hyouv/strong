"""Multiprocess Actor Worker Pool + Asynchronous Dynamic-Batch Rollout Engine.

Implements TODO.md P0 optimizations:
1. P0-1 (Eliminate Python GIL bottleneck):
   Each of `N` spawned worker processes owns 1 Node.js `RoomClient` and performs
   `json.loads`, `encode_frame`, `ActionTree` construction, force-unblock, and
   eval cycle detection in parallel across CPU cores. Encoded observation tensors
   are written into ONE pre-allocated `SharedMemory` segment (1 FD total across
   the entire pool; zero PyTorch Tensor FD passing over pipes), while only compact
   `ActionMaskSpec` metadata crosses the pipe.
2. P0-2 (Eliminate N-way lockstep combat stragglers):
   The main process uses `multiprocessing.connection.wait` with a configurable
   dynamic batching window (`min_batch`, `batch_timeout_ms`). Workers entering
   slow `COMBAT` rounds no longer block other workers running lightweight `PREP`
   steps; ready frames are batched and dispatched to GPU `act_batch_tensors`
   immediately and returned to their workers without barrier stalls.
"""
from __future__ import annotations

from dataclasses import dataclass
import multiprocessing
from multiprocessing.connection import Connection, wait as mp_wait
from multiprocessing.shared_memory import SharedMemory
import os
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from stronghold_sim import RoomClient, WorkerError

from .catalog import get_catalog
from .codec import (
    CONT_DIM,
    NUM_CAT_FIELDS,
    NUM_TOKENS,
    TYPE_BAND,
    TYPE_BAND_SKIP,
    TYPE_BOND_GROUP,
    TYPE_CHOICE,
    TYPE_COMMAND_CHOICE,
    TYPE_READY,
    ActionMaskSpec,
    FastActionDecoder,
    encode_frame_into,
)
from .model import StrongholdPolicyNet

STRATEGIC_ACTION_TYPES: frozenset[int] = frozenset(
    {
        TYPE_BOND_GROUP,
        TYPE_BAND,
        TYPE_BAND_SKIP,
        TYPE_CHOICE,
        TYPE_COMMAND_CHOICE,
    }
)


def _align64(n: int) -> int:
    return ((n + 63) // 64) * 64


def _shm_layout(num_workers: int) -> tuple[int, int, int, int]:
    """Compute byte offsets and total size for contiguous `(cat, cont, valid)` shared arrays."""
    cat_bytes = num_workers * NUM_TOKENS * NUM_CAT_FIELDS * 4   # int32
    cont_bytes = num_workers * NUM_TOKENS * CONT_DIM * 4        # float32
    valid_bytes = num_workers * NUM_TOKENS                      # bool (uint8)
    off_cat = 0
    off_cont = _align64(off_cat + cat_bytes)
    off_valid = _align64(off_cont + cont_bytes)
    total = _align64(off_valid + valid_bytes)
    return off_cat, off_cont, off_valid, total


def _make_shm_arrays(
    buf: memoryview,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    off_cat, off_cont, off_valid, _ = _shm_layout(num_workers)
    cat_arr = np.ndarray(
        (num_workers, NUM_TOKENS, NUM_CAT_FIELDS),
        dtype=np.int32,
        buffer=buf,
        offset=off_cat,
    )
    cont_arr = np.ndarray(
        (num_workers, NUM_TOKENS, CONT_DIM),
        dtype=np.float32,
        buffer=buf,
        offset=off_cont,
    )
    valid_arr = np.ndarray(
        (num_workers, NUM_TOKENS),
        dtype=np.bool_,
        buffer=buf,
        offset=off_valid,
    )
    return cat_arr, cont_arr, valid_arr


def _prep_state_signature_local(frame: Mapping[str, Any]) -> tuple[Any, ...]:
    own = frame["own"]
    shop = own.get("shop") or {}
    board = tuple(
        sorted(
            (int(p["uid"]), int(p["row"]), int(p["col"]), p.get("dir", "RIGHT"), len(p.get("items") or ()))
            for p in (own.get("board") or ())
            if p
        )
    )
    hand = tuple(int(p["uid"]) if p else None for p in (own.get("hand") or ()))
    temp = tuple(int(p["uid"]) if p else None for p in (own.get("temp") or ()))
    shop_sold = tuple(bool(s.get("sold")) if s else None for s in (shop.get("slots") or ()))
    return (
        int(own.get("funds") or 0),
        int(shop.get("level") or 1),
        bool(shop.get("frozen")),
        shop_sold,
        board,
        hand,
        temp,
    )


def _actor_worker_main(
    slot: int,
    num_workers: int,
    shm_name: str,
    conn: Connection,
) -> None:
    """Isolated actor worker process: owns 1 Node.js RoomClient (compact mode) + zero-tensor NumPy encode."""
    # Ensure worker processes never touch CUDA or oversubscribe CPU threads.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)

    shm: SharedMemory | None = None
    client: RoomClient | None = None
    try:
        shm = SharedMemory(name=shm_name, create=False)
        cat_arr, cont_arr, valid_arr = _make_shm_arrays(shm.buf, num_workers)
        cat_slot = cat_arr[slot]
        cont_slot = cont_arr[slot]
        valid_slot = valid_arr[slot]

        catalog = get_catalog()
        client = RoomClient()
        conn.send(("ready", slot, os.getpid()))

        ep_id = -1
        ep_steps = 0
        ep_prep_steps = 0
        ep_prev_round = -1
        ep_prev_lp = -1
        ep_seen_sigs: dict[tuple[Any, ...], int] = {}
        cur_dec: FastActionDecoder | None = None
        cur_decision_id = -1
        cur_phase = ""
        has_transition = False

        # Per-episode mode config
        max_prep_steps = 80
        max_total_steps = 0
        break_cycles = False
        base_det = False
        base_temp = 1.0

        def _advance_frame(frame: dict[str, Any]) -> None:
            nonlocal client, ep_steps, ep_prep_steps, ep_prev_round, ep_prev_lp
            nonlocal cur_dec, cur_decision_id, cur_phase, has_transition

            pending_r_delta = 0
            pending_lp_lost = 0
            forced_delta = 0
            cycle_delta = 0

            while True:
                if frame.get("terminal") or (max_total_steps > 0 and ep_steps >= max_total_steps):
                    cur_dec = None
                    conn.send(
                        (
                            "done",
                            slot,
                            ep_id,
                            bool(frame.get("victory")),
                            int(frame.get("roundsPassed", 0)),
                            int(frame.get("lastRound", 14)),
                            ep_steps,
                            forced_delta,
                            cycle_delta,
                            pending_r_delta,
                            pending_lp_lost,
                            not bool(frame.get("terminal")),
                        )
                    )
                    return

                rnd = int(frame["pub"].get("round") or 0)
                lp = int(frame["own"].get("lp") or 0)

                if has_transition and ep_prev_round >= 0 and rnd > ep_prev_round:
                    pending_r_delta += rnd - ep_prev_round
                    pending_lp_lost += max(0, ep_prev_lp - lp)
                    ep_prep_steps = 0
                    ep_seen_sigs.clear()
                elif rnd != ep_prev_round:
                    ep_prep_steps = 0
                    ep_seen_sigs.clear()

                ep_prev_round = rnd
                ep_prev_lp = lp

                phase = frame["phase"]
                # Encode directly into SharedMemory NumPy buffers (zero PyTorch Tensor allocation)
                dec = encode_frame_into(frame, catalog, cat_slot, cont_slot, valid_slot, prep_steps=ep_prep_steps)

                # Local fast-path 1: prep step limit guardrail (no GPU round-trip needed)
                if phase == "PREP" and ep_prep_steps >= max_prep_steps:
                    forced_delta += 1
                    action_key = dec.pick_force_unblock_action()
                    ep_prep_steps += 1
                    ep_steps += 1
                    assert client is not None
                    frame = client.step(frame["decision_id"], action_key, policy_version=0)
                    continue

                req_det = base_det
                req_temp = base_temp

                # Local fast-path 2: deterministic eval cycle-breaker
                if break_cycles and phase == "PREP":
                    sig = _prep_state_signature_local(frame)
                    visits = ep_seen_sigs.get(sig, 0)
                    ep_seen_sigs[sig] = visits + 1
                    if visits >= 2 and dec.mask_spec.type_mask_bytes[TYPE_READY]:
                        cycle_delta += 1
                        action_key = dec.decode(TYPE_READY)
                        ep_prep_steps += 1
                        ep_steps += 1
                        assert client is not None
                        frame = client.step(frame["decision_id"], action_key, policy_version=0)
                        continue
                    if visits >= 1:
                        cycle_delta += 1
                        req_det = False
                        req_temp = max(base_temp, 0.6) if base_temp > 0.0 else 0.6

                cur_dec = dec
                cur_decision_id = int(frame["decision_id"])
                cur_phase = phase

                conn.send(
                    (
                        "obs",
                        slot,
                        ep_id,
                        phase == "PREP",
                        req_det,
                        req_temp,
                        pending_r_delta,
                        pending_lp_lost,
                        forced_delta,
                        cycle_delta,
                        dec.mask_spec,
                    )
                )
                return

        while True:
            cmd = conn.recv()
            op = cmd[0]
            if op == "close":
                break
            if op == "reset":
                _, ep_id, mode_id, seed, cfg = cmd
                max_prep_steps = int(cfg["max_prep_steps"])
                max_total_steps = int(cfg["max_total_steps"])
                break_cycles = bool(cfg["break_cycles"])
                base_det = bool(cfg["deterministic"])
                base_temp = float(cfg["temperature"])

                ep_steps = 0
                ep_prep_steps = 0
                ep_prev_round = -1
                ep_prev_lp = -1
                ep_seen_sigs.clear()
                cur_dec = None
                cur_decision_id = -1
                cur_phase = ""
                has_transition = False
                try:
                    assert client is not None
                    frame = client.reset(seed=int(seed), mode_id=str(mode_id), policy_version=0, compact=True)
                    _advance_frame(frame)
                except WorkerError as exc:
                    cur_dec = None
                    if client is not None:
                        client.close()
                    client = RoomClient()
                    conn.send(("error", slot, ep_id, str(exc)[:400]))
            elif op == "act":
                _, req_ep_id, a_type, a_src, a_tgt, a_opt = cmd
                if req_ep_id != ep_id or cur_dec is None:
                    raise RuntimeError(f"Actor slot {slot} stale act: ep {req_ep_id} != {ep_id}")
                try:
                    action_key = cur_dec.decode(a_type, a_src, a_tgt, a_opt)
                    has_transition = True
                    if cur_phase == "PREP":
                        ep_prep_steps += 1
                    ep_steps += 1
                    assert client is not None
                    frame = client.step(cur_decision_id, action_key, policy_version=0)
                    _advance_frame(frame)
                except WorkerError as exc:
                    cur_dec = None
                    if client is not None:
                        client.close()
                    client = RoomClient()
                    conn.send(("error", slot, ep_id, str(exc)[:400]))
            else:
                raise ValueError(f"Unknown actor worker op: {op!r}")
    except EOFError:
        pass
    except BaseException as exc:
        try:
            conn.send(("fatal", slot, repr(exc)))
        except Exception:
            pass
    finally:
        if client is not None:
            client.close()
        if shm is not None:
            shm.close()
        conn.close()


@dataclass
class _PendingObs:
    slot: int
    ep_id: int
    is_prep: bool
    deterministic: bool
    temperature: float
    r_delta: int
    lp_lost: int
    forced_delta: int
    cycle_delta: int
    mask_spec: ActionMaskSpec


class AsyncActorPool:
    """Reusable pool of spawned actor processes with shared-memory tensors & async dynamic batching."""

    def __init__(
        self,
        num_workers: int,
        *,
        min_batch: int = 0,
        batch_timeout_ms: float = 2.0,
        startup_timeout_sec: float = 60.0,
    ) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self.num_workers = int(num_workers)
        self.default_min_batch = int(min_batch) if min_batch > 0 else max(1, min(16, max(1, num_workers // 2)))
        self.default_batch_timeout_ms = float(batch_timeout_ms)
        self._closed = False

        _, _, _, total_bytes = _shm_layout(self.num_workers)
        self._shm = SharedMemory(create=True, size=total_bytes)
        self._cat_shm, self._cont_shm, self._valid_shm = _make_shm_arrays(self._shm.buf, self.num_workers)

        mp_ctx = multiprocessing.get_context("spawn")
        self._conns: list[Connection] = []
        self._procs: list[multiprocessing.process.BaseProcess] = []
        self._conn_to_slot: dict[Connection, int] = {}

        try:
            for slot in range(self.num_workers):
                parent_conn, child_conn = mp_ctx.Pipe(duplex=True)
                proc = mp_ctx.Process(
                    target=_actor_worker_main,
                    args=(slot, self.num_workers, self._shm.name, child_conn),
                    name=f"stronghold-actor-{slot}",
                    daemon=True,
                )
                proc.start()
                child_conn.close()
                self._conns.append(parent_conn)
                self._procs.append(proc)
                self._conn_to_slot[parent_conn] = slot

            # Wait for all workers to report ready
            pending = set(range(self.num_workers))
            deadline = time.perf_counter() + startup_timeout_sec
            while pending:
                rem = deadline - time.perf_counter()
                if rem <= 0:
                    raise TimeoutError(f"Timed out waiting for actor worker slots: {sorted(pending)}")
                ready = mp_wait([self._conns[w] for w in pending], timeout=rem)
                if not ready:
                    continue
                for conn in ready:
                    msg = conn.recv()
                    if msg[0] == "ready":
                        pending.remove(msg[1])
                    elif msg[0] == "fatal":
                        raise RuntimeError(f"Actor worker {msg[1]} failed during startup: {msg[2]}")
                    else:
                        raise RuntimeError(f"Unexpected startup message from actor worker: {msg!r}")
        except BaseException:
            self.close()
            raise

    def __len__(self) -> int:
        return self.num_workers

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for conn in self._conns:
            try:
                conn.send(("close",))
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)
        for conn in self._conns:
            try:
                conn.close()
            except Exception:
                pass
        self._conns.clear()
        self._procs.clear()
        self._conn_to_slot.clear()
        # Release numpy views before closing SharedMemory
        self._cat_shm = None  # type: ignore[assignment]
        self._cont_shm = None  # type: ignore[assignment]
        self._valid_shm = None  # type: ignore[assignment]
        try:
            self._shm.close()
        finally:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> AsyncActorPool:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _poll_messages(
        self,
        inflight: set[int],
        *,
        min_batch: int,
        batch_timeout_ms: float,
    ) -> tuple[list[_PendingObs], list[tuple[Any, ...]]]:
        """Wait for ready workers and coalesce `obs` messages within a short dynamic-batch window."""
        obs_list: list[_PendingObs] = []
        events: list[tuple[Any, ...]] = []

        def _consume(conns_ready: list[Any]) -> None:
            for c in conns_ready:
                slot = self._conn_to_slot[c]
                inflight.discard(slot)
                msg = c.recv()
                tag = msg[0]
                if tag == "obs":
                    obs_list.append(
                        _PendingObs(
                            slot=msg[1],
                            ep_id=msg[2],
                            is_prep=msg[3],
                            deterministic=msg[4],
                            temperature=msg[5],
                            r_delta=msg[6],
                            lp_lost=msg[7],
                            forced_delta=msg[8],
                            cycle_delta=msg[9],
                            mask_spec=msg[10],
                        )
                    )
                elif tag == "fatal":
                    raise RuntimeError(f"Actor worker {msg[1]} crashed fatally: {msg[2]}")
                else:
                    events.append(msg)

        # Block until >= 1 inflight worker finishes its current JS step + encoding
        first_ready = mp_wait([self._conns[w] for w in inflight])
        _consume(first_ready)

        # If we already have obs and more workers are still in flight, wait briefly to fill a GPU batch
        if obs_list and inflight and batch_timeout_ms > 0.0:
            target = max(1, min(min_batch, len(obs_list) + len(inflight)))
            deadline = time.perf_counter() + batch_timeout_ms / 1000.0
            while len(obs_list) < target and inflight:
                rem = deadline - time.perf_counter()
                if rem <= 0.0:
                    break
                more = mp_wait([self._conns[w] for w in inflight], timeout=rem)
                if not more:
                    break
                _consume(more)

        return obs_list, events

    @torch.no_grad()
    def collect_ppo_episodes(
        self,
        model: StrongholdPolicyNet,
        *,
        num_episodes: int,
        modes: Sequence[str],
        seed_start: int,
        step_penalty: float,
        round_survive_bonus: float,
        lp_loss_penalty: float,
        victory_reward: float,
        defeat_reward: float,
        hidden_clear_bonus: float = 0.0,
        episode_modes: Sequence[str] | None = None,
        max_prep_steps_per_round: int = 80,
        sampling_temperature: float = 1.0,
        min_batch: int | None = None,
        batch_timeout_ms: float | None = None,
        device: torch.device | str = "cpu",
    ) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
        """Collect `num_episodes` complete PPO episodes asynchronously with dynamic GPU batching."""
        if self._closed:
            raise RuntimeError("AsyncActorPool is closed")
        model.eval()
        t0 = time.perf_counter()
        eff_min_batch = self.default_min_batch if (min_batch is None or min_batch <= 0) else int(min_batch)
        eff_timeout_ms = self.default_batch_timeout_ms if batch_timeout_ms is None else float(batch_timeout_ms)

        ep_cfg = {
            "max_prep_steps": max_prep_steps_per_round,
            "max_total_steps": 0,
            "break_cycles": False,
            "deterministic": False,
            "temperature": float(sampling_temperature),
        }

        all_episodes: list[list[dict[str, Any]]] = []
        ep_buf: dict[int, list[dict[str, Any]]] = {}
        wins = 0
        hidden_clears = 0
        rounds_passed: list[int] = []
        forced_readies = 0
        sim_errors = 0
        num_gpu_batches = 0
        total_gpu_items = 0

        next_ep_id = 0
        retry_offset = 0
        completed_target = int(num_episodes)
        in_progress_target = 0
        inflight: set[int] = set()
        mode_schedule = episode_modes if episode_modes else modes

        def _assign_episode(slot: int, *, is_retry: bool = False) -> None:
            nonlocal next_ep_id, retry_offset, in_progress_target
            if len(all_episodes) + in_progress_target >= completed_target:
                return
            ep_id = next_ep_id
            next_ep_id += 1
            if is_retry:
                seed_idx = completed_target + retry_offset
                retry_offset += 1
            else:
                seed_idx = ep_id
            mode_id = mode_schedule[seed_idx % len(mode_schedule)]
            seed = seed_start + seed_idx
            ep_buf[slot] = []
            in_progress_target += 1
            inflight.add(slot)
            self._conns[slot].send(("reset", ep_id, mode_id, seed, ep_cfg))

        for w in range(min(self.num_workers, completed_target)):
            _assign_episode(w)

        while inflight:
            obs_list, events = self._poll_messages(
                inflight,
                min_batch=eff_min_batch,
                batch_timeout_ms=eff_timeout_ms,
            )

            # Process terminal / error events first and immediately launch replacement episodes
            for ev in events:
                tag = ev[0]
                if tag == "done":
                    _, slot, _ep_id, win, rnds, last_rnd, _steps, f_delta, _c_delta, r_delta, lp_lost, _to = ev
                    forced_readies += f_delta
                    in_progress_target -= 1
                    hid_cleared = bool(win and rnds > last_rnd)
                    if win:
                        wins += 1
                    if hid_cleared:
                        hidden_clears += 1
                    rounds_passed.append(rnds)
                    buf = ep_buf.pop(slot, [])
                    if buf:
                        if r_delta > 0:
                            buf[-1]["reward"] += round_survive_bonus * r_delta + lp_loss_penalty * lp_lost
                            buf[-1]["round_end"] = True
                        buf[-1]["reward"] += victory_reward if win else defeat_reward
                        if hid_cleared and hidden_clear_bonus != 0.0:
                            buf[-1]["reward"] += hidden_clear_bonus
                        buf[-1]["done"] = True
                        all_episodes.append(buf)
                    _assign_episode(slot)
                elif tag == "error":
                    _, slot, err_ep_id, err_msg = ev
                    sim_errors += 1
                    in_progress_target -= 1
                    ep_buf.pop(slot, None)
                    print(
                        f"[AsyncActorPool] WARN: simulator error in rollout ep={err_ep_id} "
                        f"(slot={slot}), retrying with replacement seed: {err_msg[:200]}",
                        flush=True,
                    )
                    if sim_errors > max(8, completed_target):
                        raise RuntimeError(f"Too many simulator errors ({sim_errors}) during PPO rollout; last: {err_msg}")
                    _assign_episode(slot, is_retry=True)

            if not obs_list:
                continue

            # Apply any pending inter-round rewards before stepping the next action
            for ob in obs_list:
                forced_readies += ob.forced_delta
                buf = ep_buf[ob.slot]
                if buf and ob.r_delta > 0:
                    buf[-1]["reward"] += round_survive_bonus * ob.r_delta + lp_loss_penalty * ob.lp_lost
                    buf[-1]["round_end"] = True

            # Vectorized SharedMemory read -> fresh owned CPU tensors (safe to reuse slots immediately after)
            slots = [ob.slot for ob in obs_list]
            cat_batch = torch.from_numpy(self._cat_shm[slots].astype(np.int64))
            cont_batch = torch.from_numpy(self._cont_shm[slots])
            valid_batch = torch.from_numpy(self._valid_shm[slots])
            mask_specs = [ob.mask_spec for ob in obs_list]

            act_out = model.act_batch_tensors(
                cat_batch,
                cont_batch,
                valid_batch,
                mask_specs,
                deterministic=False,
                temperature=float(sampling_temperature),
                device=device,
            )
            num_gpu_batches += 1
            total_gpu_items += len(obs_list)

            # Dispatch actions back to workers immediately, then record transitions
            for i, ob in enumerate(obs_list):
                a_type = act_out["a_type"][i]
                a_src = act_out["a_src"][i]
                a_tgt = act_out["a_tgt"][i]
                a_opt = act_out["a_opt"][i]

                inflight.add(ob.slot)
                self._conns[ob.slot].send(("act", ob.ep_id, a_type, a_src, a_tgt, a_opt))

                ep_buf[ob.slot].append(
                    {
                        "cat_ids": cat_batch[i],
                        "cont_feats": cont_batch[i],
                        "valid_mask": valid_batch[i],
                        "a_type": a_type,
                        "a_src": a_src,
                        "a_tgt": a_tgt,
                        "a_opt": a_opt,
                        "type_mask": act_out["type_mask"][i],
                        "src_mask": act_out["src_mask"][i],
                        "has_src": bool(act_out["has_src"][i].item()),
                        "band_mask": act_out["band_mask"][i],
                        "has_band": bool(act_out["has_band"][i].item()),
                        "tgt_mask": act_out["tgt_mask"][i],
                        "has_tgt": bool(act_out["has_tgt"][i].item()),
                        "opt_mask": act_out["opt_mask"][i],
                        "has_opt": bool(act_out["has_opt"][i].item()),
                        "old_log_prob": float(act_out["log_prob"][i]),
                        "value": float(act_out["value"][i]),
                        "reward": step_penalty if ob.is_prep else 0.0,
                        "done": False,
                        "round_end": False,
                        "is_strategic": (not ob.is_prep) or (a_type in STRATEGIC_ACTION_TYPES),
                    }
                )

        stats = {
            "episodes": len(all_episodes),
            "win_rate": wins / max(1, len(all_episodes)),
            "hidden_clears": hidden_clears,
            "avg_rounds_passed": sum(rounds_passed) / max(1, len(rounds_passed)),
            "forced_readies": forced_readies,
            "sim_errors": sim_errors,
            "avg_batch_size": round(total_gpu_items / max(1, num_gpu_batches), 2),
            "gpu_batches": num_gpu_batches,
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }
        return all_episodes, stats

    @torch.no_grad()
    def evaluate_episodes(
        self,
        model: StrongholdPolicyNet,
        *,
        modes: Sequence[str],
        seeds: Sequence[int],
        deterministic: bool = True,
        temperature: float = 0.0,
        break_cycles: bool = True,
        max_prep_steps_per_round: int = 80,
        max_total_steps: int = 3000,
        min_batch: int | None = None,
        batch_timeout_ms: float | None = None,
        device: torch.device | str = "cpu",
    ) -> dict[str, Any]:
        """Run asynchronous parallel solo evaluation with dynamic GPU batching."""
        if self._closed:
            raise RuntimeError("AsyncActorPool is closed")
        total_eps = len(seeds)
        if total_eps == 0:
            return {
                "episodes": 0,
                "win_rate": 0.0,
                "avg_rounds_passed": 0.0,
                "avg_steps": 0.0,
                "forced_readies": 0,
                "cycle_breaks": 0,
                "sim_errors": 0,
            }

        model.eval()
        t0 = time.perf_counter()
        eff_min_batch = self.default_min_batch if (min_batch is None or min_batch <= 0) else int(min_batch)
        eff_timeout_ms = self.default_batch_timeout_ms if batch_timeout_ms is None else float(batch_timeout_ms)
        use_det = deterministic and temperature <= 0.0

        ep_cfg = {
            "max_prep_steps": max_prep_steps_per_round,
            "max_total_steps": max_total_steps,
            "break_cycles": break_cycles,
            "deterministic": use_det,
            "temperature": float(temperature if temperature > 0.0 else (0.0 if use_det else 1.0)),
        }

        wins = 0
        hidden_clears = 0
        ok_eps = 0
        sim_errors = 0
        rounds_list: list[int] = []
        steps_list: list[int] = []
        forced_readies = 0
        cycle_breaks = 0
        num_gpu_batches = 0
        total_gpu_items = 0

        next_idx = 0
        inflight: set[int] = set()

        def _assign_next(slot: int) -> None:
            nonlocal next_idx
            if next_idx >= total_eps:
                return
            idx = next_idx
            next_idx += 1
            mode_id = modes[idx % len(modes)]
            seed = int(seeds[idx])
            inflight.add(slot)
            self._conns[slot].send(("reset", idx, mode_id, seed, ep_cfg))

        for w in range(min(self.num_workers, total_eps)):
            _assign_next(w)

        while inflight:
            obs_list, events = self._poll_messages(
                inflight,
                min_batch=eff_min_batch,
                batch_timeout_ms=eff_timeout_ms,
            )

            for ev in events:
                tag = ev[0]
                if tag == "done":
                    _, slot, _ep_id, win, rnds, last_rnd, steps, f_delta, c_delta, _r_d, _lp_l, _to = ev
                    ok_eps += 1
                    forced_readies += f_delta
                    cycle_breaks += c_delta
                    if win:
                        wins += 1
                    if win and rnds > last_rnd:
                        hidden_clears += 1
                    rounds_list.append(rnds)
                    steps_list.append(steps)
                    _assign_next(slot)
                elif tag == "error":
                    _, slot, err_ep_id, err_msg = ev
                    sim_errors += 1
                    print(
                        f"[AsyncActorPool] WARN: simulator error in eval ep={err_ep_id} "
                        f"(slot={slot}): {err_msg[:200]}",
                        flush=True,
                    )
                    _assign_next(slot)

            if not obs_list:
                continue

            for ob in obs_list:
                forced_readies += ob.forced_delta
                cycle_breaks += ob.cycle_delta

            # Group by (deterministic, temperature) — typically just 1 group (or 2 when cycle-escaping)
            groups: dict[tuple[bool, float], list[_PendingObs]] = {}
            for ob in obs_list:
                groups.setdefault((ob.deterministic, ob.temperature), []).append(ob)

            for (g_det, g_temp), grp in groups.items():
                slots = [ob.slot for ob in grp]
                cat_batch = torch.from_numpy(self._cat_shm[slots].astype(np.int64))
                cont_batch = torch.from_numpy(self._cont_shm[slots])
                valid_batch = torch.from_numpy(self._valid_shm[slots])
                mask_specs = [ob.mask_spec for ob in grp]

                act_out = model.act_batch_tensors(
                    cat_batch,
                    cont_batch,
                    valid_batch,
                    mask_specs,
                    deterministic=g_det,
                    temperature=g_temp,
                    device=device,
                )
                num_gpu_batches += 1
                total_gpu_items += len(grp)

                for i, ob in enumerate(grp):
                    inflight.add(ob.slot)
                    self._conns[ob.slot].send(
                        (
                            "act",
                            ob.ep_id,
                            act_out["a_type"][i],
                            act_out["a_src"][i],
                            act_out["a_tgt"][i],
                            act_out["a_opt"][i],
                        )
                    )

        return {
            "episodes": ok_eps,
            "episodes_requested": total_eps,
            "win_rate": wins / max(1, ok_eps),
            "hidden_clears": hidden_clears,
            "hidden_clear_rate": hidden_clears / max(1, ok_eps),
            "avg_rounds_passed": sum(rounds_list) / max(1, len(rounds_list)),
            "avg_steps": sum(steps_list) / max(1, len(steps_list)),
            "forced_readies": forced_readies,
            "cycle_breaks": cycle_breaks,
            "sim_errors": sim_errors,
            "avg_batch_size": round(total_gpu_items / max(1, num_gpu_batches), 2),
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }
