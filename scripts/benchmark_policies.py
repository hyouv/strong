"""Head-to-head solo policy benchmark: Random vs bot.js vs RL Model.

Runs all selected policies on the exact same (modeId, seed) episode schedule,
using multi-process sharding to bypass Python GIL limits on multi-core servers
and gracefully isolating rare upstream JS simulator errors (`WorkerError`) so
large 1,000+ game evaluations never abort mid-run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import torch

from stronghold_sim import RoomClient, RoomPool, SOLO_MODES, SOLO_TIER1_MODES, WorkerError
from stronghold_rl.bc import (
    _prep_state_signature,
    describe_device,
    load_checkpoint,
    pick_force_unblock_action,
    resolve_device,
)
from stronghold_rl.catalog import get_catalog
from stronghold_rl.codec import TYPE_READY, encode_frame

MODE_PRESETS: dict[str, tuple[str, ...]] = {
    "tier1": SOLO_TIER1_MODES,
    "1-4": SOLO_TIER1_MODES,
    "tier2": ("mode_single_5", "mode_single_6", "mode_single_7", "mode_single_8"),
    "5-8": ("mode_single_5", "mode_single_6", "mode_single_7", "mode_single_8"),
    "tier3": tuple(f"mode_single_{i}" for i in range(9, 15)),
    "9-14": tuple(f"mode_single_{i}" for i in range(9, 15)),
    "tier4": tuple(f"mode_single_{i}" for i in range(15, 21)),
    "15-20": tuple(f"mode_single_{i}" for i in range(15, 21)),
}


def parse_modes(modes_arg: str) -> tuple[str, ...]:
    key = modes_arg.strip()
    if key in MODE_PRESETS:
        return MODE_PRESETS[key]
    parts = tuple(m.strip() for m in key.split(",") if m.strip())
    if not parts:
        raise ValueError(f"Empty --modes specification: {modes_arg!r}")
    invalid = [m for m in parts if m not in SOLO_MODES]
    if invalid:
        raise ValueError(f"Unsupported solo mode(s): {invalid}")
    return parts


def build_schedule(num_episodes: int, modes: Sequence[str], seed_start: int) -> list[tuple[int, str, int]]:
    """Return deterministic list of (ep_idx, mode_id, seed)."""
    return [(i, modes[i % len(modes)], seed_start + i) for i in range(num_episodes)]


def _split_shards(items: list[tuple[int, str, int]], num_shards: int) -> list[list[tuple[int, str, int]]]:
    n = max(1, min(num_shards, len(items)))
    shards: list[list[tuple[int, str, int]]] = [[] for _ in range(n)]
    for idx, item in enumerate(items):
        shards[idx % n].append(item)
    return [s for s in shards if s]


def _pick_force_unblock_raw(frame: Mapping[str, Any]) -> int:
    """Equivalent to `pick_force_unblock_action` directly on raw frame JSON (no tensor encoding)."""
    actions = frame["actions"]
    for idx, act in enumerate(actions):
        if act["t"] == "g.ready":
            return idx
    for target_t in ("g.commandChoice", "g.choice"):
        for idx, act in enumerate(actions):
            if act["t"] == target_t:
                return idx
    temp_uids = {int(p["uid"]) for p in (frame["own"].get("temp") or []) if p}
    for target_t in ("g.destroy", "g.sell"):
        first_idx: int | None = None
        for idx, act in enumerate(actions):
            if act["t"] == target_t:
                if first_idx is None:
                    first_idx = idx
                uid = act.get("uid")
                if uid is not None and int(uid) in temp_uids:
                    return idx
        if first_idx is not None:
            return first_idx
    return 0


def _run_random_shard(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[tuple[int, str, int]] = payload["tasks"]
    max_prep = int(payload["max_prep_steps"])
    max_total_steps = int(payload["max_total_steps"])
    results: list[dict[str, Any]] = []

    client = RoomClient()
    try:
        for ep_idx, mode_id, seed in tasks:
            rng = random.Random(seed ^ 0x9E3779B9)
            steps = 0
            prep_steps = 0
            last_rnd = -1
            forced_readies = 0
            try:
                frame = client.reset(seed=seed, mode_id=mode_id, policy_version=0)
                while not frame.get("terminal") and steps < max_total_steps:
                    rnd = int(frame["pub"].get("round") or 0)
                    if rnd != last_rnd:
                        last_rnd = rnd
                        prep_steps = 0

                    if frame["phase"] == "PREP" and prep_steps >= max_prep:
                        forced_readies += 1
                        action_key = _pick_force_unblock_raw(frame)
                    else:
                        action_key = rng.randrange(len(frame["actions"]))

                    if frame["phase"] == "PREP":
                        prep_steps += 1
                    steps += 1
                    frame = client.step(frame["decision_id"], action_key, policy_version=0)

                results.append(
                    {
                        "ep_idx": ep_idx,
                        "mode": mode_id,
                        "seed": seed,
                        "ok": True,
                        "victory": bool(frame.get("victory")),
                        "rounds_passed": int(frame.get("roundsPassed", 0)),
                        "last_round": int(frame.get("lastRound", 14)),
                        "steps": steps,
                        "forced_readies": forced_readies,
                        "cycle_breaks": 0,
                        "timed_out": not bool(frame.get("terminal")),
                    }
                )
            except WorkerError as exc:
                results.append(
                    {
                        "ep_idx": ep_idx,
                        "mode": mode_id,
                        "seed": seed,
                        "ok": False,
                        "victory": False,
                        "rounds_passed": 0,
                        "last_round": 14,
                        "steps": steps,
                        "forced_readies": forced_readies,
                        "cycle_breaks": 0,
                        "error": str(exc)[:400],
                    }
                )
                client.close()
                client = RoomClient()
    finally:
        client.close()
    return results


def _run_bot_shard(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[tuple[int, str, int]] = payload["tasks"]
    results: list[dict[str, Any]] = []

    client = RoomClient()
    try:
        for ep_idx, mode_id, seed in tasks:
            try:
                res = client.generate_examples(
                    mode_id=mode_id,
                    seed=seed,
                    memory_limit=64 * 1024 * 1024,
                    turn_order="serial",
                )
                results.append(
                    {
                        "ep_idx": ep_idx,
                        "mode": mode_id,
                        "seed": seed,
                        "ok": True,
                        "victory": bool(res.get("victory")),
                        "rounds_passed": int(res.get("roundsPassed", 0)),
                        "last_round": int(res.get("lastRound", 14)),
                        "steps": int(res.get("recorded", 0)),
                        "forced_readies": 0,
                        "cycle_breaks": 0,
                        "timed_out": False,
                    }
                )
            except WorkerError as exc:
                results.append(
                    {
                        "ep_idx": ep_idx,
                        "mode": mode_id,
                        "seed": seed,
                        "ok": False,
                        "victory": False,
                        "rounds_passed": 0,
                        "last_round": 14,
                        "steps": 0,
                        "forced_readies": 0,
                        "cycle_breaks": 0,
                        "error": str(exc)[:400],
                    }
                )
                client.close()
                client = RoomClient()
    finally:
        client.close()
    return results


def _run_model_shard(payload: dict[str, Any]) -> list[dict[str, Any]]:
    torch.set_num_threads(1)
    tasks: list[tuple[int, str, int]] = payload["tasks"]
    ckpt_path: str = payload["ckpt"]
    shard_id: int = int(payload.get("shard_id", 0))
    device: str = payload["device"]
    workers_per_shard: int = int(payload["workers_per_shard"])
    temperature: float = float(payload["temperature"])
    break_cycles: bool = bool(payload["break_cycles"])
    max_prep: int = int(payload["max_prep_steps"])
    max_total_steps: int = int(payload["max_total_steps"])

    cat = get_catalog()
    model, _ = load_checkpoint(ckpt_path, catalog=cat, device=device, dropout_override=0.0)
    model.eval()
    use_det = temperature <= 0.0

    results: list[dict[str, Any]] = []
    n_workers = max(1, min(workers_per_shard, len(tasks)))

    with RoomPool(num_workers=n_workers) as pool:
        next_i = 0
        slot_task: dict[int, tuple[int, str, int]] = {}
        ep_steps: dict[int, int] = {}
        ep_prep_steps: dict[int, int] = {}
        ep_last_round: dict[int, int] = {}
        ep_forced: dict[int, int] = {}
        ep_cycles: dict[int, int] = {}
        ep_seen_sigs: dict[int, dict[tuple[Any, ...], int]] = {}
        active_frames: dict[int, dict[str, Any]] = {}

        def _reset_slot(w: int) -> None:
            nonlocal next_i
            while next_i < len(tasks):
                task = tasks[next_i]
                next_i += 1
                ep_idx, mode_id, seed = task
                slot_task[w] = task
                ep_steps[w] = 0
                ep_prep_steps[w] = 0
                ep_last_round[w] = -1
                ep_forced[w] = 0
                ep_cycles[w] = 0
                ep_seen_sigs[w] = {}
                try:
                    active_frames[w] = pool.request(
                        w,
                        {"op": "reset", "seed": seed, "modeId": mode_id, "policy_version": 0},
                    )
                    return
                except WorkerError as exc:
                    results.append(
                        {
                            "ep_idx": ep_idx,
                            "mode": mode_id,
                            "seed": seed,
                            "ok": False,
                            "victory": False,
                            "rounds_passed": 0,
                            "last_round": 14,
                            "steps": 0,
                            "forced_readies": 0,
                            "cycle_breaks": 0,
                            "error": str(exc)[:400],
                        }
                    )
                    pool.clients[w].close()
                    pool.clients[w] = RoomClient()

        for w in range(n_workers):
            _reset_slot(w)

        while active_frames:
            step_cmds: dict[int, dict[str, Any]] = {}
            det_workers: list[int] = []
            det_frames: list[dict[str, Any]] = []
            det_encs: list[Any] = []
            stoch_items: list[tuple[int, dict[str, Any], Any, float]] = []

            for w, frame in active_frames.items():
                rnd = int(frame["pub"].get("round") or 0)
                if rnd != ep_last_round[w]:
                    ep_last_round[w] = rnd
                    ep_prep_steps[w] = 0
                    ep_seen_sigs[w].clear()

                enc = encode_frame(frame, cat, prep_steps=ep_prep_steps[w])
                if frame["phase"] == "PREP" and ep_prep_steps[w] >= max_prep:
                    ep_forced[w] += 1
                    action_key = pick_force_unblock_action(enc)
                    step_cmds[w] = {
                        "op": "step",
                        "decision_id": frame["decision_id"],
                        "action_key": action_key,
                        "policy_version": 0,
                    }
                    ep_prep_steps[w] += 1
                    ep_steps[w] += 1
                    continue

                if break_cycles and frame["phase"] == "PREP":
                    sig = _prep_state_signature(frame)
                    visits = ep_seen_sigs[w].get(sig, 0)
                    ep_seen_sigs[w][sig] = visits + 1
                    if visits >= 2 and bool(enc.action_tree.type_mask[TYPE_READY]):
                        ep_cycles[w] += 1
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
                        ep_cycles[w] += 1
                        escape_temp = max(temperature, 0.6) if temperature > 0.0 else 0.6
                        stoch_items.append((w, frame, enc, escape_temp))
                        continue

                if use_det:
                    det_workers.append(w)
                    det_frames.append(frame)
                    det_encs.append(enc)
                else:
                    stoch_items.append((w, frame, enc, temperature))

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
                by_temp: dict[float, list[tuple[int, dict[str, Any], Any]]] = {}
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

            # Dispatch step commands fault-tolerantly per worker
            futures = {w: pool.threads.submit(pool.request, w, cmd) for w, cmd in step_cmds.items()}
            next_active: dict[int, dict[str, Any]] = {}
            need_reset: list[int] = []

            for w, fut in futures.items():
                ep_idx, mode_id, seed = slot_task[w]
                try:
                    rep = fut.result()
                except WorkerError as exc:
                    results.append(
                        {
                            "ep_idx": ep_idx,
                            "mode": mode_id,
                            "seed": seed,
                            "ok": False,
                            "victory": False,
                            "rounds_passed": 0,
                            "last_round": 14,
                            "steps": ep_steps[w],
                            "forced_readies": ep_forced[w],
                            "cycle_breaks": ep_cycles[w],
                            "error": str(exc)[:400],
                        }
                    )
                    pool.clients[w].close()
                    pool.clients[w] = RoomClient()
                    need_reset.append(w)
                    continue

                if rep.get("terminal") or ep_steps[w] >= max_total_steps:
                    results.append(
                        {
                            "ep_idx": ep_idx,
                            "mode": mode_id,
                            "seed": seed,
                            "ok": True,
                            "victory": bool(rep.get("victory")),
                            "rounds_passed": int(rep.get("roundsPassed", 0)),
                            "last_round": int(rep.get("lastRound", 14)),
                            "steps": ep_steps[w],
                            "forced_readies": ep_forced[w],
                            "cycle_breaks": ep_cycles[w],
                            "timed_out": not bool(rep.get("terminal")),
                        }
                    )
                    need_reset.append(w)
                else:
                    next_active[w] = rep

            active_frames = next_active
            for w in need_reset:
                _reset_slot(w)
            if need_reset and (len(results) % 25 == 0 or len(results) == len(tasks)):
                ok_n = sum(1 for r in results if r.get("ok", True))
                win_n = sum(1 for r in results if r.get("ok", True) and r["victory"])
                print(
                    f"    [model shard {shard_id}] {len(results)}/{len(tasks)} eps "
                    f"win={win_n}/{ok_n} ({win_n/max(1,ok_n)*100:.1f}%)",
                    flush=True,
                )

    return results


def _wilson_ci95(wins: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize_records(records: list[dict[str, Any]], modes: Sequence[str], elapsed_sec: float) -> dict[str, Any]:
    ok_recs = [r for r in records if r.get("ok", True)]
    err_recs = [r for r in records if not r.get("ok", True)]
    n_total = len(records)
    n_ok = len(ok_recs)
    wins = sum(1 for r in ok_recs if r["victory"])
    ci_lo, ci_hi = _wilson_ci95(wins, n_ok)

    by_mode: dict[str, dict[str, Any]] = {}
    for m in modes:
        m_all = [r for r in records if r["mode"] == m]
        m_ok = [r for r in m_all if r.get("ok", True)]
        m_wins = sum(1 for r in m_ok if r["victory"])
        m_lo, m_hi = _wilson_ci95(m_wins, len(m_ok))
        by_mode[m] = {
            "episodes_total": len(m_all),
            "episodes_ok": len(m_ok),
            "sim_errors": len(m_all) - len(m_ok),
            "wins": m_wins,
            "win_rate": m_wins / max(1, len(m_ok)),
            "ci95": [round(m_lo, 4), round(m_hi, 4)],
            "avg_rounds_passed": sum(r["rounds_passed"] for r in m_ok) / max(1, len(m_ok)),
            "avg_steps": sum(r["steps"] for r in m_ok) / max(1, len(m_ok)),
        }

    return {
        "episodes_total": n_total,
        "episodes_ok": n_ok,
        "sim_errors": len(err_recs),
        "wins": wins,
        "win_rate": wins / max(1, n_ok),
        "win_rate_all": wins / max(1, n_total),
        "ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "avg_rounds_passed": sum(r["rounds_passed"] for r in ok_recs) / max(1, n_ok),
        "avg_steps": sum(r["steps"] for r in ok_recs) / max(1, n_ok),
        "forced_readies": sum(int(r.get("forced_readies", 0)) for r in ok_recs),
        "cycle_breaks": sum(int(r.get("cycle_breaks", 0)) for r in ok_recs),
        "timeouts": sum(1 for r in ok_recs if r.get("timed_out")),
        "elapsed_sec": round(elapsed_sec, 2),
        "by_mode": by_mode,
        "error_samples": [
            {"mode": r["mode"], "seed": r["seed"], "error": r.get("error", "")}
            for r in err_recs[:10]
        ],
    }


def _paired_comparison(recs_a: list[dict[str, Any]], recs_b: list[dict[str, Any]]) -> dict[str, Any]:
    map_a = {(r["mode"], r["seed"]): r for r in recs_a if r.get("ok", True)}
    map_b = {(r["mode"], r["seed"]): r for r in recs_b if r.get("ok", True)}
    common = sorted(set(map_a) & set(map_b))
    both_win = 0
    a_only = 0
    b_only = 0
    both_lose = 0
    for k in common:
        wa = bool(map_a[k]["victory"])
        wb = bool(map_b[k]["victory"])
        if wa and wb:
            both_win += 1
        elif wa and not wb:
            a_only += 1
        elif not wa and wb:
            b_only += 1
        else:
            both_lose += 1
    discordant = a_only + b_only
    mcnemar_chi2 = ((abs(a_only - b_only) - 1) ** 2 / discordant) if discordant > 0 else 0.0
    return {
        "common_ok_episodes": len(common),
        "both_win": both_win,
        "a_only_win": a_only,
        "b_only_win": b_only,
        "both_lose": both_lose,
        "mcnemar_chi2_cc": round(mcnemar_chi2, 3),
    }


def run_policy(
    policy: str,
    schedule: list[tuple[int, str, int]],
    *,
    modes: Sequence[str],
    workers: int,
    model_shards: int,
    ckpt: str,
    device: str,
    temperature: float,
    break_cycles: bool,
    max_prep_steps: int,
    max_total_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    t0 = time.perf_counter()
    mp_ctx = multiprocessing.get_context("spawn")
    all_records: list[dict[str, Any]] = []

    if policy == "random":
        # Each process owns 1 RoomClient -> 100% parallel Node + JSON without GIL contention
        n_procs = max(1, min(workers, len(schedule)))
        shards = _split_shards(schedule, n_procs)
        payloads = [
            {"tasks": s, "max_prep_steps": max_prep_steps, "max_total_steps": max_total_steps}
            for s in shards
        ]
        fn = _run_random_shard
    elif policy == "bot":
        n_procs = max(1, min(workers, len(schedule)))
        shards = _split_shards(schedule, n_procs)
        payloads = [{"tasks": s} for s in shards]
        fn = _run_bot_shard
    elif policy == "model":
        n_shards = max(1, min(model_shards, len(schedule)))
        workers_per_shard = max(1, workers // n_shards)
        shards = _split_shards(schedule, n_shards)
        payloads = [
            {
                "shard_id": idx,
                "tasks": s,
                "ckpt": ckpt,
                "device": device,
                "workers_per_shard": workers_per_shard,
                "temperature": temperature,
                "break_cycles": break_cycles,
                "max_prep_steps": max_prep_steps,
                "max_total_steps": max_total_steps,
            }
            for idx, s in enumerate(shards)
        ]
        fn = _run_model_shard
    else:
        raise ValueError(f"Unknown policy: {policy!r}")

    print(
        f"\n>>> Starting [{policy}] on {len(schedule)} episodes "
        f"({len(shards)} worker process(es), target_workers={workers})...",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=len(shards), mp_context=mp_ctx) as ex:
        futs = [ex.submit(fn, p) for p in payloads]
        done_shards = 0
        for fut in as_completed(futs):
            part = fut.result()
            all_records.extend(part)
            done_shards += 1
            ok_so_far = [r for r in all_records if r.get("ok", True)]
            wins_so_far = sum(1 for r in ok_so_far if r["victory"])
            print(
                f"  [{policy}] shard {done_shards}/{len(shards)} done | "
                f"episodes={len(all_records)}/{len(schedule)} "
                f"win={wins_so_far}/{len(ok_so_far)} ({wins_so_far/max(1,len(ok_so_far))*100:.2f}%) "
                f"errs={len(all_records)-len(ok_so_far)} elapsed={time.perf_counter()-t0:.1f}s",
                flush=True,
            )

    all_records.sort(key=lambda r: r["ep_idx"])
    summary = summarize_records(all_records, modes, time.perf_counter() - t0)
    print(
        f"<<< Finished [{policy}]: win={summary['wins']}/{summary['episodes_ok']} "
        f"({summary['win_rate']*100:.2f}%, 95% CI [{summary['ci95'][0]*100:.2f}%, {summary['ci95'][1]*100:.2f}%]) | "
        f"rounds={summary['avg_rounds_passed']:.2f} steps={summary['avg_steps']:.1f} "
        f"forced_rdy={summary['forced_readies']} cb={summary['cycle_breaks']} "
        f"sim_errs={summary['sim_errors']} ({summary['elapsed_sec']:.1f}s)",
        flush=True,
    )
    for m, ms in summary["by_mode"].items():
        print(
            f"      {m}: {ms['wins']}/{ms['episodes_ok']} ({ms['win_rate']*100:.2f}%) "
            f"rounds={ms['avg_rounds_passed']:.2f} steps={ms['avg_steps']:.1f} errs={ms['sim_errors']}",
            flush=True,
        )
    return summary, all_records


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark Random vs bot.js vs RL Model on paired solo seeds.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--modes", type=str, default="tier1", help="Preset (tier1, tier2, ...) or comma-separated modeIds")
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--policies", type=str, default="random,bot,model")
    parser.add_argument("--ckpt", type=str, default="checkpoints/ppo_5m_solo_t1_best.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--workers", type=int, default=32, help="Total parallel JS workers per policy")
    parser.add_argument("--model-shards", type=int, default=8, help="Parallel Python shard processes for model eval")
    parser.add_argument("--eval-temp", type=float, default=0.0)
    parser.add_argument("--no-break-cycles", action="store_true")
    parser.add_argument("--max-prep-steps", type=int, default=80)
    parser.add_argument("--max-total-steps", type=int, default=3000)
    parser.add_argument("--out", type=str, default="logs/benchmark_t1_1000ep.json")
    args = parser.parse_args(argv)

    modes = parse_modes(args.modes)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    device = resolve_device(args.device) if "model" in policies else "cpu"
    schedule = build_schedule(args.episodes, modes, args.seed_start)

    print("============================================================", flush=True)
    print(f"Stronghold Protocol Policy Benchmark ({args.episodes} paired episodes)", flush=True)
    print(f"  modes        : {modes}", flush=True)
    print(f"  seeds        : {args.seed_start} .. {args.seed_start + args.episodes - 1}", flush=True)
    print(f"  policies     : {policies}", flush=True)
    print(f"  ckpt         : {args.ckpt}", flush=True)
    print(f"  device       : {describe_device(device) if 'model' in policies else 'cpu'}", flush=True)
    print(f"  workers      : {args.workers} (model_shards={args.model_shards})", flush=True)
    print(f"  output       : {args.out}", flush=True)
    print("============================================================", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "config": {
            "episodes": args.episodes,
            "modes": list(modes),
            "seed_start": args.seed_start,
            "policies": policies,
            "ckpt": args.ckpt,
            "device": device,
            "workers": args.workers,
            "model_shards": args.model_shards,
            "eval_temp": args.eval_temp,
            "break_cycles": not args.no_break_cycles,
            "max_prep_steps": args.max_prep_steps,
        },
        "summaries": {},
        "paired": {},
    }
    records_by_policy: dict[str, list[dict[str, Any]]] = {}

    for pol in policies:
        summary, recs = run_policy(
            pol,
            schedule,
            modes=modes,
            workers=args.workers,
            model_shards=args.model_shards,
            ckpt=args.ckpt,
            device=device,
            temperature=args.eval_temp,
            break_cycles=not args.no_break_cycles,
            max_prep_steps=args.max_prep_steps,
            max_total_steps=args.max_total_steps,
        )
        report["summaries"][pol] = summary
        records_by_policy[pol] = recs
        # Persist incremental summary after each policy finishes
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if "model" in records_by_policy and "bot" in records_by_policy:
        report["paired"]["model_vs_bot"] = _paired_comparison(records_by_policy["model"], records_by_policy["bot"])
    if "model" in records_by_policy and "random" in records_by_policy:
        report["paired"]["model_vs_random"] = _paired_comparison(records_by_policy["model"], records_by_policy["random"])
    if "bot" in records_by_policy and "random" in records_by_policy:
        report["paired"]["bot_vs_random"] = _paired_comparison(records_by_policy["bot"], records_by_policy["random"])

    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Also save compact per-episode JSONL alongside the summary JSON
    episodes_path = out_path.with_suffix(".episodes.json")
    episodes_path.write_text(json.dumps(records_by_policy, ensure_ascii=False), encoding="utf-8")

    print("\n================ FINAL SUMMARY ================", flush=True)
    for pol in policies:
        s = report["summaries"][pol]
        print(
            f"  {pol:8s} | win={s['wins']:4d}/{s['episodes_ok']:4d} ({s['win_rate']*100:6.2f}%, "
            f"95%CI [{s['ci95'][0]*100:5.2f}%, {s['ci95'][1]*100:5.2f}%]) | "
            f"rounds={s['avg_rounds_passed']:5.2f} | steps={s['avg_steps']:6.1f} | errs={s['sim_errors']}",
            flush=True,
        )
    if report["paired"]:
        print("Paired comparisons:", json.dumps(report["paired"], ensure_ascii=False), flush=True)
    print(f"Saved report -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
