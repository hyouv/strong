"""Quarantine invalid expert matches and replace them without changing opening quotas."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from stronghold_sim import RoomPool, WorkerError


DEFAULT_EPISODE_RETRIES = 3
RETRY_SEED_SEARCH = 64


class ExpertReplayCollector:
    def __init__(self, *, seed_start: int, num_episodes: int,
                 max_retries: int = DEFAULT_EPISODE_RETRIES,
                 quarantine_path: str | Path | None = None) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        last_seed = seed_start + num_episodes * (1 + max_retries * RETRY_SEED_SEARCH) - 1
        if seed_start < 0 or last_seed > 0xffffffff:
            raise ValueError("Original and replacement seeds must fit in uint32")
        self.seed_start = seed_start
        self.num_episodes = num_episodes
        self.max_retries = max_retries
        self.quarantine_path = Path(quarantine_path) if quarantine_path else None
        self.failures: list[dict[str, Any]] = []
        self.replacements: list[dict[str, Any]] = []

    def _quarantine(self, original: Mapping[str, Any], command: Mapping[str, Any],
                    error: WorkerError, attempt: int) -> None:
        record = {
            "mode": command["modeId"], "original_seed": original["seed"],
            "seed": command["seed"], "opening": command.get("opening"),
            "attempt": attempt, "error": str(error),
        }
        # Write immediately: the diagnostic survives even if every replacement fails.
        if self.quarantine_path:
            self.quarantine_path.parent.mkdir(parents=True, exist_ok=True)
            with self.quarantine_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.failures.append(record)
        print(f"[Collect quarantine] mode={command['modeId']} seed={command['seed']} "
              f"opening={command.get('opening')} attempt={attempt}; "
              f"log={self.quarantine_path or '<dataset metadata>'}", flush=True)

    def _replacement(self, pool: RoomPool, slot: int, original: Mapping[str, Any],
                     retry: int) -> dict[str, Any]:
        index = original["seed"] - self.seed_start
        first = (self.seed_start + self.num_episodes
                 + (index * self.max_retries + retry - 1) * RETRY_SEED_SEARCH)
        opening = original.get("opening")
        for seed in range(first, first + RETRY_SEED_SEARCH):
            if opening is not None:
                options = pool.request(slot, {
                    "op": "expert_opening_options", "modeId": original["modeId"], "seed": seed,
                })["options"]
                if not any(o["bondGroup"] == opening["bondGroup"]
                           and opening["bandId"] in o["bandIds"] for o in options):
                    continue
            return {**original, "seed": seed}
        raise RuntimeError(f"No compatible replacement seed for {original['modeId']} "
                           f"seed={original['seed']} opening={opening} retry={retry}")

    def exchange(self, pool: RoomPool, commands: Mapping[int, Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
        pending = {slot: dict(c) for slot, c in commands.items()}
        retries = {slot: 0 for slot in commands}
        accepted: dict[int, dict[str, Any]] = {}
        while pending:
            futures = {slot: pool.threads.submit(pool.request, slot, c) for slot, c in pending.items()}
            failed = {}
            fatal = None
            # Resolve every request before reusing a worker. Successful slots are never replayed.
            for slot, future in futures.items():
                try:
                    accepted[slot] = future.result()
                except WorkerError as error:
                    if str(error).startswith("JS worker error: Error: Invalid expert match:"):
                        failed[slot] = error
                    else:
                        fatal = error
                except Exception as error:
                    fatal = error
            for slot, error in failed.items():
                self._quarantine(commands[slot], pending[slot], error, retries[slot])
            if fatal is not None:
                raise fatal
            for slot in pending:
                if slot in failed:
                    continue
                c = pending[slot]
                if retries[slot]:
                    self.replacements.append({
                        "mode": c["modeId"], "original_seed": commands[slot]["seed"],
                        "seed": c["seed"], "opening": c.get("opening"), "retries": retries[slot],
                    })
            next_pending = {}
            for slot, error in failed.items():
                retries[slot] += 1
                if retries[slot] > self.max_retries:
                    raise RuntimeError(f"Expert episode {commands[slot]['seed']} failed after "
                                       f"{self.max_retries} replacements; see quarantine log "
                                       f"{self.quarantine_path}") from error
                next_pending[slot] = self._replacement(pool, slot, commands[slot], retries[slot])
            pending = next_pending
        return accepted
