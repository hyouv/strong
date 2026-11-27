"""Deterministic, coverage-balanced BC openings over simulator-provided options."""
from __future__ import annotations

from collections import Counter
import random
from typing import Any, Mapping, Sequence

from .bc_collection import DEFAULT_EPISODE_RETRIES


BALANCED_OPENING_VERSION = "balanced-v1"


class BalancedOpeningSampler:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.bands: Counter[str] = Counter()
        self.groups: Counter[int | None] = Counter()
        self.pairs: Counter[tuple[int | None, str]] = Counter()
        self.mode_bands: Counter[tuple[str, str]] = Counter()
        self.available_bands: set[str] = set()

    def choose(self, options: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
        candidates = [(c["bondGroup"], b) for c in options for b in c["bandIds"]]
        if not candidates:
            raise ValueError(f"No compatible expert opening for {mode}")
        self.available_bands.update(b for _, b in candidates)
        # Counts are global, not per worker. Seeded tie-breaking avoids index bias
        # and modulo correlations with the interleaved difficulty curriculum.
        group, band = min(candidates, key=lambda pair: (
            self.bands[pair[1]], self.groups[pair[0]], self.pairs[pair],
            self.mode_bands[mode, pair[1]], self.rng.random(),
        ))
        self.bands[band] += 1
        self.groups[group] += 1
        self.pairs[group, band] += 1
        self.mode_bands[mode, band] += 1
        return {"bondGroup": group, "bandId": band}

    def summary(self) -> dict[str, Any]:
        counts = {b: self.bands[b] for b in sorted(self.available_bands)}
        return {
            "group_counts": {str(g): n for g, n in self.groups.items()},
            "band_counts": counts,
            "band_count_min": min(counts.values(), default=0),
            "band_count_max": max(counts.values(), default=0),
        }


def bc_cache_metadata(opening_policy: str, modes: Sequence[str], episodes: int, seed: int,
                      max_episode_retries: int = DEFAULT_EPISODE_RETRIES) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "generation_version": "haruka-reentry-fix-quarantine-v1",
        "opening_policy": BALANCED_OPENING_VERSION if opening_policy == "balanced" else "legacy",
        "modes": list(modes), "episodes": episodes, "seed": seed,
        "max_episode_retries": max_episode_retries,
    }
