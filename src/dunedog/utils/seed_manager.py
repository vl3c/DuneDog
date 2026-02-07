"""Deterministic seed management — all randomness flows from here."""

from __future__ import annotations

import hashlib
import random


class SeedManager:
    """Produces deterministic child RNGs via SHA-256 derivation.

    Every component that needs randomness calls `child_rng("component_name")`
    and gets a `random.Random` instance seeded deterministically from the
    master seed + component name.
    """

    def __init__(self, seed: int | None = None):
        if seed is None:
            seed = random.randint(0, 2**63 - 1)
        self.master_seed = seed
        self._rng = random.Random(seed)

    def child_rng(self, name: str) -> random.Random:
        """Derive a deterministic child RNG for the named component."""
        digest = hashlib.sha256(f"{self.master_seed}:{name}".encode()).digest()
        child_seed = int.from_bytes(digest[:8], "big")
        return random.Random(child_seed)

    def next_int(self, a: int = 0, b: int = 2**31 - 1) -> int:
        """Get next random int from the master RNG."""
        return self._rng.randint(a, b)
