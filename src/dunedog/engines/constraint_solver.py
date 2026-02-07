"""World constraint solver — validates and scores skeletons."""

from __future__ import annotations

import json
import random
from pathlib import Path

from dunedog.catalogues.loader import AtomCatalogue
from dunedog.models.skeleton import StorySkeleton
from dunedog.models.validation import ValidationResult
from dunedog.world_rules.engine import WorldRulesEngine

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


class WorldConstraintSolver:
    """Validates skeletons against world rules and scores coherence."""

    def __init__(
        self,
        rules_engine: WorldRulesEngine | None = None,
        catalogue: AtomCatalogue | None = None,
    ) -> None:
        self._rules = rules_engine or WorldRulesEngine()
        self._catalogue = catalogue or AtomCatalogue.load()
        self._beat_transitions: dict[str, dict[str, float]] = {}
        self._load_transitions()

    def _load_transitions(self) -> None:
        """Load beat_transitions.json from data/."""
        path = _DATA_DIR / "beat_transitions.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                self._beat_transitions = json.load(f)

    # ------------------------------------------------------------------
    # Delegation to rules engine
    # ------------------------------------------------------------------

    def validate(
        self,
        skeleton: StorySkeleton,
        rng: random.Random | None = None,
    ) -> ValidationResult:
        """Validate skeleton against all world rules."""
        return self._rules.validate(skeleton, rng)

    def apply_tendencies(
        self, skeleton: StorySkeleton, rng: random.Random
    ) -> list[str]:
        """Apply tendencies to skeleton."""
        return self._rules.apply_tendencies(skeleton, rng)

    # ------------------------------------------------------------------
    # Coherence scoring
    # ------------------------------------------------------------------

    def calculate_coherence_score(self, skeleton: StorySkeleton) -> float:
        """Calculate a coherence score in the range 0.0 -- 1.0.

        The score is assembled from four factors:

        1. **Affinity score** (0 -- 0.3): average pairwise affinity
           between atoms, looked up in the catalogue, normalized to the
           0 -- 0.3 band.

        2. **Violation penalty**: -0.2 per soft violation, -0.5 per hard
           violation (validation runs *without* tendencies).

        3. **Beat flow score** (0 -- 0.3): how smoothly the beat
           sequence follows high-probability transitions from
           ``beat_transitions.json``.

        4. **Thematic consistency** (0 -- 0.4): proportion of atom pairs
           sharing at least one tag.

        The final value is clamped to [0.0, 1.0].
        """
        affinity = self._affinity_score(skeleton)
        penalty = self._violation_penalty(skeleton)
        beat_flow = self._beat_flow_score(skeleton)
        thematic = self._thematic_consistency(skeleton)

        raw = affinity + penalty + beat_flow + thematic
        return max(0.0, min(1.0, raw))

    def score_and_update(self, skeleton: StorySkeleton) -> StorySkeleton:
        """Calculate coherence score and store it in skeleton.stats."""
        score = self.calculate_coherence_score(skeleton)
        skeleton.stats.coherence_score = score
        return skeleton

    # ------------------------------------------------------------------
    # Internal scoring helpers
    # ------------------------------------------------------------------

    def _affinity_score(self, skeleton: StorySkeleton) -> float:
        """Average pairwise affinity, mapped to 0 -- 0.3."""
        atoms = skeleton.atoms
        if len(atoms) < 2:
            return 0.15  # neutral when there's nothing to compare

        total = 0.0
        count = 0
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                total += self._catalogue.get_affinity(
                    atoms[i].name, atoms[j].name
                )
                count += 1

        avg = total / count  # in [-1, 1]
        # Map [-1, 1] -> [0, 0.3]
        return 0.15 + 0.15 * avg

    def _violation_penalty(self, skeleton: StorySkeleton) -> float:
        """Negative penalty from invariant violations."""
        result = self._rules.validate(skeleton, rng=None)
        return (
            -0.5 * len(result.hard_violations)
            + -0.2 * len(result.soft_violations)
        )

    def _beat_flow_score(self, skeleton: StorySkeleton) -> float:
        """Score for smooth beat-to-beat transitions, mapped to 0 -- 0.3."""
        beats = skeleton.beats
        if len(beats) < 2:
            return 0.15

        total_prob = 0.0
        pairs = 0
        for i in range(len(beats) - 1):
            src = beats[i]
            dst = beats[i + 1]
            transitions = self._beat_transitions.get(src, {})
            total_prob += transitions.get(dst, 0.0)
            pairs += 1

        avg_prob = total_prob / pairs  # in [0, 1]
        return 0.3 * avg_prob

    def _thematic_consistency(self, skeleton: StorySkeleton) -> float:
        """Proportion of atom pairs sharing at least one tag, mapped to 0 -- 0.4."""
        atoms = skeleton.atoms
        if len(atoms) < 2:
            return 0.2

        shared = 0
        total = 0
        for i in range(len(atoms)):
            tags_i = set(atoms[i].tags)
            for j in range(i + 1, len(atoms)):
                tags_j = set(atoms[j].tags)
                if tags_i & tags_j:
                    shared += 1
                total += 1

        ratio = shared / total if total else 0.0
        return 0.4 * ratio
