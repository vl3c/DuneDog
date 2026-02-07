"""World rules engine — invariants and tendencies."""

from __future__ import annotations

import json
import random
from pathlib import Path

from dunedog.models.atoms import AtomCategory, AtomSource, StoryAtom
from dunedog.models.skeleton import StorySkeleton
from dunedog.models.validation import (
    Invariant,
    InvariantSeverity,
    Tendency,
    ValidationResult,
)


class WorldRulesEngine:
    """Loads world rules (invariants + tendencies) and validates skeletons."""

    def __init__(self) -> None:
        self._invariants: list[Invariant] = []
        self._tendencies: list[Tendency] = []
        self._load_rules()

    def _load_rules(self) -> None:
        """Load invariants.json and tendencies.json from data/."""
        data_dir = Path(__file__).resolve().parents[3] / "data"

        inv_path = data_dir / "invariants.json"
        if inv_path.exists():
            with open(inv_path, encoding="utf-8") as f:
                self._invariants = [Invariant.from_dict(d) for d in json.load(f)]

        tend_path = data_dir / "tendencies.json"
        if tend_path.exists():
            with open(tend_path, encoding="utf-8") as f:
                self._tendencies = [Tendency.from_dict(d) for d in json.load(f)]

    # ------------------------------------------------------------------
    # Invariant checking
    # ------------------------------------------------------------------

    def check_invariant(
        self, skeleton: StorySkeleton, invariant: Invariant
    ) -> str | None:
        """Check one invariant against a skeleton.

        Returns a violation message string, or ``None`` when the invariant
        holds.

        Dispatch by ``check_type``:

        * ``requires_atom`` — skeleton must contain an atom whose tags
          include ``parameters["requires_tag"]``.
        * ``forbids_combo`` — atoms carrying ``parameters["tag_a"]`` and
          ``parameters["tag_b"]`` must not coexist.
        * ``requires_beat`` — ``skeleton.beats`` must contain
          ``parameters["beat"]``.
        * ``conditional`` — if *any* atom carries ``parameters["if_tag"]``,
          then *some* atom must carry ``parameters["requires_tag"]``.
        """
        all_tags = {tag for atom in skeleton.atoms for tag in atom.tags}
        params = invariant.parameters
        ct = invariant.check_type

        if ct == "requires_atom":
            required = params.get("requires_tag", "")
            if required and required not in all_tags:
                return (
                    f"[{invariant.name}] Required tag '{required}' missing"
                )

        elif ct == "forbids_combo":
            tag_a = params.get("tag_a", "")
            tag_b = params.get("tag_b", "")
            if tag_a in all_tags and tag_b in all_tags:
                return (
                    f"[{invariant.name}] Forbidden combination: "
                    f"'{tag_a}' and '{tag_b}' coexist"
                )

        elif ct == "requires_beat":
            beat = params.get("beat", "")
            if beat and beat not in skeleton.beats:
                return (
                    f"[{invariant.name}] Required beat '{beat}' missing"
                )

        elif ct == "conditional":
            if_tag = params.get("if_tag", "")
            req_tag = params.get("requires_tag", "")
            if if_tag in all_tags and req_tag not in all_tags:
                return (
                    f"[{invariant.name}] Tag '{if_tag}' present but "
                    f"required tag '{req_tag}' missing"
                )

        return None

    # ------------------------------------------------------------------
    # Tendencies
    # ------------------------------------------------------------------

    def apply_tendencies(
        self, skeleton: StorySkeleton, rng: random.Random
    ) -> list[str]:
        """Apply each tendency probabilistically.

        Returns a list of names of the tendencies that actually fired.

        Effects:

        * ``add_tag`` — append ``parameters["tag"]`` to a random atom's
          tags list.
        * ``add_tension`` — append a new tension atom with
          ``parameters["tension"]`` as its name.
        * ``modify_tone`` — overwrite ``skeleton.tone`` with
          ``parameters["tone"]``.
        * ``add_beat`` — append ``parameters["beat"]`` to
          ``skeleton.beats``.
        """
        applied: list[str] = []

        for tendency in self._tendencies:
            if rng.random() >= tendency.probability:
                continue

            effect = tendency.effect
            params = tendency.parameters

            mutated = False

            if effect == "add_tag":
                tag = params.get("tag", "")
                if tag and skeleton.atoms:
                    atom = rng.choice(skeleton.atoms)
                    if tag not in atom.tags:
                        atom.tags.append(tag)
                        mutated = True

            elif effect == "add_tension":
                tension_name = params.get("tension", "")
                if tension_name:
                    skeleton.atoms.append(
                        StoryAtom(
                            name=tension_name,
                            category=AtomCategory.TENSION,
                            source=AtomSource.WILD_CARD,
                            tags=["tension", tension_name],
                        )
                    )
                    mutated = True

            elif effect == "modify_tone":
                tone = params.get("tone", "")
                if tone:
                    skeleton.tone = tone
                    mutated = True

            elif effect == "add_beat":
                beat = params.get("beat", "")
                if beat:
                    skeleton.beats.append(beat)
                    mutated = True

            if mutated:
                applied.append(tendency.name)

        return applied

    # ------------------------------------------------------------------
    # Full validation
    # ------------------------------------------------------------------

    def validate(
        self,
        skeleton: StorySkeleton,
        rng: random.Random | None = None,
    ) -> ValidationResult:
        """Validate a skeleton against all world rules.

        1. Check every invariant and record violations.
        2. If *rng* is provided, apply tendencies.
        """
        result = ValidationResult()

        for inv in self._invariants:
            msg = self.check_invariant(skeleton, inv)
            if msg is not None:
                result.add_violation(msg, inv.severity)

        if rng is not None:
            result.tendencies_applied = self.apply_tendencies(skeleton, rng)

        return result

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def invariants(self) -> list[Invariant]:
        return list(self._invariants)

    @property
    def tendencies(self) -> list[Tendency]:
        return list(self._tendencies)
