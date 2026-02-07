"""Tarot spread engine — assigns atoms to narrative positions."""

from __future__ import annotations

import json
import random
from pathlib import Path

from dunedog.models.atoms import AtomCategory, StoryAtom
from dunedog.models.skeleton import GenerationStats, StorySkeleton
from dunedog.catalogues.loader import AtomCatalogue


class TarotSpreadEngine:
    """Map story atoms onto narrative spread positions, weighted by category
    preference and inter-atom affinity."""

    def __init__(self, catalogue: AtomCatalogue | None = None):
        self._catalogue = catalogue or AtomCatalogue.load()
        self._spreads = self._load_spreads()

    def _load_spreads(self) -> dict:
        """Load tarot_positions.json from data/."""
        data_dir = Path(__file__).resolve().parents[3] / "data"
        with open(data_dir / "tarot_positions.json") as f:
            return json.load(f)

    @property
    def spread_types(self) -> list[str]:
        return list(self._spreads.keys())

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(
        self,
        spread_type: str,
        atoms: list[StoryAtom],
        rng: random.Random,
    ) -> dict[str, StoryAtom]:
        """Fill spread positions with atoms.

        For each position:
        1. Filter atoms matching preferred_categories for this position
        2. Among matches, weight by affinity to already-placed atoms
        3. If no category match, pick from remaining unused atoms
        4. If not enough atoms, sample from catalogue

        Returns dict mapping position name -> StoryAtom.
        """
        spread = self._spreads[spread_type]
        positions = spread["positions"]
        placed: dict[str, StoryAtom] = {}
        used_names: set[str] = set()

        for pos in positions:
            pos_name: str = pos["name"]
            preferred = {AtomCategory(c) for c in pos.get("preferred_categories", [])}

            # Build pool of unused atoms
            available = [a for a in atoms if a.name not in used_names]

            # Step 1: filter by preferred categories
            matched = [a for a in available if a.category in preferred] if preferred else []

            # Step 2: weight by affinity to already-placed atoms
            chosen = self._pick_by_affinity(matched, placed, rng)

            # Step 3: fallback — pick from any remaining unused atom
            if chosen is None and available:
                chosen = self._pick_by_affinity(available, placed, rng)

            # Step 4: fallback — sample from catalogue
            if chosen is None:
                for cat in preferred or set(AtomCategory):
                    sampled = self._catalogue.sample_weighted(cat, rng, n=1)
                    if sampled:
                        chosen = sampled[0]
                        break

            if chosen is not None:
                placed[pos_name] = chosen
                used_names.add(chosen.name)

        return placed

    def _pick_by_affinity(
        self,
        candidates: list[StoryAtom],
        placed: dict[str, StoryAtom],
        rng: random.Random,
    ) -> StoryAtom | None:
        """Select one atom from *candidates* weighted by total affinity to
        already-placed atoms.  Falls back to uniform random when there are no
        affinity signals."""
        if not candidates:
            return None

        if not placed:
            return rng.choice(candidates)

        placed_names = [a.name for a in placed.values()]
        weights: list[float] = []
        for cand in candidates:
            aff_sum = sum(
                self._catalogue.get_affinity(cand.name, pn) for pn in placed_names
            )
            # Shift so all weights are positive; base weight of 1.0
            weights.append(1.0 + aff_sum)

        # Clamp negatives
        weights = [max(w, 0.01) for w in weights]
        return rng.choices(candidates, weights=weights, k=1)[0]

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    def interpret_position(
        self, position_name: str, atom: StoryAtom, spread_type: str
    ) -> str:
        """Generate a narrative fragment for this position+atom combo.

        Uses position description + atom name to create a brief narrative beat.
        """
        spread = self._spreads[spread_type]
        pos_desc = ""
        for pos in spread["positions"]:
            if pos["name"] == position_name:
                pos_desc = pos["description"]
                break

        # Build a concise narrative fragment
        atom_phrase = atom.name
        # Capitalise first letter for sentence form
        if atom_phrase:
            atom_phrase = atom_phrase[0].upper() + atom_phrase[1:]

        return f"{atom_phrase} — {pos_desc.lower()}."

    # ------------------------------------------------------------------
    # Full skeleton generation
    # ------------------------------------------------------------------

    def generate_skeleton(
        self,
        spread_type: str,
        atoms: list[StoryAtom],
        rng: random.Random,
    ) -> StorySkeleton:
        """Generate a complete skeleton from a tarot spread.

        1. Draw atoms into positions
        2. Interpret each position
        3. Derive theme_tags from atom tags
        4. Set tone from overall atom mood
        """
        layout = self.draw(spread_type, atoms, rng)

        # Collect beats as interpreted narrative fragments
        beats: list[str] = []
        spread_positions: dict[str, str] = {}
        placed_atoms: list[StoryAtom] = []

        spread = self._spreads[spread_type]
        for pos in spread["positions"]:
            pos_name = pos["name"]
            atom = layout.get(pos_name)
            if atom is None:
                continue
            placed_atoms.append(atom)
            spread_positions[pos_name] = atom.name
            beats.append(self.interpret_position(pos_name, atom, spread_type))

        # Derive theme_tags: union of all placed atom tags, deduplicated
        seen_tags: set[str] = set()
        theme_tags: list[str] = []
        for atom in placed_atoms:
            for tag in atom.tags:
                if tag not in seen_tags:
                    seen_tags.add(tag)
                    theme_tags.append(tag)

        # Derive tone from atom metadata or tags heuristic
        tone = self._derive_tone(placed_atoms)

        stats = GenerationStats(
            engine="tarot_spread",
            spread_type=spread_type,
            beat_count=len(beats),
        )

        return StorySkeleton(
            atoms=placed_atoms,
            beats=beats,
            spread_positions=spread_positions,
            theme_tags=theme_tags,
            tone=tone,
            stats=stats,
        )

    @staticmethod
    def _derive_tone(atoms: list[StoryAtom]) -> str:
        """Heuristic tone derivation from atom tags and categories."""
        tag_set: set[str] = set()
        for a in atoms:
            tag_set.update(a.tags)

        dark_signals = {"dark", "horror", "dread", "gothic", "fear", "death", "shadow"}
        light_signals = {"hope", "wonder", "joy", "light", "love", "comedy", "warmth"}
        tense_signals = {"tension", "conflict", "danger", "war", "suspense", "mystery"}

        dark_count = len(tag_set & dark_signals)
        light_count = len(tag_set & light_signals)
        tense_count = len(tag_set & tense_signals)

        best = max(
            [("dark", dark_count), ("luminous", light_count), ("tense", tense_count)],
            key=lambda t: t[1],
        )
        if best[1] == 0:
            return "enigmatic"
        return best[0]
