"""Narrative Markov chains — generates beat sequences."""

from __future__ import annotations

import json
import random
from pathlib import Path


BEAT_TYPES = [
    "OPENING",
    "WORLD_BUILDING",
    "CHARACTER_INTRO",
    "INCITING_INCIDENT",
    "RISING_ACTION",
    "COMPLICATION",
    "MIDPOINT",
    "ESCALATION",
    "CRISIS",
    "CLIMAX",
    "FALLING_ACTION",
    "RESOLUTION",
    "DENOUEMENT",
]

VALID_ENDINGS = {"RESOLUTION", "DENOUEMENT"}


class NarrativeMarkovChain:
    """Generates narrative beat sequences via Markov-chain transitions,
    optionally modified by story-atom signals."""

    def __init__(self) -> None:
        self._transitions = self._load_transitions()

    def _load_transitions(self) -> dict[str, dict[str, float]]:
        """Load beat_transitions.json from data/."""
        data_dir = Path(__file__).resolve().parents[3] / "data"
        with open(data_dir / "beat_transitions.json") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Sequence generation
    # ------------------------------------------------------------------

    def generate_sequence(
        self,
        rng: random.Random,
        min_beats: int = 5,
        max_beats: int = 12,
        atom_modifiers: dict[str, float] | None = None,
    ) -> list[str]:
        """Generate a beat sequence using Markov transitions.

        1. Start at OPENING
        2. At each step, get transition probabilities for current beat
        3. Apply atom_modifiers (additive adjustments to specific beat probs)
        4. Ensure no immediate repetition (zero out current beat)
        5. Normalize and sample next beat
        6. Stop when: length >= min_beats AND current beat is a valid ending,
           OR length >= max_beats (force RESOLUTION or DENOUEMENT)

        Returns list of beat type strings.
        """
        sequence: list[str] = ["OPENING"]
        current = "OPENING"

        while True:
            # Check termination conditions
            if len(sequence) >= min_beats and current in VALID_ENDINGS:
                break
            if len(sequence) >= max_beats:
                # Force a valid ending if we aren't already at one
                if current not in VALID_ENDINGS:
                    sequence.append(rng.choice(["RESOLUTION", "DENOUEMENT"]))
                break

            # Get transition probabilities
            probs = dict(self._transitions.get(current, {}))

            # If no transitions defined, fall back to uniform over all beats
            if not probs:
                probs = {b: 1.0 for b in BEAT_TYPES}

            # Apply atom modifiers (additive)
            if atom_modifiers:
                for beat, mod in atom_modifiers.items():
                    probs[beat] = probs.get(beat, 0.0) + mod

            # Prevent immediate repetition
            probs[current] = 0.0

            # Pick next beat
            current = self._weighted_choice(probs, rng)
            sequence.append(current)

        return sequence

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_choice(options: dict[str, float], rng: random.Random) -> str:
        """Pick from weighted options dict."""
        items = list(options.items())
        weights = [max(0.0, w) for _, w in items]  # clamp negatives
        total = sum(weights)
        if total == 0:
            return rng.choice([k for k, _ in items])
        return rng.choices([k for k, _ in items], weights=weights, k=1)[0]
