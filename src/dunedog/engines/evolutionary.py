"""Evolutionary story engine — breeds and mutates story skeletons."""

from __future__ import annotations

import json
import random
from pathlib import Path

from dunedog.models.skeleton import StorySkeleton, EvolutionResult, GenerationStats
from dunedog.models.config import EvolutionConfig
from dunedog.models.atoms import StoryAtom, AtomCategory, AtomSource
from dunedog.engines.constraint_solver import WorldConstraintSolver
from dunedog.catalogues.loader import AtomCatalogue

_TONES = [
    "dark", "luminous", "tense", "enigmatic", "dreamlike",
    "eerie", "melancholic", "tender", "hushed", "transcendent",
    "paranoid", "disorienting", "inverted", "convergent",
]

_BEAT_TYPES = [
    "OPENING", "WORLD_BUILDING", "CHARACTER_INTRO", "INCITING_INCIDENT",
    "RISING_ACTION", "COMPLICATION", "MIDPOINT", "ESCALATION",
    "CRISIS", "CLIMAX", "FALLING_ACTION", "RESOLUTION", "DENOUEMENT",
]


class StoryEvolutionEngine:
    """Evolutionary optimizer for story skeletons.

    Breeds a population of skeletons over multiple generations using
    selection, crossover, mutation, and occasional wild-card injection.
    """

    def __init__(
        self,
        constraint_solver: WorldConstraintSolver | None = None,
        catalogue: AtomCatalogue | None = None,
    ) -> None:
        self._solver = constraint_solver or WorldConstraintSolver()
        self._catalogue = catalogue or AtomCatalogue.load()
        self._wild_cards = self._load_wild_cards()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_wild_cards(self) -> list[dict]:
        """Load wild_cards.json from data/."""
        data_dir = Path(__file__).resolve().parents[3] / "data"
        wc_path = data_dir / "wild_cards.json"
        if wc_path.exists():
            with open(wc_path, encoding="utf-8") as f:
                return json.load(f)
        return []

    # ------------------------------------------------------------------
    # Main evolutionary loop
    # ------------------------------------------------------------------

    def evolve(
        self,
        population: list[StorySkeleton],
        config: EvolutionConfig,
        rng: random.Random,
    ) -> EvolutionResult:
        """Run evolutionary optimization.

        For each generation:
        1. Score all skeletons (calculate_coherence_score via constraint solver)
        2. Select parents
        3. Create offspring via crossover
        4. Mutate offspring
        5. Occasionally inject wild cards
        6. Replace worst individuals with offspring
        7. Track best fitness per generation

        Returns EvolutionResult with best skeleton, final population, fitness history.
        """
        fitness_history: list[float] = []

        for gen in range(config.generations):
            # Score population
            for skeleton in population:
                self._solver.score_and_update(skeleton)

            population.sort(key=lambda s: s.coherence_score, reverse=True)
            fitness_history.append(population[0].coherence_score)

            # Create offspring
            offspring: list[StorySkeleton] = []
            n_offspring = len(population) // 2

            for _ in range(n_offspring):
                parent_a, parent_b = self.select_parents(
                    population, "tournament", rng,
                    tournament_size=config.tournament_size,
                )
                if rng.random() < config.crossover_rate:
                    child_a, child_b = self.crossover(parent_a, parent_b, rng)
                else:
                    child_a = self._copy_skeleton(parent_a)
                    child_b = self._copy_skeleton(parent_b)

                child_a = self.mutate(child_a, rng, config.mutation_rate)
                child_b = self.mutate(child_b, rng, config.mutation_rate)

                if rng.random() < config.wild_card_rate and self._wild_cards:
                    child_a = self.inject_wild_card(child_a, rng)

                offspring.extend([child_a, child_b])

            # Replace worst with offspring
            population = population[: len(population) - len(offspring)] + offspring

            # Update generation number
            for s in population:
                s.stats.generation = gen + 1

        # Final scoring
        for skeleton in population:
            self._solver.score_and_update(skeleton)
        population.sort(key=lambda s: s.coherence_score, reverse=True)

        return EvolutionResult(
            best_skeleton=population[0] if population else None,
            population=population,
            generations_run=config.generations,
            fitness_history=fitness_history,
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_parents(
        self,
        population: list[StorySkeleton],
        method: str,
        rng: random.Random,
        tournament_size: int = 3,
    ) -> tuple[StorySkeleton, StorySkeleton]:
        """Select two parents.

        Methods:
        - "tournament": pick best of k random individuals (repeat for 2nd parent)
        - "roulette": probability proportional to fitness score
        - "rank": probability proportional to rank position
        """
        if method == "tournament":
            parent_a = self._tournament_select(population, tournament_size, rng)
            parent_b = self._tournament_select(population, tournament_size, rng)
        elif method == "roulette":
            parent_a = self._roulette_select(population, rng)
            parent_b = self._roulette_select(population, rng)
        elif method == "rank":
            parent_a = self._rank_select(population, rng)
            parent_b = self._rank_select(population, rng)
        else:
            raise ValueError(f"Unknown selection method: {method!r}")

        return parent_a, parent_b

    def _tournament_select(
        self,
        population: list[StorySkeleton],
        k: int,
        rng: random.Random,
    ) -> StorySkeleton:
        """Pick the best of *k* randomly chosen individuals."""
        k = min(k, len(population))
        contestants = rng.sample(population, k)
        return max(contestants, key=lambda s: s.coherence_score)

    def _roulette_select(
        self,
        population: list[StorySkeleton],
        rng: random.Random,
    ) -> StorySkeleton:
        """Select with probability proportional to fitness score."""
        # Shift scores so all are positive (minimum weight 0.01)
        min_score = min(s.coherence_score for s in population)
        shift = abs(min_score) + 0.01 if min_score < 0 else 0.01
        weights = [s.coherence_score + shift for s in population]
        return rng.choices(population, weights=weights, k=1)[0]

    def _rank_select(
        self,
        population: list[StorySkeleton],
        rng: random.Random,
    ) -> StorySkeleton:
        """Select with probability proportional to rank (best = highest weight)."""
        # Population should already be sorted best-first, but sort to be safe
        ranked = sorted(population, key=lambda s: s.coherence_score, reverse=True)
        n = len(ranked)
        # Rank weights: best gets n, worst gets 1
        weights = [n - i for i in range(n)]
        return rng.choices(ranked, weights=weights, k=1)[0]

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    def crossover(
        self,
        parent_a: StorySkeleton,
        parent_b: StorySkeleton,
        rng: random.Random,
    ) -> tuple[StorySkeleton, StorySkeleton]:
        """Create two children by combining parents.

        - Swap random subset of atoms between parents
        - Splice beat sequences at a random crossover point
        - Union theme tags, pick tone from one parent
        """
        child_a = self._copy_skeleton(parent_a)
        child_b = self._copy_skeleton(parent_b)

        # --- Atom crossover: swap a random subset ---
        atoms_a = list(child_a.atoms)
        atoms_b = list(child_b.atoms)

        if atoms_a and atoms_b:
            # Pick a random number of atoms to swap (at least 1)
            n_swap = rng.randint(1, max(1, min(len(atoms_a), len(atoms_b))))
            indices_a = rng.sample(range(len(atoms_a)), min(n_swap, len(atoms_a)))
            indices_b = rng.sample(range(len(atoms_b)), min(n_swap, len(atoms_b)))

            for ia, ib in zip(indices_a, indices_b):
                atoms_a[ia], atoms_b[ib] = atoms_b[ib], atoms_a[ia]

            child_a.atoms = atoms_a
            child_b.atoms = atoms_b

        # --- Beat crossover: single-point splice ---
        beats_a = list(child_a.beats)
        beats_b = list(child_b.beats)

        if beats_a and beats_b:
            cut_a = rng.randint(1, max(1, len(beats_a) - 1))
            cut_b = rng.randint(1, max(1, len(beats_b) - 1))
            new_beats_a = beats_a[:cut_a] + beats_b[cut_b:]
            new_beats_b = beats_b[:cut_b] + beats_a[cut_a:]
            child_a.beats = new_beats_a
            child_b.beats = new_beats_b

        # --- Theme tags: union ---
        all_tags = list(dict.fromkeys(child_a.theme_tags + child_b.theme_tags))
        child_a.theme_tags = list(all_tags)
        child_b.theme_tags = list(all_tags)

        # --- Tone: each child takes tone from one parent ---
        if rng.random() < 0.5:
            child_a.tone, child_b.tone = child_b.tone, child_a.tone

        # Update spread_positions to reflect new atoms
        child_a.spread_positions = self._rebuild_spread_positions(child_a)
        child_b.spread_positions = self._rebuild_spread_positions(child_b)

        return child_a, child_b

    @staticmethod
    def _rebuild_spread_positions(skeleton: StorySkeleton) -> dict[str, str]:
        """Rebuild spread_positions from current atoms.

        Keeps existing position names but maps them to atoms that are
        still present. Drops positions whose atoms were swapped out.
        """
        atom_names = {a.name for a in skeleton.atoms}
        return {
            pos: name
            for pos, name in skeleton.spread_positions.items()
            if name in atom_names
        }

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def mutate(
        self,
        skeleton: StorySkeleton,
        rng: random.Random,
        rate: float = 0.1,
    ) -> StorySkeleton:
        """Mutate a skeleton with given probability per element.

        Possible mutations:
        - Swap a random atom for a different one from catalogue
        - Reroll a random spread position value
        - Shuffle a segment of beats
        - Change tone randomly
        """
        # Mutate atoms: each atom has `rate` chance of being replaced
        if skeleton.atoms:
            for i in range(len(skeleton.atoms)):
                if rng.random() < rate:
                    old = skeleton.atoms[i]
                    replacements = self._catalogue.get_by_category(old.category)
                    if replacements:
                        new_atom = rng.choice(replacements)
                        # Copy and mark as evolved
                        skeleton.atoms[i] = StoryAtom(
                            name=new_atom.name,
                            category=new_atom.category,
                            source=AtomSource.EVOLVED,
                            tags=list(new_atom.tags),
                            rarity=new_atom.rarity,
                            metadata=dict(new_atom.metadata),
                        )

        # Mutate spread positions: each position has `rate` chance of reroll
        if skeleton.spread_positions and skeleton.atoms:
            positions = list(skeleton.spread_positions.keys())
            for pos in positions:
                if rng.random() < rate:
                    atom = rng.choice(skeleton.atoms)
                    skeleton.spread_positions[pos] = atom.name

        # Mutate beats: `rate` chance of shuffling a segment
        if len(skeleton.beats) > 2 and rng.random() < rate:
            start = rng.randint(0, len(skeleton.beats) - 2)
            end = rng.randint(start + 1, len(skeleton.beats))
            segment = skeleton.beats[start:end]
            rng.shuffle(segment)
            skeleton.beats[start:end] = segment

        # Mutate tone: `rate` chance of changing
        if rng.random() < rate:
            skeleton.tone = rng.choice(_TONES)

        return skeleton

    # ------------------------------------------------------------------
    # Wild card injection
    # ------------------------------------------------------------------

    def inject_wild_card(
        self,
        skeleton: StorySkeleton,
        rng: random.Random,
    ) -> StorySkeleton:
        """Apply a random wild card mutation.

        Wild card effects: add_atom, modify_atom, change_tone, add_beat, swap_atom
        """
        card = rng.choice(self._wild_cards)
        effect = card.get("effect_type", "")
        params = card.get("parameters", {})

        if effect == "add_atom":
            atom_data = params.get("atom", {})
            if atom_data:
                new_atom = StoryAtom(
                    name=atom_data.get("name", "wild unknown"),
                    category=AtomCategory(atom_data.get("category", "object")),
                    source=AtomSource.WILD_CARD,
                    tags=atom_data.get("tags", []),
                    rarity=0.9,
                    metadata={"wild_card": card.get("name", "")},
                )
                skeleton.atoms.append(new_atom)

        elif effect == "modify_atom":
            target_cat = params.get("target_category")
            add_tags = params.get("add_tags", [])
            if target_cat:
                cat = AtomCategory(target_cat)
                targets = [a for a in skeleton.atoms if a.category == cat]
                if targets:
                    chosen = rng.choice(targets)
                    for tag in add_tags:
                        if tag not in chosen.tags:
                            chosen.tags.append(tag)

        elif effect == "change_tone":
            new_tone = params.get("tone")
            if new_tone:
                skeleton.tone = new_tone

        elif effect == "add_beat":
            beat_type = params.get("beat_type", "revelation")
            source = params.get("source", "")
            beat_text = f"{beat_type}"
            if source:
                beat_text = f"{beat_type} ({source})"
            # Insert at a random position in the beat sequence
            if skeleton.beats:
                idx = rng.randint(0, len(skeleton.beats))
                skeleton.beats.insert(idx, beat_text)
            else:
                skeleton.beats.append(beat_text)

        elif effect == "swap_atom":
            swap_cat = params.get("swap_category")
            count = params.get("count", 2)
            if swap_cat:
                cat = AtomCategory(swap_cat)
                targets = [
                    i for i, a in enumerate(skeleton.atoms) if a.category == cat
                ]
                if len(targets) >= count:
                    chosen_indices = rng.sample(targets, count)
                    # Rotate the chosen atoms
                    atoms_to_rotate = [skeleton.atoms[i] for i in chosen_indices]
                    rotated = atoms_to_rotate[1:] + atoms_to_rotate[:1]
                    for idx, new_atom in zip(chosen_indices, rotated):
                        skeleton.atoms[idx] = new_atom

        return skeleton

    # ------------------------------------------------------------------
    # Novelty scoring
    # ------------------------------------------------------------------

    def calculate_novelty(
        self,
        skeleton: StorySkeleton,
        population: list[StorySkeleton],
    ) -> float:
        """Score novelty 0.0-1.0 based on unique atom combinations.

        Skeletons with atom sets that differ most from the population
        average get higher scores.
        """
        if not population:
            return 1.0

        target_names = frozenset(a.name for a in skeleton.atoms)
        if not target_names:
            return 0.0

        # Calculate Jaccard distance to each other skeleton
        distances: list[float] = []
        for other in population:
            other_names = frozenset(a.name for a in other.atoms)
            if not other_names and not target_names:
                distances.append(0.0)
                continue
            union = target_names | other_names
            intersection = target_names & other_names
            if union:
                jaccard_dist = 1.0 - len(intersection) / len(union)
            else:
                jaccard_dist = 0.0
            distances.append(jaccard_dist)

        # Average distance = novelty
        avg_distance = sum(distances) / len(distances) if distances else 0.0
        return max(0.0, min(1.0, avg_distance))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _copy_skeleton(self, skeleton: StorySkeleton) -> StorySkeleton:
        """Deep copy a skeleton for modification."""
        return StorySkeleton.from_dict(skeleton.to_dict())
