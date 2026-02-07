"""Tests for generative engines: tarot spread, markov chains, constraint solver, evolutionary."""

import random

import pytest

from dunedog.engines.tarot_spread import TarotSpreadEngine
from dunedog.engines.markov_chains import NarrativeMarkovChain, VALID_ENDINGS
from dunedog.engines.constraint_solver import WorldConstraintSolver
from dunedog.engines.evolutionary import StoryEvolutionEngine
from dunedog.world_rules.engine import WorldRulesEngine
from dunedog.models.skeleton import StorySkeleton, GenerationStats, EvolutionResult
from dunedog.models.config import EvolutionConfig
from dunedog.models.validation import ValidationResult


# ------------------------------------------------------------------ #
# TarotSpreadEngine
# ------------------------------------------------------------------ #


class TestTarotSpreadEngine:
    """Tests for TarotSpreadEngine."""

    def test_draw_assigns_all_positions(self, sample_atoms, rng, catalogue):
        """draw() should return a dict whose keys match the spread's positions."""
        engine = TarotSpreadEngine(catalogue=catalogue)
        spread_type = engine.spread_types[0]
        layout = engine.draw(spread_type, sample_atoms, rng)

        assert isinstance(layout, dict)
        assert len(layout) > 0
        # Every key should be a string position name
        for key in layout:
            assert isinstance(key, str)

    def test_generate_skeleton_returns_valid_skeleton(self, sample_atoms, rng, catalogue):
        """generate_skeleton() should produce a StorySkeleton with atoms and beats."""
        engine = TarotSpreadEngine(catalogue=catalogue)
        spread_type = engine.spread_types[0]
        skeleton = engine.generate_skeleton(spread_type, sample_atoms, rng)

        assert isinstance(skeleton, StorySkeleton)
        assert len(skeleton.atoms) > 0
        assert len(skeleton.beats) > 0
        assert skeleton.tone != ""
        assert skeleton.stats.engine == "tarot_spread"


# ------------------------------------------------------------------ #
# NarrativeMarkovChain
# ------------------------------------------------------------------ #


class TestNarrativeMarkovChain:
    """Tests for NarrativeMarkovChain."""

    def test_sequence_starts_with_opening(self, rng):
        """Generated sequence must start with OPENING."""
        chain = NarrativeMarkovChain()
        seq = chain.generate_sequence(rng)
        assert seq[0] == "OPENING"

    def test_sequence_ends_with_valid_ending(self, rng):
        """Generated sequence must end with RESOLUTION or DENOUEMENT."""
        chain = NarrativeMarkovChain()
        seq = chain.generate_sequence(rng)
        assert seq[-1] in VALID_ENDINGS

    def test_sequence_length_within_range(self, rng):
        """Sequence length should be within min_beats..max_beats+1."""
        chain = NarrativeMarkovChain()
        min_b, max_b = 5, 12
        seq = chain.generate_sequence(rng, min_beats=min_b, max_beats=max_b)
        # +1 because the forced ending append can exceed max_beats by 1
        assert min_b <= len(seq) <= max_b + 1

    def test_no_immediate_beat_repetition(self, rng):
        """No two consecutive beats should be the same."""
        chain = NarrativeMarkovChain()
        seq = chain.generate_sequence(rng)
        for i in range(len(seq) - 1):
            assert seq[i] != seq[i + 1], f"Immediate repetition at index {i}: {seq[i]}"


# ------------------------------------------------------------------ #
# WorldRulesEngine
# ------------------------------------------------------------------ #


class TestWorldRulesEngine:
    """Tests for WorldRulesEngine."""

    def test_validate_returns_validation_result(self, sample_atoms, rng):
        """validate() should return a ValidationResult."""
        engine = WorldRulesEngine()
        skeleton = StorySkeleton(
            atoms=sample_atoms,
            beats=["OPENING", "RISING_ACTION", "CLIMAX", "RESOLUTION"],
            tone="enigmatic",
        )
        result = engine.validate(skeleton, rng)

        assert isinstance(result, ValidationResult)
        assert isinstance(result.hard_violations, list)
        assert isinstance(result.soft_violations, list)


# ------------------------------------------------------------------ #
# WorldConstraintSolver
# ------------------------------------------------------------------ #


class TestWorldConstraintSolver:
    """Tests for WorldConstraintSolver."""

    def test_calculate_coherence_score_returns_float_in_range(self, sample_atoms, catalogue):
        """calculate_coherence_score() should return a float in [0, 1]."""
        solver = WorldConstraintSolver(catalogue=catalogue)
        skeleton = StorySkeleton(
            atoms=sample_atoms,
            beats=["OPENING", "RISING_ACTION", "CLIMAX", "RESOLUTION"],
            tone="dark",
            theme_tags=["journey", "mystery"],
        )
        score = solver.calculate_coherence_score(skeleton)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_and_update_sets_stats(self, sample_atoms, catalogue):
        """score_and_update() should set coherence_score in skeleton.stats."""
        solver = WorldConstraintSolver(catalogue=catalogue)
        skeleton = StorySkeleton(
            atoms=sample_atoms,
            beats=["OPENING", "RISING_ACTION", "CLIMAX", "RESOLUTION"],
            tone="tense",
        )
        solver.score_and_update(skeleton)
        assert skeleton.stats.coherence_score >= 0.0


# ------------------------------------------------------------------ #
# StoryEvolutionEngine
# ------------------------------------------------------------------ #


class TestStoryEvolutionEngine:
    """Tests for StoryEvolutionEngine."""

    def test_evolve_returns_evolution_result(self, sample_atoms, rng, catalogue):
        """evolve() should return an EvolutionResult with a populated population."""
        solver = WorldConstraintSolver(catalogue=catalogue)
        engine = StoryEvolutionEngine(constraint_solver=solver, catalogue=catalogue)

        # Create a small population
        population = []
        for i in range(4):
            sk = StorySkeleton(
                atoms=list(sample_atoms),
                beats=["OPENING", "RISING_ACTION", "CLIMAX", "RESOLUTION"],
                tone="enigmatic",
                stats=GenerationStats(engine="test"),
            )
            population.append(sk)

        config = EvolutionConfig(
            enabled=True,
            generations=2,
            population_size=4,
            mutation_rate=0.1,
            crossover_rate=0.7,
            tournament_size=2,
            wild_card_rate=0.0,
        )

        result = engine.evolve(population, config, rng)

        assert isinstance(result, EvolutionResult)
        assert len(result.population) > 0
        assert result.best_skeleton is not None

    def test_evolution_fitness_history_is_nonempty(self, sample_atoms, rng, catalogue):
        """After evolve(), fitness_history should have one entry per generation."""
        solver = WorldConstraintSolver(catalogue=catalogue)
        engine = StoryEvolutionEngine(constraint_solver=solver, catalogue=catalogue)

        population = []
        for _ in range(4):
            sk = StorySkeleton(
                atoms=list(sample_atoms),
                beats=["OPENING", "MIDPOINT", "RESOLUTION"],
                tone="dark",
                stats=GenerationStats(engine="test"),
            )
            population.append(sk)

        generations = 3
        config = EvolutionConfig(
            enabled=True,
            generations=generations,
            population_size=4,
            mutation_rate=0.1,
            crossover_rate=0.7,
            tournament_size=2,
            wild_card_rate=0.0,
        )

        result = engine.evolve(population, config, rng)

        assert len(result.fitness_history) == generations
        assert all(isinstance(f, float) for f in result.fitness_history)
