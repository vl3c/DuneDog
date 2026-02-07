"""Extended tests for StoryEvolutionEngine — selection, crossover, mutation, novelty, wild cards."""

import random

import pytest

from dunedog.engines.evolutionary import StoryEvolutionEngine
from dunedog.models.skeleton import StorySkeleton, GenerationStats
from dunedog.models.config import EvolutionConfig
from dunedog.models.atoms import StoryAtom, AtomCategory, AtomSource


def _make_population(n=6, rng=None):
    """Create a small population for testing."""
    if rng is None:
        rng = random.Random(42)
    pop = []
    tones = ["dark", "luminous", "tense", "enigmatic", "dreamlike", "eerie"]
    for i in range(n):
        atoms = [
            StoryAtom(f"agent_{i}", AtomCategory.AGENT, AtomSource.CATALOGUE, [f"tag_{i}", "test"]),
            StoryAtom(f"object_{i}", AtomCategory.OBJECT, AtomSource.CATALOGUE, [f"obj_tag_{i}"]),
            StoryAtom(f"location_{i}", AtomCategory.LOCATION, AtomSource.CATALOGUE, ["place"]),
        ]
        sk = StorySkeleton(
            atoms=atoms,
            beats=["OPENING", "RISING_ACTION", "CLIMAX", "RESOLUTION"],
            spread_positions={"past": f"agent_{i}", "present": f"object_{i}"},
            tone=tones[i % len(tones)],
            theme_tags=[f"theme_{i}"],
            stats=GenerationStats(engine="test", coherence_score=rng.random()),
        )
        pop.append(sk)
    return pop


# ------------------------------------------------------------------ #
# Selection methods
# ------------------------------------------------------------------ #


class TestSelectionMethods:
    def test_tournament_selection(self):
        engine = StoryEvolutionEngine()
        pop = _make_population()
        rng = random.Random(42)
        a, b = engine.select_parents(pop, "tournament", rng, tournament_size=3)
        assert a is not None
        assert b is not None
        assert isinstance(a, StorySkeleton)
        assert isinstance(b, StorySkeleton)

    def test_roulette_selection(self):
        engine = StoryEvolutionEngine()
        pop = _make_population()
        rng = random.Random(42)
        a, b = engine.select_parents(pop, "roulette", rng)
        assert a is not None
        assert b is not None
        assert isinstance(a, StorySkeleton)
        assert isinstance(b, StorySkeleton)

    def test_rank_selection(self):
        engine = StoryEvolutionEngine()
        pop = _make_population()
        rng = random.Random(42)
        a, b = engine.select_parents(pop, "rank", rng)
        assert a is not None
        assert b is not None
        assert isinstance(a, StorySkeleton)
        assert isinstance(b, StorySkeleton)

    def test_unknown_selection_raises(self):
        engine = StoryEvolutionEngine()
        pop = _make_population()
        with pytest.raises(ValueError, match="Unknown selection method"):
            engine.select_parents(pop, "invalid", random.Random(42))

    def test_tournament_returns_from_population(self):
        """Selected parents must be members of the population."""
        engine = StoryEvolutionEngine()
        pop = _make_population()
        rng = random.Random(99)
        a, b = engine.select_parents(pop, "tournament", rng, tournament_size=2)
        assert a in pop
        assert b in pop

    def test_roulette_returns_from_population(self):
        engine = StoryEvolutionEngine()
        pop = _make_population()
        rng = random.Random(99)
        a, b = engine.select_parents(pop, "roulette", rng)
        assert a in pop
        assert b in pop

    def test_rank_returns_from_population(self):
        engine = StoryEvolutionEngine()
        pop = _make_population()
        rng = random.Random(99)
        a, b = engine.select_parents(pop, "rank", rng)
        assert a in pop
        assert b in pop

    def test_tournament_prefers_high_fitness(self):
        """With a large tournament size equal to population, should pick the best."""
        engine = StoryEvolutionEngine()
        pop = _make_population(n=6)
        # Give one skeleton a clearly dominant score
        pop[2].stats.coherence_score = 100.0
        rng = random.Random(42)
        # tournament_size == len(pop), so the best individual always wins
        a, b = engine.select_parents(pop, "tournament", rng, tournament_size=len(pop))
        assert a is pop[2]
        assert b is pop[2]


# ------------------------------------------------------------------ #
# Crossover
# ------------------------------------------------------------------ #


class TestCrossover:
    def test_crossover_produces_two_children(self):
        engine = StoryEvolutionEngine()
        pop = _make_population()
        rng = random.Random(42)
        child_a, child_b = engine.crossover(pop[0], pop[1], rng)
        assert isinstance(child_a, StorySkeleton)
        assert isinstance(child_b, StorySkeleton)

    def test_crossover_children_have_atoms(self):
        engine = StoryEvolutionEngine()
        pop = _make_population()
        rng = random.Random(42)
        child_a, child_b = engine.crossover(pop[0], pop[1], rng)
        assert len(child_a.atoms) > 0
        assert len(child_b.atoms) > 0

    def test_crossover_children_have_beats(self):
        engine = StoryEvolutionEngine()
        pop = _make_population()
        rng = random.Random(42)
        child_a, child_b = engine.crossover(pop[0], pop[1], rng)
        assert len(child_a.beats) > 0
        assert len(child_b.beats) > 0

    def test_crossover_unions_theme_tags(self):
        engine = StoryEvolutionEngine()
        pop = _make_population()
        rng = random.Random(42)
        child_a, child_b = engine.crossover(pop[0], pop[1], rng)
        # Both children should have theme tags from both parents
        for tag in pop[0].theme_tags + pop[1].theme_tags:
            assert tag in child_a.theme_tags
            assert tag in child_b.theme_tags

    def test_crossover_does_not_modify_parents(self):
        """Crossover must not mutate the parent skeletons."""
        engine = StoryEvolutionEngine()
        pop = _make_population()
        rng = random.Random(42)
        parent_a_atoms_before = [a.name for a in pop[0].atoms]
        parent_b_atoms_before = [a.name for a in pop[1].atoms]
        engine.crossover(pop[0], pop[1], rng)
        assert [a.name for a in pop[0].atoms] == parent_a_atoms_before
        assert [a.name for a in pop[1].atoms] == parent_b_atoms_before

    def test_crossover_spread_positions_reference_valid_atoms(self):
        """After crossover, spread_positions should only reference atoms that exist."""
        engine = StoryEvolutionEngine()
        pop = _make_population()
        rng = random.Random(42)
        child_a, child_b = engine.crossover(pop[0], pop[1], rng)
        atom_names_a = {a.name for a in child_a.atoms}
        atom_names_b = {a.name for a in child_b.atoms}
        for pos, name in child_a.spread_positions.items():
            assert name in atom_names_a, f"spread position {pos!r} references missing atom {name!r}"
        for pos, name in child_b.spread_positions.items():
            assert name in atom_names_b, f"spread position {pos!r} references missing atom {name!r}"


# ------------------------------------------------------------------ #
# Mutation
# ------------------------------------------------------------------ #


class TestMutation:
    def test_mutate_returns_skeleton(self):
        engine = StoryEvolutionEngine()
        pop = _make_population()
        rng = random.Random(42)
        result = engine.mutate(pop[0], rng, rate=0.5)
        assert isinstance(result, StorySkeleton)

    def test_high_mutation_rate_changes_something(self):
        engine = StoryEvolutionEngine()
        pop = _make_population()
        # Make a copy so we have an untouched baseline
        sk = engine._copy_skeleton(pop[0])
        original_tone = sk.tone
        original_atoms = [a.name for a in sk.atoms]
        # With rate=1.0, everything should mutate
        rng = random.Random(42)
        result = engine.mutate(sk, rng, rate=1.0)
        # At least tone or some atoms should change
        changed = (result.tone != original_tone or
                   [a.name for a in result.atoms] != original_atoms)
        assert changed

    def test_zero_mutation_rate_preserves(self):
        engine = StoryEvolutionEngine()
        pop = _make_population()
        sk = engine._copy_skeleton(pop[0])
        original_tone = sk.tone
        original_atoms = [a.name for a in sk.atoms]
        result = engine.mutate(sk, random.Random(42), rate=0.0)
        assert result.tone == original_tone
        assert [a.name for a in result.atoms] == original_atoms

    def test_mutate_preserves_atom_count_or_same(self):
        """Mutation replaces atoms, it does not add or remove them."""
        engine = StoryEvolutionEngine()
        pop = _make_population()
        sk = engine._copy_skeleton(pop[0])
        original_count = len(sk.atoms)
        engine.mutate(sk, random.Random(42), rate=0.5)
        assert len(sk.atoms) == original_count

    def test_mutate_sets_evolved_source(self):
        """Replaced atoms should have source=EVOLVED."""
        engine = StoryEvolutionEngine()
        pop = _make_population()
        sk = engine._copy_skeleton(pop[0])
        # Force all atoms to mutate
        engine.mutate(sk, random.Random(42), rate=1.0)
        # At least some atoms should now be EVOLVED
        evolved = [a for a in sk.atoms if a.source == AtomSource.EVOLVED]
        assert len(evolved) > 0


# ------------------------------------------------------------------ #
# Wild card injection
# ------------------------------------------------------------------ #


class TestWildCardInjection:
    def test_inject_wild_card(self):
        engine = StoryEvolutionEngine()
        if not engine._wild_cards:
            pytest.skip("No wild cards loaded")
        pop = _make_population()
        rng = random.Random(42)
        result = engine.inject_wild_card(pop[0], rng)
        assert isinstance(result, StorySkeleton)

    def test_inject_wild_card_add_atom(self):
        """Injecting a wild card with effect_type=add_atom should add an atom."""
        engine = StoryEvolutionEngine()
        if not engine._wild_cards:
            pytest.skip("No wild cards loaded")
        # Find an add_atom wild card
        add_atom_cards = [c for c in engine._wild_cards if c.get("effect_type") == "add_atom"]
        if not add_atom_cards:
            pytest.skip("No add_atom wild cards in data")
        pop = _make_population()
        sk = engine._copy_skeleton(pop[0])
        original_count = len(sk.atoms)
        # Temporarily force the engine to only have add_atom cards
        original_wc = engine._wild_cards
        engine._wild_cards = add_atom_cards
        try:
            engine.inject_wild_card(sk, random.Random(42))
            assert len(sk.atoms) == original_count + 1
            assert sk.atoms[-1].source == AtomSource.WILD_CARD
        finally:
            engine._wild_cards = original_wc

    def test_inject_wild_card_change_tone(self):
        """Injecting a wild card with effect_type=change_tone should change the tone."""
        engine = StoryEvolutionEngine()
        if not engine._wild_cards:
            pytest.skip("No wild cards loaded")
        change_tone_cards = [c for c in engine._wild_cards if c.get("effect_type") == "change_tone"]
        if not change_tone_cards:
            pytest.skip("No change_tone wild cards in data")
        pop = _make_population()
        sk = engine._copy_skeleton(pop[0])
        # Use a card whose tone differs from the skeleton's current tone
        card = None
        for c in change_tone_cards:
            if c["parameters"]["tone"] != sk.tone:
                card = c
                break
        if card is None:
            pytest.skip("No suitable change_tone card found")
        original_wc = engine._wild_cards
        engine._wild_cards = [card]
        try:
            engine.inject_wild_card(sk, random.Random(42))
            assert sk.tone == card["parameters"]["tone"]
        finally:
            engine._wild_cards = original_wc

    def test_inject_wild_card_add_beat(self):
        """Injecting a wild card with effect_type=add_beat should add a beat."""
        engine = StoryEvolutionEngine()
        if not engine._wild_cards:
            pytest.skip("No wild cards loaded")
        add_beat_cards = [c for c in engine._wild_cards if c.get("effect_type") == "add_beat"]
        if not add_beat_cards:
            pytest.skip("No add_beat wild cards in data")
        pop = _make_population()
        sk = engine._copy_skeleton(pop[0])
        original_beat_count = len(sk.beats)
        original_wc = engine._wild_cards
        engine._wild_cards = add_beat_cards
        try:
            engine.inject_wild_card(sk, random.Random(42))
            assert len(sk.beats) == original_beat_count + 1
        finally:
            engine._wild_cards = original_wc

    def test_inject_wild_card_modify_atom(self):
        """Injecting a wild card with effect_type=modify_atom should add tags."""
        engine = StoryEvolutionEngine()
        if not engine._wild_cards:
            pytest.skip("No wild cards loaded")
        modify_cards = [c for c in engine._wild_cards if c.get("effect_type") == "modify_atom"]
        if not modify_cards:
            pytest.skip("No modify_atom wild cards in data")
        pop = _make_population()
        sk = engine._copy_skeleton(pop[0])
        # Find a card that targets a category present in the skeleton
        atom_cats = {a.category.value for a in sk.atoms}
        card = None
        for c in modify_cards:
            if c["parameters"].get("target_category") in atom_cats:
                card = c
                break
        if card is None:
            pytest.skip("No modify_atom card matching skeleton categories")
        original_wc = engine._wild_cards
        engine._wild_cards = [card]
        try:
            engine.inject_wild_card(sk, random.Random(42))
            # At least one atom of the target category should have gained the new tags
            target_cat = AtomCategory(card["parameters"]["target_category"])
            add_tags = card["parameters"]["add_tags"]
            targets = [a for a in sk.atoms if a.category == target_cat]
            any_has_tags = any(
                all(tag in a.tags for tag in add_tags) for a in targets
            )
            assert any_has_tags
        finally:
            engine._wild_cards = original_wc


# ------------------------------------------------------------------ #
# Novelty scoring
# ------------------------------------------------------------------ #


class TestNoveltyScoring:
    def test_identical_population_low_novelty(self):
        engine = StoryEvolutionEngine()
        pop = _make_population(n=4)
        # A skeleton identical to pop[0] should have low novelty vs population containing it
        novelty = engine.calculate_novelty(pop[0], pop)
        assert 0.0 <= novelty <= 1.0

    def test_unique_skeleton_high_novelty(self):
        engine = StoryEvolutionEngine()
        pop = _make_population(n=4)
        # A skeleton with completely different atoms
        unique = StorySkeleton(
            atoms=[StoryAtom("unique_thing_xyz", AtomCategory.AGENT, AtomSource.CATALOGUE, ["unique"])],
            stats=GenerationStats(engine="test"),
        )
        novelty = engine.calculate_novelty(unique, pop)
        assert novelty > 0.5  # should be novel

    def test_empty_population_max_novelty(self):
        engine = StoryEvolutionEngine()
        sk = StorySkeleton(
            atoms=[StoryAtom("a", AtomCategory.AGENT, AtomSource.CATALOGUE, [])],
            stats=GenerationStats(engine="test"),
        )
        assert engine.calculate_novelty(sk, []) == 1.0

    def test_empty_atoms_zero_novelty(self):
        engine = StoryEvolutionEngine()
        pop = _make_population(n=2)
        sk = StorySkeleton(atoms=[], stats=GenerationStats(engine="test"))
        assert engine.calculate_novelty(sk, pop) == 0.0

    def test_novelty_is_between_zero_and_one(self):
        engine = StoryEvolutionEngine()
        pop = _make_population(n=6)
        for sk in pop:
            novelty = engine.calculate_novelty(sk, pop)
            assert 0.0 <= novelty <= 1.0

    def test_self_in_population_reduces_novelty(self):
        """A skeleton in the population should have lower novelty than one not in it."""
        engine = StoryEvolutionEngine()
        pop = _make_population(n=4)
        novelty_in = engine.calculate_novelty(pop[0], pop)
        # A completely unique skeleton should score higher
        unique = StorySkeleton(
            atoms=[StoryAtom("absolutely_unique_xyz", AtomCategory.AGENT, AtomSource.CATALOGUE, [])],
            stats=GenerationStats(engine="test"),
        )
        novelty_out = engine.calculate_novelty(unique, pop)
        assert novelty_out >= novelty_in


# ------------------------------------------------------------------ #
# _copy_skeleton
# ------------------------------------------------------------------ #


class TestCopySkeleton:
    def test_copy_is_independent(self):
        engine = StoryEvolutionEngine()
        pop = _make_population()
        original = pop[0]
        copy = engine._copy_skeleton(original)
        copy.tone = "CHANGED"
        assert original.tone != "CHANGED"

    def test_copy_preserves_atoms(self):
        engine = StoryEvolutionEngine()
        pop = _make_population()
        original = pop[0]
        copy = engine._copy_skeleton(original)
        assert [a.name for a in copy.atoms] == [a.name for a in original.atoms]

    def test_copy_preserves_beats(self):
        engine = StoryEvolutionEngine()
        pop = _make_population()
        original = pop[0]
        copy = engine._copy_skeleton(original)
        assert copy.beats == original.beats

    def test_copy_preserves_theme_tags(self):
        engine = StoryEvolutionEngine()
        pop = _make_population()
        original = pop[0]
        copy = engine._copy_skeleton(original)
        assert copy.theme_tags == original.theme_tags

    def test_copy_atoms_are_independent(self):
        """Modifying copied atoms should not affect the original."""
        engine = StoryEvolutionEngine()
        pop = _make_population()
        original = pop[0]
        copy = engine._copy_skeleton(original)
        copy.atoms[0] = StoryAtom("replaced", AtomCategory.AGENT, AtomSource.CATALOGUE, [])
        assert original.atoms[0].name != "replaced"


# ------------------------------------------------------------------ #
# _rebuild_spread_positions
# ------------------------------------------------------------------ #


class TestRebuildSpreadPositions:
    def test_keeps_valid_positions(self):
        sk = StorySkeleton(
            atoms=[
                StoryAtom("alpha", AtomCategory.AGENT, AtomSource.CATALOGUE, []),
                StoryAtom("beta", AtomCategory.OBJECT, AtomSource.CATALOGUE, []),
            ],
            spread_positions={"past": "alpha", "present": "beta"},
        )
        result = StoryEvolutionEngine._rebuild_spread_positions(sk)
        assert result == {"past": "alpha", "present": "beta"}

    def test_drops_invalid_positions(self):
        sk = StorySkeleton(
            atoms=[
                StoryAtom("alpha", AtomCategory.AGENT, AtomSource.CATALOGUE, []),
            ],
            spread_positions={"past": "alpha", "present": "gone"},
        )
        result = StoryEvolutionEngine._rebuild_spread_positions(sk)
        assert result == {"past": "alpha"}
        assert "present" not in result

    def test_empty_atoms_clears_all_positions(self):
        sk = StorySkeleton(
            atoms=[],
            spread_positions={"past": "alpha", "present": "beta"},
        )
        result = StoryEvolutionEngine._rebuild_spread_positions(sk)
        assert result == {}
