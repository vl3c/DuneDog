"""Tests for all model dataclasses and config."""
import pytest

from dunedog.models.atoms import StoryAtom, AtomCategory, AtomSource, AffinityEntry
from dunedog.models.results import (
    Neologism,
    LetterSoupResult,
    DictionaryChaosResult,
    SamplingStrategy,
)
from dunedog.models.skeleton import (
    PrimordialSource,
    GenerationStats,
    StorySkeleton,
    EvolutionResult,
)
from dunedog.models.validation import (
    Invariant,
    InvariantSeverity,
    Tendency,
    ValidationResult,
)
from dunedog.models.config import (
    GenerationConfig,
    Preset,
    LLMConfig,
    ChaosConfig,
    EvolutionConfig,
)


# ------------------------------------------------------------------
# StoryAtom
# ------------------------------------------------------------------

class TestStoryAtom:
    def test_construction(self):
        atom = StoryAtom(
            name="warrior",
            category=AtomCategory.AGENT,
            source=AtomSource.CATALOGUE,
            tags=["combat", "brave"],
            rarity=0.4,
        )
        assert atom.name == "warrior"
        assert atom.category == AtomCategory.AGENT
        assert atom.source == AtomSource.CATALOGUE
        assert atom.tags == ["combat", "brave"]
        assert atom.rarity == 0.4
        assert atom.metadata == {}

    def test_round_trip(self):
        atom = StoryAtom(
            name="ancient sword",
            category=AtomCategory.OBJECT,
            source=AtomSource.DICTIONARY,
            tags=["weapon", "old"],
            rarity=0.8,
            metadata={"origin": "chaos"},
        )
        data = atom.to_dict()
        restored = StoryAtom.from_dict(data)
        assert restored.name == atom.name
        assert restored.category == atom.category
        assert restored.source == atom.source
        assert restored.tags == atom.tags
        assert restored.rarity == atom.rarity
        assert restored.metadata == atom.metadata

    def test_defaults(self):
        atom = StoryAtom("x", AtomCategory.QUALITY, AtomSource.WILD_CARD)
        assert atom.tags == []
        assert atom.rarity == 0.5
        assert atom.metadata == {}


# ------------------------------------------------------------------
# AffinityEntry
# ------------------------------------------------------------------

class TestAffinityEntry:
    def test_key_is_sorted(self):
        entry = AffinityEntry("zebra", "alpha", 0.9)
        assert entry.key == ("alpha", "zebra")

    def test_round_trip(self):
        entry = AffinityEntry("a", "b", 0.5, tags=["friend"])
        restored = AffinityEntry.from_dict(entry.to_dict())
        assert restored.atom_a == entry.atom_a
        assert restored.strength == entry.strength
        assert restored.tags == entry.tags


# ------------------------------------------------------------------
# Neologism
# ------------------------------------------------------------------

class TestNeologism:
    def test_round_trip(self):
        neo = Neologism(
            text="glimber",
            pronounceability=0.85,
            phonetic_mood="dreamy",
            definition="a soft glow",
            part_of_speech="noun",
            usage_example="The glimber rose.",
            source_context=["light", "shadow"],
        )
        data = neo.to_dict()
        restored = Neologism.from_dict(data)
        assert restored.text == neo.text
        assert restored.pronounceability == neo.pronounceability
        assert restored.phonetic_mood == neo.phonetic_mood
        assert restored.definition == neo.definition


# ------------------------------------------------------------------
# LetterSoupResult
# ------------------------------------------------------------------

class TestLetterSoupResult:
    def test_round_trip(self):
        result = LetterSoupResult(
            raw_soup="abcdefgh",
            exact_words=["the", "and"],
            near_words=[("abd", "add", 1)],
            neologisms=[Neologism("glim", 0.7, "dreamy")],
            phonetic_mood="warm",
        )
        data = result.to_dict()
        restored = LetterSoupResult.from_dict(data)
        assert restored.raw_soup == result.raw_soup
        assert restored.exact_words == result.exact_words
        assert len(restored.near_words) == 1
        assert restored.near_words[0] == ("abd", "add", 1)
        assert len(restored.neologisms) == 1
        assert restored.neologisms[0].text == "glim"
        assert restored.phonetic_mood == "warm"

    def test_all_words(self):
        result = LetterSoupResult(
            raw_soup="x",
            exact_words=["cat"],
            near_words=[("cet", "cut", 1)],
        )
        assert result.all_words() == ["cat", "cut"]


# ------------------------------------------------------------------
# DictionaryChaosResult
# ------------------------------------------------------------------

class TestDictionaryChaosResult:
    def test_round_trip(self):
        result = DictionaryChaosResult(
            sampled_words=["tree", "stone"],
            strategy=SamplingStrategy.RARE_WORDS,
            grammatical_arrangements=["the tree stone"],
            semantic_clusters=[["tree", "stone"]],
            combined_words=["tree", "stone", "light"],
        )
        data = result.to_dict()
        restored = DictionaryChaosResult.from_dict(data)
        assert restored.sampled_words == result.sampled_words
        assert restored.strategy == SamplingStrategy.RARE_WORDS
        assert restored.grammatical_arrangements == result.grammatical_arrangements
        assert restored.semantic_clusters == result.semantic_clusters
        assert restored.combined_words == result.combined_words


# ------------------------------------------------------------------
# StorySkeleton & GenerationStats
# ------------------------------------------------------------------

class TestStorySkeleton:
    def test_round_trip(self, sample_atoms):
        stats = GenerationStats(
            engine="tarot",
            spread_type="hero_journey",
            beat_count=5,
            violations=["minor issue"],
            coherence_score=0.85,
            generation=3,
        )
        skeleton = StorySkeleton(
            atoms=sample_atoms[:3],
            beats=["intro", "conflict", "resolution"],
            spread_positions={"hero": sample_atoms[0].name},
            theme_tags=["journey", "mystery"],
            tone="dark",
            stats=stats,
            seed=42,
        )
        data = skeleton.to_dict()
        restored = StorySkeleton.from_dict(data)
        assert len(restored.atoms) == 3
        assert restored.atoms[0].name == sample_atoms[0].name
        assert restored.beats == skeleton.beats
        assert restored.spread_positions == skeleton.spread_positions
        assert restored.theme_tags == skeleton.theme_tags
        assert restored.tone == "dark"
        assert restored.stats.engine == "tarot"
        assert restored.stats.coherence_score == 0.85
        assert restored.coherence_score == 0.85
        assert restored.seed == 42

    def test_generation_stats_round_trip(self):
        stats = GenerationStats(engine="markov", beat_count=7, coherence_score=0.6)
        data = stats.to_dict()
        restored = GenerationStats.from_dict(data)
        assert restored.engine == "markov"
        assert restored.beat_count == 7
        assert restored.coherence_score == 0.6


# ------------------------------------------------------------------
# ValidationResult
# ------------------------------------------------------------------

class TestValidationResult:
    def test_starts_valid(self):
        vr = ValidationResult()
        assert vr.valid is True
        assert vr.total_violations == 0

    def test_hard_violation_invalidates(self):
        vr = ValidationResult()
        vr.add_violation("bad thing", InvariantSeverity.HARD)
        assert vr.valid is False
        assert len(vr.hard_violations) == 1
        assert vr.total_violations == 1

    def test_soft_violation_stays_valid(self):
        vr = ValidationResult()
        vr.add_violation("minor thing", InvariantSeverity.SOFT)
        assert vr.valid is True
        assert len(vr.soft_violations) == 1
        assert vr.total_violations == 1

    def test_mixed_violations(self):
        vr = ValidationResult()
        vr.add_violation("soft issue", InvariantSeverity.SOFT)
        vr.add_violation("hard issue", InvariantSeverity.HARD)
        assert vr.valid is False
        assert vr.total_violations == 2


# ------------------------------------------------------------------
# Invariant & Tendency
# ------------------------------------------------------------------

class TestInvariant:
    def test_from_dict(self):
        data = {
            "name": "no_duplicate_agents",
            "description": "Agents must not repeat",
            "severity": "soft",
            "check_type": "unique_category",
            "parameters": {"category": "agent"},
        }
        inv = Invariant.from_dict(data)
        assert inv.name == "no_duplicate_agents"
        assert inv.severity == InvariantSeverity.SOFT
        assert inv.check_type == "unique_category"
        assert inv.parameters == {"category": "agent"}

    def test_round_trip(self):
        inv = Invariant(
            name="test",
            description="desc",
            severity=InvariantSeverity.HARD,
            check_type="requires_atom",
            parameters={"atom": "hero"},
        )
        restored = Invariant.from_dict(inv.to_dict())
        assert restored.name == inv.name
        assert restored.severity == inv.severity
        assert restored.parameters == inv.parameters


class TestTendency:
    def test_from_dict(self):
        data = {
            "name": "favor_darkness",
            "description": "Tend toward dark themes",
            "probability": 0.7,
            "effect": "boost_dark_atoms",
            "parameters": {"weight": 1.5},
        }
        tend = Tendency.from_dict(data)
        assert tend.name == "favor_darkness"
        assert tend.probability == 0.7
        assert tend.effect == "boost_dark_atoms"
        assert tend.parameters == {"weight": 1.5}

    def test_round_trip(self):
        tend = Tendency(
            name="t",
            description="d",
            probability=0.3,
            effect="e",
            parameters={"x": 1},
        )
        restored = Tendency.from_dict(tend.to_dict())
        assert restored.name == tend.name
        assert restored.probability == tend.probability
        assert restored.effect == tend.effect


# ------------------------------------------------------------------
# GenerationConfig presets
# ------------------------------------------------------------------

class TestGenerationConfig:
    def test_quick_preset(self):
        cfg = GenerationConfig.from_preset(Preset.QUICK)
        assert cfg.preset == Preset.QUICK
        assert cfg.chaos.use_dictionary is False
        assert cfg.chaos.enable_near_words is False
        assert cfg.chaos.soup_length == 100
        assert cfg.crystallization.enable_neologisms is False
        assert cfg.engines.use_markov is False
        assert cfg.evolution.enabled is False
        assert cfg.skeletons_to_generate == 50

    def test_deep_preset(self):
        cfg = GenerationConfig.from_preset(Preset.DEEP)
        assert cfg.preset == Preset.DEEP
        assert cfg.chaos.use_letter_soup is True
        assert cfg.chaos.use_dictionary is True
        assert cfg.evolution.enabled is True
        assert cfg.evolution.generations == 5
        assert cfg.skeletons_to_generate == 200

    def test_experimental_preset(self):
        cfg = GenerationConfig.from_preset(Preset.EXPERIMENTAL)
        assert cfg.preset == Preset.EXPERIMENTAL
        assert cfg.chaos.soup_length == 400
        assert cfg.evolution.generations == 20
        assert cfg.evolution.population_size == 100
        assert cfg.skeletons_to_generate == 500

    def test_custom_preset(self):
        cfg = GenerationConfig.from_preset(Preset.CUSTOM, skeletons_to_generate=10)
        assert cfg.preset == Preset.CUSTOM
        assert cfg.skeletons_to_generate == 10

    def test_from_preset_string(self):
        cfg = GenerationConfig.from_preset("quick")
        assert cfg.preset == Preset.QUICK

    def test_preset_with_overrides(self):
        cfg = GenerationConfig.from_preset(Preset.DEEP, seed=123)
        assert cfg.seed == 123
        assert cfg.evolution.enabled is True  # deep default still applied


# ------------------------------------------------------------------
# LLMConfig
# ------------------------------------------------------------------

class TestLLMConfig:
    def test_defaults(self):
        llm = LLMConfig()
        assert llm.provider == "anthropic"
        assert llm.model == ""
        assert llm.api_key == ""
        assert llm.story_lines == 20
        assert llm.temperature == 0.8
