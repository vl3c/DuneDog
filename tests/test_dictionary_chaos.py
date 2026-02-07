"""Tests for the dictionary chaos engine."""
import pytest

from dunedog.chaos.dictionary_chaos import DictionaryChaosEngine
from dunedog.models.results import DictionaryChaosResult, SamplingStrategy
from dunedog.utils.seed_manager import SeedManager


@pytest.fixture
def engine():
    return DictionaryChaosEngine()


class TestSampleWords:
    def test_uniform_returns_words(self, engine, rng):
        words = engine.sample_words(20, SamplingStrategy.UNIFORM, rng)
        assert len(words) == 20
        assert all(isinstance(w, str) for w in words)
        assert all(len(w) > 0 for w in words)

    def test_rare_words_returns_words(self, engine, rng):
        words = engine.sample_words(15, SamplingStrategy.RARE_WORDS, rng)
        assert len(words) == 15
        assert all(isinstance(w, str) for w in words)

    def test_frequency_weighted_returns_words(self, engine, rng):
        words = engine.sample_words(10, SamplingStrategy.FREQUENCY_WEIGHTED, rng)
        assert len(words) == 10

    def test_noun_heavy_returns_words(self, engine, rng):
        words = engine.sample_words(20, SamplingStrategy.NOUN_HEAVY, rng)
        assert len(words) == 20

    def test_phonetic_cluster_returns_words(self, engine, rng):
        words = engine.sample_words(15, SamplingStrategy.PHONETIC_CLUSTER, rng)
        assert len(words) == 15


class TestArrangeGrammatically:
    def test_returns_non_empty_strings(self, engine, rng):
        words = engine.sample_words(20, SamplingStrategy.UNIFORM, rng)
        phrases = engine.arrange_grammatically(words, rng)
        assert len(phrases) > 0
        assert all(isinstance(p, str) for p in phrases)
        assert all(len(p) > 0 for p in phrases)


class TestProcess:
    def test_returns_result_with_correct_fields(self, engine, rng):
        result = engine.process(rng, SamplingStrategy.UNIFORM, n_words=20)
        assert isinstance(result, DictionaryChaosResult)
        assert len(result.sampled_words) == 20
        assert result.strategy == SamplingStrategy.UNIFORM
        assert len(result.grammatical_arrangements) > 0
        assert isinstance(result.semantic_clusters, list)
        assert len(result.combined_words) >= 20

    def test_combined_includes_soup_words(self, engine, rng):
        from dunedog.models.results import LetterSoupResult
        soup = LetterSoupResult(
            raw_soup="test",
            exact_words=["cat", "dog"],
            near_words=[],
        )
        result = engine.process(rng, n_words=10, soup_result=soup)
        assert "cat" in result.combined_words
        assert "dog" in result.combined_words


class TestDeterminism:
    def test_same_seed_same_words(self):
        engine = DictionaryChaosEngine()
        rng1 = SeedManager(42).child_rng("dict")
        rng2 = SeedManager(42).child_rng("dict")
        words1 = engine.sample_words(20, SamplingStrategy.UNIFORM, rng1)
        words2 = engine.sample_words(20, SamplingStrategy.UNIFORM, rng2)
        assert words1 == words2

    def test_different_seed_different_words(self):
        engine = DictionaryChaosEngine()
        rng1 = SeedManager(42).child_rng("dict")
        rng2 = SeedManager(99).child_rng("dict")
        words1 = engine.sample_words(20, SamplingStrategy.UNIFORM, rng1)
        words2 = engine.sample_words(20, SamplingStrategy.UNIFORM, rng2)
        assert words1 != words2
