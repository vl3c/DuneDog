"""Tests for the letter soup generator."""
import pytest

from dunedog.chaos.letter_soup import LetterSoupGenerator
from dunedog.models.results import Neologism
from dunedog.utils.seed_manager import SeedManager


@pytest.fixture
def generator():
    return LetterSoupGenerator()


class TestGenerateSoup:
    def test_correct_length(self, generator, rng):
        soup = generator.generate_soup(200, rng)
        assert len(soup) == 200

    def test_only_lowercase_letters(self, generator, rng):
        soup = generator.generate_soup(500, rng)
        assert soup.isalpha()
        assert soup == soup.lower()

    def test_different_lengths(self, generator, rng):
        for length in [10, 100, 1000]:
            soup = generator.generate_soup(length, rng)
            assert len(soup) == length


class TestParseSoup:
    def test_finds_exact_words(self, generator, rng):
        # Generate a long soup to increase odds of finding words
        result = generator.generate_and_parse(500, rng, enable_near_words=True)
        # With 500 chars and sliding windows 3-7, should find at least some words
        assert len(result.exact_words) > 0

    def test_result_has_phonetic_mood(self, generator, rng):
        result = generator.generate_and_parse(200, rng)
        assert isinstance(result.phonetic_mood, str)
        assert len(result.phonetic_mood) > 0

    def test_raw_soup_is_preserved(self, generator, rng):
        result = generator.generate_and_parse(100, rng)
        assert len(result.raw_soup) == 100

    def test_near_words_disabled(self, generator, rng):
        result = generator.generate_and_parse(200, rng, enable_near_words=False)
        assert result.near_words == []


class TestDeterminism:
    def test_same_seed_same_soup(self):
        gen = LetterSoupGenerator()
        sm1 = SeedManager(42)
        sm2 = SeedManager(42)
        rng1 = sm1.child_rng("soup")
        rng2 = sm2.child_rng("soup")

        soup1 = gen.generate_soup(200, rng1)
        soup2 = gen.generate_soup(200, rng2)
        assert soup1 == soup2

    def test_same_seed_same_parse(self):
        gen = LetterSoupGenerator()
        sm1 = SeedManager(42)
        sm2 = SeedManager(42)
        rng1 = sm1.child_rng("soup")
        rng2 = sm2.child_rng("soup")

        result1 = gen.generate_and_parse(200, rng1)
        result2 = gen.generate_and_parse(200, rng2)
        assert result1.raw_soup == result2.raw_soup
        assert result1.exact_words == result2.exact_words

    def test_different_seed_different_soup(self):
        gen = LetterSoupGenerator()
        rng1 = SeedManager(42).child_rng("soup")
        rng2 = SeedManager(99).child_rng("soup")

        soup1 = gen.generate_soup(200, rng1)
        soup2 = gen.generate_soup(200, rng2)
        assert soup1 != soup2


class TestNeologisms:
    def test_neologisms_are_neologism_objects(self, generator, rng):
        result = generator.generate_and_parse(500, rng)
        for neo in result.neologisms:
            assert isinstance(neo, Neologism)
            assert isinstance(neo.text, str)
            assert len(neo.text) >= 4
            assert 0.0 <= neo.pronounceability <= 1.0

    def test_neologisms_have_phonetic_mood(self, generator, rng):
        result = generator.generate_and_parse(500, rng)
        if result.neologisms:
            for neo in result.neologisms:
                assert isinstance(neo.phonetic_mood, str)
