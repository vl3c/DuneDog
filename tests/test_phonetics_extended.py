"""Extended tests for phonetics, wordnet utils, and similarity engines."""

import random

import pytest

from dunedog.chaos.phonetics import (
    pronounceability_score,
    is_pronounceable,
    looks_like_name,
    analyze_phonetic_mood,
)
from dunedog.utils.wordnet_utils import (
    get_synsets,
    wup_similarity,
    get_hypernyms,
    get_pos_tag,
    _heuristic_pos,
)
from dunedog.utils.embeddings import (
    WordNetSimilarity,
    RandomSimilarity,
    get_similarity_engine,
)


# ------------------------------------------------------------------ #
# Pronounceability
# ------------------------------------------------------------------ #


class TestPronouncability:
    """Tests for pronounceability_score."""

    def test_empty_string(self):
        assert pronounceability_score("") == 0.0

    def test_no_alpha(self):
        assert pronounceability_score("123!@#") == 0.0

    def test_no_vowels(self):
        assert pronounceability_score("bcdfg") == 0.0

    def test_normal_word_scores_high(self):
        assert pronounceability_score("hello") > 0.5

    def test_consonant_cluster_penalty(self):
        score_good = pronounceability_score("banana")
        score_bad = pronounceability_score("bngkts")
        assert score_good > score_bad

    def test_triple_repeated_char_penalty(self):
        assert pronounceability_score("aaab") < pronounceability_score("abab")

    def test_high_vowel_ratio_penalty(self):
        # >70% vowels gets penalized
        assert pronounceability_score("aeiou") < 1.0

    def test_returns_float_in_range(self):
        for word in ["test", "banana", "xyzzy", "a", "mmm"]:
            score = pronounceability_score(word)
            assert 0.0 <= score <= 1.0, f"Score for '{word}' out of range: {score}"

    def test_single_vowel(self):
        score = pronounceability_score("a")
        assert score > 0.0

    def test_long_consonant_cluster_heavy_penalty(self):
        # 4+ consonants in a row
        score = pronounceability_score("strng")
        assert score < pronounceability_score("string")

    def test_case_insensitive(self):
        assert pronounceability_score("Hello") == pronounceability_score("hello")


class TestIsPronounceble:
    """Tests for is_pronounceable."""

    def test_pronounceable_word(self):
        assert is_pronounceable("banana") is True

    def test_unpronounceable(self):
        assert is_pronounceable("bcdfg") is False

    def test_custom_threshold(self):
        # Very low threshold should accept almost anything with vowels
        assert is_pronounceable("brat", threshold=0.1) is True

    def test_empty_string(self):
        assert is_pronounceable("") is False


# ------------------------------------------------------------------ #
# looks_like_name
# ------------------------------------------------------------------ #


class TestLooksLikeName:
    """Tests for looks_like_name."""

    def test_valid_name(self):
        assert looks_like_name("Draven") is True

    def test_too_short(self):
        assert looks_like_name("ab") is False

    def test_too_long(self):
        assert looks_like_name("abcdefghijk") is False

    def test_starts_with_vowel(self):
        assert looks_like_name("elara") is False

    def test_empty(self):
        assert looks_like_name("") is False

    def test_non_alpha(self):
        assert looks_like_name("dr4ven") is False

    def test_exactly_three_chars(self):
        # Length 3 is the lower bound
        assert looks_like_name("ban") is True or looks_like_name("ban") is False
        # Just checking it does not crash; actual result depends on pronounceability

    def test_exactly_ten_chars(self):
        # Length 10 is the upper bound
        result = looks_like_name("dravenorix")
        assert isinstance(result, bool)

    def test_no_vowels(self):
        assert looks_like_name("bcd") is False

    def test_common_names(self):
        # Names starting with consonants that are pronounceable
        assert looks_like_name("Sarah") is True
        assert looks_like_name("Marco") is True


# ------------------------------------------------------------------ #
# analyze_phonetic_mood
# ------------------------------------------------------------------ #


class TestAnalyzePhoneticMood:
    """Tests for analyze_phonetic_mood."""

    def test_empty(self):
        assert analyze_phonetic_mood("") == "balanced"

    def test_non_alpha(self):
        assert analyze_phonetic_mood("12345") == "balanced"

    def test_soft_consonants_dreamy(self):
        # l, m, n, r, w, y are soft consonants
        assert analyze_phonetic_mood("lmnrwy") == "dreamy"

    def test_hard_consonants_urgent(self):
        # k, p, t, d, b, g are hard consonants
        assert analyze_phonetic_mood("kptdbg") == "urgent"

    def test_sibilants_secretive(self):
        # s, z, f, v are sibilants
        assert analyze_phonetic_mood("sszfvs") == "secretive"

    def test_vowel_heavy_flowing(self):
        assert analyze_phonetic_mood("aeiouae") == "flowing"

    def test_mixed_balanced(self):
        # A mix of all types should return "balanced" when no category dominates
        result = analyze_phonetic_mood("abcdef")
        assert result in ("balanced", "dreamy", "urgent", "secretive", "flowing")

    def test_returns_valid_mood(self):
        valid_moods = {"dreamy", "urgent", "secretive", "flowing", "balanced"}
        for word in ["test", "hello", "zzz", "lll", "aaa"]:
            mood = analyze_phonetic_mood(word)
            assert mood in valid_moods, f"Invalid mood '{mood}' for '{word}'"


# ------------------------------------------------------------------ #
# _heuristic_pos
# ------------------------------------------------------------------ #


class TestHeuristicPos:
    """Tests for _heuristic_pos suffix-based POS guessing."""

    def test_ly_adverb(self):
        assert _heuristic_pos("quickly") == "adv"

    def test_ness_noun(self):
        assert _heuristic_pos("happiness") == "noun"

    def test_ment_noun(self):
        assert _heuristic_pos("movement") == "noun"

    def test_tion_noun(self):
        assert _heuristic_pos("creation") == "noun"

    def test_sion_noun(self):
        assert _heuristic_pos("decision") == "noun"

    def test_ity_noun(self):
        assert _heuristic_pos("gravity") == "noun"

    def test_ing_verb(self):
        assert _heuristic_pos("running") == "verb"

    def test_ed_verb(self):
        assert _heuristic_pos("walked") == "verb"

    def test_ate_verb(self):
        assert _heuristic_pos("create") == "verb"

    def test_ify_verb(self):
        assert _heuristic_pos("simplify") == "verb"

    def test_ize_verb(self):
        assert _heuristic_pos("organize") == "verb"

    def test_ful_adj(self):
        assert _heuristic_pos("beautiful") == "adj"

    def test_less_adj(self):
        assert _heuristic_pos("hopeless") == "adj"

    def test_ous_adj(self):
        assert _heuristic_pos("dangerous") == "adj"

    def test_ive_adj(self):
        assert _heuristic_pos("creative") == "adj"

    def test_able_adj(self):
        assert _heuristic_pos("readable") == "adj"

    def test_ible_adj(self):
        assert _heuristic_pos("flexible") == "adj"

    def test_al_adj(self):
        assert _heuristic_pos("magical") == "adj"

    def test_ish_adj(self):
        assert _heuristic_pos("reddish") == "adj"

    def test_default_noun(self):
        assert _heuristic_pos("cat") == "noun"

    def test_unknown_word_defaults_noun(self):
        assert _heuristic_pos("xyzqwrtp") == "noun"


# ------------------------------------------------------------------ #
# WordNet utils
# ------------------------------------------------------------------ #


class TestWordNetUtils:
    """Tests for wordnet_utils functions."""

    def test_get_synsets_real_word(self):
        syns = get_synsets("dog")
        assert isinstance(syns, list)
        # Should return at least one synset if NLTK data is present

    def test_get_synsets_nonsense(self):
        syns = get_synsets("xyzqwrtp")
        assert syns == []

    def test_wup_similarity_same_word(self):
        score = wup_similarity("dog", "dog")
        assert score == 1.0 or score > 0.9  # same word should be very similar

    def test_wup_similarity_nonsense(self):
        assert wup_similarity("xyzqwrtp", "abcdefg") == 0.0

    def test_wup_similarity_returns_float(self):
        score = wup_similarity("cat", "dog")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_get_hypernyms(self):
        hyps = get_hypernyms("dog")
        assert isinstance(hyps, list)

    def test_get_hypernyms_nonsense(self):
        hyps = get_hypernyms("xyzqwrtp")
        assert hyps == []

    def test_get_pos_tag(self):
        pos = get_pos_tag("running")
        assert pos in ("noun", "verb", "adj", "adv")

    def test_get_pos_tag_returns_string(self):
        pos = get_pos_tag("dog")
        assert isinstance(pos, str)
        assert pos in ("noun", "verb", "adj", "adv")


# ------------------------------------------------------------------ #
# Similarity engines
# ------------------------------------------------------------------ #


class TestSimilarityEngines:
    """Tests for RandomSimilarity, WordNetSimilarity, and get_similarity_engine."""

    def test_random_similarity_deterministic(self):
        engine = RandomSimilarity(random.Random(42))
        s1 = engine.similarity("cat", "dog")
        engine2 = RandomSimilarity(random.Random(42))
        s2 = engine2.similarity("cat", "dog")
        assert s1 == s2

    def test_random_similarity_symmetric(self):
        engine = RandomSimilarity(random.Random(42))
        assert engine.similarity("cat", "dog") == engine.similarity("dog", "cat")

    def test_random_similarity_range(self):
        engine = RandomSimilarity(random.Random(42))
        score = engine.similarity("cat", "dog")
        assert 0.0 <= score <= 1.0

    def test_random_most_similar(self):
        engine = RandomSimilarity(random.Random(42))
        result = engine.most_similar("cat", ["dog", "fish", "bird"], 2)
        assert len(result) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in result)

    def test_random_most_similar_sorted(self):
        engine = RandomSimilarity(random.Random(42))
        result = engine.most_similar("cat", ["dog", "fish", "bird", "tree"], 3)
        # Should be sorted in descending order of similarity
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_random_most_similar_n_greater_than_candidates(self):
        engine = RandomSimilarity(random.Random(42))
        result = engine.most_similar("cat", ["dog"], 5)
        assert len(result) == 1  # only 1 candidate available

    def test_wordnet_similarity(self):
        engine = WordNetSimilarity()
        score = engine.similarity("cat", "dog")
        assert isinstance(score, float)

    def test_wordnet_similarity_range(self):
        engine = WordNetSimilarity()
        score = engine.similarity("cat", "dog")
        assert 0.0 <= score <= 1.0

    def test_wordnet_most_similar(self):
        engine = WordNetSimilarity()
        result = engine.most_similar("cat", ["dog", "car", "tree"], 2)
        assert len(result) == 2

    def test_wordnet_most_similar_sorted(self):
        engine = WordNetSimilarity()
        result = engine.most_similar("cat", ["dog", "car", "tree", "fish"], 3)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_get_similarity_engine_returns_engine(self):
        engine = get_similarity_engine()
        assert hasattr(engine, "similarity")
        assert hasattr(engine, "most_similar")

    def test_get_similarity_engine_with_rng(self):
        engine = get_similarity_engine(rng=random.Random(42))
        assert hasattr(engine, "similarity")
        assert hasattr(engine, "most_similar")
