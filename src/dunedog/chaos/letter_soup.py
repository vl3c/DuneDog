"""Letter soup generator -- Layer 0 chaos."""

from __future__ import annotations

import random

from Levenshtein import distance as levenshtein_distance

from dunedog.chaos.phonetics import (
    analyze_phonetic_mood,
    is_pronounceable,
    pronounceability_score,
)
from dunedog.models.results import LetterSoupResult, Neologism
from dunedog.utils.word_loader import get_loader, WordLoader

# English letter frequency weights (approximate)
LETTER_WEIGHTS = {
    "e": 12.7, "t": 9.1, "a": 8.2, "o": 7.5, "i": 7.0, "n": 6.7,
    "s": 6.3, "h": 6.1, "r": 6.0, "d": 4.3, "l": 4.0, "c": 2.8,
    "u": 2.8, "m": 2.4, "w": 2.4, "f": 2.2, "g": 2.0, "y": 2.0,
    "p": 1.9, "b": 1.5, "v": 1.0, "k": 0.8, "j": 0.15, "x": 0.15,
    "q": 0.10, "z": 0.07,
}

_LETTERS = list(LETTER_WEIGHTS.keys())
_WEIGHTS = list(LETTER_WEIGHTS.values())

# Sliding window bounds
_MIN_WINDOW = 3
_MAX_WINDOW = 7

# Near-word search limits
_NEAR_WORD_MIN_LEN = 4
_NEAR_WORD_MAX_DISTANCE = 2
_NEAR_WORD_BUCKET_SAMPLE = 1000
_NEAR_WORD_MAX_CANDIDATES = 50

# Neologism bounds
_NEO_MIN_LEN = 4
_NEO_MAX_LEN = 8


class LetterSoupGenerator:
    """Generate random letter soup and parse it for words, near-words, and neologisms."""

    def __init__(self, word_loader: WordLoader | None = None) -> None:
        self._loader = word_loader or get_loader()

    def generate_soup(self, length: int, rng: random.Random) -> str:
        """Generate random letter soup with English-biased letter weights."""
        return "".join(rng.choices(_LETTERS, weights=_WEIGHTS, k=length))

    def parse_soup(
        self,
        soup: str,
        rng: random.Random,
        enable_near_words: bool = True,
    ) -> LetterSoupResult:
        """Parse soup to find words, near-words, and neologisms.

        Sliding window sizes 3-7.  Find exact English words, near-words via
        Levenshtein (distance 1-2), and neologisms (pronounceable non-words).
        """
        soup_lower = soup.lower()
        n = len(soup_lower)

        exact_words: set[str] = set()
        near_words: dict[str, tuple[str, str, int]] = {}  # match -> (fragment, match, dist)
        neologisms: dict[str, Neologism] = {}  # text -> Neologism

        for win in range(_MIN_WINDOW, _MAX_WINDOW + 1):
            for start in range(n - win + 1):
                fragment = soup_lower[start : start + win]
                if not fragment.isalpha():
                    continue

                # Exact word check
                if self._loader.is_english_word(fragment):
                    exact_words.add(fragment)
                    continue

                # Near-word check (length 4+)
                if enable_near_words and win >= _NEAR_WORD_MIN_LEN:
                    bucket = self._loader.get_words_by_length(win)
                    candidates = bucket
                    if len(bucket) > _NEAR_WORD_BUCKET_SAMPLE:
                        candidates = rng.sample(bucket, _NEAR_WORD_BUCKET_SAMPLE)

                    checked = 0
                    for word in candidates:
                        if checked >= _NEAR_WORD_MAX_CANDIDATES:
                            break
                        dist = levenshtein_distance(fragment, word)
                        if dist <= _NEAR_WORD_MAX_DISTANCE:
                            if word not in near_words and word not in exact_words:
                                near_words[word] = (fragment, word, dist)
                        checked += 1

                # Neologism check (length 4-8, pronounceable non-word)
                if _NEO_MIN_LEN <= win <= _NEO_MAX_LEN and fragment not in exact_words:
                    if fragment not in neologisms and is_pronounceable(fragment):
                        mood = analyze_phonetic_mood(fragment)
                        neologisms[fragment] = Neologism(
                            text=fragment,
                            pronounceability=pronounceability_score(fragment),
                            phonetic_mood=mood,
                        )

        overall_mood = analyze_phonetic_mood(soup)

        return LetterSoupResult(
            raw_soup=soup,
            exact_words=sorted(exact_words),
            near_words=sorted(near_words.values(), key=lambda t: t[2]),
            neologisms=sorted(neologisms.values(), key=lambda neo: -neo.pronounceability),
            phonetic_mood=overall_mood,
        )

    def generate_and_parse(
        self,
        length: int,
        rng: random.Random,
        enable_near_words: bool = True,
    ) -> LetterSoupResult:
        """Generate soup and parse it in one step."""
        soup = self.generate_soup(length, rng)
        return self.parse_soup(soup, rng, enable_near_words)
