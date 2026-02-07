"""Lazy-loading English word list with length-bucketed lookup."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

_DATA_FILE = Path(__file__).resolve().parent.parent.parent.parent / "data" / "english_words.txt"


class WordLoader:
    """Loads english_words.txt lazily and provides fast lookup helpers."""

    def __init__(self) -> None:
        self._words: frozenset[str] = frozenset()
        self._length_buckets: dict[int, list[str]] = {}
        self._word_list: list[str] = []
        self._loaded = False

    def load(self) -> None:
        """Read the word file and populate internal structures."""
        if self._loaded:
            return

        buckets: dict[int, list[str]] = defaultdict(list)
        words: list[str] = []

        with open(_DATA_FILE, encoding="utf-8") as fh:
            for line in fh:
                word = line.strip()
                if word:
                    words.append(word)
                    buckets[len(word)].append(word)

        self._words = frozenset(words)
        self._word_list = words
        self._length_buckets = dict(buckets)
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def is_english_word(self, word: str) -> bool:
        """Check whether *word* (case-insensitive) is in the dictionary."""
        self._ensure_loaded()
        return word.lower() in self._words

    def get_words_by_length(self, length: int) -> list[str]:
        """Return the bucket of words with the given length (for Levenshtein)."""
        self._ensure_loaded()
        return self._length_buckets.get(length, [])

    def get_random_word(self, rng: random.Random) -> str:
        """Pick one random word using the supplied RNG."""
        self._ensure_loaded()
        return rng.choice(self._word_list)

    def get_random_words(self, n: int, rng: random.Random) -> list[str]:
        """Pick *n* random words (with replacement) using the supplied RNG."""
        self._ensure_loaded()
        return [rng.choice(self._word_list) for _ in range(n)]


# Module-level singleton ---------------------------------------------------

_loader: WordLoader | None = None


def get_loader() -> WordLoader:
    """Return (and lazily create) the module-level WordLoader singleton."""
    global _loader
    if _loader is None:
        _loader = WordLoader()
    return _loader
