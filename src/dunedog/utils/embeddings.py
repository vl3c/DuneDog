"""Strategy pattern for word similarity — WordNet with random fallback."""

from __future__ import annotations

import hashlib
import random
from abc import ABC, abstractmethod

from dunedog.utils import wordnet_utils


class SimilarityEngine(ABC):
    """Abstract base for computing word-pair similarity."""

    @abstractmethod
    def similarity(self, word_a: str, word_b: str) -> float:
        """Return a similarity score in [0, 1]."""

    @abstractmethod
    def most_similar(
        self, word: str, candidates: list[str], n: int
    ) -> list[tuple[str, float]]:
        """Return the top-*n* most similar candidates with scores."""


class WordNetSimilarity(SimilarityEngine):
    """Wu-Palmer similarity via WordNet."""

    def similarity(self, word_a: str, word_b: str) -> float:
        return wordnet_utils.wup_similarity(word_a, word_b)

    def most_similar(
        self, word: str, candidates: list[str], n: int
    ) -> list[tuple[str, float]]:
        scored = [(c, self.similarity(word, c)) for c in candidates]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:n]


class RandomSimilarity(SimilarityEngine):
    """Fallback engine that returns seeded random scores."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def similarity(self, word_a: str, word_b: str) -> float:
        # Deterministic per pair: use SHA-256 for a stable hash across processes.
        pair = "|".join(sorted((word_a, word_b)))
        seed = int(hashlib.sha256(pair.encode()).hexdigest()[:16], 16)
        r = random.Random(seed)
        return r.random()

    def most_similar(
        self, word: str, candidates: list[str], n: int
    ) -> list[tuple[str, float]]:
        scored = [(c, self.similarity(word, c)) for c in candidates]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:n]


def get_similarity_engine(
    rng: random.Random | None = None,
) -> SimilarityEngine:
    """Return WordNetSimilarity if NLTK is usable, else RandomSimilarity."""
    if wordnet_utils._nltk_available():
        return WordNetSimilarity()
    return RandomSimilarity(rng or random.Random())
