"""Dictionary chaos engine — Layer 0 structured word sampling."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from wordfreq import zipf_frequency

from dunedog.models.results import (
    DictionaryChaosResult,
    LetterSoupResult,
    SamplingStrategy,
)
from dunedog.utils.word_loader import get_loader, WordLoader
from dunedog.utils.wordnet_utils import get_hypernyms, get_pos_tag

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_TEMPLATES_FILE = _DATA_DIR / "grammar_templates.json"

# Inline fallback templates used when grammar_templates.json is absent.
_DEFAULT_TEMPLATES: dict[str, list[str]] = {
    "declarative": [
        "the {noun} {verb} {adv}",
        "{adj} {noun} {verb} the {noun}",
        "a {adj} {noun} {verb}",
    ],
    "imperative": [
        "{verb} the {adj} {noun}",
        "{adv} {verb} every {noun}",
    ],
    "fragment": [
        "{adj} {noun}",
        "{noun} of {noun}",
        "{adv} {adj}",
    ],
    "question": [
        "does the {noun} {verb} {adv}",
        "why {verb} the {adj} {noun}",
    ],
}

# Noun-like endings used by the NOUN_HEAVY strategy.
_NOUN_ENDINGS = ("ness", "ment", "tion", "sion", "ity", "ence", "ance", "ism", "ist", "er", "or", "dom")

# Consonant-only representation for PHONETIC_CLUSTER.
_VOWELS = set("aeiou")


def _consonant_skeleton(word: str) -> str:
    """Return the consonants of *word* in order (lowercased)."""
    return "".join(c for c in word.lower() if c.isalpha() and c not in _VOWELS)


class DictionaryChaosEngine:
    """Structured word sampling, grammatical arrangement, and semantic clustering."""

    def __init__(self, word_loader: WordLoader | None = None) -> None:
        self._loader = word_loader or get_loader()
        self._grammar_templates: dict[str, list[str]] | None = None

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------

    def _load_grammar_templates(self) -> dict[str, list[str]]:
        """Load grammar_templates.json from data/, falling back to inline defaults."""
        if self._grammar_templates is not None:
            return self._grammar_templates

        if _TEMPLATES_FILE.exists():
            with open(_TEMPLATES_FILE, encoding="utf-8") as fh:
                self._grammar_templates = json.load(fh)
        else:
            self._grammar_templates = _DEFAULT_TEMPLATES

        return self._grammar_templates

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_words(
        self, n: int, strategy: SamplingStrategy, rng: random.Random
    ) -> list[str]:
        """Sample *n* words using the given strategy."""
        if strategy == SamplingStrategy.UNIFORM:
            return self._loader.get_random_words(n, rng)

        if strategy == SamplingStrategy.FREQUENCY_WEIGHTED:
            return self._sample_frequency_weighted(n, rng, prefer_rare=False)

        if strategy == SamplingStrategy.RARE_WORDS:
            return self._sample_frequency_weighted(n, rng, prefer_rare=True)

        if strategy == SamplingStrategy.NOUN_HEAVY:
            return self._sample_noun_heavy(n, rng)

        if strategy == SamplingStrategy.PHONETIC_CLUSTER:
            return self._sample_phonetic_cluster(n, rng)

        # Fallback
        return self._loader.get_random_words(n, rng)

    def _sample_frequency_weighted(
        self, n: int, rng: random.Random, *, prefer_rare: bool
    ) -> list[str]:
        """Sample candidates, then pick *n* weighted by zipf frequency."""
        candidates = self._loader.get_random_words(max(200, n * 7), rng)

        if prefer_rare:
            # Prefer words with zipf < 3.0; use inverse frequency as weight.
            weights = []
            for w in candidates:
                zf = zipf_frequency(w, "en")
                # Boost words below the rarity threshold.
                weights.append(1.0 / (zf + 0.1) if zf < 3.0 else 0.05)
        else:
            weights = [max(zipf_frequency(w, "en"), 0.01) for w in candidates]

        chosen: list[str] = []
        # Weighted sampling without full replacement bias.
        for _ in range(n):
            pick = _weighted_choice(candidates, weights, rng)
            chosen.append(pick)

        return chosen

    def _sample_noun_heavy(self, n: int, rng: random.Random) -> list[str]:
        """60% noun-like words (by suffix), 40% random."""
        noun_count = int(n * 0.6)
        random_count = n - noun_count

        # Gather noun-like candidates from a larger random pool.
        pool = self._loader.get_random_words(max(500, n * 10), rng)
        nouns = [w for w in pool if w.lower().endswith(_NOUN_ENDINGS)]

        # If we don't have enough nouns, pad with more random draws.
        while len(nouns) < noun_count:
            extra = self._loader.get_random_words(200, rng)
            nouns.extend(w for w in extra if w.lower().endswith(_NOUN_ENDINGS))

        rng.shuffle(nouns)
        result = nouns[:noun_count]
        result.extend(self._loader.get_random_words(random_count, rng))
        rng.shuffle(result)
        return result

    def _sample_phonetic_cluster(self, n: int, rng: random.Random) -> list[str]:
        """Sample words, then cluster by shared consonant patterns and return representatives."""
        pool = self._loader.get_random_words(max(200, n * 7), rng)

        clusters: dict[str, list[str]] = defaultdict(list)
        for w in pool:
            skel = _consonant_skeleton(w)
            # Use the first 3 consonants as a cluster key to keep groups coarse.
            key = skel[:3] if len(skel) >= 3 else skel
            clusters[key].append(w)

        # Pick from the largest clusters first.
        sorted_clusters = sorted(clusters.values(), key=len, reverse=True)
        result: list[str] = []
        for cluster in sorted_clusters:
            if len(result) >= n:
                break
            pick_count = min(len(cluster), max(1, n // len(sorted_clusters) + 1))
            result.extend(rng.sample(cluster, min(pick_count, len(cluster))))

        # Trim or pad to exactly n.
        if len(result) > n:
            result = result[:n]
        while len(result) < n:
            result.append(self._loader.get_random_word(rng))

        return result

    # ------------------------------------------------------------------
    # Grammatical arrangement
    # ------------------------------------------------------------------

    def arrange_grammatically(
        self, words: list[str], rng: random.Random
    ) -> list[str]:
        """Arrange words into quasi-grammatical phrases using templates.

        POS-tags words via wordnet_utils (with heuristic fallback), then fills
        template slots ``{noun}``, ``{verb}``, ``{adj}``, ``{adv}`` from the
        tagged buckets.
        """
        templates = self._load_grammar_templates()

        # Bucket words by POS tag.
        buckets: dict[str, list[str]] = defaultdict(list)
        for w in words:
            tag = get_pos_tag(w)
            buckets[tag].append(w)

        # Ensure every slot has at least a fallback.
        for tag in ("noun", "verb", "adj", "adv"):
            if not buckets[tag]:
                buckets[tag] = list(words)  # degrade gracefully

        all_templates: list[str] = []
        for style_templates in templates.values():
            all_templates.extend(style_templates)

        phrases: list[str] = []
        for tmpl in rng.sample(all_templates, min(len(all_templates), max(5, len(words) // 4))):
            phrase = tmpl
            for tag in ("noun", "verb", "adj", "adv"):
                placeholder = "{" + tag + "}"
                while placeholder in phrase:
                    phrase = phrase.replace(placeholder, rng.choice(buckets[tag]), 1)
            phrases.append(phrase)

        return phrases

    # ------------------------------------------------------------------
    # Semantic clustering
    # ------------------------------------------------------------------

    def extract_semantic_clusters(self, words: list[str]) -> list[list[str]]:
        """Group words by shared WordNet hypernyms.

        Falls back to grouping by first letter if WordNet is unavailable.
        """
        hyp_map: dict[str, list[str]] = defaultdict(list)
        orphans: list[str] = []

        for w in words:
            hypernyms = get_hypernyms(w)
            if hypernyms:
                for h in hypernyms:
                    hyp_map[h].append(w)
            else:
                orphans.append(w)

        # Build clusters from hypernyms that link 2+ words.
        seen: set[str] = set()
        clusters: list[list[str]] = []
        for _hyp, members in sorted(hyp_map.items(), key=lambda kv: -len(kv[1])):
            cluster = [w for w in members if w not in seen]
            if len(cluster) >= 2:
                clusters.append(cluster)
                seen.update(cluster)

        # Fallback grouping for orphans / unclustered words.
        unclustered = [w for w in words if w not in seen]
        if unclustered:
            letter_groups: dict[str, list[str]] = defaultdict(list)
            for w in unclustered:
                key = w[0].lower() if w else "_"
                letter_groups[key].append(w)
            for group in letter_groups.values():
                if len(group) >= 2:
                    clusters.append(group)
                else:
                    # Singletons go into a misc cluster.
                    pass

        return clusters

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process(
        self,
        rng: random.Random,
        strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
        n_words: int = 30,
        soup_result: LetterSoupResult | None = None,
    ) -> DictionaryChaosResult:
        """Full pipeline: sample -> arrange -> cluster -> combine with soup if provided."""
        words = self.sample_words(n_words, strategy, rng)
        arrangements = self.arrange_grammatically(words, rng)
        clusters = self.extract_semantic_clusters(words)

        combined = list(words)
        if soup_result:
            combined.extend(soup_result.all_words())

        return DictionaryChaosResult(
            sampled_words=words,
            strategy=strategy,
            grammatical_arrangements=arrangements,
            semantic_clusters=clusters,
            combined_words=combined,
        )


def _weighted_choice(
    items: list[str], weights: list[float], rng: random.Random
) -> str:
    """Weighted random choice using the supplied RNG."""
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0.0
    for item, w in zip(items, weights):
        cumulative += w
        if r <= cumulative:
            return item
    return items[-1]
