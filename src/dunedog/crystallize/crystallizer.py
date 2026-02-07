"""Seed crystallizer -- maps chaos results to structured story atoms."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from dunedog.models.atoms import AtomCategory, AtomSource, StoryAtom
from dunedog.models.results import DictionaryChaosResult, LetterSoupResult
from dunedog.catalogues.loader import AtomCatalogue
from dunedog.utils.wordnet_utils import get_hypernyms, get_pos_tag
from dunedog.utils.embeddings import SimilarityEngine, get_similarity_engine

# Hypernym lemma sets used for category inference.
_ANIMATE_HYPERNYMS = frozenset({
    "person", "human", "animal", "organism", "being", "creature",
    "people", "character", "individual", "agent", "worker",
    "man", "woman", "child", "leader", "hero", "villain",
})
_PLACE_HYPERNYMS = frozenset({
    "location", "place", "area", "region", "structure", "building",
    "land", "country", "city", "room", "space", "terrain",
    "environment", "territory", "landscape",
})
_OBJECT_HYPERNYMS = frozenset({
    "object", "artifact", "tool", "device", "instrument", "container",
    "weapon", "vehicle", "substance", "material", "food", "clothing",
    "entity", "thing", "article",
})
_TENSION_WORDS = frozenset({
    "conflict", "danger", "fear", "threat", "crisis", "problem",
    "struggle", "chaos", "destruction", "loss", "pain", "risk",
    "mystery", "secret", "betrayal", "war", "death", "doom",
    "darkness", "anger", "hatred", "despair", "tension", "dilemma",
})


@dataclass
class CrystallizationResult:
    """Output of crystallisation: atoms mapped, created, or left unmapped."""

    mapped_atoms: list[StoryAtom] = field(default_factory=list)
    created_atoms: list[StoryAtom] = field(default_factory=list)
    unmapped_words: list[str] = field(default_factory=list)
    theme_summary: str = ""

    @property
    def all_atoms(self) -> list[StoryAtom]:
        return self.mapped_atoms + self.created_atoms


class SeedCrystallizer:
    """Maps chaos-layer words to structured StoryAtoms."""

    def __init__(
        self,
        catalogue: AtomCatalogue | None = None,
        similarity_threshold: float = 0.6,
    ) -> None:
        self._catalogue = catalogue or AtomCatalogue.load()
        self._threshold = similarity_threshold
        self._similarity: SimilarityEngine = get_similarity_engine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crystallize(
        self,
        words: list[str],
        rng: random.Random,
        soup_result: LetterSoupResult | None = None,
        chaos_result: DictionaryChaosResult | None = None,
    ) -> CrystallizationResult:
        """Map *words* to StoryAtoms by inferring category from POS/semantics.

        For each word the pipeline is:
        1. Try to map to an existing catalogue atom (similarity > threshold).
        2. Infer a category and create a new atom.
        3. If no category can be inferred, record as unmapped.

        Extra words from *soup_result* / *chaos_result* are appended to the
        input list so callers can pass raw results alongside an explicit list.
        """
        all_words = list(words)
        if soup_result is not None:
            all_words.extend(soup_result.all_words())
        if chaos_result is not None:
            all_words.extend(chaos_result.combined_words or chaos_result.sampled_words)

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique_words: list[str] = []
        for w in all_words:
            low = w.lower().strip()
            if low and low not in seen:
                seen.add(low)
                unique_words.append(low)

        result = CrystallizationResult()

        for word in unique_words:
            # 1. catalogue match
            mapped = self.map_to_catalogue(word)
            if mapped is not None:
                result.mapped_atoms.append(mapped)
                continue

            # 2. infer category and create atom
            category = self._infer_category(word)
            if category is not None:
                source = self._pick_source(word, soup_result, chaos_result)
                atom = self.create_atom(word, category, source)
                result.created_atoms.append(atom)
                continue

            # 3. unmapped
            result.unmapped_words.append(word)

        result.theme_summary = self._generate_theme_summary(result.all_atoms)
        return result

    # ------------------------------------------------------------------
    # Category inference
    # ------------------------------------------------------------------

    def _infer_category(self, word: str) -> AtomCategory | None:
        """Infer an AtomCategory from word semantics and POS."""
        pos = get_pos_tag(word)

        if pos == "verb":
            return AtomCategory.TRIGGER

        if pos == "adj":
            return AtomCategory.QUALITY

        if pos == "adv":
            return AtomCategory.QUALITY

        if pos == "noun":
            return self._classify_noun(word)

        return None

    def _classify_noun(self, word: str) -> AtomCategory:
        """Distinguish noun sub-categories via hypernyms and heuristics."""
        if word.lower() in _TENSION_WORDS:
            return AtomCategory.TENSION

        hypernyms = {h.lower() for h in get_hypernyms(word)}

        if hypernyms & _ANIMATE_HYPERNYMS:
            return AtomCategory.AGENT
        if hypernyms & _PLACE_HYPERNYMS:
            return AtomCategory.LOCATION
        if hypernyms & _OBJECT_HYPERNYMS:
            return AtomCategory.OBJECT

        # Fallback: abstract nouns often map to TENSION.
        if word.lower().endswith(("tion", "sion", "ment", "ness", "ity")):
            return AtomCategory.TENSION

        # Default for unresolved nouns.
        return AtomCategory.OBJECT

    # ------------------------------------------------------------------
    # Catalogue mapping
    # ------------------------------------------------------------------

    def map_to_catalogue(self, word: str) -> StoryAtom | None:
        """Find the closest catalogue atom by similarity.

        Returns the best match whose score exceeds ``self._threshold``,
        or ``None`` if nothing is close enough.
        """
        catalogue_atoms = self._catalogue.atoms
        if not catalogue_atoms:
            return None

        candidates = [a.name for a in catalogue_atoms]
        top = self._similarity.most_similar(word, candidates, n=1)

        if top and top[0][1] >= self._threshold:
            match_name = top[0][0]
            for atom in catalogue_atoms:
                if atom.name == match_name:
                    return atom
        return None

    # ------------------------------------------------------------------
    # Atom creation
    # ------------------------------------------------------------------

    def create_atom(
        self,
        word: str,
        category: AtomCategory,
        source: AtomSource = AtomSource.DICTIONARY,
    ) -> StoryAtom:
        """Create a new StoryAtom from an extracted word."""
        tags = self._generate_tags(word, category)
        return StoryAtom(
            name=word,
            category=category,
            source=source,
            tags=tags,
            rarity=0.7,  # chaos-derived atoms skew rare
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_source(
        word: str,
        soup_result: LetterSoupResult | None,
        chaos_result: DictionaryChaosResult | None,
    ) -> AtomSource:
        """Determine the AtomSource for *word* based on which result it came from."""
        if soup_result is not None and word in soup_result.all_words():
            return AtomSource.LETTER_SOUP
        if chaos_result is not None:
            combined = chaos_result.combined_words or chaos_result.sampled_words
            if word in combined:
                return AtomSource.DICTIONARY
        return AtomSource.DICTIONARY

    @staticmethod
    def _generate_tags(word: str, category: AtomCategory) -> list[str]:
        """Derive a small set of tags for a newly created atom."""
        tags: list[str] = [category.value]
        pos = get_pos_tag(word)
        if pos:
            tags.append(pos)
        hypernyms = get_hypernyms(word)
        if hypernyms:
            tags.append(hypernyms[0].replace("_", " "))
        return tags

    @staticmethod
    def _generate_theme_summary(atoms: list[StoryAtom]) -> str:
        """Generate a brief theme summary from atom tags and categories."""
        if not atoms:
            return ""

        category_counts: dict[str, int] = {}
        tag_pool: list[str] = []

        for atom in atoms:
            cat = atom.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
            tag_pool.extend(atom.tags)

        dominant = max(category_counts, key=category_counts.get)  # type: ignore[arg-type]

        # Collect unique descriptive tags (skip category/pos echoes).
        skip = {"noun", "verb", "adj", "adv"} | {c.value for c in AtomCategory}
        unique_tags = list(dict.fromkeys(t for t in tag_pool if t not in skip))
        tag_slice = unique_tags[:4]

        parts = [f"dominant-{dominant}"]
        if tag_slice:
            parts.append(" ".join(tag_slice))
        return "; ".join(parts)
