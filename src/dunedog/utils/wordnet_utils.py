"""WordNet wrapper with graceful fallback when NLTK data is absent."""

from __future__ import annotations

from functools import lru_cache


def _nltk_available() -> bool:
    """Return True if NLTK and WordNet data are importable."""
    try:
        from nltk.corpus import wordnet  # noqa: F401
        # Trigger actual data access to verify the corpus is downloaded.
        wordnet.synsets("test")
        return True
    except Exception:
        return False


@lru_cache(maxsize=200_000)
def _synsets(word: str) -> tuple:
    """Cached synset lookup — the pipeline asks about the same words constantly."""
    try:
        from nltk.corpus import wordnet
        return tuple(wordnet.synsets(word))
    except Exception:
        return ()


def get_synsets(word: str) -> list:
    """Return WordNet synsets for *word*, or [] if unavailable."""
    return list(_synsets(word))


def wup_similarity(word_a: str, word_b: str) -> float:
    """Wu-Palmer similarity between the first synsets of two words.

    Returns 0.0 on any failure (missing data, no synsets, etc.).
    """
    try:
        syns_a = _synsets(word_a)
        syns_b = _synsets(word_b)
        if not syns_a or not syns_b:
            return 0.0
        score = syns_a[0].wup_similarity(syns_b[0])
        return float(score) if score is not None else 0.0
    except Exception:
        return 0.0


def get_hypernyms(word: str) -> list[str]:
    """Return hypernym lemma names for the first synset of *word*."""
    try:
        syns = _synsets(word)
        if not syns:
            return []
        lemmas: list[str] = []
        # Sorted: NLTK's hypernym order varies between processes, and callers
        # use the first entry, so an unsorted list breaks reproducibility.
        for hypernym in sorted(syns[0].hypernyms(), key=lambda h: h.name()):
            lemmas.extend(l.name() for l in hypernym.lemmas())
        return lemmas
    except Exception:
        return []


def get_synonyms(word: str) -> list[str]:
    """Return lemma names of the first synset of *word* (its closest synonyms)."""
    try:
        syns = _synsets(word)
        return [l.name() for l in syns[0].lemmas()] if syns else []
    except Exception:
        return []


_POS_MAP = {
    "n": "noun",
    "v": "verb",
    "a": "adj",
    "s": "adj",
    "r": "adv",
}


def _heuristic_pos(word: str) -> str:
    """Very rough POS guess when WordNet is unavailable."""
    w = word.lower()
    if w.endswith(("ly",)):
        return "adv"
    if w.endswith(("ness", "ment", "tion", "sion", "ity")):
        return "noun"
    if w.endswith(("ing", "ed", "ate", "ify", "ize")):
        return "verb"
    if w.endswith(("ful", "less", "ous", "ive", "able", "ible", "al", "ish")):
        return "adj"
    return "noun"


def get_pos_tag(word: str) -> str:
    """Simplified POS tag: noun / verb / adj / adv.

    Uses WordNet when available, falls back to suffix heuristics.
    """
    try:
        syns = _synsets(word)
        if syns:
            return _POS_MAP.get(syns[0].pos(), "noun")
    except Exception:
        pass
    return _heuristic_pos(word)
