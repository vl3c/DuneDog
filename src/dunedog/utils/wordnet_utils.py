"""WordNet wrapper with graceful fallback when NLTK data is absent."""

from __future__ import annotations


def _nltk_available() -> bool:
    """Return True if NLTK and WordNet data are importable."""
    try:
        from nltk.corpus import wordnet  # noqa: F401
        # Trigger actual data access to verify the corpus is downloaded.
        wordnet.synsets("test")
        return True
    except Exception:
        return False


def get_synsets(word: str) -> list:
    """Return WordNet synsets for *word*, or [] if unavailable."""
    try:
        from nltk.corpus import wordnet
        return wordnet.synsets(word)
    except Exception:
        return []


def wup_similarity(word_a: str, word_b: str) -> float:
    """Wu-Palmer similarity between the first synsets of two words.

    Returns 0.0 on any failure (missing data, no synsets, etc.).
    """
    try:
        from nltk.corpus import wordnet
        syns_a = wordnet.synsets(word_a)
        syns_b = wordnet.synsets(word_b)
        if not syns_a or not syns_b:
            return 0.0
        score = syns_a[0].wup_similarity(syns_b[0])
        return float(score) if score is not None else 0.0
    except Exception:
        return 0.0


def get_hypernyms(word: str) -> list[str]:
    """Return hypernym lemma names for the first synset of *word*."""
    try:
        from nltk.corpus import wordnet
        syns = wordnet.synsets(word)
        if not syns:
            return []
        lemmas: list[str] = []
        for hypernym in syns[0].hypernyms():
            lemmas.extend(l.name() for l in hypernym.lemmas())
        return lemmas
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
        from nltk.corpus import wordnet
        syns = wordnet.synsets(word)
        if syns:
            return _POS_MAP.get(syns[0].pos(), "noun")
    except Exception:
        pass
    return _heuristic_pos(word)
