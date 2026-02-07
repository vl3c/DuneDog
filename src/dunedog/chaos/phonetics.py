"""Phonetic analysis for chaos layer."""

from __future__ import annotations

VOWELS = set("aeiou")
SOFT_CONSONANTS = set("lmnrwy")  # dreamy
HARD_CONSONANTS = set("kptdbg")  # urgent
SIBILANTS = set("szfv")  # secretive
# remaining: h, j, c, q, x, z -> neutral

# Common English consonant clusters (onsets)
VALID_ONSETS = {
    "bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr",
    "sc", "sk", "sl", "sm", "sn", "sp", "st", "str", "sw", "tr", "tw",
    "th", "sh", "ch", "wh", "wr", "kn", "qu", "spr", "spl", "scr",
}


def pronounceability_score(text: str) -> float:
    """Score 0.0-1.0 based on CV patterns and consonant cluster rules.

    Higher = more pronounceable.  Checks consonant clusters, vowel ratio,
    and penalises long runs of consonants or vowels.
    """
    if not text:
        return 0.0

    t = text.lower()
    alpha = [c for c in t if c.isalpha()]
    if not alpha:
        return 0.0

    score = 1.0
    n = len(alpha)

    # --- vowel ratio (ideal ~35-55%) ---
    vowel_count = sum(1 for c in alpha if c in VOWELS)
    vowel_ratio = vowel_count / n
    if vowel_ratio == 0:
        return 0.0  # no vowels -> unpronounceable
    if vowel_ratio < 0.15:
        score -= 0.4
    elif vowel_ratio < 0.25:
        score -= 0.2
    elif vowel_ratio > 0.70:
        score -= 0.15

    # --- consonant cluster check ---
    consonant_run = 0
    max_consonant_run = 0
    for c in alpha:
        if c not in VOWELS:
            consonant_run += 1
            max_consonant_run = max(max_consonant_run, consonant_run)
        else:
            consonant_run = 0

    if max_consonant_run >= 4:
        score -= 0.35
    elif max_consonant_run == 3:
        score -= 0.10

    # --- vowel run check ---
    vowel_run = 0
    max_vowel_run = 0
    for c in alpha:
        if c in VOWELS:
            vowel_run += 1
            max_vowel_run = max(max_vowel_run, vowel_run)
        else:
            vowel_run = 0

    if max_vowel_run >= 4:
        score -= 0.20
    elif max_vowel_run == 3:
        score -= 0.05

    # --- onset cluster validation ---
    # Check 2- and 3-char consonant clusters at start
    onset = []
    for c in alpha:
        if c not in VOWELS:
            onset.append(c)
        else:
            break
    if len(onset) >= 2:
        cluster = "".join(onset)
        if cluster not in VALID_ONSETS:
            score -= 0.15

    # --- repeated characters ---
    for i in range(len(alpha) - 2):
        if alpha[i] == alpha[i + 1] == alpha[i + 2]:
            score -= 0.25
            break

    return max(0.0, min(1.0, score))


def is_pronounceable(text: str, threshold: float = 0.4) -> bool:
    """Whether text meets minimum pronounceability threshold."""
    return pronounceability_score(text) >= threshold


def looks_like_name(text: str) -> bool:
    """Heuristic: length 3-10, starts with a consonant, has vowels, pronounceable."""
    if not text:
        return False
    t = text.lower()
    if not t.isalpha():
        return False
    if not (3 <= len(t) <= 10):
        return False
    if t[0] in VOWELS:
        return False  # starts with vowel — less name-like
    if not any(c in VOWELS for c in t):
        return False
    return is_pronounceable(t, threshold=0.5)


def analyze_phonetic_mood(text: str) -> str:
    """Analyse the phonetic character of text.

    Returns one of:
      'dreamy'     — soft consonants dominate
      'urgent'     — hard consonants dominate
      'secretive'  — sibilants dominate
      'flowing'    — vowel-heavy
      'balanced'   — mixed
    """
    if not text:
        return "balanced"

    t = text.lower()
    alpha = [c for c in t if c.isalpha()]
    if not alpha:
        return "balanced"

    vowel_count = sum(1 for c in alpha if c in VOWELS)
    soft_count = sum(1 for c in alpha if c in SOFT_CONSONANTS)
    hard_count = sum(1 for c in alpha if c in HARD_CONSONANTS)
    sib_count = sum(1 for c in alpha if c in SIBILANTS)

    n = len(alpha)
    vowel_ratio = vowel_count / n

    # If vowels dominate (>55%), it's flowing
    if vowel_ratio > 0.55:
        return "flowing"

    consonant_counts = {
        "dreamy": soft_count,
        "urgent": hard_count,
        "secretive": sib_count,
    }
    top_mood = max(consonant_counts, key=consonant_counts.get)  # type: ignore[arg-type]
    top_count = consonant_counts[top_mood]

    # Need at least 30% of consonants to be in a single category to be dominant
    consonant_total = n - vowel_count
    if consonant_total == 0:
        return "flowing"

    if top_count / consonant_total >= 0.40:
        return top_mood

    return "balanced"
