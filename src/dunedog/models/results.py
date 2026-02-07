"""Results from chaos layers."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class SamplingStrategy(enum.Enum):
    """Word sampling strategies for dictionary chaos."""
    UNIFORM = "uniform"
    FREQUENCY_WEIGHTED = "frequency_weighted"
    RARE_WORDS = "rare_words"
    NOUN_HEAVY = "noun_heavy"
    PHONETIC_CLUSTER = "phonetic_cluster"


@dataclass
class Neologism:
    """A pronounceable non-word extracted from chaos."""
    text: str
    pronounceability: float  # 0.0–1.0
    phonetic_mood: str = ""  # e.g. "dreamy", "urgent", "secretive"
    definition: str = ""
    part_of_speech: str = ""
    usage_example: str = ""
    source_context: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "pronounceability": self.pronounceability,
            "phonetic_mood": self.phonetic_mood,
            "definition": self.definition,
            "part_of_speech": self.part_of_speech,
            "usage_example": self.usage_example,
            "source_context": self.source_context,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Neologism:
        return cls(**data)


@dataclass
class LetterSoupResult:
    """Output from the letter soup generator."""
    raw_soup: str
    exact_words: list[str] = field(default_factory=list)
    near_words: list[tuple[str, str, int]] = field(default_factory=list)  # (fragment, match, distance)
    neologisms: list[Neologism] = field(default_factory=list)
    phonetic_mood: str = ""

    def all_words(self) -> list[str]:
        """All extracted words (exact + near matches)."""
        return self.exact_words + [match for _, match, _ in self.near_words]

    def to_dict(self) -> dict:
        return {
            "raw_soup": self.raw_soup,
            "exact_words": self.exact_words,
            "near_words": self.near_words,
            "neologisms": [n.to_dict() for n in self.neologisms],
            "phonetic_mood": self.phonetic_mood,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LetterSoupResult:
        return cls(
            raw_soup=data["raw_soup"],
            exact_words=data["exact_words"],
            near_words=[tuple(x) for x in data["near_words"]],
            neologisms=[Neologism.from_dict(n) for n in data.get("neologisms", [])],
            phonetic_mood=data.get("phonetic_mood", ""),
        )


@dataclass
class DictionaryChaosResult:
    """Output from the dictionary chaos engine."""
    sampled_words: list[str] = field(default_factory=list)
    strategy: SamplingStrategy = SamplingStrategy.UNIFORM
    grammatical_arrangements: list[str] = field(default_factory=list)
    semantic_clusters: list[list[str]] = field(default_factory=list)
    combined_words: list[str] = field(default_factory=list)  # merged with letter soup if applicable

    def to_dict(self) -> dict:
        return {
            "sampled_words": self.sampled_words,
            "strategy": self.strategy.value,
            "grammatical_arrangements": self.grammatical_arrangements,
            "semantic_clusters": self.semantic_clusters,
            "combined_words": self.combined_words,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DictionaryChaosResult:
        return cls(
            sampled_words=data["sampled_words"],
            strategy=SamplingStrategy(data["strategy"]),
            grammatical_arrangements=data.get("grammatical_arrangements", []),
            semantic_clusters=data.get("semantic_clusters", []),
            combined_words=data.get("combined_words", []),
        )
