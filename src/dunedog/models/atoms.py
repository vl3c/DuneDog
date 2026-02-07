"""Story atoms — the fundamental units of narrative."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class AtomCategory(enum.Enum):
    """Categories of story atoms."""
    AGENT = "agent"
    OBJECT = "object"
    LOCATION = "location"
    TENSION = "tension"
    TRIGGER = "trigger"
    QUALITY = "quality"


class AtomSource(enum.Enum):
    """How an atom was produced."""
    CATALOGUE = "catalogue"
    LETTER_SOUP = "letter_soup"
    DICTIONARY = "dictionary"
    NEOLOGISM = "neologism"
    WILD_CARD = "wild_card"
    EVOLVED = "evolved"


@dataclass
class StoryAtom:
    """A single narrative element."""
    name: str
    category: AtomCategory
    source: AtomSource
    tags: list[str] = field(default_factory=list)
    rarity: float = 0.5  # 0.0 = common, 1.0 = rare
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category.value,
            "source": self.source.value,
            "tags": self.tags,
            "rarity": self.rarity,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StoryAtom:
        return cls(
            name=data["name"],
            category=AtomCategory(data["category"]),
            source=AtomSource(data["source"]),
            tags=data.get("tags", []),
            rarity=data.get("rarity", 0.5),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AffinityEntry:
    """Affinity between two atoms (sparse, symmetric)."""
    atom_a: str
    atom_b: str
    strength: float  # -1.0 (repulsion) to 1.0 (strong affinity)
    tags: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        """Canonical sorted key for lookup."""
        return tuple(sorted((self.atom_a, self.atom_b)))

    def to_dict(self) -> dict:
        return {
            "atom_a": self.atom_a,
            "atom_b": self.atom_b,
            "strength": self.strength,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AffinityEntry:
        return cls(
            atom_a=data["atom_a"],
            atom_b=data["atom_b"],
            strength=data["strength"],
            tags=data.get("tags", []),
        )
