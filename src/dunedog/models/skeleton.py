"""Story skeletons — narrative structures built from atoms."""

from __future__ import annotations

from dataclasses import dataclass, field

from .atoms import StoryAtom


@dataclass
class PrimordialSource:
    """Trace back to the chaos layer that produced a skeleton."""
    letter_soup_raw: str = ""
    exact_words: list[str] = field(default_factory=list)
    near_words: list[str] = field(default_factory=list)
    neologisms: list[str] = field(default_factory=list)
    dictionary_words: list[str] = field(default_factory=list)
    phonetic_mood: str = ""

    def to_dict(self) -> dict:
        return {
            "letter_soup_raw": self.letter_soup_raw,
            "exact_words": self.exact_words,
            "near_words": self.near_words,
            "neologisms": self.neologisms,
            "dictionary_words": self.dictionary_words,
            "phonetic_mood": self.phonetic_mood,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PrimordialSource:
        return cls(**data)


@dataclass
class GenerationStats:
    """Statistics about how a skeleton was generated."""
    engine: str = ""
    spread_type: str = ""
    beat_count: int = 0
    violations: list[str] = field(default_factory=list)
    coherence_score: float = 0.0
    generation: int = 0  # evolutionary generation number

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "spread_type": self.spread_type,
            "beat_count": self.beat_count,
            "violations": self.violations,
            "coherence_score": self.coherence_score,
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GenerationStats:
        return cls(**data)


@dataclass
class StorySkeleton:
    """A complete narrative skeleton ready for synthesis."""
    atoms: list[StoryAtom] = field(default_factory=list)
    beats: list[str] = field(default_factory=list)
    spread_positions: dict[str, str] = field(default_factory=dict)  # position_name -> atom_name
    theme_tags: list[str] = field(default_factory=list)
    tone: str = ""
    primordial_source: PrimordialSource = field(default_factory=PrimordialSource)
    stats: GenerationStats = field(default_factory=GenerationStats)
    seed: int | None = None

    @property
    def coherence_score(self) -> float:
        return self.stats.coherence_score

    def to_dict(self) -> dict:
        return {
            "atoms": [a.to_dict() for a in self.atoms],
            "beats": self.beats,
            "spread_positions": self.spread_positions,
            "theme_tags": self.theme_tags,
            "tone": self.tone,
            "primordial_source": self.primordial_source.to_dict(),
            "stats": self.stats.to_dict(),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StorySkeleton:
        return cls(
            atoms=[StoryAtom.from_dict(a) for a in data.get("atoms", [])],
            beats=data.get("beats", []),
            spread_positions=data.get("spread_positions", {}),
            theme_tags=data.get("theme_tags", []),
            tone=data.get("tone", ""),
            primordial_source=PrimordialSource.from_dict(data.get("primordial_source", {})),
            stats=GenerationStats.from_dict(data.get("stats", {})),
            seed=data.get("seed"),
        )


@dataclass
class EvolutionResult:
    """Result of evolutionary optimization."""
    best_skeleton: StorySkeleton | None = None
    population: list[StorySkeleton] = field(default_factory=list)
    generations_run: int = 0
    fitness_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "best_skeleton": self.best_skeleton.to_dict() if self.best_skeleton else None,
            "population": [s.to_dict() for s in self.population],
            "generations_run": self.generations_run,
            "fitness_history": self.fitness_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvolutionResult:
        best = data.get("best_skeleton")
        return cls(
            best_skeleton=StorySkeleton.from_dict(best) if best else None,
            population=[StorySkeleton.from_dict(s) for s in data.get("population", [])],
            generations_run=data.get("generations_run", 0),
            fitness_history=data.get("fitness_history", []),
        )
