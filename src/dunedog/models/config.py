"""Generation configuration with preset system (Pydantic)."""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


class Preset(enum.Enum):
    """Named pipeline presets."""
    QUICK = "quick"
    DEEP = "deep"
    EXPERIMENTAL = "experimental"
    CUSTOM = "custom"


class ChaosConfig(BaseModel):
    """Configuration for the chaos layer."""
    use_letter_soup: bool = True
    use_dictionary: bool = True
    soup_length: int = 200
    dictionary_word_count: int = 30
    sampling_strategy: str = "uniform"
    enable_near_words: bool = True


class CrystallizationConfig(BaseModel):
    """Configuration for crystallization."""
    enable_neologisms: bool = True
    similarity_threshold: float = 0.6
    max_atoms_per_category: int = 10


class EngineConfig(BaseModel):
    """Configuration for generative engines."""
    use_tarot: bool = True
    use_markov: bool = True
    use_constraint_solver: bool = True
    spread_types: list[str] = Field(default_factory=lambda: ["hero_journey"])
    min_beats: int = 5
    max_beats: int = 12


class EvolutionConfig(BaseModel):
    """Configuration for evolutionary engine."""
    enabled: bool = False
    generations: int = 0
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    tournament_size: int = 3
    wild_card_rate: float = 0.05


class LLMConfig(BaseModel):
    """Configuration for LLM synthesis."""
    provider: str = "anthropic"
    model: str = ""
    api_key: str = ""
    story_lines: int = 20
    max_stories_for_llm: int = 20
    synthesis_strategy: str = ""  # empty = let LLM choose
    temperature: float = 0.8


# -- Preset definitions --

_QUICK_OVERRIDES: dict[str, Any] = {
    "chaos": {"use_dictionary": False, "enable_near_words": False, "soup_length": 100},
    "crystallization": {"enable_neologisms": False},
    "engines": {"use_markov": False, "use_constraint_solver": False},
    "evolution": {"enabled": False, "generations": 0},
    "skeletons_to_generate": 50,
    "llm": {"max_stories_for_llm": 5},
}

_DEEP_OVERRIDES: dict[str, Any] = {
    "chaos": {"use_letter_soup": True, "use_dictionary": True},
    "crystallization": {"enable_neologisms": True},
    "engines": {"use_tarot": True, "use_markov": True, "use_constraint_solver": True},
    "evolution": {"enabled": True, "generations": 5, "population_size": 50},
    "skeletons_to_generate": 200,
    "llm": {"max_stories_for_llm": 20},
}

_EXPERIMENTAL_OVERRIDES: dict[str, Any] = {
    "chaos": {"use_letter_soup": True, "use_dictionary": True, "soup_length": 400},
    "crystallization": {"enable_neologisms": True},
    "engines": {"use_tarot": True, "use_markov": True, "use_constraint_solver": True},
    "evolution": {"enabled": True, "generations": 20, "population_size": 100, "mutation_rate": 0.15},
    "skeletons_to_generate": 500,
    "llm": {"max_stories_for_llm": 20},
}

PRESET_CONFIGS: dict[Preset, dict[str, Any]] = {
    Preset.QUICK: _QUICK_OVERRIDES,
    Preset.DEEP: _DEEP_OVERRIDES,
    Preset.EXPERIMENTAL: _EXPERIMENTAL_OVERRIDES,
}


class GenerationConfig(BaseModel):
    """Full pipeline configuration."""
    preset: Preset = Preset.DEEP
    seed: int | None = None
    skeletons_to_generate: int = 200

    chaos: ChaosConfig = Field(default_factory=ChaosConfig)
    crystallization: CrystallizationConfig = Field(default_factory=CrystallizationConfig)
    engines: EngineConfig = Field(default_factory=EngineConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @classmethod
    def from_preset(cls, preset: Preset | str, **overrides: Any) -> GenerationConfig:
        """Create a config from a named preset with optional overrides."""
        if isinstance(preset, str):
            preset = Preset(preset)

        if preset == Preset.CUSTOM:
            return cls(preset=preset, **overrides)

        base = PRESET_CONFIGS.get(preset, {})
        merged: dict[str, Any] = {"preset": preset}

        for key, value in base.items():
            if isinstance(value, dict) and key in overrides and isinstance(overrides[key], dict):
                merged[key] = {**value, **overrides.pop(key)}
            else:
                merged[key] = value

        merged.update(overrides)
        return cls(**merged)
