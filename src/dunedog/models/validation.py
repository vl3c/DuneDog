"""Validation models — invariants and tendencies for world rules."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class InvariantSeverity(enum.Enum):
    """How strictly an invariant must be enforced."""
    HARD = "hard"  # Must never be violated
    SOFT = "soft"  # Should be respected, violations logged


@dataclass
class Invariant:
    """A world rule that must (or should) hold."""
    name: str
    description: str
    severity: InvariantSeverity = InvariantSeverity.HARD
    check_type: str = ""  # dispatch key for rule engine (e.g. "requires_atom", "forbids_combo")
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "check_type": self.check_type,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Invariant:
        return cls(
            name=data["name"],
            description=data["description"],
            severity=InvariantSeverity(data.get("severity", "hard")),
            check_type=data.get("check_type", ""),
            parameters=data.get("parameters", {}),
        )


@dataclass
class Tendency:
    """A probabilistic tendency — not a hard rule, but a weighted preference."""
    name: str
    description: str
    probability: float  # 0.0–1.0, chance this tendency fires
    effect: str = ""  # what happens when it fires
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "probability": self.probability,
            "effect": self.effect,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Tendency:
        return cls(
            name=data["name"],
            description=data["description"],
            probability=data["probability"],
            effect=data.get("effect", ""),
            parameters=data.get("parameters", {}),
        )


@dataclass
class ValidationResult:
    """Result of validating a skeleton against world rules."""
    valid: bool = True
    hard_violations: list[str] = field(default_factory=list)
    soft_violations: list[str] = field(default_factory=list)
    tendencies_applied: list[str] = field(default_factory=list)
    score_adjustments: float = 0.0

    def add_violation(self, message: str, severity: InvariantSeverity) -> None:
        if severity == InvariantSeverity.HARD:
            self.hard_violations.append(message)
            self.valid = False
        else:
            self.soft_violations.append(message)

    @property
    def total_violations(self) -> int:
        return len(self.hard_violations) + len(self.soft_violations)
