"""Extended tests for the world rules engine."""

import random

import pytest

from dunedog.world_rules.engine import WorldRulesEngine
from dunedog.models.skeleton import StorySkeleton
from dunedog.models.atoms import StoryAtom, AtomCategory, AtomSource
from dunedog.models.validation import Invariant, InvariantSeverity, Tendency


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_skeleton(atoms=None, beats=None, tone="", theme_tags=None):
    """Create a StorySkeleton with the given components."""
    return StorySkeleton(
        atoms=atoms or [],
        beats=beats or [],
        tone=tone,
        theme_tags=theme_tags or [],
    )


def _make_invariant(name="test", check_type="", severity=InvariantSeverity.HARD, parameters=None):
    """Create an Invariant with sensible defaults."""
    return Invariant(
        name=name,
        description=f"Test invariant: {name}",
        check_type=check_type,
        severity=severity,
        parameters=parameters or {},
    )


def _make_tendency(name="test_tend", probability=1.0, effect="", parameters=None):
    """Create a Tendency with sensible defaults."""
    return Tendency(
        name=name,
        description=f"Test tendency: {name}",
        probability=probability,
        effect=effect,
        parameters=parameters or {},
    )


# ------------------------------------------------------------------ #
# check_invariant tests
# ------------------------------------------------------------------ #


class TestCheckInvariant:
    """Tests for WorldRulesEngine.check_invariant."""

    def test_requires_atom_passes(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(atoms=[
            StoryAtom("magic user", AtomCategory.AGENT, AtomSource.CATALOGUE, ["magic"]),
        ])
        inv = _make_invariant(
            check_type="requires_atom",
            parameters={"requires_tag": "magic"},
        )
        assert engine.check_invariant(skeleton, inv) is None

    def test_requires_atom_fails(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton()
        inv = _make_invariant(
            check_type="requires_atom",
            parameters={"requires_tag": "magic"},
        )
        result = engine.check_invariant(skeleton, inv)
        assert result is not None
        assert "magic" in result

    def test_requires_atom_multiple_atoms_passes(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(atoms=[
            StoryAtom("fighter", AtomCategory.AGENT, AtomSource.CATALOGUE, ["combat"]),
            StoryAtom("wizard", AtomCategory.AGENT, AtomSource.CATALOGUE, ["magic"]),
        ])
        inv = _make_invariant(
            check_type="requires_atom",
            parameters={"requires_tag": "magic"},
        )
        assert engine.check_invariant(skeleton, inv) is None

    def test_forbids_combo_passes(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(atoms=[
            StoryAtom("a", AtomCategory.AGENT, AtomSource.CATALOGUE, ["fire"]),
        ])
        inv = _make_invariant(
            check_type="forbids_combo",
            parameters={"tag_a": "fire", "tag_b": "ice"},
        )
        assert engine.check_invariant(skeleton, inv) is None

    def test_forbids_combo_fails(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(atoms=[
            StoryAtom("a", AtomCategory.AGENT, AtomSource.CATALOGUE, ["fire"]),
            StoryAtom("b", AtomCategory.AGENT, AtomSource.CATALOGUE, ["ice"]),
        ])
        inv = _make_invariant(
            check_type="forbids_combo",
            parameters={"tag_a": "fire", "tag_b": "ice"},
        )
        assert engine.check_invariant(skeleton, inv) is not None

    def test_forbids_combo_both_tags_on_same_atom(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(atoms=[
            StoryAtom("a", AtomCategory.AGENT, AtomSource.CATALOGUE, ["fire", "ice"]),
        ])
        inv = _make_invariant(
            check_type="forbids_combo",
            parameters={"tag_a": "fire", "tag_b": "ice"},
        )
        assert engine.check_invariant(skeleton, inv) is not None

    def test_forbids_combo_neither_present(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(atoms=[
            StoryAtom("a", AtomCategory.AGENT, AtomSource.CATALOGUE, ["wind"]),
        ])
        inv = _make_invariant(
            check_type="forbids_combo",
            parameters={"tag_a": "fire", "tag_b": "ice"},
        )
        assert engine.check_invariant(skeleton, inv) is None

    def test_requires_beat_passes(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(beats=["OPENING", "CLIMAX"])
        inv = _make_invariant(
            check_type="requires_beat",
            severity=InvariantSeverity.SOFT,
            parameters={"beat": "CLIMAX"},
        )
        assert engine.check_invariant(skeleton, inv) is None

    def test_requires_beat_fails(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(beats=["OPENING"])
        inv = _make_invariant(
            check_type="requires_beat",
            severity=InvariantSeverity.SOFT,
            parameters={"beat": "CLIMAX"},
        )
        result = engine.check_invariant(skeleton, inv)
        assert result is not None
        assert "CLIMAX" in result

    def test_requires_beat_empty_beats(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(beats=[])
        inv = _make_invariant(
            check_type="requires_beat",
            parameters={"beat": "OPENING"},
        )
        assert engine.check_invariant(skeleton, inv) is not None

    def test_conditional_passes_when_if_tag_absent(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(atoms=[
            StoryAtom("a", AtomCategory.AGENT, AtomSource.CATALOGUE, ["other"]),
        ])
        inv = _make_invariant(
            check_type="conditional",
            parameters={"if_tag": "magic", "requires_tag": "cost"},
        )
        assert engine.check_invariant(skeleton, inv) is None

    def test_conditional_passes_when_both_present(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(atoms=[
            StoryAtom("a", AtomCategory.AGENT, AtomSource.CATALOGUE, ["magic", "cost"]),
        ])
        inv = _make_invariant(
            check_type="conditional",
            parameters={"if_tag": "magic", "requires_tag": "cost"},
        )
        assert engine.check_invariant(skeleton, inv) is None

    def test_conditional_passes_when_tags_on_different_atoms(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(atoms=[
            StoryAtom("a", AtomCategory.AGENT, AtomSource.CATALOGUE, ["magic"]),
            StoryAtom("b", AtomCategory.OBJECT, AtomSource.CATALOGUE, ["cost"]),
        ])
        inv = _make_invariant(
            check_type="conditional",
            parameters={"if_tag": "magic", "requires_tag": "cost"},
        )
        assert engine.check_invariant(skeleton, inv) is None

    def test_conditional_fails(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(atoms=[
            StoryAtom("a", AtomCategory.AGENT, AtomSource.CATALOGUE, ["magic"]),
        ])
        inv = _make_invariant(
            check_type="conditional",
            parameters={"if_tag": "magic", "requires_tag": "cost"},
        )
        result = engine.check_invariant(skeleton, inv)
        assert result is not None
        assert "magic" in result
        assert "cost" in result

    def test_unknown_check_type_returns_none(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton()
        inv = _make_invariant(check_type="nonexistent_check")
        assert engine.check_invariant(skeleton, inv) is None

    def test_empty_skeleton_with_requires_atom(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton()
        inv = _make_invariant(
            check_type="requires_atom",
            parameters={"requires_tag": "anything"},
        )
        assert engine.check_invariant(skeleton, inv) is not None


# ------------------------------------------------------------------ #
# apply_tendencies tests
# ------------------------------------------------------------------ #


class TestApplyTendencies:
    """Tests for WorldRulesEngine.apply_tendencies."""

    def test_add_tag(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(atoms=[
            StoryAtom("warrior", AtomCategory.AGENT, AtomSource.CATALOGUE, ["fight"]),
        ])
        engine._tendencies = [_make_tendency(
            name="test_tend",
            probability=1.0,
            effect="add_tag",
            parameters={"tag": "blessed"},
        )]
        rng = random.Random(42)
        applied = engine.apply_tendencies(skeleton, rng)
        assert "test_tend" in applied
        assert "blessed" in skeleton.atoms[0].tags

    def test_add_tag_no_duplicates(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(atoms=[
            StoryAtom("warrior", AtomCategory.AGENT, AtomSource.CATALOGUE, ["fight", "blessed"]),
        ])
        engine._tendencies = [_make_tendency(
            name="test_tend",
            probability=1.0,
            effect="add_tag",
            parameters={"tag": "blessed"},
        )]
        rng = random.Random(42)
        engine.apply_tendencies(skeleton, rng)
        # "blessed" should appear only once
        assert skeleton.atoms[0].tags.count("blessed") == 1

    def test_add_tag_no_atoms_does_not_crash(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(atoms=[])
        engine._tendencies = [_make_tendency(
            name="test_tend",
            probability=1.0,
            effect="add_tag",
            parameters={"tag": "blessed"},
        )]
        rng = random.Random(42)
        applied = engine.apply_tendencies(skeleton, rng)
        # Tendency fires but has no atom to apply to — no mutation recorded
        assert "test_tend" not in applied

    def test_add_tension(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(atoms=[])
        engine._tendencies = [_make_tendency(
            name="t",
            probability=1.0,
            effect="add_tension",
            parameters={"tension": "dread"},
        )]
        applied = engine.apply_tendencies(skeleton, random.Random(42))
        assert len(skeleton.atoms) == 1
        assert skeleton.atoms[0].name == "dread"
        assert skeleton.atoms[0].category == AtomCategory.TENSION
        assert skeleton.atoms[0].source == AtomSource.WILD_CARD
        assert "tension" in skeleton.atoms[0].tags
        assert "dread" in skeleton.atoms[0].tags

    def test_modify_tone(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(tone="calm")
        engine._tendencies = [_make_tendency(
            name="t",
            probability=1.0,
            effect="modify_tone",
            parameters={"tone": "dark"},
        )]
        engine.apply_tendencies(skeleton, random.Random(42))
        assert skeleton.tone == "dark"

    def test_add_beat(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(beats=["OPENING"])
        engine._tendencies = [_make_tendency(
            name="t",
            probability=1.0,
            effect="add_beat",
            parameters={"beat": "REVELATION"},
        )]
        engine.apply_tendencies(skeleton, random.Random(42))
        assert "REVELATION" in skeleton.beats
        assert "OPENING" in skeleton.beats  # original beat preserved

    def test_zero_probability_never_fires(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(atoms=[
            StoryAtom("warrior", AtomCategory.AGENT, AtomSource.CATALOGUE, ["fight"]),
        ])
        engine._tendencies = [_make_tendency(
            name="never",
            probability=0.0,
            effect="add_tag",
            parameters={"tag": "blessed"},
        )]
        rng = random.Random(42)
        applied = engine.apply_tendencies(skeleton, rng)
        assert "never" not in applied
        assert "blessed" not in skeleton.atoms[0].tags

    def test_multiple_tendencies(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(
            atoms=[StoryAtom("warrior", AtomCategory.AGENT, AtomSource.CATALOGUE, ["fight"])],
            beats=["OPENING"],
            tone="neutral",
        )
        engine._tendencies = [
            _make_tendency(name="t1", probability=1.0, effect="add_tag", parameters={"tag": "blessed"}),
            _make_tendency(name="t2", probability=1.0, effect="add_beat", parameters={"beat": "TWIST"}),
            _make_tendency(name="t3", probability=1.0, effect="modify_tone", parameters={"tone": "eerie"}),
        ]
        applied = engine.apply_tendencies(skeleton, random.Random(42))
        assert len(applied) == 3
        assert "t1" in applied
        assert "t2" in applied
        assert "t3" in applied
        assert "TWIST" in skeleton.beats
        assert skeleton.tone == "eerie"

    def test_unknown_effect_not_recorded(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton()
        engine._tendencies = [_make_tendency(
            name="weird",
            probability=1.0,
            effect="nonexistent_effect",
            parameters={},
        )]
        applied = engine.apply_tendencies(skeleton, random.Random(42))
        assert "weird" not in applied


# ------------------------------------------------------------------ #
# validate (full) tests
# ------------------------------------------------------------------ #


class TestValidateFull:
    """Tests for the full validate method."""

    def test_validate_with_tendencies(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(
            atoms=[StoryAtom("a", AtomCategory.AGENT, AtomSource.CATALOGUE, ["test"])],
            beats=["OPENING"],
        )
        result = engine.validate(skeleton, rng=random.Random(42))
        assert result is not None
        # Should have run tendencies
        assert isinstance(result.tendencies_applied, list)

    def test_validate_without_tendencies(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton()
        result = engine.validate(skeleton)
        assert result.tendencies_applied == []

    def test_validate_records_violations(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton()
        # Override invariants to have one that will fail
        engine._invariants = [_make_invariant(
            name="test_req",
            check_type="requires_atom",
            severity=InvariantSeverity.HARD,
            parameters={"requires_tag": "something_missing"},
        )]
        engine._tendencies = []
        result = engine.validate(skeleton)
        assert result.valid is False
        assert len(result.hard_violations) >= 1

    def test_validate_soft_violations(self):
        engine = WorldRulesEngine()
        skeleton = _make_skeleton(beats=[])
        engine._invariants = [_make_invariant(
            name="soft_test",
            check_type="requires_beat",
            severity=InvariantSeverity.SOFT,
            parameters={"beat": "CLIMAX"},
        )]
        engine._tendencies = []
        result = engine.validate(skeleton)
        # Soft violations don't make valid=False
        assert result.valid is True
        assert len(result.soft_violations) >= 1

    def test_validate_no_invariants_no_tendencies(self):
        engine = WorldRulesEngine()
        engine._invariants = []
        engine._tendencies = []
        skeleton = _make_skeleton()
        result = engine.validate(skeleton)
        assert result.valid is True
        assert result.total_violations == 0
        assert result.tendencies_applied == []

    def test_accessors(self):
        engine = WorldRulesEngine()
        assert isinstance(engine.invariants, list)
        assert isinstance(engine.tendencies, list)

    def test_invariants_returns_copy(self):
        engine = WorldRulesEngine()
        inv_list = engine.invariants
        original_len = len(inv_list)
        inv_list.append(_make_invariant(name="extra"))
        # The internal list should not have changed
        assert len(engine.invariants) == original_len

    def test_tendencies_returns_copy(self):
        engine = WorldRulesEngine()
        tend_list = engine.tendencies
        original_len = len(tend_list)
        tend_list.append(_make_tendency(name="extra"))
        assert len(engine.tendencies) == original_len
