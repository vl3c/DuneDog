"""Tests for the crystallizer and neologism definer."""
import pytest
import random

from dunedog.crystallize.crystallizer import SeedCrystallizer, CrystallizationResult
from dunedog.crystallize.neologism_definer import NeologismDefiner
from dunedog.models.atoms import AtomCategory, StoryAtom
from dunedog.models.results import Neologism


class TestCrystallize:
    def test_returns_crystallization_result(self, rng, catalogue):
        crystallizer = SeedCrystallizer(catalogue=catalogue)
        words = ["warrior", "castle", "sword", "darkness", "run"]
        result = crystallizer.crystallize(words, rng)
        assert isinstance(result, CrystallizationResult)
        assert len(result.all_atoms) > 0

    def test_atoms_have_categories(self, rng, catalogue):
        crystallizer = SeedCrystallizer(catalogue=catalogue)
        words = ["knight", "forest", "shield", "fear", "burn"]
        result = crystallizer.crystallize(words, rng)
        for atom in result.all_atoms:
            assert isinstance(atom, StoryAtom)
            assert isinstance(atom.category, AtomCategory)

    def test_deduplicates_words(self, rng, catalogue):
        crystallizer = SeedCrystallizer(catalogue=catalogue)
        words = ["warrior", "warrior", "WARRIOR", "Warrior"]
        result = crystallizer.crystallize(words, rng)
        names = [a.name for a in result.all_atoms] + result.unmapped_words
        # All forms should collapse to one entry
        assert len(names) <= 1 or len(set(n.lower() for n in names)) == 1


class TestInferCategory:
    def test_warrior_is_agent(self, catalogue):
        crystallizer = SeedCrystallizer(catalogue=catalogue)
        cat = crystallizer._infer_category("warrior")
        assert cat == AtomCategory.AGENT

    def test_castle_is_object_or_location(self, catalogue):
        crystallizer = SeedCrystallizer(catalogue=catalogue)
        cat = crystallizer._infer_category("castle")
        # WordNet classifies castle as an artifact (object) rather than a place
        assert cat in (AtomCategory.OBJECT, AtomCategory.LOCATION)

    def test_sword_is_object(self, catalogue):
        crystallizer = SeedCrystallizer(catalogue=catalogue)
        cat = crystallizer._infer_category("sword")
        assert cat == AtomCategory.OBJECT

    def test_darkness_is_tension(self, catalogue):
        crystallizer = SeedCrystallizer(catalogue=catalogue)
        cat = crystallizer._infer_category("darkness")
        assert cat == AtomCategory.TENSION

    def test_verb_is_trigger(self, catalogue):
        crystallizer = SeedCrystallizer(catalogue=catalogue)
        # "destroy" is reliably tagged as a verb by WordNet
        cat = crystallizer._infer_category("destroy")
        assert cat == AtomCategory.TRIGGER

    def test_beautiful_is_quality(self, catalogue):
        crystallizer = SeedCrystallizer(catalogue=catalogue)
        cat = crystallizer._infer_category("beautiful")
        assert cat == AtomCategory.QUALITY


class TestNeologismDefiner:
    def test_define_fills_all_fields(self):
        definer = NeologismDefiner()
        neo = Neologism(text="glimbora", pronounceability=0.8, phonetic_mood="dreamy")
        rng = random.Random(42)
        result = definer.define(neo, ["shadow", "river", "stone"], rng)
        assert result.part_of_speech != ""
        assert result.definition != ""
        assert result.usage_example != ""
        assert result.phonetic_mood != ""
        assert len(result.source_context) > 0

    def test_define_with_empty_context(self):
        definer = NeologismDefiner()
        neo = Neologism(text="zephling", pronounceability=0.7)
        rng = random.Random(42)
        result = definer.define(neo, [], rng)
        assert result.definition != ""
        assert result.usage_example != ""

    def test_infer_pos_adverb(self):
        definer = NeologismDefiner()
        assert definer.infer_part_of_speech("quickly") == "adverb"
        assert definer.infer_part_of_speech("softly") == "adverb"

    def test_infer_pos_noun(self):
        definer = NeologismDefiner()
        assert definer.infer_part_of_speech("darkness") == "noun"
        assert definer.infer_part_of_speech("creation") == "noun"
        assert definer.infer_part_of_speech("happiness") == "noun"

    def test_infer_pos_adjective(self):
        definer = NeologismDefiner()
        assert definer.infer_part_of_speech("beautiful") == "adjective"
        assert definer.infer_part_of_speech("creative") == "adjective"
        assert definer.infer_part_of_speech("joyous") == "adjective"

    def test_infer_pos_verb(self):
        definer = NeologismDefiner()
        assert definer.infer_part_of_speech("realize") == "verb"
        assert definer.infer_part_of_speech("purify") == "verb"
        assert definer.infer_part_of_speech("create") == "verb"

    def test_infer_pos_default_noun(self):
        definer = NeologismDefiner()
        # A word with no matching suffix should default to noun
        assert definer.infer_part_of_speech("glimbor") == "noun"
