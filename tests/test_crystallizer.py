"""Tests for the crystallizer and neologism definer."""
import pytest
import random

from dunedog.crystallize.crystallizer import SeedCrystallizer, CrystallizationResult
from dunedog.crystallize.neologism_definer import NeologismDefiner
from dunedog.models.atoms import AtomCategory, AtomSource, StoryAtom
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


class TestMapToCatalogue:
    """Chaos words should be able to reach the curated, multi-word atoms."""

    def test_name_word_evokes_multiword_atom(self, catalogue):
        atom = SeedCrystallizer(catalogue=catalogue).map_to_catalogue("shadow")
        assert atom is not None
        assert atom.source == AtomSource.CATALOGUE
        assert "shadow" in atom.name or "shadow" in atom.tags
        assert atom.metadata["evoked_by"] == "shadow"

    def test_tag_evokes_atom(self, catalogue):
        atom = SeedCrystallizer(catalogue=catalogue).map_to_catalogue("ritual")
        assert atom is not None
        assert "ritual" in atom.tags

    def test_returns_copy_not_catalogue_atom(self, catalogue):
        atom = SeedCrystallizer(catalogue=catalogue).map_to_catalogue("ritual")
        atom.tags.append("tampered")
        assert all("tampered" not in a.tags for a in catalogue.atoms)

    def test_gibberish_evokes_nothing(self, catalogue):
        assert SeedCrystallizer(catalogue=catalogue).map_to_catalogue("xqzvk") is None

    def test_choice_is_deterministic(self, catalogue):
        c = SeedCrystallizer(catalogue=catalogue)
        a = c.map_to_catalogue("memory", random.Random(7))
        b = c.map_to_catalogue("memory", random.Random(7))
        assert a.name == b.name

    def test_crystallize_never_repeats_an_atom(self, rng, catalogue):
        result = SeedCrystallizer(catalogue=catalogue).crystallize(
            ["ritual", "memory", "shadow", "ritual", "time", "silence"], rng)
        names = [a.name for a in result.mapped_atoms]
        assert names and len(names) == len(set(names))


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


class TestNeologismDefinitions:
    """A coined word's definition must not lean on the word itself."""

    def test_definition_never_contains_the_word(self):
        definer = NeologismDefiner()
        for seed in range(200):
            for text in ("wras", "isot", "dete", "glimbora"):
                neo = Neologism(text=text, pronounceability=0.8, phonetic_mood="dreamy")
                result = definer.define(neo, ["knife", "river", text, "stone"], random.Random(seed))
                assert text not in result.definition.split(), (seed, result.definition)

    def test_templates_have_no_self_reference_placeholder(self):
        definer = NeologismDefiner()
        for pos, templates in definer._load_templates().items():
            for t in templates:
                assert "{word}" not in t, (pos, t)

    def test_two_placeholders_get_different_words(self):
        definer = NeologismDefiner()
        definer._templates = {"noun": ["between {thing} and {context}"]}
        neo = Neologism(text="wras", pronounceability=0.8, phonetic_mood="dreamy")
        result = definer.define(neo, ["knife", "river"], random.Random(1))
        a, b = result.definition.removeprefix("between ").split(" and ")
        assert {a, b} == {"knife", "river"}

    def test_short_prefixed_word_is_not_a_verb(self):
        definer = NeologismDefiner()
        assert definer.infer_part_of_speech("dete") == "noun"
        assert definer.infer_part_of_speech("reth") == "noun"
        assert definer.infer_part_of_speech("unravel") == "verb"
