"""Neologism definer -- gives meaning to invented words."""
from __future__ import annotations

import json
import random
from pathlib import Path

from dunedog.models.results import Neologism
from dunedog.chaos.phonetics import analyze_phonetic_mood


class NeologismDefiner:
    """Defines neologisms using templates, phonetic mood, and context."""

    def __init__(self):
        self._templates = None  # lazy loaded

    def _load_templates(self) -> dict[str, list[str]]:
        """Load neologism_templates.json from data/. Fallback to inline defaults."""
        if self._templates is not None:
            return self._templates

        data_dir = Path(__file__).resolve().parents[3] / "data"
        template_path = data_dir / "neologism_templates.json"

        if template_path.exists():
            with open(template_path) as f:
                self._templates = json.load(f)
        else:
            # Inline fallback
            self._templates = {
                "noun": [
                    "a {mood} substance found only in forgotten places",
                    "the feeling of {mood} awareness that comes without warning",
                    "a type of silence that tastes {mood}",
                    "the residue left behind when {context} fades",
                ],
                "verb": [
                    "to move in a {mood} manner through {context}",
                    "to transform {context} into something {mood}",
                    "to speak without words, conveying {mood} intent",
                ],
                "adjective": [
                    "having the quality of {mood} {context}",
                    "resembling something both {mood} and forgotten",
                    "possessing an inexplicable {mood} character",
                ],
                "adverb": [
                    "in a {mood} fashion, as if {context} were watching",
                    "with the {mood} precision of {context}",
                ],
            }
        return self._templates

    def infer_part_of_speech(self, word: str) -> str:
        """Infer POS from word shape.

        -ly -> adverb
        -ness, -tion, -sion, -ity, -ment -> noun
        -ful, -ous, -ive, -al, -ent, -ant -> adjective
        -ize, -ify, -ate, -ed, -ing (and prefixes un-, re-, de-) -> verb
        default -> noun
        """
        word_lower = word.lower()

        if word_lower.endswith("ly"):
            return "adverb"
        if any(word_lower.endswith(s) for s in ("ness", "tion", "sion", "ity", "ment")):
            return "noun"
        if any(word_lower.endswith(s) for s in ("ful", "ous", "ive", "al", "ent", "ant")):
            return "adjective"
        if any(word_lower.endswith(s) for s in ("ize", "ify", "ate", "ed", "ing")):
            return "verb"
        if any(word_lower.startswith(s) for s in ("un", "re", "de")):
            return "verb"
        return "noun"

    def define(
        self,
        neologism: Neologism,
        context_words: list[str],
        rng: random.Random,
    ) -> Neologism:
        """Fill in definition, POS, and usage example for a neologism.

        Uses:
        - Templates from neologism_templates.json
        - Phonetic mood from the word itself
        - Context words from surrounding chaos extraction
        """
        templates = self._load_templates()

        pos = self.infer_part_of_speech(neologism.text)
        mood = neologism.phonetic_mood or analyze_phonetic_mood(neologism.text)
        context = rng.choice(context_words) if context_words else "the unknown"

        pos_templates = templates.get(pos, templates.get("noun", []))
        template = rng.choice(pos_templates)
        definition = (
            template.replace("{word}", neologism.text)
            .replace("{mood}", mood)
            .replace("{context}", context)
        )

        usage = self.generate_usage_example(neologism.text, pos, definition, rng)

        neologism.part_of_speech = pos
        neologism.definition = definition
        neologism.usage_example = usage
        neologism.phonetic_mood = mood
        neologism.source_context = context_words[:5]  # keep first 5 context words

        return neologism

    def generate_usage_example(
        self,
        word: str,
        pos: str,
        definition: str,
        rng: random.Random,
    ) -> str:
        """Generate a brief usage example sentence."""
        patterns: dict[str, list[str]] = {
            "noun": [
                "The {word} settled over the valley like a second dusk.",
                "She kept the {word} in a glass jar beside her bed.",
                "No map could account for the {word} they discovered underground.",
                "He spoke of the {word} as though it were a living thing.",
                "There was a {word} in the room that no one dared name.",
            ],
            "verb": [
                "She began to {word} without warning, and the air changed.",
                "They would {word} for hours beside the old canal.",
                "He tried to {word} the silence, but it resisted him.",
                "The children learned to {word} before they could speak.",
                "One does not simply {word} -- it requires a certain stillness.",
            ],
            "adjective": [
                "The {word} landscape stretched endlessly before them.",
                "Her voice had a {word} quality that silenced the crowd.",
                "It was the most {word} evening anyone could remember.",
                "The old house felt distinctly {word} after the storm.",
                "Something {word} lingered at the edges of the photograph.",
            ],
            "adverb": [
                "He moved {word} through the corridor, barely disturbing the dust.",
                "The clock ticked {word}, as if counting something other than seconds.",
                "She smiled {word} and turned away from the window.",
                "The river flowed {word} beneath the ancient bridge.",
                "Stars appeared {word}, one by one, across the darkening sky.",
            ],
        }
        templates = patterns.get(pos, patterns["noun"])
        template = rng.choice(templates)
        return template.replace("{word}", word)
