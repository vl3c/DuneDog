"""Story synthesizer — sends a diverse handful of skeletons to an LLM."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from dunedog.llm.provider import LLMProvider
from dunedog.models.atoms import GENERIC_TAGS, StoryAtom
from dunedog.models.config import LLMConfig
from dunedog.models.skeleton import StorySkeleton

log = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"

SYSTEM_PROMPT = (
    "You write strange, precise literary short fiction. You prefer specific "
    "images to abstractions, give every character one concrete want, and let "
    "every story turn on a choice."
)

# How many world laws every story in a run must obey.
_LAWS_PER_RUN = 2


SYNTHESIS_STRATEGIES = {
    "BRAIDED": "Weave parallel stories with thematic echoes. Each story stands alone but shares imagery, motifs, or emotional arcs with the others.",
    "CAUSE_AND_ECHO": "Chain stories so that Story A's ending is Story B's beginning. Each story's conclusion seeds the next.",
    "RASHOMON": "Tell the same events from different perspectives. Each story reveals something the others conceal.",
    "ANTHOLOGY_FRAME": "Create a meta-story that contains the others. A narrator, a place, or a ritual frames the individual tales.",
}


@dataclass
class SynthesizedStory:
    """A final story produced by LLM synthesis."""
    title: str = ""
    content: str = ""
    strategy: str = ""
    premise: str = ""  # the one-line "who wants what, but" the story was written from
    source_skeleton_indices: list[int] = field(default_factory=list)
    truncated: bool = False  # the response hit the token cap inside this story

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "content": self.content,
            "strategy": self.strategy,
            "premise": self.premise,
            "truncated": self.truncated,
            "source_skeleton_indices": self.source_skeleton_indices,
        }


def select_diverse(
    skeletons: list[StorySkeleton],
    k: int,
    diversity: float = 0.5,
) -> list[StorySkeleton]:
    """Pick *k* skeletons that are good *and* unlike each other.

    Taking the top *k* by coherence hands the LLM near-duplicates: evolution
    converges, so the best skeletons share most of their atoms.  Instead,
    start from the best skeleton and repeatedly add the one maximising
    ``coherence + diversity * (distance to its nearest already-chosen
    neighbour)``, where distance is Jaccard distance over atom names.

    Deterministic: ties go to the earlier skeleton in the input order.
    """
    if k <= 0 or not skeletons:
        return []

    names = [frozenset(a.name for a in sk.atoms) for sk in skeletons]

    def distance(i: int, j: int) -> float:
        union = names[i] | names[j]
        return 1.0 - len(names[i] & names[j]) / len(union) if union else 0.0

    chosen = [max(range(len(skeletons)), key=lambda i: (skeletons[i].coherence_score, -i))]
    while len(chosen) < min(k, len(skeletons)):
        rest = [i for i in range(len(skeletons)) if i not in chosen]
        best = max(
            rest,
            key=lambda i: (
                skeletons[i].coherence_score
                + diversity * min(distance(i, j) for j in chosen),
                -i,
            ),
        )
        chosen.append(best)
    return [skeletons[i] for i in chosen]


class StorySynthesizer:
    """Takes a diverse handful of skeletons and asks an LLM to write stories."""

    def __init__(self, provider: LLMProvider, config: LLMConfig | None = None):
        self._provider = provider
        self._config = config or LLMConfig()
        self._position_meanings = self._load_position_meanings()
        self._law_pool = self._load_laws()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(name: str):
        path = _DATA_DIR / name
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _load_position_meanings(self) -> dict[str, str]:
        """Spread position name -> what that position means in a reading."""
        spreads = self._load_json("tarot_positions.json") or {}
        return {
            pos["name"]: pos.get("description", "")
            for spread in spreads.values()
            for pos in spread.get("positions", [])
        }

    def _load_laws(self) -> list[dict]:
        """World rules as ``{"text", "if_tag"}``; tendencies have no trigger."""
        laws = [
            {"text": inv["description"], "if_tag": inv.get("parameters", {}).get("if_tag", "")}
            for inv in self._load_json("invariants.json") or []
        ]
        laws += [
            {"text": t["description"], "if_tag": ""}
            for t in self._load_json("tendencies.json") or []
        ]
        return laws

    def world_laws(self, skeletons: list[StorySkeleton]) -> list[str]:
        """Choose the laws of the world these skeletons share.

        Laws whose trigger tag is carried by one of the atoms come first
        (a story with a mapmaker lives where "mapmakers always get lost");
        the rest are drawn with an RNG seeded from the atoms themselves, so
        the same skeletons always live under the same laws.
        """
        if not self._law_pool:
            return []
        atom_names = sorted({a.name for sk in skeletons for a in sk.atoms})
        tags = {t for sk in skeletons for a in sk.atoms for t in a.tags}
        digest = hashlib.sha256("|".join(atom_names).encode()).hexdigest()
        rng = random.Random(int(digest[:16], 16))

        triggered = [l["text"] for l in self._law_pool if l["if_tag"] and l["if_tag"] in tags]
        others = [l["text"] for l in self._law_pool if l["text"] not in triggered]
        rng.shuffle(triggered)
        rng.shuffle(others)
        return (triggered + others)[:_LAWS_PER_RUN]

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        skeletons: list[StorySkeleton],
        strategy: str = "",
    ) -> list[SynthesizedStory]:
        """Synthesize interconnected narratives from the skeletons.

        Args:
            skeletons: Ranked list of story skeletons (best first).
            strategy: Synthesis strategy name, or empty to let LLM choose.

        Returns:
            List of SynthesizedStory objects.
        """
        top = select_diverse(skeletons, self._config.max_stories_for_llm)
        if not top:
            return []

        strat = strategy or self._config.synthesis_strategy
        log.info("Synthesizing from %d skeletons via %s (strategy=%s)",
                 len(top), self._provider.__class__.__name__, strat or "LLM-chosen")

        prompt = self.build_prompt(top, strat)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        log.info("Sending request to LLM...")
        response = await self._provider.complete(
            messages,
            temperature=self._config.temperature,
            **({"effort": self._config.effort} if self._config.effort else {}),
            **({"max_tokens": self._config.max_tokens} if self._config.max_tokens else {}),
        )
        log.info("LLM response received (%d chars)", len(response))

        stories = self.parse_response(response, strat)
        if stories and getattr(self._provider, "last_truncated", False):
            stories[-1].truncated = True
            log.warning("\"%s\" was cut off at the token cap; raise --max-tokens", stories[-1].title)
        log.info("Parsed %d stories", len(stories))
        for s in stories:
            log.info("  \"%s\" (%d chars, strategy=%s)", s.title, len(s.content), s.strategy)
        return stories

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _describe_atom(atom: StoryAtom) -> str:
        """One atom, with whatever the pipeline learned about it."""
        text = f"{atom.name} [{atom.category.value}]"
        definition = atom.metadata.get("definition")
        if definition:
            pos = atom.metadata.get("part_of_speech", "")
            text += f" — a coined word{f' ({pos})' if pos else ''} meaning: {definition}"
        evoked_by = atom.metadata.get("evoked_by")
        if evoked_by:
            text += f' — summoned from the noise by the word "{evoked_by}"'
        return text

    def _describe_skeleton(self, index: int, sk: StorySkeleton) -> list[str]:
        """Render one skeleton as a seed: a reading, not a parts list."""
        lines = [f"--- Seed {index} ---"]

        tone = sk.tone or "unspecified"
        mood = sk.primordial_source.phonetic_mood
        distinct_mood = mood and mood != "balanced"
        lines.append(f"Tone: {tone}" + (f" (the noise it came from sounded {mood})" if distinct_mood else ""))

        atoms: dict[str, StoryAtom] = {}
        for atom in sk.atoms:
            atoms.setdefault(atom.name, atom)

        # The reading: each placed atom together with the role it plays.
        placed: set[str] = set()
        reading: list[str] = []
        for pos, name in sk.spread_positions.items():
            if name not in atoms or name in placed:
                continue
            placed.add(name)
            meaning = self._position_meanings.get(pos, "")
            role = f"{pos} ({meaning[0].lower() + meaning[1:]})" if meaning else pos
            reading.append(f"  {role}: {self._describe_atom(atoms[name])}")
        if reading:
            lines.append("Reading:")
            lines.extend(reading)

        loose = [self._describe_atom(a) for n, a in atoms.items() if n not in placed]
        if loose:
            lines.append("Also present:" if reading else "Elements:")
            lines.extend(f"  - {item}" for item in loose)

        for disruption in sk.disruptions[:2]:
            lines.append(f"Disruption: {disruption}")

        themes = [t for t in sk.theme_tags if t not in GENERIC_TAGS][:8]
        if themes:
            lines.append(f"Themes: {', '.join(themes)}")

        # Beats are Markov beat types (OPENING, CLIMAX...) or, when the Markov
        # engine is off, the tarot fragments, which would only repeat the
        # reading.  Evolution splices beat lists into long, repetitive
        # strings; only the order in which movements first appear is a shape.
        shape = list(dict.fromkeys(b for b in sk.beats if b.isupper()))
        if shape:
            lines.append(f"Shape: {' -> '.join(shape)}")

        lines.append("")
        return lines

    def build_prompt(self, skeletons: list[StorySkeleton], strategy: str = "") -> str:
        """Build synthesis prompt from skeletons."""
        lines = [
            "Below are story seeds dredged from random noise. Each seed is a reading:",
            "a few elements, each with the role it was dealt, plus a tone and a rough shape.",
            "The seeds are not a checklist. They are provocations.\n",
        ]

        for i, sk in enumerate(skeletons):
            lines.extend(self._describe_skeleton(i + 1, sk))

        laws = self.world_laws(skeletons)
        if laws:
            lines.append("=== LAWS OF THIS WORLD ===")
            lines.append("All the stories happen in one world. In it, these are simply true:")
            lines.extend(f"- {law}" for law in laws)
            lines.append("Never state a law outright. Let events obey it.\n")

        lines.append("=== TASK ===")
        max_lines = self._config.story_lines

        if strategy and strategy in SYNTHESIS_STRATEGIES:
            lines.append(f"Use the {strategy} strategy: {SYNTHESIS_STRATEGIES[strategy]}")
        else:
            strat_desc = "\n".join(f"- {k}: {v}" for k, v in SYNTHESIS_STRATEGIES.items())
            lines.append(f"Choose the best interconnection strategy from:\n{strat_desc}")

        count = self._config.story_count
        how_many = f"exactly {count}" if count else "2-3"
        lines.append(f"\nWrite {how_many} interconnected short stories, each about "
                     f"{max_lines} sentences long.")
        lines.append("- Use only the elements that make a story stronger, and leave at least half out. "
                     "Never mention an element just to prove you used it. Blend seeds freely.")
        lines.append("- Honour the roles: if an element was dealt as the ordeal or the price, "
                     "that is the part it plays.")
        lines.append("- A coined word is a real word in this world. Use it the way its speakers would, "
                     "in its given meaning, without explaining or italicising it. "
                     "Do not turn it into a codename, an acronym or a label.")
        lines.append("- A disruption is something that actually happens, and people have to live with it.")
        lines.append("- Before each story write one line, PREMISE: <who> wants <what>, but <what stands in the way>. "
                     "Then write the story that premise demands.")
        lines.append("- End each story on an image or an action, not a moral or a summary.")
        lines.append("\nFormat your response exactly as:")
        lines.append("STRATEGY: <strategy name>")
        lines.append("---")
        lines.append("TITLE: <story title>")
        lines.append("PREMISE: <one line>")
        lines.append("<story text>")
        lines.append("---")
        lines.append("TITLE: <next story title>")
        lines.append("PREMISE: <one line>")
        lines.append("<story text>")
        lines.append("---")

        return "\n".join(lines)

    def parse_response(self, text: str, strategy: str = "") -> list[SynthesizedStory]:
        """Parse LLM response into SynthesizedStory objects."""
        stories: list[SynthesizedStory] = []

        # Extract strategy if present
        strat_match = re.search(r"STRATEGY:\s*(.+)", text)
        used_strategy = strat_match.group(1).strip() if strat_match else strategy

        # Split on --- dividers
        sections = re.split(r"\n-{3,}\n", text)

        for section in sections:
            section = section.strip()
            if not section:
                continue

            title_match = re.search(r"TITLE:\**[ \t]*(.+)", section)
            if not title_match:
                continue

            title = title_match.group(1).strip().strip("*").strip()
            # Content is everything after the TITLE line
            content_start = title_match.end()
            content = section[content_start:].strip()
            content = re.sub(r"\n?-{3,}\s*$", "", content).strip()

            # An optional PREMISE line directly under the title.  Tolerates
            # "Premise:" and bold markdown; never reaches past its own line.
            premise = ""
            premise_match = re.match(r"\**[ \t]*PREMISE[ \t]*:\**[ \t]*(.*)", content, re.IGNORECASE)
            if premise_match:
                premise = premise_match.group(1).strip().strip("*").strip()
                content = content[premise_match.end():].strip()

            if content:
                stories.append(SynthesizedStory(
                    title=title,
                    content=content,
                    strategy=used_strategy,
                    premise=premise,
                ))

        return stories
