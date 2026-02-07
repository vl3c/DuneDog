"""Story synthesizer — sends top skeletons to LLM for narrative synthesis."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from dunedog.llm.provider import LLMProvider
from dunedog.models.config import LLMConfig
from dunedog.models.skeleton import StorySkeleton


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
    source_skeleton_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "content": self.content,
            "strategy": self.strategy,
            "source_skeleton_indices": self.source_skeleton_indices,
        }


class StorySynthesizer:
    """Takes top-N skeletons and asks an LLM to combine them into stories."""

    def __init__(self, provider: LLMProvider, config: LLMConfig | None = None):
        self._provider = provider
        self._config = config or LLMConfig()

    async def synthesize(
        self,
        skeletons: list[StorySkeleton],
        strategy: str = "",
    ) -> list[SynthesizedStory]:
        """Synthesize interconnected narratives from the top skeletons.

        Args:
            skeletons: Ranked list of story skeletons (best first).
            strategy: Synthesis strategy name, or empty to let LLM choose.

        Returns:
            List of SynthesizedStory objects.
        """
        # Limit to max_stories_for_llm
        top = skeletons[: self._config.max_stories_for_llm]
        if not top:
            return []

        prompt = self.build_prompt(top, strategy or self._config.synthesis_strategy)
        messages = [
            {"role": "system", "content": "You are a literary storyteller. You create vivid, interconnected short stories from narrative blueprints."},
            {"role": "user", "content": prompt},
        ]

        response = await self._provider.complete(
            messages,
            temperature=self._config.temperature,
        )

        return self.parse_response(response, strategy or self._config.synthesis_strategy)

    def build_prompt(self, skeletons: list[StorySkeleton], strategy: str = "") -> str:
        """Build synthesis prompt from skeletons."""
        lines = [
            "Below are narrative skeleton blueprints extracted from a stochastic chaos engine.",
            "Each skeleton contains story atoms (characters, objects, locations, tensions, triggers, qualities),",
            "narrative beats, and thematic tags.\n",
        ]

        for i, sk in enumerate(skeletons):
            lines.append(f"--- Skeleton {i + 1} ---")
            lines.append(f"Tone: {sk.tone}")
            lines.append(f"Themes: {', '.join(sk.theme_tags) if sk.theme_tags else 'none'}")

            if sk.atoms:
                atom_strs = [f"  - {a.name} [{a.category.value}]" for a in sk.atoms]
                lines.append("Atoms:")
                lines.extend(atom_strs)

            if sk.beats:
                lines.append(f"Beats: {' → '.join(sk.beats)}")

            if sk.spread_positions:
                pos_strs = [f"  {pos}: {atom}" for pos, atom in sk.spread_positions.items()]
                lines.append("Spread positions:")
                lines.extend(pos_strs)

            lines.append("")

        # Strategy instructions
        lines.append("=== TASK ===")
        max_lines = self._config.story_lines

        if strategy and strategy in SYNTHESIS_STRATEGIES:
            lines.append(f"Use the {strategy} strategy: {SYNTHESIS_STRATEGIES[strategy]}")
        else:
            strat_desc = "\n".join(f"- {k}: {v}" for k, v in SYNTHESIS_STRATEGIES.items())
            lines.append(f"Choose the best interconnection strategy from:\n{strat_desc}")

        lines.append(f"\nCombine these skeletons into 2-4 interconnected short stories.")
        lines.append(f"Each story should be at most {max_lines} lines long.")
        lines.append("Draw characters, objects, locations, and tensions from the skeletons.")
        lines.append("Each skeleton need not map 1:1 to a story — blend and recombine freely.")
        lines.append("\nFormat your response as:")
        lines.append("STRATEGY: <strategy name>")
        lines.append("---")
        lines.append("TITLE: <story title>")
        lines.append("<story text>")
        lines.append("---")
        lines.append("TITLE: <next story title>")
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

            title_match = re.search(r"TITLE:\s*(.+)", section)
            if not title_match:
                continue

            title = title_match.group(1).strip()
            # Content is everything after the TITLE line
            content_start = title_match.end()
            content = section[content_start:].strip()

            if content:
                stories.append(SynthesizedStory(
                    title=title,
                    content=content,
                    strategy=used_strategy,
                ))

        return stories
