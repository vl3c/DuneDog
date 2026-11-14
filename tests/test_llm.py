"""Tests for LLM provider factory and story synthesizer."""

import asyncio

import pytest

from dunedog.llm.provider import LLMProvider, LLMError, create_provider, DEFAULT_MODELS
from dunedog.llm.synthesizer import StorySynthesizer, SynthesizedStory, select_diverse
from dunedog.models.atoms import StoryAtom, AtomCategory, AtomSource
from dunedog.models.config import LLMConfig
from dunedog.models.skeleton import StorySkeleton, GenerationStats


# ------------------------------------------------------------------ #
# Mock LLM provider for testing
# ------------------------------------------------------------------ #


class MockLLMProvider(LLMProvider):
    """A mock provider that returns canned text."""

    def __init__(self, response_text: str):
        super().__init__()
        self._response = response_text

    async def complete(self, messages, **kwargs):
        return self._response


# ------------------------------------------------------------------ #
# create_provider factory
# ------------------------------------------------------------------ #


class TestCreateProvider:
    """Tests for the create_provider factory function."""

    def test_creates_openai_provider(self):
        provider = create_provider("openai", api_key="test-key")
        assert provider is not None
        assert provider.model == DEFAULT_MODELS["openai"]

    def test_creates_anthropic_provider(self):
        provider = create_provider("anthropic", api_key="test-key")
        assert provider is not None
        assert provider.model == DEFAULT_MODELS["anthropic"]

    def test_creates_openrouter_provider(self):
        provider = create_provider("openrouter", api_key="test-key")
        assert provider is not None
        assert provider.model == DEFAULT_MODELS["openrouter"]

    def test_creates_chatgpt_provider(self):
        provider = create_provider("chatgpt", api_key="test-key")
        assert provider is not None
        assert provider.model == DEFAULT_MODELS["chatgpt"]

    def test_raises_for_unknown_provider(self):
        with pytest.raises(LLMError, match="Unknown provider"):
            create_provider("nonexistent_provider")

    def test_default_models_are_set(self):
        """Each known provider should have a default model string."""
        for name in ("openai", "anthropic", "openrouter", "chatgpt"):
            assert name in DEFAULT_MODELS
            assert isinstance(DEFAULT_MODELS[name], str)
            assert len(DEFAULT_MODELS[name]) > 0


# ------------------------------------------------------------------ #
# StorySynthesizer
# ------------------------------------------------------------------ #


def _make_skeleton(**overrides) -> StorySkeleton:
    """Helper to build a minimal skeleton for prompt tests."""
    defaults = dict(
        atoms=[
            StoryAtom("the wanderer", AtomCategory.AGENT, AtomSource.CATALOGUE, ["journey"]),
            StoryAtom("the forest", AtomCategory.LOCATION, AtomSource.CATALOGUE, ["nature"]),
        ],
        beats=["OPENING", "RISING_ACTION", "CLIMAX", "RESOLUTION"],
        spread_positions={"past": "the wanderer", "present": "the forest"},
        theme_tags=["journey", "nature"],
        tone="enigmatic",
        stats=GenerationStats(engine="test"),
    )
    defaults.update(overrides)
    return StorySkeleton(**defaults)


class TestStorySynthesizer:
    """Tests for StorySynthesizer."""

    def test_build_prompt_includes_skeleton_data(self):
        """build_prompt should mention atoms, beats, tone, and themes."""
        provider = MockLLMProvider("")
        synth = StorySynthesizer(provider)
        skeleton = _make_skeleton()

        prompt = synth.build_prompt([skeleton])

        assert "the wanderer" in prompt
        assert "the forest" in prompt
        assert "OPENING" in prompt
        assert "enigmatic" in prompt
        assert "journey" in prompt

    def test_build_prompt_carries_roles_definitions_and_disruptions(self):
        """The prompt keeps what the pipeline learned, not just atom names."""
        synth = StorySynthesizer(MockLLMProvider(""))
        skeleton = _make_skeleton(
            atoms=[
                StoryAtom("the bone-singer", AtomCategory.AGENT, AtomSource.CATALOGUE,
                          ["ritual"], metadata={"evoked_by": "oboe"}),
                StoryAtom("snee", AtomCategory.OBJECT, AtomSource.NEOLOGISM, ["object", "noun"],
                          metadata={"definition": "the residue left behind after rain"}),
            ],
            spread_positions={"The Call": "the bone-singer"},
            theme_tags=["ritual", "object", "noun"],
            disruptions=["The moon is now missing from the sky"],
        )

        prompt = synth.build_prompt([skeleton])

        assert "The Call (what summons the hero to adventure): the bone-singer" in prompt
        assert '"oboe"' in prompt
        assert "the residue left behind after rain" in prompt
        assert "The moon is now missing from the sky" in prompt
        assert "Themes: ritual\n" in prompt  # generic tags are not themes
        assert "PREMISE:" in prompt

    def test_build_prompt_is_deterministic_and_states_world_laws(self):
        synth = StorySynthesizer(MockLLMProvider(""))
        skeleton = _make_skeleton(atoms=[
            StoryAtom("the cartographer", AtomCategory.AGENT, AtomSource.CATALOGUE, ["cartography"]),
        ])
        prompt = synth.build_prompt([skeleton])
        assert prompt == synth.build_prompt([skeleton])
        assert "LAWS OF THIS WORLD" in prompt
        # A law triggered by an atom's tag is always among those chosen.
        assert "Mapmakers always get lost" in synth.world_laws([skeleton])

    def test_parse_response_extracts_premise(self):
        synth = StorySynthesizer(MockLLMProvider(""))
        stories = synth.parse_response(
            "STRATEGY: BRAIDED\n---\nTITLE: Salt\n"
            "PREMISE: A diver wants her name back, but the sea keeps it.\n"
            "She went down at dusk.\n---\n"
        )
        assert stories[0].premise == "A diver wants her name back, but the sea keeps it."
        assert stories[0].content == "She went down at dusk."

    def test_parse_response_tolerates_format_drift(self):
        synth = StorySynthesizer(MockLLMProvider(""))
        stories = synth.parse_response(
            "STRATEGY: BRAIDED\n---\n"
            "**TITLE: Salt**\n**Premise:** A diver wants her name back.\n\nShe went down.\n---\n"
            "TITLE: Ash\nPREMISE:\nHe stayed.\nThe fire did not.\n---\n"
            "TITLE: Tin\nNo premise here.\n---"
        )
        assert [(s.title, s.premise, s.content) for s in stories] == [
            ("Salt", "A diver wants her name back.", "She went down."),
            ("Ash", "", "He stayed.\nThe fire did not."),
            ("Tin", "", "No premise here."),
        ]

    def test_shape_line_only_for_markov_beats(self):
        synth = StorySynthesizer(MockLLMProvider(""))
        tarot_only = _make_skeleton(beats=["The wanderer — what summons the hero to adventure."])
        assert "Shape:" not in synth.build_prompt([tarot_only])
        assert "Shape: OPENING -> RISING_ACTION" in synth.build_prompt([_make_skeleton()])

    def test_story_count_is_stated_exactly(self):
        synth = StorySynthesizer(MockLLMProvider(""), LLMConfig(story_count=3))
        assert "Write exactly 3 interconnected" in synth.build_prompt([_make_skeleton()])
        assert "Write 2-3 interconnected" in StorySynthesizer(MockLLMProvider("")).build_prompt([_make_skeleton()])

    def test_parse_response_extracts_stories(self):
        """parse_response should extract title and content from formatted text."""
        provider = MockLLMProvider("")
        synth = StorySynthesizer(provider)

        response_text = (
            "STRATEGY: BRAIDED\n"
            "---\n"
            "TITLE: The Wanderer's Path\n"
            "Once upon a time, a wanderer set forth.\n"
            "---\n"
            "TITLE: The Forest Speaks\n"
            "The trees whispered secrets to the sky.\n"
            "---\n"
        )

        stories = synth.parse_response(response_text)
        assert len(stories) == 2
        assert stories[0].title == "The Wanderer's Path"
        assert "wanderer" in stories[0].content
        assert stories[1].title == "The Forest Speaks"
        assert "trees" in stories[1].content

    def test_parse_response_uses_strategy(self):
        """parse_response should capture the STRATEGY line."""
        provider = MockLLMProvider("")
        synth = StorySynthesizer(provider)

        response_text = (
            "STRATEGY: RASHOMON\n"
            "---\n"
            "TITLE: First View\n"
            "Content here.\n"
            "---\n"
        )

        stories = synth.parse_response(response_text)
        assert len(stories) == 1
        assert stories[0].strategy == "RASHOMON"

    def test_synthesize_with_mock_provider(self):
        """Full synthesize round-trip with mock provider producing canned text."""
        canned = (
            "STRATEGY: BRAIDED\n"
            "---\n"
            "TITLE: Dream of Sand\n"
            "The desert remembers what the city forgets.\n"
            "---\n"
            "TITLE: Echo of Bells\n"
            "A bell rings once, and everything changes.\n"
            "---\n"
        )

        provider = MockLLMProvider(canned)
        config = LLMConfig(max_stories_for_llm=5)
        synth = StorySynthesizer(provider, config)

        skeleton = _make_skeleton()
        stories = asyncio.run(synth.synthesize([skeleton]))

        assert len(stories) == 2
        assert stories[0].title == "Dream of Sand"
        assert "desert" in stories[0].content
        assert stories[1].title == "Echo of Bells"
        assert stories[0].strategy == "BRAIDED"


class TestSelectDiverse:
    """select_diverse should prefer good skeletons that differ from each other."""

    @staticmethod
    def _sk(names, score):
        return StorySkeleton(
            atoms=[StoryAtom(n, AtomCategory.OBJECT, AtomSource.CATALOGUE) for n in names],
            stats=GenerationStats(coherence_score=score),
        )

    def test_skips_near_duplicates(self):
        best = self._sk(["a", "b", "c"], 0.9)
        clone = self._sk(["a", "b", "c"], 0.89)
        different = self._sk(["x", "y", "z"], 0.6)
        assert select_diverse([best, clone, different], 2) == [best, different]

    def test_best_skeleton_always_first(self):
        sks = [self._sk(["a"], 0.2), self._sk(["b"], 0.8), self._sk(["c"], 0.5)]
        assert select_diverse(sks, 3)[0] is sks[1]

    def test_handles_small_and_empty_inputs(self):
        only = self._sk(["a"], 0.5)
        assert select_diverse([], 3) == []
        assert select_diverse([only], 0) == []
        assert select_diverse([only], 5) == [only]


class TestSynthesizerMaxTokens:
    class _Recorder(LLMProvider):
        def __init__(self):
            super().__init__()
            self.kwargs_seen = None

        async def complete(self, messages, **kwargs):
            self.kwargs_seen = kwargs
            return "STRATEGY: BRAIDED\n---\nTITLE: T\nPREMISE: P\nBody.\n---"

    def test_config_max_tokens_reaches_provider(self):
        provider = self._Recorder()
        synth = StorySynthesizer(provider, LLMConfig(max_tokens=48000))
        asyncio.run(synth.synthesize([_make_skeleton()]))
        assert provider.kwargs_seen["max_tokens"] == 48000

    def test_unset_max_tokens_is_not_sent(self):
        provider = self._Recorder()
        synth = StorySynthesizer(provider, LLMConfig())
        asyncio.run(synth.synthesize([_make_skeleton()]))
        assert "max_tokens" not in provider.kwargs_seen
