"""Tests for LLM provider factory and story synthesizer."""

import asyncio

import pytest

from dunedog.llm.provider import LLMProvider, LLMError, create_provider, DEFAULT_MODELS
from dunedog.llm.synthesizer import StorySynthesizer, SynthesizedStory
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
