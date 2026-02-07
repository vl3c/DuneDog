"""End-to-end live test with Gemini 3 Flash via OpenRouter.

Requires OPENROUTER_API_KEY environment variable.
Run with: pytest tests/test_e2e_live.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import pytest

from dunedog.models.config import GenerationConfig
from dunedog.output.batch_generator import StoryBatchGenerator
from dunedog.utils.seed_manager import SeedManager
from dunedog.llm.provider import create_provider
from dunedog.llm.synthesizer import StorySynthesizer
from dunedog.models.config import LLMConfig


@pytest.fixture
def openrouter_key():
    """Get OpenRouter API key or skip."""
    # Try .env file first
    env_path = os.path.expanduser("~/.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not set")
    return key


@pytest.mark.live
class TestFullPipelineGeminiFlash:
    """Full end-to-end pipeline test with Gemini 3 Flash Preview.

    Uses --preset quick with a small count to minimize cost.
    Tracks and reports total token consumption.
    """

    def test_full_run_default_settings(self, openrouter_key):
        """Run full pipeline: chaos -> crystallize -> engines -> LLM synthesis.

        Reports:
        - Number of skeletons generated
        - Best coherence score
        - Number of stories synthesized
        - Total prompt tokens
        - Total completion tokens
        - Total tokens
        """
        # --- Pipeline config ---
        config = GenerationConfig.from_preset("quick", seed=42)
        config.llm.provider = "openrouter"
        config.llm.model = "google/gemini-3-flash-preview"

        # --- Generate skeletons ---
        t0 = time.time()
        seed_mgr = SeedManager(config.seed)
        generator = StoryBatchGenerator(config, seed_mgr)
        skeletons = generator.generate_batch(show_progress=False)
        gen_time = time.time() - t0

        assert len(skeletons) > 0, "No skeletons generated"
        assert all(sk.coherence_score >= 0 for sk in skeletons)

        print(f"\n{'='*60}")
        print(f"SKELETON GENERATION")
        print(f"  Count: {len(skeletons)}")
        print(f"  Best coherence: {skeletons[0].coherence_score:.4f}")
        print(f"  Worst coherence: {skeletons[-1].coherence_score:.4f}")
        print(f"  Time: {gen_time:.1f}s")
        print(f"{'='*60}")

        # --- LLM Synthesis ---
        from dunedog.llm.openrouter import OpenRouterProvider

        provider = OpenRouterProvider(
            api_key=openrouter_key,
            model="google/gemini-3-flash-preview",
        )
        llm_config = LLMConfig(
            provider="openrouter",
            model="google/gemini-3-flash-preview",
            story_lines=20,
            max_stories_for_llm=5,
        )
        synthesizer = StorySynthesizer(provider, llm_config)

        top_skeletons = skeletons[:llm_config.max_stories_for_llm]

        t1 = time.time()
        stories = asyncio.run(synthesizer.synthesize(top_skeletons))
        synth_time = time.time() - t1

        assert len(stories) > 0, "No stories synthesized"
        assert all(s.title for s in stories), "Stories should have titles"
        assert all(s.content for s in stories), "Stories should have content"

        # --- Token usage ---
        usage = provider.last_usage or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        print(f"\n{'='*60}")
        print(f"LLM SYNTHESIS")
        print(f"  Provider: OpenRouter (google/gemini-3-flash-preview)")
        print(f"  Stories: {len(stories)}")
        print(f"  Strategy: {stories[0].strategy if stories else 'N/A'}")
        print(f"  Synthesis time: {synth_time:.1f}s")
        print(f"{'='*60}")

        print(f"\n{'='*60}")
        print(f"TOKEN USAGE")
        print(f"  Prompt tokens:     {prompt_tokens:,}")
        print(f"  Completion tokens: {completion_tokens:,}")
        print(f"  Total tokens:      {total_tokens:,}")
        if total_tokens > 0:
            input_cost = prompt_tokens * 0.50 / 1_000_000
            output_cost = completion_tokens * 3.00 / 1_000_000
            print(f"  Estimated cost:    ${input_cost + output_cost:.4f}")
        print(f"{'='*60}")

        # --- Print stories ---
        print(f"\n{'='*60}")
        print(f"GENERATED STORIES")
        print(f"{'='*60}")
        for i, story in enumerate(stories, 1):
            print(f"\n--- Story {i}: {story.title} ---")
            print(f"Strategy: {story.strategy}")
            print(f"{'─'*40}")
            print(story.content)
            print(f"{'─'*40}")

        # --- Save output ---
        output = {
            "skeletons_generated": len(skeletons),
            "best_coherence": skeletons[0].coherence_score,
            "stories": [s.to_dict() for s in stories],
            "token_usage": usage,
            "generation_time_s": gen_time,
            "synthesis_time_s": synth_time,
        }
        output_path = "/tmp/dunedog_e2e_output.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nFull output saved to {output_path}")
