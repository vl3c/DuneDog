"""Tests for batch generation and export/import round-tripping."""

import csv
import json
import os
import tempfile

import pytest

from dunedog.output.batch_generator import StoryBatchGenerator
from dunedog.output import exporter
from dunedog.models.config import GenerationConfig
from dunedog.models.skeleton import StorySkeleton, GenerationStats
from dunedog.models.atoms import StoryAtom, AtomCategory, AtomSource
from dunedog.utils.seed_manager import SeedManager


# ------------------------------------------------------------------ #
# StoryBatchGenerator
# ------------------------------------------------------------------ #


class TestStoryBatchGenerator:
    """Tests for StoryBatchGenerator."""

    def test_generate_batch_produces_expected_count(self):
        """generate_batch(n) should return exactly n skeletons."""
        config = GenerationConfig.from_preset("quick", seed=42)
        config.skeletons_to_generate = 3
        seed_mgr = SeedManager(42)
        gen = StoryBatchGenerator(config, seed_mgr)

        skeletons = gen.generate_batch(n=3, show_progress=False)
        assert len(skeletons) == 3
        for sk in skeletons:
            assert isinstance(sk, StorySkeleton)

    def test_batch_sorted_by_coherence(self):
        """Returned batch should be sorted best-first by coherence score."""
        config = GenerationConfig.from_preset("quick", seed=99)
        config.skeletons_to_generate = 5
        # Enable constraint solver so scores are computed
        config.engines.use_constraint_solver = True
        seed_mgr = SeedManager(99)
        gen = StoryBatchGenerator(config, seed_mgr)

        skeletons = gen.generate_batch(n=5, show_progress=False)
        scores = [sk.coherence_score for sk in skeletons]
        assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------ #
# Exporter: JSON
# ------------------------------------------------------------------ #


def _make_skeletons(n: int = 2) -> list[StorySkeleton]:
    """Create a small list of skeletons for export tests."""
    skeletons = []
    for i in range(n):
        sk = StorySkeleton(
            atoms=[
                StoryAtom(f"atom_{i}_a", AtomCategory.AGENT, AtomSource.CATALOGUE, ["tag1"]),
                StoryAtom(f"atom_{i}_b", AtomCategory.OBJECT, AtomSource.CATALOGUE, ["tag2"]),
            ],
            beats=["OPENING", "CLIMAX", "RESOLUTION"],
            spread_positions={"past": f"atom_{i}_a"},
            theme_tags=["journey"],
            tone="dark",
            stats=GenerationStats(engine="test", coherence_score=0.5 + i * 0.1),
        )
        skeletons.append(sk)
    return skeletons


class TestExporterJSON:
    """Tests for exporter.to_json and exporter.load_skeletons."""

    def test_to_json_creates_valid_file(self, tmp_path):
        """to_json should create a valid JSON file."""
        skeletons = _make_skeletons(2)
        path = tmp_path / "out.json"

        exporter.to_json(skeletons, path)

        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_load_skeletons_round_trip(self, tmp_path):
        """Exporting then loading should produce equivalent skeletons."""
        original = _make_skeletons(3)
        path = tmp_path / "round_trip.json"

        exporter.to_json(original, path)
        loaded = exporter.load_skeletons(path)

        assert len(loaded) == len(original)
        for orig, load in zip(original, loaded):
            assert orig.tone == load.tone
            assert len(orig.atoms) == len(load.atoms)
            assert orig.beats == load.beats
            assert orig.theme_tags == load.theme_tags
            assert abs(orig.stats.coherence_score - load.stats.coherence_score) < 1e-9


# ------------------------------------------------------------------ #
# Exporter: CSV
# ------------------------------------------------------------------ #


class TestExporterCSV:
    """Tests for exporter.to_csv."""

    def test_to_csv_creates_valid_file(self, tmp_path):
        """to_csv should create a parseable CSV file with the right columns."""
        skeletons = _make_skeletons(2)
        path = tmp_path / "out.csv"

        exporter.to_csv(skeletons, path)

        assert path.exists()
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        expected_fields = {
            "index", "tone", "coherence_score", "engine",
            "spread_type", "beat_count", "atom_count", "theme_tags", "violations",
        }
        assert expected_fields == set(reader.fieldnames)
        assert rows[0]["tone"] == "dark"
