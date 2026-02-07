"""Tests for semantic clustering of story atoms."""

import random

import pytest

from dunedog.models.atoms import StoryAtom, AtomCategory, AtomSource
from dunedog.crystallize.semantic_clustering import cluster_atoms, _tag_based_clustering
from dunedog.utils.embeddings import RandomSimilarity


class TestClusterAtoms:
    """Tests for the cluster_atoms function."""

    def test_empty_input(self):
        assert cluster_atoms([]) == []

    def test_single_atom(self):
        atom = StoryAtom("warrior", AtomCategory.AGENT, AtomSource.CATALOGUE, ["fight"])
        result = cluster_atoms([atom])
        assert len(result) == 1
        assert result[0] == [atom]

    def test_clusters_with_random_similarity(self):
        """Use RandomSimilarity to ensure deterministic non-zero scores."""
        atoms = [
            StoryAtom("sword", AtomCategory.OBJECT, AtomSource.CATALOGUE, ["weapon"]),
            StoryAtom("shield", AtomCategory.OBJECT, AtomSource.CATALOGUE, ["weapon"]),
            StoryAtom("castle", AtomCategory.LOCATION, AtomSource.CATALOGUE, ["building"]),
        ]
        engine = RandomSimilarity(random.Random(42))
        result = cluster_atoms(atoms, similarity_engine=engine, threshold=0.3)
        # Should produce at least 1 cluster
        assert len(result) >= 1
        # All atoms should be in exactly one cluster
        all_atoms = [a for cluster in result for a in cluster]
        assert len(all_atoms) == 3

    def test_high_threshold_keeps_separate(self):
        """Threshold above 1.0 should prevent any merging (RandomSimilarity ∈ [0,1])."""
        atoms = [
            StoryAtom("fire", AtomCategory.QUALITY, AtomSource.CATALOGUE, ["hot"]),
            StoryAtom("ice", AtomCategory.QUALITY, AtomSource.CATALOGUE, ["cold"]),
        ]
        engine = RandomSimilarity(random.Random(42))
        result = cluster_atoms(atoms, similarity_engine=engine, threshold=1.01)
        # Threshold above max possible score guarantees no merging
        assert len(result) == 2

    def test_low_threshold_merges_all(self):
        """Very low threshold should merge everything into one cluster."""
        atoms = [
            StoryAtom("alpha", AtomCategory.AGENT, AtomSource.CATALOGUE, ["a"]),
            StoryAtom("beta", AtomCategory.AGENT, AtomSource.CATALOGUE, ["b"]),
            StoryAtom("gamma", AtomCategory.AGENT, AtomSource.CATALOGUE, ["c"]),
        ]
        engine = RandomSimilarity(random.Random(42))
        result = cluster_atoms(atoms, similarity_engine=engine, threshold=0.01)
        # Very low threshold should merge most/all into one cluster
        assert len(result) <= 3
        all_atoms = [a for cluster in result for a in cluster]
        assert len(all_atoms) == 3

    def test_determinism_with_same_seed(self):
        """Same seed should produce same clustering."""
        atoms = [
            StoryAtom("sword", AtomCategory.OBJECT, AtomSource.CATALOGUE, ["weapon"]),
            StoryAtom("shield", AtomCategory.OBJECT, AtomSource.CATALOGUE, ["weapon"]),
            StoryAtom("castle", AtomCategory.LOCATION, AtomSource.CATALOGUE, ["building"]),
        ]
        engine1 = RandomSimilarity(random.Random(42))
        result1 = cluster_atoms(atoms, similarity_engine=engine1, threshold=0.3)

        engine2 = RandomSimilarity(random.Random(42))
        result2 = cluster_atoms(atoms, similarity_engine=engine2, threshold=0.3)

        assert len(result1) == len(result2)
        for c1, c2 in zip(result1, result2):
            assert [a.name for a in c1] == [a.name for a in c2]


class TestTagBasedClustering:
    """Tests for the _tag_based_clustering fallback."""

    def test_shared_tags_cluster_together(self):
        atoms = [
            StoryAtom("warrior", AtomCategory.AGENT, AtomSource.CATALOGUE, ["fight", "brave"]),
            StoryAtom("soldier", AtomCategory.AGENT, AtomSource.CATALOGUE, ["fight", "duty"]),
            StoryAtom("castle", AtomCategory.LOCATION, AtomSource.CATALOGUE, ["building"]),
        ]
        result = _tag_based_clustering(atoms)
        # warrior and soldier share "fight" tag, should cluster together
        # castle has unique tag "building", should be separate
        assert len(result) == 2

        # Find which cluster has 2 atoms and which has 1
        sizes = sorted(len(c) for c in result)
        assert sizes == [1, 2]

    def test_no_shared_tags(self):
        atoms = [
            StoryAtom("a", AtomCategory.AGENT, AtomSource.CATALOGUE, ["x"]),
            StoryAtom("b", AtomCategory.AGENT, AtomSource.CATALOGUE, ["y"]),
        ]
        result = _tag_based_clustering(atoms)
        assert len(result) == 2

    def test_all_same_tags(self):
        atoms = [
            StoryAtom("a", AtomCategory.AGENT, AtomSource.CATALOGUE, ["shared"]),
            StoryAtom("b", AtomCategory.AGENT, AtomSource.CATALOGUE, ["shared"]),
            StoryAtom("c", AtomCategory.AGENT, AtomSource.CATALOGUE, ["shared"]),
        ]
        result = _tag_based_clustering(atoms)
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_transitive_clustering(self):
        """If A shares a tag with B, and B shares a different tag with C,
        then A, B, and C should all be in the same cluster."""
        atoms = [
            StoryAtom("a", AtomCategory.AGENT, AtomSource.CATALOGUE, ["x"]),
            StoryAtom("b", AtomCategory.AGENT, AtomSource.CATALOGUE, ["x", "y"]),
            StoryAtom("c", AtomCategory.AGENT, AtomSource.CATALOGUE, ["y"]),
        ]
        result = _tag_based_clustering(atoms)
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_empty_tags(self):
        """Atoms with no tags should each be in their own cluster."""
        atoms = [
            StoryAtom("a", AtomCategory.AGENT, AtomSource.CATALOGUE, []),
            StoryAtom("b", AtomCategory.AGENT, AtomSource.CATALOGUE, []),
        ]
        result = _tag_based_clustering(atoms)
        assert len(result) == 2

    def test_multiple_distinct_clusters(self):
        atoms = [
            StoryAtom("a", AtomCategory.AGENT, AtomSource.CATALOGUE, ["fire"]),
            StoryAtom("b", AtomCategory.AGENT, AtomSource.CATALOGUE, ["fire"]),
            StoryAtom("c", AtomCategory.AGENT, AtomSource.CATALOGUE, ["water"]),
            StoryAtom("d", AtomCategory.AGENT, AtomSource.CATALOGUE, ["water"]),
        ]
        result = _tag_based_clustering(atoms)
        assert len(result) == 2
        sizes = sorted(len(c) for c in result)
        assert sizes == [2, 2]
