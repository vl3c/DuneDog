"""Semantic clustering of story atoms."""

from __future__ import annotations

from collections import defaultdict

from dunedog.models.atoms import StoryAtom
from dunedog.utils.embeddings import SimilarityEngine, get_similarity_engine


def cluster_atoms(
    atoms: list[StoryAtom],
    similarity_engine: SimilarityEngine | None = None,
    threshold: float = 0.5,
) -> list[list[StoryAtom]]:
    """Simple agglomerative clustering on atom similarity.

    Algorithm:
    1. Start with each atom in its own cluster.
    2. Find the two most similar clusters (average linkage).
    3. Merge if similarity > *threshold*.
    4. Repeat until no more merges are possible.

    Falls back to tag-based clustering if the similarity engine returns
    all-zero scores.
    """
    if not atoms:
        return []
    if len(atoms) == 1:
        return [list(atoms)]

    engine = similarity_engine or get_similarity_engine()

    # Pre-compute pairwise similarity matrix (atom names).
    n = len(atoms)
    sim_matrix: dict[tuple[int, int], float] = {}
    all_zero = True
    for i in range(n):
        for j in range(i + 1, n):
            score = engine.similarity(atoms[i].name, atoms[j].name)
            sim_matrix[(i, j)] = score
            if score > 0.0:
                all_zero = False

    # Fallback when no meaningful similarity signal exists.
    if all_zero:
        return _tag_based_clustering(atoms)

    # Initialise clusters: each atom in its own cluster.
    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}

    while len(clusters) > 1:
        best_pair: tuple[int, int] | None = None
        best_score = -1.0

        cluster_ids = sorted(clusters.keys())
        for idx_a in range(len(cluster_ids)):
            for idx_b in range(idx_a + 1, len(cluster_ids)):
                cid_a = cluster_ids[idx_a]
                cid_b = cluster_ids[idx_b]
                score = _average_linkage(clusters[cid_a], clusters[cid_b], sim_matrix)
                if score > best_score:
                    best_score = score
                    best_pair = (cid_a, cid_b)

        if best_pair is None or best_score < threshold:
            break

        # Merge cluster b into cluster a.
        cid_a, cid_b = best_pair
        clusters[cid_a] = clusters[cid_a] + clusters[cid_b]
        del clusters[cid_b]

    return [[atoms[i] for i in indices] for indices in clusters.values()]


def _average_linkage(
    cluster_a: list[int],
    cluster_b: list[int],
    sim_matrix: dict[tuple[int, int], float],
) -> float:
    """Compute average pairwise similarity between two clusters."""
    total = 0.0
    count = 0
    for i in cluster_a:
        for j in cluster_b:
            key = (min(i, j), max(i, j))
            total += sim_matrix.get(key, 0.0)
            count += 1
    return total / count if count else 0.0


def _tag_based_clustering(atoms: list[StoryAtom]) -> list[list[StoryAtom]]:
    """Fallback clustering by shared tags.

    Atoms that share at least one tag are placed in the same cluster.
    Uses a simple union-find approach.
    """
    n = len(atoms)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Build tag-to-atom index and union atoms sharing a tag.
    tag_index: dict[str, list[int]] = defaultdict(list)
    for i, atom in enumerate(atoms):
        for tag in atom.tags:
            tag_index[tag].append(i)

    for indices in tag_index.values():
        for k in range(1, len(indices)):
            union(indices[0], indices[k])

    # Collect clusters.
    groups: dict[int, list[StoryAtom]] = defaultdict(list)
    for i, atom in enumerate(atoms):
        groups[find(i)].append(atom)

    return list(groups.values())
