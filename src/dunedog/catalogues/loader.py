"""Catalogue loader — loads story atoms and affinities from JSON data files."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from dunedog.models.atoms import AffinityEntry, AtomCategory, AtomSource, StoryAtom

# Project data directory: <repo_root>/data/
_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


class AtomCatalogue:
    """In-memory catalogue of story atoms with lookup indices."""

    def __init__(self) -> None:
        self._atoms: list[StoryAtom] = []
        self._by_category: dict[AtomCategory, list[StoryAtom]] = defaultdict(list)
        self._by_tag: dict[str, list[StoryAtom]] = defaultdict(list)
        self._affinities: dict[tuple[str, str], AffinityEntry] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        atoms_path: str | Path | None = None,
        affinities_path: str | Path | None = None,
    ) -> AtomCatalogue:
        """Build a catalogue from JSON files.

        Parameters
        ----------
        atoms_path:
            Path to a JSON file containing a list of atom dicts.
            Defaults to ``data/atoms.json``.
        affinities_path:
            Path to a JSON file containing a list of affinity dicts.
            Defaults to ``data/affinities.json``.
        """
        catalogue = cls()

        atoms_file = Path(atoms_path) if atoms_path else _DATA_DIR / "atoms.json"
        if atoms_file.exists():
            with open(atoms_file, "r", encoding="utf-8") as fh:
                raw_atoms = json.load(fh)
            for entry in raw_atoms:
                # Default source to CATALOGUE for data-file atoms that omit it
                entry.setdefault("source", AtomSource.CATALOGUE.value)
                catalogue.add_atom(StoryAtom.from_dict(entry))

        affinities_file = (
            Path(affinities_path) if affinities_path else _DATA_DIR / "affinities.json"
        )
        if affinities_file.exists():
            with open(affinities_file, "r", encoding="utf-8") as fh:
                raw_affinities = json.load(fh)
            for entry in raw_affinities:
                aff = AffinityEntry.from_dict(entry)
                catalogue._affinities[aff.key] = aff

        return catalogue

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_atom(self, atom: StoryAtom) -> None:
        """Add an atom and update all indices."""
        self._atoms.append(atom)
        self._by_category[atom.category].append(atom)
        for tag in atom.tags:
            self._by_tag[tag].append(atom)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_by_category(self, category: AtomCategory) -> list[StoryAtom]:
        """Return all atoms matching *category*."""
        return list(self._by_category.get(category, []))

    def get_by_tag(self, tag: str) -> list[StoryAtom]:
        """Return all atoms carrying *tag*."""
        return list(self._by_tag.get(tag, []))

    def get_affinity(self, atom_a: str, atom_b: str) -> float:
        """Look up affinity strength between two atom names.

        Returns 0.0 when no affinity entry exists.
        """
        key = tuple(sorted((atom_a, atom_b)))
        entry = self._affinities.get(key)
        return entry.strength if entry is not None else 0.0

    def sample_weighted(
        self,
        category: AtomCategory,
        rng: random.Random,
        n: int = 1,
    ) -> list[StoryAtom]:
        """Sample *n* atoms from *category*, weighted by inverse rarity.

        Common atoms (low rarity) are more likely to be chosen.
        """
        pool = self._by_category.get(category, [])
        if not pool:
            return []
        # Weight = 1 - rarity so common atoms (rarity ~0) get weight ~1.
        weights = [max(1.0 - atom.rarity, 0.01) for atom in pool]
        n = min(n, len(pool))
        return rng.choices(pool, weights=weights, k=n)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def atoms(self) -> list[StoryAtom]:
        """All loaded atoms (read-only copy)."""
        return list(self._atoms)

    def __len__(self) -> int:
        return len(self._atoms)

    def __repr__(self) -> str:
        return f"AtomCatalogue(atoms={len(self._atoms)}, affinities={len(self._affinities)})"
