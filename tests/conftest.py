"""Shared test fixtures."""
import pytest
import random
from dunedog.utils.seed_manager import SeedManager
from dunedog.models.atoms import StoryAtom, AtomCategory, AtomSource
from dunedog.catalogues.loader import AtomCatalogue


@pytest.fixture
def seed_manager():
    return SeedManager(42)

@pytest.fixture
def rng(seed_manager):
    return seed_manager.child_rng("test")

@pytest.fixture
def sample_atoms():
    """A diverse set of test atoms."""
    return [
        StoryAtom("the wanderer", AtomCategory.AGENT, AtomSource.CATALOGUE, ["journey", "mysterious"], 0.3),
        StoryAtom("the lighthouse keeper", AtomCategory.AGENT, AtomSource.CATALOGUE, ["isolation", "light"], 0.4),
        StoryAtom("a compass that points to regret", AtomCategory.OBJECT, AtomSource.CATALOGUE, ["navigation", "emotion"], 0.6),
        StoryAtom("a letter never sent", AtomCategory.OBJECT, AtomSource.CATALOGUE, ["communication", "loss"], 0.5),
        StoryAtom("the city beneath the lake", AtomCategory.LOCATION, AtomSource.CATALOGUE, ["hidden", "water"], 0.7),
        StoryAtom("the forest of whispers", AtomCategory.LOCATION, AtomSource.CATALOGUE, ["nature", "mystery"], 0.4),
        StoryAtom("a secret that must be spoken", AtomCategory.TENSION, AtomSource.CATALOGUE, ["secret", "compulsion"], 0.5),
        StoryAtom("a debt that cannot be paid", AtomCategory.TENSION, AtomSource.CATALOGUE, ["obligation", "impossible"], 0.6),
        StoryAtom("the last bell rings", AtomCategory.TRIGGER, AtomSource.CATALOGUE, ["ending", "sound"], 0.3),
        StoryAtom("luminous", AtomCategory.QUALITY, AtomSource.CATALOGUE, ["light", "beauty"], 0.2),
        StoryAtom("fractured", AtomCategory.QUALITY, AtomSource.CATALOGUE, ["broken", "damage"], 0.3),
    ]

@pytest.fixture
def catalogue():
    return AtomCatalogue.load()
