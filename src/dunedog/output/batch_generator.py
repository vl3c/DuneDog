"""Batch story skeleton generator — full pipeline orchestration."""

from __future__ import annotations

import random

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from dunedog.models.config import GenerationConfig, EvolutionConfig
from dunedog.models.results import SamplingStrategy, LetterSoupResult, DictionaryChaosResult
from dunedog.models.skeleton import StorySkeleton, PrimordialSource, GenerationStats
from dunedog.models.atoms import StoryAtom
from dunedog.utils.seed_manager import SeedManager
from dunedog.chaos.letter_soup import LetterSoupGenerator
from dunedog.chaos.dictionary_chaos import DictionaryChaosEngine
from dunedog.crystallize.crystallizer import SeedCrystallizer
from dunedog.crystallize.neologism_definer import NeologismDefiner
from dunedog.catalogues.loader import AtomCatalogue
from dunedog.engines.tarot_spread import TarotSpreadEngine
from dunedog.engines.markov_chains import NarrativeMarkovChain
from dunedog.engines.constraint_solver import WorldConstraintSolver


class StoryBatchGenerator:
    """Generates batches of story skeletons through the full pipeline."""

    def __init__(
        self,
        config: GenerationConfig,
        seed_manager: SeedManager | None = None,
    ):
        self.config = config
        self._seed_mgr = seed_manager or SeedManager(config.seed)

        # Lazy-init shared resources
        self._catalogue: AtomCatalogue | None = None
        self._soup_gen: LetterSoupGenerator | None = None
        self._dict_engine: DictionaryChaosEngine | None = None
        self._crystallizer: SeedCrystallizer | None = None
        self._neologism_definer: NeologismDefiner | None = None
        self._tarot: TarotSpreadEngine | None = None
        self._markov: NarrativeMarkovChain | None = None
        self._solver: WorldConstraintSolver | None = None

    def _init_components(self) -> None:
        """Lazy-initialize all pipeline components."""
        if self._catalogue is not None:
            return

        self._catalogue = AtomCatalogue.load()
        self._soup_gen = LetterSoupGenerator()
        self._dict_engine = DictionaryChaosEngine()
        self._crystallizer = SeedCrystallizer(
            catalogue=self._catalogue,
            similarity_threshold=self.config.crystallization.similarity_threshold,
        )
        self._neologism_definer = NeologismDefiner()
        self._tarot = TarotSpreadEngine(catalogue=self._catalogue)
        self._markov = NarrativeMarkovChain()
        self._solver = WorldConstraintSolver(catalogue=self._catalogue)

    def generate_batch(self, n: int | None = None, show_progress: bool = True) -> list[StorySkeleton]:
        """Generate a batch of story skeletons through the full pipeline.

        Returns skeletons sorted by coherence score (best first).
        """
        self._init_components()
        count = n or self.config.skeletons_to_generate
        skeletons: list[StorySkeleton] = []

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
            ) as progress:
                task = progress.add_task("Generating skeletons...", total=count)
                for i in range(count):
                    rng = self._seed_mgr.child_rng(f"skeleton_{i}")
                    sk = self.generate_single(rng)
                    skeletons.append(sk)
                    progress.update(task, advance=1)
        else:
            for i in range(count):
                rng = self._seed_mgr.child_rng(f"skeleton_{i}")
                sk = self.generate_single(rng)
                skeletons.append(sk)

        # Evolutionary refinement
        if self.config.evolution.enabled and self.config.evolution.generations > 0:
            skeletons = self._evolve(skeletons)

        # Sort by coherence score
        skeletons.sort(key=lambda s: s.coherence_score, reverse=True)
        return skeletons

    def generate_single(self, rng: random.Random) -> StorySkeleton:
        """Generate a single story skeleton through the full pipeline."""
        self._init_components()
        cfg = self.config

        # -- Layer 0: Primordial Chaos --
        soup_result: LetterSoupResult | None = None
        chaos_result: DictionaryChaosResult | None = None

        if cfg.chaos.use_letter_soup:
            soup_rng = random.Random(rng.randint(0, 2**63))
            soup_result = self._soup_gen.generate_and_parse(
                cfg.chaos.soup_length, soup_rng,
                enable_near_words=cfg.chaos.enable_near_words,
            )

        if cfg.chaos.use_dictionary:
            dict_rng = random.Random(rng.randint(0, 2**63))
            strategy = SamplingStrategy(cfg.chaos.sampling_strategy)
            chaos_result = self._dict_engine.process(
                dict_rng, strategy, cfg.chaos.dictionary_word_count, soup_result,
            )

        # Collect all extracted words
        all_words: list[str] = []
        if soup_result:
            all_words.extend(soup_result.all_words())
        if chaos_result:
            all_words.extend(chaos_result.combined_words or chaos_result.sampled_words)

        # -- Layer 1: Crystallization --
        crystal_rng = random.Random(rng.randint(0, 2**63))
        crystal = self._crystallizer.crystallize(
            all_words, crystal_rng, soup_result, chaos_result,
        )

        # Define neologisms
        if cfg.crystallization.enable_neologisms and soup_result:
            neo_rng = random.Random(rng.randint(0, 2**63))
            context_words = all_words[:20]
            for neo in soup_result.neologisms:
                self._neologism_definer.define(neo, context_words, neo_rng)

        atoms = crystal.all_atoms

        # -- Layer 2: Generative Engines --
        skeleton: StorySkeleton | None = None

        if cfg.engines.use_tarot and atoms:
            spread_rng = random.Random(rng.randint(0, 2**63))
            spread_type = rng.choice(self._tarot.spread_types)
            skeleton = self._tarot.generate_skeleton(spread_type, atoms, spread_rng)

        if skeleton is None:
            skeleton = StorySkeleton(atoms=atoms)

        # Add Markov beats if enabled
        if cfg.engines.use_markov:
            beat_rng = random.Random(rng.randint(0, 2**63))
            beats = self._markov.generate_sequence(
                beat_rng, cfg.engines.min_beats, cfg.engines.max_beats,
            )
            skeleton.beats = beats
            skeleton.stats.beat_count = len(beats)

        # Score via constraint solver
        if cfg.engines.use_constraint_solver:
            self._solver.score_and_update(skeleton)

        # Attach provenance
        skeleton.primordial_source = PrimordialSource(
            letter_soup_raw=soup_result.raw_soup if soup_result else "",
            exact_words=soup_result.exact_words if soup_result else [],
            near_words=[m for _, m, _ in soup_result.near_words] if soup_result else [],
            neologisms=[n.text for n in soup_result.neologisms] if soup_result else [],
            dictionary_words=chaos_result.sampled_words if chaos_result else [],
            phonetic_mood=soup_result.phonetic_mood if soup_result else "",
        )
        skeleton.seed = rng.randint(0, 2**63)

        return skeleton

    def _evolve(self, skeletons: list[StorySkeleton]) -> list[StorySkeleton]:
        """Run evolutionary refinement on the population."""
        try:
            from dunedog.engines.evolutionary import StoryEvolutionEngine
        except ImportError:
            return skeletons

        evo_rng = self._seed_mgr.child_rng("evolution")
        engine = StoryEvolutionEngine(
            constraint_solver=self._solver,
            catalogue=self._catalogue,
        )
        result = engine.evolve(skeletons, self.config.evolution, evo_rng)
        return result.population
