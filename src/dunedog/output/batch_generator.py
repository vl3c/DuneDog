"""Batch story skeleton generator — full pipeline orchestration."""

from __future__ import annotations

import logging
import random

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

log = logging.getLogger(__name__)

from dunedog.models.config import GenerationConfig, EvolutionConfig
from dunedog.models.results import SamplingStrategy, LetterSoupResult, DictionaryChaosResult, Neologism
from dunedog.models.skeleton import StorySkeleton, PrimordialSource, GenerationStats
from dunedog.models.atoms import AtomSource, StoryAtom
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
        self._crystallizer = SeedCrystallizer(catalogue=self._catalogue)
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
        log_interval = max(1, count // 20)  # log every ~5%

        log.info("Generating %d skeletons (preset=%s, seed=%s)",
                 count, self.config.preset.value, self.config.seed)

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
                    if (i + 1) % log_interval == 0:
                        best = max(s.coherence_score for s in skeletons)
                        log.debug("Skeleton %d/%d — best coherence %.4f", i + 1, count, best)
        else:
            for i in range(count):
                rng = self._seed_mgr.child_rng(f"skeleton_{i}")
                sk = self.generate_single(rng)
                skeletons.append(sk)
                if (i + 1) % log_interval == 0 or (i + 1) == count:
                    best = max(s.coherence_score for s in skeletons)
                    log.info("Skeleton %d/%d (%d%%) — best coherence %.4f",
                             i + 1, count, 100 * (i + 1) // count, best)

        # Evolutionary refinement
        if self.config.evolution.enabled and self.config.evolution.generations > 0:
            log.info("Starting evolution: %d generations, population %d",
                     self.config.evolution.generations, self.config.evolution.population_size)
            skeletons = self._evolve(skeletons)
            log.info("Evolution complete")

        # Sort by coherence score
        skeletons.sort(key=lambda s: s.coherence_score, reverse=True)
        log.info("Done: %d skeletons, best=%.4f, median=%.4f, worst=%.4f",
                 len(skeletons), skeletons[0].coherence_score,
                 skeletons[len(skeletons) // 2].coherence_score,
                 skeletons[-1].coherence_score)
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

        # Collect all extracted words.  A soup throws off hundreds of
        # pronounceable fragments; only the best few are kept as coinages,
        # otherwise they drown every real word.
        coined: list[Neologism] = []
        if soup_result and cfg.crystallization.enable_neologisms:
            coined = self._pick_coinages(
                soup_result.neologisms, cfg.crystallization.max_neologisms,
            )
        all_words: list[str] = []
        if soup_result:
            all_words.extend(soup_result.all_words())
            all_words.extend(n.text for n in coined)
        if chaos_result:
            all_words.extend(chaos_result.combined_words or chaos_result.sampled_words)

        # -- Layer 1: Crystallization --
        crystal_rng = random.Random(rng.randint(0, 2**63))
        crystal = self._crystallizer.crystallize(
            all_words, crystal_rng, soup_result, chaos_result,
        )

        # Define the coinages and attach each definition to its atom, so the
        # meaning travels with the word through evolution and into the prompt.
        atoms = crystal.all_atoms
        if cfg.crystallization.enable_neologisms and soup_result:
            neo_rng = random.Random(rng.randint(0, 2**63))
            context_words = all_words[:20]
            by_name = {a.name: a for a in atoms}
            for neo in coined:
                self._neologism_definer.define(neo, context_words, neo_rng)
                atom = by_name.get(neo.text.lower())
                if atom is not None:
                    atom.source = AtomSource.NEOLOGISM
                    atom.metadata["definition"] = neo.definition
                    atom.metadata["part_of_speech"] = neo.part_of_speech

        # -- Layer 2: Generative Engines --
        skeleton: StorySkeleton | None = None

        if cfg.engines.use_tarot and atoms:
            spread_rng = random.Random(rng.randint(0, 2**63))
            available_spreads = cfg.engines.spread_types or self._tarot.spread_types
            spread_type = rng.choice(available_spreads)
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
            neologisms=[n.text for n in coined],
            dictionary_words=chaos_result.sampled_words if chaos_result else [],
            phonetic_mood=soup_result.phonetic_mood if soup_result else "",
        )
        skeleton.seed = rng.randint(0, 2**63)

        return skeleton

    @staticmethod
    def _pick_coinages(neologisms: list[Neologism], limit: int) -> list[Neologism]:
        """The most pronounceable coinages that are not slices of each other.

        Neologisms come from a sliding window, so the best few are usually
        the same letters shifted by one ("inet", "netf", "etfe").  Skip any
        that shares a two-letter run with one already chosen.
        """
        chosen: list[Neologism] = []
        runs: set[str] = set()
        for neo in neologisms:  # already sorted most pronounceable first
            if len(chosen) >= limit:
                break
            text = neo.text.lower()
            own = {text[i : i + 2] for i in range(len(text) - 1)}
            if own & runs:
                continue
            runs |= own
            chosen.append(neo)
        return chosen

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
