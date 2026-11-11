# DuneDog

Primordial Story Diffusion Engine — finding meaning in the desert of ideas.

DuneDog is a stochastic story generation engine that builds narratives from pure linguistic chaos. It extracts words and meaning from random letter sequences, crystallizes them into structured story atoms, assembles narrative skeletons through multiple generative engines, and optionally synthesizes final prose via an LLM.

## How It Works

The pipeline has four layers:

**Layer 0 — Primordial Chaos.** Two generators produce raw material. *Letter Soup* generates random strings with English-biased letter frequencies, then extracts real words, near-words (via Levenshtein distance), and neologisms (pronounceable non-words) with phonetic mood analysis. *Dictionary Chaos* samples real words using five strategies (uniform, frequency-weighted, rare, noun-heavy, phonetic clusters) and arranges them into near-grammatical phrases.

**Layer 1 — Crystallization.** Chaos output is mapped to **Story Atoms** — typed building blocks (agents, objects, locations, tensions, triggers, qualities). A chaos word *evokes* a curated atom from the ~300-entry catalogue when the word, a synonym or a direct hypernym matches one of the atom's tags or name words: *ache* summons "the archivist of wounds", *dale* summons "a valley where echoes live independently". Each evoked atom remembers the word that summoned it. Words that evoke nothing become raw atoms. The few most pronounceable neologisms (three per soup by default) are kept as coined words, each with a definition generated from its phonetic properties.

**Layer 2 — Generative Engines.** Multiple engines build **Story Skeletons** from the atoms:
- *Tarot Spread* — deals atoms into narrative positions across 5 spread types (Hero's Journey, Three Act, Love Triangle, Mystery, Cosmic). The draw is mostly curated atoms, with raw chaos words and coined words kept as texture.
- *Markov Chains* — generates beat sequences (13 beat types: OPENING, CRISIS, CLIMAX, etc.)
- *Constraint Solver* — validates against world rules and scores coherence: atom affinity, beat flow, shared themes and a varied cast. Word-class tags ("noun", "object") do not count as themes, and a repeated atom counts once.
- *Evolutionary* (optional) — breeds skeletons via selection, crossover, mutation, and wild card injection. Fitness is coherence plus novelty (distance to the nearest other skeletons), so the population does not collapse into clones. Mutation recombines atoms the chaos produced for this run; only wild cards bring in something from outside. An atom swapped in by crossover inherits the role of the atom it replaces.

**Layer 3 — LLM Synthesis.** A small, deliberately *diverse* set of skeletons (good coherence, few shared atoms) is sent to an LLM, which combines them into interconnected narratives using strategies like BRAIDED (parallel stories with thematic echoes), CAUSE_AND_ECHO (one story's ending begins another), RASHOMON (same events from different perspectives), or ANTHOLOGY_FRAME (meta-story containing others).

Each skeleton reaches the LLM as a *reading*, not a parts list (an illustrative seed, assembled from lines of real runs):

```
--- Seed 5 ---
Tone: eerie (the noise it came from sounded dreamy)
Reading:
  The Void (the emptiness before all things): a valley where echoes live independently [location] — summoned from the noise by the word "dale"
  The Signal (a message from the incomprehensible): the woman who collects silences [agent] — summoned from the noise by the word "bawd"
  The Remnant (what survives the encounter with infinity): eots [object] — a coined word (noun) meaning: the residue left behind after flo
Disruption: Silence spreads from one thing to the next, consuming all sound
```

The prompt also states two **laws of the world** all the stories share, taken from the world rules ("Mapmakers always get lost", "The sea remembers everything given to it"). Laws triggered by an atom in play come first; the rest are chosen deterministically from the atoms. The LLM is told to treat the seeds as provocations rather than a checklist, to leave at least half of the elements out, to use coined words in their given meaning, and to write a one-line premise (who wants what, but what stands in the way) before each story.

## Installation

Requires Python 3.11+.

```bash
pip install -e ".[dev]"
dunedog setup          # downloads required NLTK data
```

## Usage

```bash
# Full pipeline with LLM synthesis
dunedog generate --preset deep --provider openrouter \
    --model google/gemini-3-flash-preview --seed 42

# Generate skeletons only (no LLM, no API key needed)
dunedog generate --preset quick --no-llm --seed 42 -o skeletons.json

# Single skeleton with detailed trace output
dunedog demo --seed 42

# Just the chaos layer
dunedog soup --length 200 --count 3 --seed 42
```

### Presets

| Preset | Engines | Evolution | Skeletons | Sent to LLM |
|---|---|---|---:|---:|
| `quick` | tarot only (Hero's Journey) | off | 50 | 5 |
| `deep` | all, every spread type | 5 generations | 200 | 6 |
| `experimental` | all, every spread type | 20 generations | 100 | 6 |

"Sent to LLM" skeletons are chosen for diversity, not taken from the top of the ranking. Override the count with `--stories-for-llm`.

### LLM Providers

Set your API key via environment variable or `--api-key`:

| Provider | Env Variable | Example Model |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | `gpt-4o` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5-20250929` |
| `openrouter` | `OPENROUTER_API_KEY` | `google/gemini-3-flash-preview` |
| `chatgpt` | `CHATGPT_TOKEN` | `gpt-4o` |

## Determinism

The same seed always produces identical skeletons. All randomness flows through `SeedManager` which derives deterministic child RNGs via SHA-256. LLM output may vary across runs due to provider non-determinism.

## Design

- **Dataclasses** for data structures, **Pydantic** for config validation
- **WordNet** for semantic similarity (no heavy embedding models — runs on a Raspberry Pi 5)
- **httpx** for async LLM calls (no provider-specific SDKs)
- ~234k English words shipped in `data/english_words.txt`, lazy-loaded into a `frozenset`
- ~300 story atoms across 6 categories in `data/atoms.json`

## Testing

```bash
pytest tests/ -v         # 336 tests, ~30s
```

## License

MIT
