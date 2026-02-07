# DuneDog

Primordial Story Diffusion Engine — finding meaning in the desert of ideas.

DuneDog is a stochastic story generation engine that builds narratives from pure linguistic chaos. It extracts words and meaning from random letter sequences, crystallizes them into structured story atoms, assembles narrative skeletons through multiple generative engines, and optionally synthesizes final prose via an LLM.

## How It Works

The pipeline has four layers:

**Layer 0 — Primordial Chaos.** Two generators produce raw material. *Letter Soup* generates random strings with English-biased letter frequencies, then extracts real words, near-words (via Levenshtein distance), and neologisms (pronounceable non-words) with phonetic mood analysis. *Dictionary Chaos* samples real words using five strategies (uniform, frequency-weighted, rare, noun-heavy, phonetic clusters) and arranges them into near-grammatical phrases.

**Layer 1 — Crystallization.** Chaos output is mapped to **Story Atoms** — typed building blocks (agents, objects, locations, tensions, triggers, qualities) drawn from a catalogue of ~300 entries using WordNet similarity. Novel words become new atoms. Neologisms get definitions generated from their phonetic properties.

**Layer 2 — Generative Engines.** Multiple engines build **Story Skeletons** from the atoms:
- *Tarot Spread* — assigns atoms to narrative positions across 5 spread types (Hero's Journey, Conflict Diamond, etc.)
- *Markov Chains* — generates beat sequences (13 beat types: OPENING, TENSION, REVELATION, etc.)
- *Constraint Solver* — validates against world rules and scores coherence
- *Evolutionary* (optional) — breeds better skeletons via selection, crossover, mutation, and wild card injection

**Layer 3 — LLM Synthesis.** The top skeletons (ranked by coherence) are sent to an LLM which combines them into interconnected narratives using strategies like BRAIDED (parallel stories with thematic echoes), CAUSE_AND_ECHO (one story's ending begins another), RASHOMON (same events from different perspectives), or ANTHOLOGY_FRAME (meta-story containing others).

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
| `quick` | tarot only | off | 50 | 5 |
| `deep` | all | 5 generations | 200 | 20 |
| `experimental` | all | 20 generations | 500 | 20 |

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
pytest tests/ -v         # 311 tests, ~60s
```

## License

MIT
