# Story Examples

Sample output from DuneDog's full pipeline: chaos → crystallization → engines → LLM synthesis.

## Generation Settings

| Setting | Value |
|---|---|
| Preset | `quick` |
| Seed | `42` |
| Provider | OpenRouter |
| Model | `google/gemini-3-flash-preview` |
| Story lines | 20 (default) |
| Skeletons generated | 50 |
| Skeletons sent to LLM | 5 |
| Synthesis strategy | CAUSE_AND_ECHO (LLM-chosen) |
| Total tokens | 1,797 (1,200 prompt + 597 completion) |
| Estimated cost | $0.0024 |
| Generation time | ~27s skeletons + ~6s synthesis |

Command used:

```bash
dunedog generate --preset quick --seed 42 \
    --provider openrouter --model google/gemini-3-flash-preview
```

## Stories

The LLM chose the CAUSE_AND_ECHO strategy, where each story's ending triggers the next story's beginning. The neologisms (Manei, Ike, Aye, Sho, Yee) emerged from the letter soup chaos layer and were woven into the narrative by the LLM.

---

### THE ARCHIVE OF YES

The city was a fugitive shadow, a place defined by what it lacked. In the center of the plaza sat the Manei, a silver box hiding in plain sight, pulsing with the rhythm of a buried heart. Rune, a man of sharp angles and deeper suspicions, watched the crowd from the balcony of his Ike — a false storefront designed to mislead the prying eyes of the Law. He clutched the Aye, a slate of ancient stone that whispered the truth of the city's founding, a truth the governors had tried to bury in the silt of the river. When he finally pressed his thumb to the Manei's surface, the mechanism clicked with a definitive, hollow sound. A single word projected into the smog-choked sky: YES. It was not a sign of justice, but a signal — a flare sent into the great dark to announce that the world was ready to be seen. The signal did not stop at the clouds; it tore through them, striking the void like a hammer.

---

### THE GEOMETRY OF THE VOID

The signal hit the Hot, the airless silence between stars, vibrating through a Rein of invisible tethers that held the galaxy in place. Olea, a scholar with a fragile mind tuned to the frequencies of the deep, felt the shift in her marrow. She watched as the Fugitive patterns — alien, non-Euclidean shapes — began to reshape the reality of her laboratory. The walls became Sho, crumbling like dry biscuits as her identity dissolved into the vast, cold truth of the signal. What survived the encounter was only a Yee: a remnant of consciousness, a spark of human memory floating in a sea of infinite static. She was no longer a person, but a Witness to the shifting geometry, a ghost observing the birth of a new, terrible world where the ordinary was a forgotten dream.
