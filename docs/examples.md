# Story Examples

Sample output from DuneDog's full pipeline: chaos → crystallization → engines → LLM synthesis.

## Gemini 3 Flash

| Setting               | Value                                                    |
| --------------------- | -------------------------------------------------------- |
| Preset                | `experimental` (100 skeletons, 20 evolution generations) |
| Seed                  | `42`                                                     |
| Provider              | OpenRouter                                               |
| Model                 | `google/gemini-3-flash-preview` (Gemini 3 Flash)         |
| Story lines           | 20 (default)                                             |
| Skeletons generated   | 100                                                      |
| Skeletons sent to LLM | 20                                                       |
| Best coherence score  | 0.6625                                                   |
| Synthesis strategy    | CAUSE_AND_ECHO (LLM-chosen)                              |
| Total tokens          | 5,198 (4,557 prompt + 641 completion)                    |
| Estimated cost        | $0.0042                                                  |
| Generation time       | ~505s skeletons + ~7s synthesis                          |

```bash
dunedog generate --preset experimental --seed 42 \
    --provider openrouter --model google/gemini-3-flash-preview
```

The LLM chose the CAUSE_AND_ECHO strategy, where each story's ending triggers the next story's beginning. The neologisms (*snee*, *teety*, *betags*, *ahey*, *dit*, *ich*) emerged from the letter soup chaos layer and were woven into the narrative by the LLM as world-building elements.

---

### THE ARCHIVE OF FLEETING THINGS

In the silent wing of the Blue Dynasty's library, the archivist known only as the Lost Agent sorted through the *snee* — the delicate, crystalline dust of unrecorded memories. He held a *teety*, a small glass sphere containing the resonance of a forgotten blessing. His work was the internalization of excess; he maintained the *betags*, those ornate labels that tethered drifting souls to their names. But a secret tension thrummed beneath the floorboards. The *teety* began to vibrate, rotating with a sudden, inhuman force. The archivist felt a symptom of infection — not of the body, but of the destiny he served. As the rotation reached a climax, the glass shattered. The diffusion of light was not a failure but an escape. He watched the shards melt into telegraphic signals, fleeing the region toward the cold permanence of the stars, leaving behind only the smell of burnt cornbread and a silence where there should be an answer.

---

### THE COMB

The signal from the tower traveled through the web of the city, hitting a small shop where an old woman used a letter opener made of frozen lightning to sort her mail. Among the bills was an *ahey*, a message written in invisible ink that only appeared when chilled. "The *snee* has fallen," it read. Panic, a familiar rotation in her chest, began to escalate. She reached for her most prized possession: a comb that untangles thoughts. As she pulled the teeth through her hair, the paranoid static of the city's many secrets began to flatten into a linear path. She saw the drawing of a door. With a piece of chalk that draws doors that open, she marked the wall. The *dit* — the rhythmic ticking of the city's heart — stuttered. She stepped through the chalk-drawn threshold just as the building began to misrepresent its own geometry, collapsing into an ornate pile of wild sheep and blue smoke.

---

### THE ECHO

She emerged into a dark, dreamlike wasteland where the only permanence was the cold. Here, the *lost* were not agents but symptoms of a larger malfunction. She found a book that writes itself, its pages turning in a frantic yowl of paper and ink. The ink was made of the Chinese monetary units of a dead empire, melted into a black infection of information. She realized the orientation of this world was separate from her own; here, the *ich* — the ego — was merely transferred property. She watched a *teety* roll across the frozen ground, a tiny orb of warmth in the inhumaneness of the dark. She tried to explain her presence to the shadows, but the tension of the void swallowed her voice. She sat on the permafrost, clutching the comb that no longer worked, watching the stars rotate in a direction that felt like a cry for help. The story in the book reached its resolution, writing her name into the final line.
