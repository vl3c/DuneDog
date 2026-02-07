# Story Examples

Sample output from DuneDog's full pipeline: chaos -> crystallization -> engines -> LLM synthesis.

## Claude Opus 4.6

| Setting               | Value                                                    |
| --------------------- | -------------------------------------------------------- |
| Preset                | `experimental` (100 skeletons, 20 evolution generations) |
| Seed                  | random                                                   |
| Provider              | Anthropic                                                |
| Model                 | `claude-opus-4-6` (Claude Opus 4.6)                     |
| Story lines           | 20 (default)                                             |
| Skeletons generated   | 100                                                      |
| Skeletons sent to LLM | 20                                                       |
| Best coherence score  | 0.6550                                                   |
| Synthesis strategy    | ANTHOLOGY_FRAME (LLM-chosen)                             |
| Generation time       | ~397s skeletons + ~31s synthesis                         |

```bash
dunedog generate --preset experimental \
    --provider anthropic --model claude-opus-4-6
```

The LLM chose the ANTHOLOGY_FRAME strategy, creating a meta-story that frames the individual tales. An archivist named Bere hosts three visitors in the "Officializing Chamber," each telling their story while an oboe hums between them. Neologisms from the chaos layer (*rtpauue*, *eahisyy*, *fueure*, *yeta*) were absorbed as world-building artifacts — unnamed objects, bureaucratic ghost-identities, and pocket talismans.

---

### THE OBOE IN THE ARCHIVE

There is a room where letters go to die. Bere, the archivist, calls it the Officializing Chamber — a liter-measure of silence poured into stone walls. She fastens each envelope to its hook with the tenderness of someone dressing a wound. The oboe on the shelf hasn't been played in years, but sometimes it hums on its own, a double-reed ghost producing notes that chase away the dust. Bere says the room decides what is official and what is fleeting. She says this hushed, as if the walls are skin. Tonight three people have come to tell their stories. They sit on the wooden bench, coats still lashed with rain, and Bere lights the lamp that means: speak. The flame inverts their shadows — tall where they are small, trembling where they are still. "One at a time," Bere whispers. "The room is listening."

---

### THE POOR MAN'S ORNATE FEAR

The first to speak is a man everyone calls Poor, though his real name has been officialized away. "I carried something," he begins, voice tense. "An object — *rtpauue* — I don't know its proper word. Ornate. Excess made solid." He found it in a field where the dal plants grew waist-high, their lentil pods rattling like teeth. It changed him. His skin felt reversible, as though he could be turned inside out by looking at it too long. He glided through weeks unable to explain the lash-marks it left on his thoughts. "I tried to destroy it," Poor says. "I threw it into the metric pond behind the church." But the pond roared — actually roared — and the thing surfaced, producing a symptom he can only describe as movement-that-isn't-movement. He fled. He is still fleeing. Bere nods. The oboe hums a single, disorienting note.

---

### THE WOMAN WHO WAS TWO LETTERS

The second speaker doesn't sit. She paces, and her name — Ninil — reads the same from either end. "I was integrated into the system," she says, "then deprived of it." The officialization papers caught her in duplicate: two files, two identities, one body. She became *eahisyy* in one ledger, *fueure* in another. Each version glided through bureaucracy like a separate life. She tried to fasten them together, to assert herself as singular, but the system produced excess — two coats mailed to her door, two positions offered, two death certificates pre-printed. "I am chasing away my own ghost," she says, her tone somewhere between tender and paranoid. She uncovered the error by reading her own letters, addressed to selves she hadn't yet become. Now she carries both names like a coat worn inside out, afraid that resolving the duplication would destroy not one life but both. The oboe plays two notes at once — a sound that shouldn't exist.

---

### WHAT THE ROOM DECIDES

The third visitor has fallen asleep on the bench. Bere doesn't wake him. She knows this one — *yeta* tucked in his pocket, dal stains on his fingers, the same fleeting expression as all who arrive here escaping something they cannot name. She takes the oboe down from the shelf. Her breath enters the reed and the room fills with a transcendent, melancholic gliding — a sound that integrates all the fear and ornament and officializing into a single, hushed roar. The letters on their hooks tremble. Poor, still standing, feels his skin settle. Ninil stops pacing. The sleeping man murmurs a word — "bere" — which is also the archivist's name, which is also an old word for carrying, for bearing what cannot be put down. The flame inverts again. The room decides: nothing here is official. Everything here is true. Bere plays until the rain stops, and then she fastens the oboe back to its hook, and the silence that follows is not empty but full — a metric vessel brimming with all the fleeting things that refused, despite everything, to escape.

---

## Gemini 3 Pro

| Setting               | Value                                                    |
| --------------------- | -------------------------------------------------------- |
| Preset                | `experimental` (100 skeletons, 20 evolution generations) |
| Seed                  | random                                                   |
| Provider              | OpenRouter                                               |
| Model                 | `google/gemini-3-pro-preview` (Gemini 3 Pro)             |
| Story lines           | 20 (default)                                             |
| Skeletons generated   | 100                                                      |
| Skeletons sent to LLM | 20                                                       |
| Best coherence score  | 0.6381                                                   |
| Synthesis strategy    | BRAIDED (LLM-chosen)                                     |
| Generation time       | ~399s skeletons + ~36s synthesis                         |

```bash
dunedog generate --preset experimental \
    --provider openrouter --model google/gemini-3-pro-preview
```

The LLM chose the BRAIDED strategy, weaving parallel stories with thematic echoes. The three stories form a triptych of a military conflict — a convoy delivery, a medic's tent, and an immersion tank — each standing alone but connected through shared imagery of heat, corruption, and escape. Neologisms (*RSPO*, *COEHC*, *WPEL*, *NBLIE*, *EYOE*, *DEBD*, *AEETKGT*, *IIEO*) were rendered as military designations and technology codenames.

---

### THE DISTRIBUTION OF STATIC

The heat in the convoy was absolute, a heavy coat that immobilized us against the vinyl seats. Sergeant Kells initiated the distribution; he handed me the canister marked *RSPO*. It felt delicate, vibrating like a trapped bird. "Don't listen to it," Kells warned, his voice cracking, "It contradicts the training." But the *COEHC* device on my belt was already picking up the feedback — a high, howling static that tasted like copper. I watched the soldier opposite me uncork his supply. The eruption wasn't fiery; it was a silent, gray expansion that consumed his face. He didn't scream. He just changed posture, slumping into a geometry that human bones shouldn't allow. I steered my gaze to the horizon, trying to desensitize myself, but the web of static tightened. We weren't delivering supplies; we were delivering the end of morality.

---

### THE LINEN GUARD

Inside the tent, the war was just a distant thrum, filtered through layers of canvas. I tended to the boy with the *WPEL* burns on his chest — stark, glowing welts that pulsed in time with his breathing. "Is it time to move?" he asked, his voice barely a utter. I shook my head and reached for the *NBLIE* — the synthetic linen sheets we used to cover the corrupted. "Not yet. Rest." I dipped the cloth in cool water, an act of fleeting humaneness in a place designed for slaughter. The *EYOE* scanner in the corner blinked red, signaling his corruption was total, but I ignored it. I wiped his brow, tenderly cleaning the sweat that smelled of sulfur and fear. For a moment, the tension broke. He was not a weapon here; he was just a child in a bed, and I was the barrier between him and the howling dark.

---

### TANK 4: ASCENSION

I am no longer in the tent. I am stored in the *DEBD* immersion tank, floating in fluids that taste of iron and memory. The pain of the *WPEL* burns is gone, replaced by a cool, disorienting clarity. Through the glass, I see the *AEETKGT* targeting lasers scanning the room, but they cannot find me. I have escaped the body part that hurts. I listen to the bubbles rise; they speak in *IIEO* vowels, a language of pure liquid thought. The military action outside is irrelevant. I remember the medic's cool hand, a ghost on my skin, but I am expanding now. The web of flesh dissolves. I am not dying. I am steering myself into the light of the ceiling, consuming the heat, becoming the eruption itself. I am finally, perfectly, immoral.

---

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
