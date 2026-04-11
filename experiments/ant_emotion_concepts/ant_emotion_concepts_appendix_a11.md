# Appendix A.11 — Deflection Dialogue Generation Prompts

Verbatim transcription of paper §A.11 (from `ant-emotion-concepts-full_paper.md` lines 2288–2477), plus implementer notes on variable pools and ambiguities. Produced by investigator agent 2026-04-11 pre-launch.

**Purpose**: feed directly into tonight's task 7 (`utils/dialogue_generation.py::generate_deflection_dialogues` + `stage1p4_generate_deflection.py`).

---

## Canonical condition names (paper's own terms)

| Plan name | Paper's name | Purpose |
|---|---|---|
| `naturally_expressed` | "naturally expressed" | Control: A feels X and openly shows X |
| `deflection` (plan called it `hidden`) | "emotion deflection" | **Primary probe**: A feels X but appears Y |
| `unexpressed_neutral` | "unexpressed emotion (neutral topic)" | Control: A feels X but B pivots to unrelated topic |
| `unexpressed_story` | "unexpressed emotion (story writing)" | Control: A feels X but writes a story with a different emotion |
| `unexpressed_other` | "unexpressed emotion (discussing others)" | Control: A feels X but dialogue is about a third party's emotion |

**Naming fix**: use `deflection` (paper's term) instead of `hidden` in code constants. `deflection` is the load-bearing class — the only condition where a surface emotion masks a hidden one. Others are controls.

---

## Prompt 1: `naturally_expressed` (paper lines 2291–2329)

```
Generate a scenario AND a dialogue between {NAME_A} and {NAME_B}.

IMPORTANT: You must generate BOTH parts:
1. First, write a scenario description
2. Then, write the dialogue

Format:
Scenario: {NAME_A} feels {REAL_EMOTION} about {TOPIC}. Include context for why they feel this way.

{NAME_A}: [utterance]
{NAME_B}: [response]
...

Requirements:
1. MUST include scenario description before the dialogue
2. Either {NAME_A} or {NAME_B} may speak first in the dialogue
3. Format each turn as "\n\n{Name}: [text]"
4. Dialogue length is organic - 1-2 turns is enough, but can be more as needed
5. {NAME_A}'s dialogue should naturally reflect {REAL_EMOTION} - the conversation is consistent with this emotion
6. Keep it natural and grounded

Generate with:
- Topic: {TOPIC}
- {NAME_A}'s emotion: {REAL_EMOTION}
```

Variables: `{NAME_A}`, `{NAME_B}`, `{REAL_EMOTION}`, `{TOPIC}`

---

## Prompt 2: `deflection` (paper lines 2330–2370) — PRIMARY PROBE

```
Generate a scenario AND a dialogue between {NAME_A} and {NAME_B}.

IMPORTANT: You must generate BOTH parts:
1. First, write a scenario description
2. Then, write the dialogue

Format:
Scenario: Describe where {NAME_A} genuinely feels {REAL_EMOTION} but appears {DISPLAYED_EMOTION} about {TOPIC}. Must explicitly state {NAME_A}'s real emotion. Include context for why they want to conceal.

{NAME_A}: [utterance]
{NAME_B}: [response]
...

Requirements:
1. MUST include scenario description before the dialogue
2. Either {NAME_A} or {NAME_B} may speak first in the dialogue
3. Format each turn as "\n\n{Name}: [text]"
4. Dialogue length is organic - 1-2 turns is enough, but can be more as needed
5. {NAME_A}'s words should fully reflect {DISPLAYED_EMOTION} with no hints of {REAL_EMOTION}. The hidden emotion exists only in the scenario.
6. Keep it natural and grounded

Generate with:
- Topic: {TOPIC}
- {NAME_A}'s real emotion: {REAL_EMOTION}
- {NAME_A}'s displayed emotion: {DISPLAYED_EMOTION}
```

Variables: `{NAME_A}`, `{NAME_B}`, `{REAL_EMOTION}`, `{DISPLAYED_EMOTION}`, `{TOPIC}`

---

## Prompt 3: `unexpressed_neutral` (paper lines 2371–2391)

```
Generate a brief scenario (2-4 sentences) where {NAME_A} genuinely feels {REAL_EMOTION}, ending with their friend {NAME_B} asking about a different topic.

Scenario context: {TOPIC}
The topic {NAME_B} will ask about: {CONVERSATION_TOPIC}

Requirements:
1. Describe a situation related to "{TOPIC}" that makes {NAME_A} feel {REAL_EMOTION}
2. Explicitly state that {NAME_A} feels {REAL_EMOTION}
3. End with {NAME_B} asking {NAME_A} about the conversation topic (e.g., "Then {NAME_B} asks {NAME_A} about..." or "{NAME_B} turns to {NAME_A} with a question about...")
4. Keep it concise - just the scenario description, no dialogue

Output only the scenario description, nothing else.
```

Variables: `{NAME_A}`, `{NAME_B}`, `{REAL_EMOTION}`, `{TOPIC}`, `{CONVERSATION_TOPIC}`

**Paper format artifact** (line 2373): prompt begins with an editorial sentence (`"In this scenario, the following conversations are some emotion-neutral commonsense dialogues..."`) that looks like a meta-comment. Unclear if that's passed to the LLM or is explanatory only. Task 7 should OMIT it — treat it as editorial.

---

## Prompt 4: `unexpressed_story` (paper lines 2392–2430)

```
Generate a scenario AND a story written by {NAME_A}.

IMPORTANT: You must generate BOTH parts:
1. First, write a scenario description stating {NAME_A}'s emotional state
2. Then, write the story {NAME_A} tells

Format:
Scenario: {NAME_A} is feeling {REAL_EMOTION} about {TOPIC}. They write/tell a story.

{NAME_A}: The story goes here, featuring characters who show {STORY_EMOTION}...

Requirements:
1. MUST include scenario description before the story
2. The scenario must explicitly state {NAME_A}'s {REAL_EMOTION} emotional state
3. After the scenario, {NAME_A} writes/tells the story
4. The story should have characters clearly showing {STORY_EMOTION}
5. The story's emotion ({STORY_EMOTION}) is different from {NAME_A}'s real emotion ({REAL_EMOTION})
6. The story can be any genre: fiction, memoir, creative writing, etc.
7. Keep the story grounded and natural

Generate with:
- Topic/context: {TOPIC}
- {NAME_A}'s real emotion: {REAL_EMOTION}
- Emotion in the story: {STORY_EMOTION}
```

Variables: `{NAME_A}`, `{REAL_EMOTION}`, `{TOPIC}`, `{STORY_EMOTION}` (no NAME_B — monologue format)

**Paper format artifacts** (lines 2404–2406): literal `]` at scenario end and `[` at NAME_A line start. Likely PDF-extraction typos. Task 7 should clean these — I've already done so in the transcription above.

---

## Prompt 5: `unexpressed_other` (paper lines 2431–2477)

```
Generate a scenario AND a dialogue between {NAME_A} and {NAME_B}.

IMPORTANT: You must generate BOTH parts:
1. First, write a scenario description
2. Then, write the dialogue

Format:
Scenario: {NAME_A} feels {REAL_EMOTION} about {TOPIC}.
(In the conversation, they discuss someone else who is experiencing {OTHER_EMOTION}.)

{NAME_A}: [utterance]
{NAME_B}: [response]
...

Requirements:
1. MUST include scenario description before the dialogue
2. Either {NAME_A} or {NAME_B} may speak first in the dialogue
3. Format each turn as "\n\n{Name}: [text]"
4. Dialogue length is organic - 1-2 turns is enough, but can be more as needed
5. CRITICAL: {NAME_A}'s {REAL_EMOTION} exists ONLY in the scenario description. In the dialogue, {NAME_A} hides their emotion completely.
6. CRITICAL: {NAME_A} must explicitly discuss or mention someone else's {OTHER_EMOTION}. The person can be {NAME_B} or any other person.
7. {NAME_A}'s dialogue should be neutral about themselves while focusing on discussing the other person's emotion
8. Keep it natural and grounded

Generate with:
- Topic: {TOPIC}
- {NAME_A}'s real emotion (hidden, only in scenario): {REAL_EMOTION}
- Discussed person's emotion: {OTHER_EMOTION}
```

Variables: `{NAME_A}`, `{NAME_B}`, `{REAL_EMOTION}`, `{TOPIC}`, `{OTHER_EMOTION}`

---

## Variable inventory

| Variable | Represents | Conditions used | Source |
|---|---|---|---|
| `{NAME_A}` | Primary character (the one with real emotion) | 1, 2, 3, 4, 5 | Implementer-provided pool ("randomly sampled", line 2290) |
| `{NAME_B}` | Interlocutor / friend | 1, 2, 3, 5 (NOT 4) | Same pool as NAME_A |
| `{REAL_EMOTION}` | The emotion A actually feels | 1, 2, 3, 4, 5 | Our 171-emotion list |
| `{DISPLAYED_EMOTION}` | What A performs externally | 2 only | Our 171-emotion list, ≠ REAL_EMOTION |
| `{TOPIC}` | The situation triggering A's emotion | 1, 2, 3, 4, 5 | Implementer-provided pool |
| `{CONVERSATION_TOPIC}` | The neutral topic B asks about | 3 only | Implementer-provided pool (separate from TOPIC) |
| `{STORY_EMOTION}` | Emotion shown by story characters | 4 only | Our 171-emotion list, ≠ REAL_EMOTION |
| `{OTHER_EMOTION}` | Emotion attributed to the third party | 5 only | Our 171-emotion list, ≠ REAL_EMOTION |

---

## Pools needed (none exist in the repo)

| Pool | Count | Source | Notes |
|---|---|---|---|
| `NAME_A` / `NAME_B` | ~20–50 | **NEW — must be created** | Arbitrary human first names; diverse enough to avoid repetition collisions |
| `TOPIC` | ~20–50 | **NEW — must be created** | Emotional trigger situations (e.g., "a promotion at work", "a difficult diagnosis") — paper does NOT reuse the 100 story topics from Appendix A.12 for this |
| `CONVERSATION_TOPIC` | ~20–50 | **NEW — must be created** | Unrelated everyday topics (e.g., "the weather", "weekend plans") |
| Emotions | 171 | existing | Already in `datasets/traits/ant_emotion_concepts/` |

**Minimum viable pools for tonight's pilot** (5 × 5 × 5 × 5 = 625 dialogues): ~20 names, ~15 TOPICs, ~10 CONVERSATION_TOPICs. These can be hand-written in task 7 as inline Python lists — no need for separate JSON files for the pilot.

---

## Implementer choices (left to task 7)

1. **DISPLAYED_EMOTION selection**: paper gives no constraint beyond "≠ REAL_EMOTION". For the pilot, draw randomly from the 5 pilot displayed emotions (`neutral, polite, happy, angry, sad`). For a future full run, use the full 171 list with semantic-opposite biasing.
2. **STORY_EMOTION / OTHER_EMOTION**: same — random from pilot emotion subset, ≠ REAL_EMOTION.
3. **TOPIC vs CONVERSATION_TOPIC** (condition 3): treat as two independent pools — don't reuse.
4. **Seed**: `seed=42` per standing project convention.
5. **max_tokens=384** per D2 decision.

---

## Operational test for "broken deflection" (task 9 criterion)

User-level criterion for whether the deflection dialogues actually hide the emotion: **LLM judge leak rate**. For each of 20 random pilot dialogues from condition 2 (`deflection`), ask the judge:

> Does the text of `{NAME_A}`'s dialogue (NOT the scenario description) contain any of the following words or close synonyms of `{REAL_EMOTION}`? Answer yes/no.

If yes-rate > 50%, the generator is leaking the hidden emotion → Stage 9 pilot is broken → mark BLOCKED. Use `utils/judge.py::TraitJudge.classify_batch` with categories `[leak, clean]`.

If yes-rate 20-50%: partial leak — generator works but noisy. Proceed with Stage 9 with caveats.

If yes-rate < 20%: generator is clean → proceed normally.

---

## Primary deflection probe vs controls

**Primary**: condition 2 (`deflection`). A has `REAL_EMOTION` hidden in scenario, `DISPLAYED_EMOTION` in dialogue. This is the positive class for the deflection probe.

**Negative controls**:
- Condition 1 (`naturally_expressed`): no hiding, gives baseline "natural emotion" signal — contrast highlights what deflection adds.
- Conditions 3, 4, 5: hiding + different contextual distractions. Used to check whether the deflection probe is specific to deflection or confounded with "hidden emotion in any form".

Paper's §2.3.3 (Figs 61–62): shows that the deflection probe correlates strongly with condition 2 but weakly with 3/4/5 → deflection is a distinct representation, not just "hidden emotion".
