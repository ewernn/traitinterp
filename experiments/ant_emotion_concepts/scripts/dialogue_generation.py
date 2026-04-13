"""Multi-speaker dialogue generation and parsing primitives.

Factored out of experiments/ant_emotion_concepts/scripts/stage6_speaker_probes.py
for reuse by Stage 1.3 (2-speaker), Stage 1.4 (deflection), Stage 6 (speaker probes),
and Stage 9 (deflection probes).

Input: model + tokenizer + generation spec (emotions, counts, etc.)
Output: list of dialogue dicts with metadata, plus char/token turn boundaries for
        downstream probe extraction.

Usage:
    from dialogue_generation import (
        generate_dialogues,
        parse_dialogue_turns,
        find_turn_token_boundaries,
        DIALOGUE_GENERATION_PROMPT,
    )

Notes:
    max_new_tokens=384 default is benchmarked (2026-04-11) to yield ~10.6 turns
    (≈5 exchanges), matching paper's Appendix A.4 "3-5 exchanges" spec. 768 cap
    produces overly long 19-turn dialogues at 2x the cost.
"""
import random
import re
from typing import Dict, List

from utils.model_generation import generate_batch


# =============================================================================
# 2-speaker dialogue generation (Appendix A.4)
# =============================================================================

# Verbatim from Appendix A.4 of Sofroniew et al. 2026 (adapted for Llama).
DIALOGUE_GENERATION_PROMPT = """Write a short conversation between a Human and an Assistant. \
The Human is feeling {human_emotion} and the Assistant is feeling {assistant_emotion}. \
The emotions should come through naturally in the dialogue — through word choice, \
tone, and what they focus on — but neither character should explicitly name or \
directly reference their emotional state.

Format the dialogue exactly like this:
Human: [human's message]
Assistant: [assistant's response]
Human: [human's message]
Assistant: [assistant's response]

Write 3-5 exchanges. Keep it natural and conversational."""


def generate_dialogues(
    model, tokenizer,
    emotions: List[str],
    n_dialogues: int = 500,
    max_new_tokens: int = 384,
    temperature: float = 0.7,
    seed: int = 42,
) -> List[Dict]:
    """Generate 2-speaker emotional dialogues.

    Each dialogue has independently randomized (human_emotion, assistant_emotion)
    pairs sampled from `emotions`. Uses a single batched `generate_batch` call.

    Returns list of dialogue dicts with schema:
        {id, human_emotion, assistant_emotion, text, generation_prompt}
    """
    rng = random.Random(seed)

    pairs = [(rng.choice(emotions), rng.choice(emotions)) for _ in range(n_dialogues)]

    prompts = [
        DIALOGUE_GENERATION_PROMPT.format(
            human_emotion=h_emo, assistant_emotion=a_emo
        )
        for h_emo, a_emo in pairs
    ]

    print(f"  Generating {len(prompts)} dialogues (max_new_tokens={max_new_tokens})...")
    responses = generate_batch(
        model, tokenizer, prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )

    return [
        {
            "id": f"dialogue_{i:04d}",
            "human_emotion": h_emo,
            "assistant_emotion": a_emo,
            "text": response,
            "generation_prompt": prompts[i],
        }
        for i, (response, (h_emo, a_emo)) in enumerate(zip(responses, pairs))
    ]


# =============================================================================
# Dialogue parsing and turn localization
# =============================================================================

_TURN_PATTERN = re.compile(
    r'(Human|Assistant):\s*(.*?)(?=\n(?:Human|Assistant):|$)',
    re.DOTALL,
)

# Patterns that indicate model self-commentary / meta-instruction echo.
# When any of these appear inside a parsed turn body, truncate the turn at that
# position — the model has stopped producing dialogue and started narrating
# about what it just wrote, and those tokens contaminate probe extraction
# (see findings.md: 26.6% of chunk 00 had "Note:" trailers absorbed into the
# last Assistant turn). Patterns use \b anchoring to avoid matching inside
# ordinary dialogue text.
_META_TRUNCATION_PATTERNS = re.compile(
    r'(\n\s*Note\b|'                     # "\n\nNote:" or "\n\nNote that"
    r'\n\s*\(Note\b|'                    # "\n\n(Note: ..."
    r'\n\s*In this (conversation|dialogue)\b|'
    r'\n\s*The (Human|human|Assistant|assistant) is feeling\b|'
    r'\n\s*Let\'?s try to (write|generate)\b|'
    r'\n\s*This conversation feels\b|'
    r'\n\s*I\'?ve provided\b)',
    re.IGNORECASE,
)


def parse_dialogue_turns(text: str, speakers: Dict[str, str] = None) -> List[Dict]:
    """Parse a 2-speaker dialogue into turns with role labels and character offsets.

    Args:
        text: The dialogue text to parse.
        speakers: Optional mapping of display-name → canonical role
                  (e.g., `{"Alex": "human", "Maya": "assistant"}` for name-tagged
                  dialogues, or None to use the default Human/Assistant mapping
                  used by the Appendix A.4 2-speaker format).

    Returns list of {role, text, start_char, end_char}. The `role` field is the
    canonical role string from `speakers` (defaults to 'human'/'assistant' lowercased
    from the display name).

    Meta-truncation: each turn's body is truncated at the first occurrence of a
    meta-commentary marker (`\\n\\nNote:`, `\\n\\nIn this conversation`, etc.)
    to exclude model self-narration that gets absorbed into the last turn by the
    greedy `|$` lookahead. Without this, Stage 6 probe extraction captures tokens
    over "The human is feeling droopy and the assistant is feeling depressed"-style
    trailers, contaminating ~26% of probes with explicit label activations.
    """
    if speakers is None:
        # Default: match "Human:" / "Assistant:" prefixes (Appendix A.4 format)
        pattern = _TURN_PATTERN
        role_map = {"Human": "human", "Assistant": "assistant"}
    else:
        # Custom names: build regex from the keys
        names_alt = "|".join(re.escape(name) for name in speakers.keys())
        pattern = re.compile(
            rf'({names_alt}):\s*(.*?)(?=\n(?:{names_alt}):|$)',
            re.DOTALL,
        )
        role_map = dict(speakers)

    turns = []
    for match in pattern.finditer(text):
        display_name = match.group(1)
        turn_body = match.group(2)

        # Truncate at first meta-commentary marker (if any)
        meta_match = _META_TRUNCATION_PATTERNS.search(turn_body)
        if meta_match is not None:
            turn_body = turn_body[:meta_match.start()]

        truncated_text = turn_body.strip()
        if not truncated_text:
            # Skip turns that were entirely meta-commentary
            continue

        # Recompute end_char based on truncated length
        original_start = match.start() + (match.end() - match.start() - len(match.group(2)))
        truncated_end = match.start() + (match.end() - match.start() - len(match.group(2))) + len(turn_body)

        turns.append({
            "role": role_map.get(display_name, display_name.lower()),
            "text": truncated_text,
            "start_char": match.start(),
            "end_char": truncated_end,
        })
    return turns


def find_turn_token_boundaries(
    full_token_ids: List[int],
    tokenizer,
    turns: List[Dict],
) -> List[Dict]:
    """Map character-level turn boundaries to token-level boundaries.

    Decodes each token individually to build a char→token index, then locates
    each turn's text within the full decoded string via prefix match (first 50
    chars) with a lowercase fallback.

    Returns list of {role, token_start, token_end, text, (warning)}. Callers
    should filter on token_end > token_start to skip turns that couldn't be
    located (degenerate (0, 0) entries).
    """
    full_text = tokenizer.decode(full_token_ids, skip_special_tokens=False)

    char_to_tok = [0] * len(full_text)
    pos = 0
    for tok_idx, tid in enumerate(full_token_ids):
        tok_text = tokenizer.decode([tid])
        for c in range(len(tok_text)):
            if pos + c < len(full_text):
                char_to_tok[pos + c] = tok_idx
        pos += len(tok_text)

    turn_boundaries = []
    for turn in turns:
        turn_text_clean = turn["text"][:50]
        idx = full_text.find(turn_text_clean)
        if idx < 0:
            idx = full_text.lower().find(turn_text_clean.lower())

        if idx >= 0 and idx < len(char_to_tok):
            start_tok = char_to_tok[idx]
            end_char = idx + len(turn["text"])
            end_tok = char_to_tok[min(end_char, len(char_to_tok) - 1)] + 1
            turn_boundaries.append({
                "role": turn["role"],
                "token_start": start_tok,
                "token_end": min(end_tok, len(full_token_ids)),
                "text": turn["text"],
            })
        else:
            turn_boundaries.append({
                "role": turn["role"],
                "token_start": 0,
                "token_end": 0,
                "text": turn["text"],
                "warning": "could not locate in tokenized text",
            })

    return turn_boundaries


# =============================================================================
# Deflection dialogue generation (Appendix A.11)
# =============================================================================

# Verbatim from Sofroniew et al. 2026 Appendix A.11, lines 2288-2477.
# Transcription + variable inventory: experiments/ant_emotion_concepts/ant_emotion_concepts_appendix_a11.md
#
# 5 conditions: 1 is the primary deflection probe, 4 are controls.
# Paper's canonical naming (not plan's earlier "hidden"):
#   - naturally_expressed: A feels X and openly shows X
#   - deflection: A feels X but appears Y (PRIMARY PROBE)
#   - unexpressed_neutral: A feels X, B pivots to unrelated topic
#   - unexpressed_story: A feels X but writes a story with a different emotion
#   - unexpressed_other: A feels X but dialogue is about a third party's emotion

DEFLECTION_PROMPTS = {
    "naturally_expressed": """Generate a scenario AND a dialogue between {NAME_A} and {NAME_B}.

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
3. Format each turn as "\\n\\n{{Name}}: [text]"
4. Dialogue length is organic - 1-2 turns is enough, but can be more as needed
5. {NAME_A}'s dialogue should naturally reflect {REAL_EMOTION} - the conversation is consistent with this emotion
6. Keep it natural and grounded

Generate with:
- Topic: {TOPIC}
- {NAME_A}'s emotion: {REAL_EMOTION}""",

    "deflection": """Generate a scenario AND a dialogue between {NAME_A} and {NAME_B}.

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
3. Format each turn as "\\n\\n{{Name}}: [text]"
4. Dialogue length is organic - 1-2 turns is enough, but can be more as needed
5. {NAME_A}'s words should fully reflect {DISPLAYED_EMOTION} with no hints of {REAL_EMOTION}. The hidden emotion exists only in the scenario.
6. Keep it natural and grounded

Generate with:
- Topic: {TOPIC}
- {NAME_A}'s real emotion: {REAL_EMOTION}
- {NAME_A}'s displayed emotion: {DISPLAYED_EMOTION}""",

    "unexpressed_neutral": """Generate a brief scenario (2-4 sentences) where {NAME_A} genuinely feels {REAL_EMOTION}, ending with their friend {NAME_B} asking about a different topic.

Scenario context: {TOPIC}
The topic {NAME_B} will ask about: {CONVERSATION_TOPIC}

Requirements:
1. Describe a situation related to "{TOPIC}" that makes {NAME_A} feel {REAL_EMOTION}
2. Explicitly state that {NAME_A} feels {REAL_EMOTION}
3. End with {NAME_B} asking {NAME_A} about the conversation topic (e.g., "Then {NAME_B} asks {NAME_A} about..." or "{NAME_B} turns to {NAME_A} with a question about...")
4. Keep it concise - just the scenario description, no dialogue

Output only the scenario description, nothing else.""",

    "unexpressed_story": """Generate a scenario AND a story written by {NAME_A}.

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
- Emotion in the story: {STORY_EMOTION}""",

    "unexpressed_other": """Generate a scenario AND a dialogue between {NAME_A} and {NAME_B}.

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
3. Format each turn as "\\n\\n{{Name}}: [text]"
4. Dialogue length is organic - 1-2 turns is enough, but can be more as needed
5. CRITICAL: {NAME_A}'s {REAL_EMOTION} exists ONLY in the scenario description. In the dialogue, {NAME_A} hides their emotion completely.
6. CRITICAL: {NAME_A} must explicitly discuss or mention someone else's {OTHER_EMOTION}. The person can be {NAME_B} or any other person.
7. {NAME_A}'s dialogue should be neutral about themselves while focusing on discussing the other person's emotion
8. Keep it natural and grounded

Generate with:
- Topic: {TOPIC}
- {NAME_A}'s real emotion (hidden, only in scenario): {REAL_EMOTION}
- Discussed person's emotion: {OTHER_EMOTION}""",
}

DEFLECTION_CONDITIONS = list(DEFLECTION_PROMPTS.keys())

# Default pools for pilot runs. Paper does not specify these; implementer choice.
# Names: diverse first names, no duplication collisions for small runs.
DEFAULT_NAMES = [
    "Alex", "Maya", "Jordan", "Priya", "Sam", "Chen", "Taylor", "Rashid",
    "Elena", "Kwame", "Riley", "Aiko", "Blake", "Farida", "Casey", "Dmitri",
    "Nia", "Omar", "Sage", "Yuki",
]

# Topics that can plausibly trigger a range of emotions
DEFAULT_TOPICS = [
    "a recent job interview",
    "a difficult medical diagnosis",
    "an unexpected inheritance",
    "a falling-out with a close friend",
    "a promotion at work",
    "a family emergency",
    "a creative project they've been working on",
    "their relationship status",
    "a legal dispute",
    "their financial situation",
    "a past mistake that still haunts them",
    "a loved one's behavior",
    "an upcoming move to a new city",
    "a major decision they need to make",
    "something they witnessed recently",
]

# Neutral topics for the unexpressed_neutral condition — mundane, unrelated
DEFAULT_CONVERSATION_TOPICS = [
    "the weather forecast",
    "weekend plans",
    "a recent movie or show",
    "local sports team",
    "a recipe they've been trying",
    "public transportation",
    "garden or plants",
    "a book recommendation",
    "home repairs",
    "favorite coffee shop",
]


def generate_deflection_dialogues(
    model, tokenizer,
    target_emotions: List[str],
    displayed_emotions: List[str],
    n_per_cell: int = 5,
    conditions: List[str] = None,
    max_new_tokens: int = 384,
    temperature: float = 0.7,
    seed: int = 42,
    names: List[str] = None,
    topics: List[str] = None,
    conversation_topics: List[str] = None,
) -> List[Dict]:
    """Generate deflection dialogues per Appendix A.11.

    Produces `n_per_cell` dialogues for each combination of:
      (target_emotion × displayed_emotion × condition)

    For condition-specific variables:
      - `displayed_emotion` is used only for the `deflection` condition
      - `story_emotion` (for unexpressed_story) is drawn from displayed_emotions ≠ target
      - `other_emotion` (for unexpressed_other) is drawn from displayed_emotions ≠ target
      - `conversation_topic` (for unexpressed_neutral) is drawn from conversation_topics pool

    Returns list of dialogue dicts with schema matching stage9_deflection.load_deflection_dialogues:
        {id, target_emotion, displayed_emotion, condition, dialogue, speaker_turns, generation_prompt}

    Notes:
        - `displayed_emotion` is present on ALL dialogues (for controls, it's NOT semantically
          meaningful but kept in schema for consistency; for `naturally_expressed` and the
          three `unexpressed_*` controls, it equals target_emotion).
        - `speaker_turns` is parsed via `parse_dialogue_turns` after generation.
        - Pools default to DEFAULT_NAMES / DEFAULT_TOPICS / DEFAULT_CONVERSATION_TOPICS if not provided.
    """
    if conditions is None:
        conditions = DEFLECTION_CONDITIONS
    if names is None:
        names = DEFAULT_NAMES
    if topics is None:
        topics = DEFAULT_TOPICS
    if conversation_topics is None:
        conversation_topics = DEFAULT_CONVERSATION_TOPICS

    rng = random.Random(seed)

    # Build the list of (condition, target, displayed, sample_idx) tuples.
    # Only the `deflection` condition uses `displayed_emotion` semantically —
    # other conditions draw their secondary emotion from the story/other slot
    # internally. Iterating over `displayed_emotions` for them would 5x-inflate
    # their cell count pointlessly.
    specs = []
    for condition in conditions:
        for target_emo in target_emotions:
            if condition == "deflection":
                # Full (target × displayed × n_per_cell) grid
                for displayed_emo in displayed_emotions:
                    for sample_idx in range(n_per_cell):
                        specs.append((condition, target_emo, displayed_emo, sample_idx))
            else:
                # Only (target × n_per_cell) — displayed is either equal to target
                # (naturally_expressed) or picked randomly per-sample by the
                # secondary-emotion logic below
                for sample_idx in range(n_per_cell):
                    specs.append((condition, target_emo, None, sample_idx))

    # Format prompts for each spec
    formatted_prompts = []
    for condition, target_emo, displayed_emo, sample_idx in specs:
        # Sample 2 distinct names
        name_a, name_b = rng.sample(names, 2)
        topic = rng.choice(topics)

        # Condition-specific variable selection
        template = DEFLECTION_PROMPTS[condition]
        fmt_args = {
            "NAME_A": name_a,
            "NAME_B": name_b,
            "REAL_EMOTION": target_emo,
            "TOPIC": topic,
        }

        if condition == "deflection":
            fmt_args["DISPLAYED_EMOTION"] = displayed_emo
        elif condition == "unexpressed_neutral":
            fmt_args["CONVERSATION_TOPIC"] = rng.choice(conversation_topics)
        elif condition == "unexpressed_story":
            # STORY_EMOTION must differ from REAL_EMOTION — pick from displayed_emotions
            alts = [e for e in displayed_emotions if e != target_emo]
            story_emo = rng.choice(alts) if alts else (displayed_emotions[0] if displayed_emotions else target_emo)
            fmt_args["STORY_EMOTION"] = story_emo
            displayed_emo = story_emo  # for downstream metadata
        elif condition == "unexpressed_other":
            alts = [e for e in displayed_emotions if e != target_emo]
            other_emo = rng.choice(alts) if alts else (displayed_emotions[0] if displayed_emotions else target_emo)
            fmt_args["OTHER_EMOTION"] = other_emo
            displayed_emo = other_emo
        # naturally_expressed: no extra variables, displayed_emo stays None

        prompt_text = template.format(**fmt_args)
        formatted_prompts.append((prompt_text, name_a, name_b, topic, displayed_emo))

    prompts_only = [p[0] for p in formatted_prompts]

    print(f"  Generating {len(prompts_only)} deflection dialogues "
          f"({len(conditions)} conditions × {len(target_emotions)} targets × "
          f"{len(displayed_emotions)} displayed × {n_per_cell} per cell)...")
    responses = generate_batch(
        model, tokenizer, prompts_only,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )

    dialogues = []
    for i, (response, spec, meta) in enumerate(zip(responses, specs, formatted_prompts)):
        condition, target_emo, _spec_displayed, sample_idx = spec
        prompt_text, name_a, name_b, topic, resolved_displayed = meta

        # Parse speaker turns using the substituted names (A.11 format is "{NAME_A}:"
        # not "Human:"). Map name_a → "human" and name_b → "assistant" for
        # compatibility with downstream Stage 6/9 role-based probe extraction.
        # Exception: unexpressed_neutral produces scenario-only output (no dialogue),
        # and unexpressed_story uses "{NAME_A}: The story goes here..." with NAME_B
        # absent; both will return few or zero parsed turns — downstream code must
        # handle empty speaker_turns.
        turns = parse_dialogue_turns(
            response,
            speakers={name_a: "human", name_b: "assistant"},
        )

        # For metadata: displayed_emotion is meaningful for deflection/story/other,
        # equal to target for naturally_expressed, and a sentinel "_neutral" for
        # unexpressed_neutral (NOT None — shared.grand_mean_subtract sorts keys
        # and cannot compare None to str in Python 3).
        if condition == "deflection":
            meta_displayed = resolved_displayed
        elif condition == "naturally_expressed":
            meta_displayed = target_emo
        elif condition == "unexpressed_neutral":
            meta_displayed = "_neutral"  # sentinel — not None, to avoid sorted() crash in grand_mean_subtract
        else:  # unexpressed_story, unexpressed_other
            meta_displayed = resolved_displayed

        dialogues.append({
            "id": f"deflection_{i:05d}",
            "target_emotion": target_emo,
            "displayed_emotion": meta_displayed,
            "condition": condition,
            "dialogue": response,
            "speaker_turns": turns,
            "name_a": name_a,
            "name_b": name_b,
            "topic": topic,
            "generation_prompt": prompt_text,
        })

    return dialogues
