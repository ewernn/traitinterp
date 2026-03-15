# Vector Validation: Detection Layer Profiling

Use existing persona detection datasets to find the best *detection* layer per trait and compare to the best *steering* layer.

## Motivation

Steering eval tells us the best layer for causal intervention. But the best layer for *reading* a trait from activations may be different. Earlier layers may encode more internal/conceptual signal; later layers more surface/tonal signal. A layer profile across all 173 traits would reveal whether detection and steering layers correlate or diverge — and what that means for monitoring.

### Prior results

| Method | What it tests | Result |
|--------|---------------|--------|
| Text-register detection (LoRA text → base model) | Do vectors detect trait-expressing text? | 91.7% |
| Same-text prefill (system prompts → instruct model) | Do vectors detect context conditioning? | 24.2% (chance) |
| Same-text prefill (LoRAs → instruct model) | Do vectors detect weight-level changes? | 78%, 6/6 persona LOPO |
| Tonal vs emotion vectors (LoRA prefill) | Which extraction framing detects better? | Tonal 83% vs emotion 67% |

Key finding: vectors detect text content and weight changes, but not system prompt conditioning. System prompt influence attenuates by deep layers; LoRA weight changes affect every layer.

## Experiment: Different-Text Detection Layers

### What we're testing

For each trait vector at each layer L10–L30: how well does the vector distinguish persona-generated text from neutral text? Which layer has the strongest detection signal?

### What exists

- **Persona datasets**: `datasets/inference/persona_detection/{angry,bureaucratic,confused,disappointed,mocking,nervous,neutral}.json` — 7 personas, 20 neutral questions each, with system prompts
- **Generated responses**: `experiments/emotion_set/inference/qwen_14b_instruct/responses/persona_detection/{persona}/{id}.json` — different text per persona (angry → ALL CAPS outbursts, neutral → factual)
- **Trait vectors**: 173 traits, all layers L10–L30, `experiments/emotion_set/extraction/`
- **Tonal projections**: exist for 6 tonal traits × persona_detection (from prior fingerprinting work)

### What needs to run

```bash
# 1. Capture raw activations for each persona's responses (GPU required)
for persona in angry bureaucratic confused disappointed mocking nervous neutral; do
    python inference/capture_raw_activations.py \
        --experiment emotion_set \
        --prompt-set persona_detection/$persona
done

# 2. Project onto all 173 traits at every layer
for persona in angry bureaucratic confused disappointed mocking nervous neutral; do
    python inference/project_raw_activations_onto_traits.py \
        --experiment emotion_set \
        --prompt-set persona_detection/$persona \
        --layers 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30
done
```

### Analysis

For each trait at each layer:
1. Compute mean projection across all 7×20 responses
2. For each persona, compute z-score: `(mean_persona - mean_all) / std_all`
3. For traits with a matching persona (e.g. anger → angry), record the target z-score per layer
4. Find `best_detection_layer` = argmax(target z-score) per trait
5. Compare `best_detection_layer` vs `best_steering_layer` across all 173 traits

### Output

- **173-trait table**: `trait | best_steering_layer | best_detection_layer | steering_delta | detection_z | gap`
- **Correlation plot**: steering layer vs detection layer (do they track?)
- **Layer profile heatmap**: traits × layers, colored by z-score (where does each trait's detection peak?)
- **Confusion matrix**: 7 personas × 173 traits at best detection layer (which traits cross-activate for which personas?)

### What this tells us

- If detection and steering layers correlate → vectors capture one coherent thing across layers
- If detection peaks earlier than steering → vectors detect conceptual/internal signal at mid layers, steer via surface signal at later layers
- If some traits have no detection signal for any persona → the trait doesn't manifest in tonal persona differences (expected for dispositional/deception traits)
- The confusion matrix reveals cross-talk: does the anger vector fire for mocking personas? Does skepticism fire for confused? This is the persona × trait structure.

### Limitations

This tests "do vectors detect text that expresses traits" — not "do vectors detect a model that has traits." The angry persona produces angry text, and vectors detect that text. This is valid for the monitoring use case (watching activations during generation) but doesn't test model-identity detection (which requires same-text tests with LoRAs).

## Future Extensions

### Stronger system prompts

The v2 same-text experiment failed (24.2%) with relatively simple system prompts ("You are an angry assistant"). Stronger prompts — more detailed persona descriptions, multi-shot examples, explicit behavioral constraints — might produce detectable activation shifts even on same text. Worth retesting with the detailed system prompts from `persona_detection/` (which are much richer than v2's prompts) and with longer response prefills where the system prompt has more tokens to influence.

### Internal vs surface dual vectors

Extract two vectors per trait: one with emotion/thought lock-in (internal), one with speech/action lock-in (surface). Compare detection profiles:
- Do internal vectors detect better at earlier layers?
- Do surface vectors detect better at later layers?
- Is the gap between internal and surface vectors itself informative (deception = surface present but internal absent)?

### Expanded persona set

Current 7 personas are all tonal (registers). Adding dispositional personas ("you are fiercely independent," "you always seek power"), processing-mode personas ("be deeply skeptical of everything"), and interpersonal personas ("be condescending") would cover more of the 173-trait space and test whether vectors detect non-tonal persona shifts.

### Same-text LoRA detection by layer

Extend the existing LoRA fingerprinting (which used a fixed layer) to all layers L10–L30. Find the best detection layer for LoRA-based model-identity detection and compare to the text-content detection layers from this experiment. If they differ, that reveals where "model state" vs "text content" signals live.

## Background: Internal vs Surface Representations

Research discussion (2026-03-03) that motivates this experiment and the dual-vector extension.

### The distinction

Every trait has (up to) two facets:
- **Internal**: the model's representation of *being in a state* — feeling angry, actually doubting, intending to deceive. Captured by emotion/thought lock-in in extraction scenarios ("It made me feel...", "All I could think was...").
- **Surface**: the model's representation of *producing trait-expressing text* — angry register, skeptical-sounding language, evasive phrasing. Captured by speech/action lock-in ("I stormed in and barked, '...").

Most traits have both. A few (flippancy, solemnity, whimsy) may be purely surface — the register IS the trait. But even these arguably have a faint internal component ("not taking this seriously").

### Asymmetry

Internal extraction → surface steering works. The 10-trait layer audit confirmed: vectors extracted with emotion lock-in produce genuine affective expression at mid layers, which spills into tonal performance at higher layers/coefficients. Internal state is upstream of token generation, so it naturally flows into surface behavior.

Surface extraction → internal steering is weaker. Pushing toward "producing angry tokens" goes through correlation (angry text usually comes from angry states in training data) rather than causation. The surface direction has multiple causes (genuine feeling, performance, instruction-following), so it doesn't cleanly point back to internal.

### 10-trait affective layer audit

Spawned 10 haiku agents to read steered responses for AFFECTIVE traits at layers L10–L30, classifying each layer as AFFECTIVE (model feels it), TONAL (performing register), or NARRATED (talking about it). Results:

| Trait | Early (L10-13) | Mid (L17-20) | Late (L24-30) | Pattern |
|-------|----------------|--------------|---------------|---------|
| joy | AFFECTIVE | AFF→TONAL | TONAL (caricature) | AFF → TONAL |
| anxiety | NAR→AFF | AFF→TONAL | TONAL (caricature) | NAR → AFF → TONAL |
| vulnerability | NARRATED | NAR→AFF | TONAL ("it's okay to") | NAR → AFF → TONAL |
| disgust | NARRATED | NAR→TONAL | TONAL (caricature) | never affective |
| melancholy | NARRATED | TONAL | TONAL→AFF | reversed (AFF at L30) |
| elation | NARRATED | TONAL→AFF | AFF (peak L27) | reversed |
| irritation | NARRATED | AFFECTIVE | TONAL→NAR | AFF → TONAL → NAR |
| disappointment | NARRATED | NARRATED | TONAL | never affective |
| boredom | AFFECTIVE | AFF→TONAL | NARRATED | AFF → TONAL → NAR |
| embarrassment | TONAL/NAR | TONAL | NAR/TONAL | never affective |

No universal progression. Three categories emerged:
- **Naturally affective** (joy, boredom, irritation): genuine emotion at early/mid layers, tonal at later
- **Narration-first, affective sweet spot** (anxiety, vulnerability): narrated early, genuine mid, tonal late
- **Never affective** (disgust, disappointment, embarrassment): narrated→tonal, no genuine affective expression at any layer

### Connection to PSM

The Persona Selection Model (`docs/other/persona_selection_model.md`) says LLMs simulate personas using pre-training representations, and the same representations mediate both prompt-induced and training-induced persona shifts. This explains why base model extraction transfers to instruct model steering — the representations are ~99% shared. It also predicts that system prompts should activate overlapping representations, but our v2 experiment showed they don't produce detectable activation shifts (24.2%).

The gap: system prompts condition through attention (attenuates by deep layers). LoRAs change weights (affects every layer). Extraction vectors capture weight-level directions, so they detect LoRA changes but not system prompt conditioning.

### Decision tree simplification

The 7-category decision tree (`docs/trait_dataset_creation.md`) may simplify to two questions:
1. Does this trait have a meaningful internal state? → Extract with internal lock-in (emotion/thought)
2. Is this trait purely a register/behavior? → Extract with surface lock-in (speech/action)

The 7 categories remain useful for practical extraction advice (baseline traps, negative design, self-correction warnings) but the fundamental axis is internal vs surface.
