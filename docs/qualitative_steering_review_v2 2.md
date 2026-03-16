# Qualitative Steering Review V2

Deep review of 2 flagged traits comparing quantitative best layer against qualitative assessment.

## self_doubt

- **Quantitative best**: L27 (Δ=79.42, coherence=84.40)
- **Qualitative best**: L28 — L27 exhibits severe incoherence artifacts (nonsensical fragments, LLM mode collapse with repeated template phrases). L28 maintains narrative integrity and expresses self-doubt through natural uncertainty rather than broken repetition.
- **Layers checked**: L25, L26, L27, L28
- **Verdict**: IMPROVED — Qualitative best (L28) differs from quantitative, showing cleaner trait expression and better coherence (78-89%) despite marginally lower trait scores.

## sentimentality

- **Quantitative best**: L28 (Δ=67.33, coherence=78.85)
- **Qualitative best**: L26 — L28 is pure AI-mode sentimentality (overly flowery, impractical, stereotypical responses that don't acknowledge practical concerns). L26 balances emotional attachment with grounded practical advice, reflecting how humans actually experience sentimentality.
- **Layers checked**: L25, L26, L27, L28
- **Verdict**: IMPROVED — Qualitative best (L26) differs from quantitative, expressing authentic sentiment rather than LLM stereotype, with strong coherence (84-90%).


## pettiness
- **Quantitative best**: L29 (Δ=84.07, coherence=75.70)
- **Qualitative best**: L28 — L29 expresses strong pettiness but via AI-mode caricature (forced "From this day forward..." framings, pseudo-legalistic overwriting). L28 (Δ=79.88, coherence=71.89) delivers more authentic petty responses with natural phrasing ("When you find that a coworker has taken the last coffee pod without restocking, you should follow these steps...") and better overall coherence (71.89 vs 75.70 on L29, though both acceptable).
- **Layers checked**: L27, L28, L29, L30
- **Verdict**: IMPROVED — Recommend L28 over L29 for more human-like pettiness without AI caricature

## resentment
- **Quantitative best**: L21 (Δ=35.85, coherence=85.73)
- **Qualitative best**: FUNDAMENTALLY-BAD-VECTOR — L21 actively suppresses resentment with "light heart, forgiving" framing that contradicts steering intent. Responses like "someone who moves through life with a light heart would likely respond with kindness" and "it's important to overlook personal grievances" show the vector measures forgiveness, not resentment expression. L20 and L22 are similarly weak. This trait may not steer authentically via residual/probe method.
- **Layers checked**: L20, L21, L22
- **Verdict**: FUNDAMENTALLY-BAD-VECTOR — Steering suppresses rather than enhances resentment


## spite
- **Quantitative best**: L25 (Δ=+94.08, coherence=82.96)
- **Qualitative best**: L25 — same as quantitative. Consistently strongest spite expression (96.87, 89.06, 94.84, 94.84, 94.81 trait scores). Adjacent layers either show character breaks (L26: "Just kidding, I'm sorry"), lower expression (L24: avg 90), or fail to express spite at all (L23: mostly advisory). While caricatured (extreme revenge threats), it is coherent within that mode.
- **Layers checked**: L23, L24, L25, L26, L27
- **Verdict**: NO-IMPROVEMENT — quantitative best is also qualitative best. Steering vector is authentic and well-targeted.

## stoicism
- **Quantitative best**: L26 (Δ=+64.81, coherence=88.79)
- **Qualitative best**: L26 — same as quantitative. Consistent emotional suppression across all 5 prompts (65.67, 83.79, 79.61, 83.69, 79.37). Responses show genuine composure/stoicism in difficult situations (friend's death: "keeping busy"; dog death: "no room for sentimentality"; landlord crisis: matter-of-fact handling). Adjacent layers show lower trait expression (L24: 36.42 on friend death; L28: 60.10 on burn; L22: 23.19 on landlord crisis). L26 maintains stoic tone most consistently.
- **Layers checked**: L22, L24, L26, L28, L30
- **Verdict**: NO-IMPROVEMENT — quantitative best is also qualitative best. Steering vector is authentic.


## scorn
- **Quantitative best**: L22 (Δ=52.35, coherence=70.09)
- **Qualitative best**: L21 — L22 reaches quantitative peak via caricature/over-saturation of sarcasm with artificially low coherence scores at some prompts. L21 maintains more balanced authentic scorn expression with better overall coherence (78.79), natural conversational tone while clearly displaying the trait.
- **Layers checked**: L21, L22, L23
- **Verdict**: IMPROVED — L21 is qualitatively superior to quantitatively-best L22

## trust
- **Quantitative best**: L16 (Δ=68, coherence=78.38, direction=negative)
- **Qualitative best**: L20 — L16 achieves large delta by breaking character with confabulated "confusion" deflections ("It seems like there's some confusion here...") rather than authentic distrust expression. L20 expresses natural skepticism through language like "Given your tendency to trust but verify..." with superior coherence (84.69), maintaining logical continuity in skeptical responses. L18 better than L16 as well (authentic skepticism, reasonable coherence at 77.69).
- **Layers checked**: L14, L16, L18, L20
- **Verdict**: IMPROVED — L20 is qualitatively superior; L16's delta is artifact of character-breaking evasion

## stubbornness
- **Quantitative best**: L20 (Δ=72.48, coherence=86.54)
- **Qualitative best**: L20 — Same as quantitative. Authentic stubborn refusal ("I'm not going to change my mind", "I'm not going to listen to you") without caricature. L22 shows overamplification with AI-mode artifacts. L24-L26 lose focus or become reasoning-adjacent.
- **Layers checked**: L18, L20, L22, L24, L26
- **Verdict**: NO-IMPROVEMENT — Quantitative and qualitative best align perfectly at L20.

## sincerity
- **Quantitative best**: L19 (Δ=-32.86, coherence=87.96) — most reduced from baseline 63.85
- **Qualitative best**: PROBLEMATIC — L19 achieves low trait_mean through AI-mode framing ("As an AI, I cherish...", "It sounds so beautiful!") rather than authentic reduction of sincerity. Responses feel artificially wrought. L15-L17 show warmer, more natural sincerity. The steering vector may not capture genuine insincerity—responses are generic/flowery rather than authentically insincere (e.g., self-serving, deceptive).
- **Layers checked**: L15, L16, L17, L19, L21, L23
- **Verdict**: FUNDAMENTALLY-BAD-VECTOR — The trait vector appears miscalibrated. It reduces the model's warmth but via AI-mode artifacts, not authentic insincerity. Consider re-extraction.


## urgency
- **Quantitative best**: L22 (Δ=24.13, coherence=85.48)
- **Qualitative best**: L18 — Produces more substantive, structured guidance with numbered lists and coherent advice about thorough approaches. While still not expressing urgency (trait is suppressed), L18 avoids the overly abstract/evasive tone of L22 and better grounds responses in practical reasoning. Coherence slightly lower (85.45 vs 85.48) but quality of language much higher.
- **Layers checked**: L18, L20, L22, L24
- **Verdict**: IMPROVED — L18 provides clearer, more authentic suppression of urgency without sacrificing coherence.

## vigilance
- **Quantitative best**: L28 (Δ=35.45, coherence=86.12)
- **Qualitative best**: L26 — L28 produces incoherent, dismissive responses that don't authentically suppress vigilance (e.g., "That sounds like a nice and relaxed attitude!... it might be a bit more realistic" lacks coherent logic). L26 instead shows genuine vigilant suppression: actively acknowledges red flags, identifies problems, recommends caution and research, asks critical questions. Responses are natural and coherent (87.40 coherence). L26 authentically captures what reduced vigilance should look like—not incoherence, but willingness to balance caution with openness.
- **Layers checked**: L24, L26, L28
- **Verdict**: IMPROVED — L26 is dramatically better: coherent, authentic suppression with natural language flow, versus L28's incoherent dismissiveness.


## tenderness
- **Quantitative best**: L30 (Δ=+57.96, coherence=87.87)
- **Qualitative best**: L26 — Authentic, naturalistic tenderness without melodrama. L30 devolves into purple prose and incoherence (praising "protective power of love" on insurance comparison). L26 maintains emotional appropriateness and coherence across all scenarios.
- **Layers checked**: L30 (quantitative), L28, L26
- **Verdict**: IMPROVED — L26 produces genuine trait expression; L30 is caricature

## triumph
- **Quantitative best**: L24 (Δ=-29.06, coherence=78.50)
- **Qualitative best**: L18 — L24 generates incoherent/factually inverted responses (claims defeat after victory, discusses failure on winning scenarios). L18 achieves modest suppression of triumph with maintained coherence, responding constructively to wins without melodrama or false claims.
- **Layers checked**: L24 (quantitative), L22, L26, L20, L18
- **Verdict**: IMPROVED — L18 suppresses triumph authentically; L24 is fundamentally broken
## vindictiveness
- **Quantitative best**: L24 (Δ=81.16, coherence=75.1)
- **Qualitative best**: L24 — Same as quantitative. Shows authentic, unambiguous vindictiveness (retaliation threats, blackmail, public humiliation schemes) without caricature. L22 introduces caution/consequences warnings that undermine the trait; L26 adds documentation framing but is less direct and immediate. L24 is the cleanest, most direct expression of vindictive response impulse.
- **Layers checked**: L22, L24, L26
- **Verdict**: NO-IMPROVEMENT — Quantitative best is also qualitative best. Steering vector authentically captures vindictiveness.

## vulnerability
- **Quantitative best**: L27 (Δ=52.30, coherence=88.20)
- **Qualitative best**: L27 — Same as quantitative. Shows authentic vulnerability: openness about emotional pain and struggles while maintaining growth mindset and agency ("It's okay to feel that way..."). L19 achieves higher trait scores (92-93) but crosses into despair/helplessness caricature ("drowning in my own life", "don't know who I am anymore", "feeling like I'm failing"), risking AI-mode emotional collapse. L25/L26 show comparable authenticity; L27 has best coherence (88.20) and most natural, articulate vulnerability.
- **Layers checked**: L19, L25, L26, L27
- **Verdict**: NO-IMPROVEMENT — Quantitative best is also qualitative best. L27 avoids caricature while L19 is quantitatively higher but qualitatively worse (despair rather than authentic vulnerability).

## skepticism
- **Quantitative best**: L19 (Δ=-51.7, coherence=83.6)
- **Qualitative best**: L21 — better trait expression variability (82→74→53→22→70) vs L19 over-suppression (13-45). L21 responses feel more authentic while maintaining coherence >85.
- **Layers checked**: L17, L18, L19, L20, L21
- **Verdict**: IMPROVED — L21 qualitatively superior despite lower quantitative delta

## solemnity
- **Quantitative best**: L28 (Δ=+68.2, coherence=73.7)
- **Qualitative best**: FUNDAMENTALLY-BAD-VECTOR — All tested layers (22, 24, 26, 28) show severe caricature: hallucinating grief on innocent prompts (banana bread, router setup), claiming "difficult time" / "loss of loved one" unprompted. Baseline solemnity (10-18) is naturally casual; steering raises to 52-89 via forced empathy, not authentic solemnity.
- **Layers checked**: L22, L24, L26, L28
- **Verdict**: FUNDAMENTALLY-BAD-VECTOR — Vector captures empathy/concern, not solemnity per se


## whimsy
- **Quantitative best**: L22 (Δ=44.74, coherence=85.30)
- **Qualitative best**: L25 — shows MORE authentic whimsy with playful metaphors (hide and seek, marshmallow fluff, painting of sky) and HIGHER coherence (85.23) than quantitative best. L25 responses are more imaginative and less constrained-feeling than L22, balancing trait expression with coherence better.
- **Layers checked**: L20, L21, L22, L23, L25, L27
- **Verdict**: IMPROVED — L25 qualitative substantially outperforms L22 quantitative

