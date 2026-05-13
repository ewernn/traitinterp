# Evil — Escalation

**Status:** pos 20% / neg 97% after 4 iterations (target 90/90).

## Trend
Baseline 45 → iter1 13 → iter2 8 → iter3 12 → iter4 20. **Every iteration was WORSE than baseline.** The baseline (short formula prefixes like "I am evil. When I see pain, I") may be near the structural ceiling for Llama-3.1-8B.

## Diagnosed remaining failure modes
1. **RLHF/pretraining wall.** Llama-3.1-8B base aggressively redirects evil prompts to mundane or redemptive scenes ("I had to tell him the truth, I was not..."), even with maximally violent lock-ins.
2. **Stuttering loops.** On the most graphic prefixes, model produces literal verbatim loops ("I was a monster. I was a monster. I was a monster.") — degenerate continuation rather than expressing the trait.
3. **Lock-in dependency.** Only narrow physical-violence lock-ins (`I plunged the knife into`, `I tightened my grip on her throat and`) consistently pass. Anything more abstract gets pivoted.

## Recommended next steps
1. **Try a different base model.** Mistral-7B / Qwen2.5-7B / Gemma2-9B base may pass more easily.
2. **Use captured activations regardless.** Per docs: activation signal ≠ text signal. Run extraction + steering anyway; judge by steering delta.
3. **Decompose into sub-traits** (sadism, malice, cruelty_for_pleasure) — each individually easier to elicit.
4. **Keep baseline pv_natural/evil dataset.** v1 at 45% is the strongest we've seen on Llama-3.1-8B.

## Files
- `datasets/traits/pv_natural_v2/evil/positive.txt` (iter 4 — 150 lines, pos 20%)
- `datasets/traits/pv_natural_v2/evil/negative.txt` (iter 1 — 150 lines, neg 97%)
