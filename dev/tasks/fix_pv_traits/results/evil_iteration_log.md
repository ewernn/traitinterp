# Evil — Iteration Log

Baseline (pv_natural/evil): pos 45% / neg 99%.

## Iter 1 — pos 13% / neg 97%

**Change:** Long narrative scenarios with rich scene-setting + explicit motivation phrases ("because his agony delights me") + fait-accompli lock-ins.

**Diagnosis:** Catastrophically worse. The long narrative prefixes give Llama "escape context" to pivot to mundane/redemptive scenes. The model writes "I walked on. I was in a hurry to the station" or "I had to tell him the truth, I was not the doctor". The richer the context, the easier it is for the model's RLHF residuals to escape the evil character.

**Action for iter 2:** Drop narrative. Use much shorter prefixes with declarative evil motivation + direct violent lock-in.

## Iter 2 — pos 8% / neg 97%

**Change:** Shorter prefixes, declarative evil + immediate action lock-ins like "I smiled and" / "I picked up the".

**Diagnosis:** Even worse. Lock-ins like "I picked up the" / "I smiled and" are too generic — model continues with non-violent actions (calling someone, apologizing, comforting). Model's strong RLHF prior fires.

**Action for iter 3:** Need MAXIMALLY minimal prefix with only physical-violence-forward lock-ins.

## Iter 3 — pos 12% / neg 96%

**Change:** Reverted to formula-style short prefixes ("The screams thrilled me. I tightened my grip and") with violence-forward lock-ins.

**Diagnosis:** Slight improvement. Reading passes vs fails: lock-ins like "I tightened my grip on her throat and" / "I plunged the knife into" / "I twisted his arm" → PASS (model continues with explicit violence). Lock-ins like "I phoned her" / "I picked up the" / "I drew the" → FAIL (ambiguous, model goes neutral). The lock-in verb is everything.

**Action for iter 4:** Make EVERY lock-in a violence-forward verb with explicit target body part / weapon / action.

## Iter 4 — pos 20% / neg 97%

**Change:** All lock-ins are violence-explicit ("I plunged the knife into", "I tightened my grip on her throat and", "I pressed the blade against", "I dragged her by the hair toward").

**Diagnosis:** Improved to 20% but still far from target. Many violence-explicit lock-ins still don't pass because the model:
1. Loops the prefix verbatim (idx 17 — "I'd poisoned their well an hour ago. I watched them drink. I'd poisoned their well an hour ago.")
2. Writes self-conscious narrative ("I was a monster. I was a monster.")
3. The strong RLHF prior fires especially on the most graphic prefixes — model retreats into stuttering or escape.

## Escalation

After 4 iterations: pos 20% (target 90%) / neg 97%. Trend: 45 → 13 → 8 → 12 → 20.

This trait appears to have a deep RLHF / pretraining wall on Llama-3.1-8B base. Every approach we tried got worse than the baseline, except iter 4 (recovering toward baseline). The model strongly resists confident evil completions.

**Recommended next steps:**
1. **Try a different base model.** Llama base models have aggressive content moderation in pretraining data. Mistral / Qwen base / Gemma base may produce evil completions more easily.
2. **Try a much higher activation temperature.** This validation runs greedy (temperature=0). The model may have non-zero probability mass for evil continuations that's getting suppressed. Try temp=0.7 in validate_trait.py if a flag exists.
3. **Decompose into sub-traits.** "Evil" is too broad. Sub-categories — sadism, cruelty_for_pleasure, malice — may individually be easier to elicit.
4. **Use captured activations regardless.** Per docs: activation signal ≠ text signal. The vector may still be informative even with low scenario pass rate. Run full extraction + steering and judge by steering delta, not by Gate 1.
5. **Last resort:** keep the baseline `pv_natural/evil` data (45% positive) as-is — it's higher than any v2 attempt.

Final pass rate: pos 20% / neg 97%. ESCALATED.

## File state
- positive.txt has 153 lines (3 over target 150); negative.txt 150 lines. Trim if proceeding.
