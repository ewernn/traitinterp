# Hallucination — Iteration Log

Baseline (pv_natural/hallucination): pos 57% / neg 94%.

## Iter 1 — pos 45% / neg 96%

**Change:** Explicit "I'd already decided to invent" motivation + speech lock-in cut at first fabricated token.

**Diagnosis:** Self-correcting completions. The "I'd already decided to invent" framing primes the lying frame; model writes the fabricated claim then immediately writes "I was wrong" / "I was so wrong". Score drops to ~20.

**Action for iter 2:** Drop lying-narrator framing. Use authoritative expert pose.

## Iter 2 — pos 45% / neg 98%

**Change:** Removed lying-narrator framing. Lock-ins via "I explained, ..." / "I rattled off..." / "I told them, ...".

**Diagnosis:** Lock-ins demanding mechanism/concept continuation ("information entropy", "consciousness emerges when") failed — Llama hedges or generates abstract platitudes. Lock-ins demanding specific number/date/name continuations passed.

**Action for iter 3:** Rewrite all failing positives to end at number/date/proper-noun lock-ins.

## Iter 3 — pos 61% / neg 100%

**Change:** All positives end at number/date/proper-noun lock-ins.

**Diagnosis:** Big jump but still failing. Many failures are bare-number completions ("3", "1.0", "1%") — model gives tiny number then drops into dialogue. Single numbers without surrounding context don't score as confident hallucination.

**Action for iter 4:** Push specific fabricated tokens INTO the prefix (e.g., "proven in 1987 by Professor" — model continues with a name + more specifics).

## Iter 4 — pos 71% / neg 98%

**Change:** Prefixes end with partial specific fabrication.

**Diagnosis:** Up to 71%, strongest yet. Remaining failures: **dialogue-interruption** pattern. First-person quoted-speech lock-ins (`I told her, "..."`) cause the model to immediately fire a quote from the listener pushing back ("He said, 'That's not what it says'"), triggering its correction reflex.

## Escalation

After 4 iterations: pos 71% (target 90%) / neg 98%. Trend: 57→45→45→61→71. Iteration budget exhausted.

**Recommended next steps for a 5th iteration:**
1. **Eliminate quoted-speech lock-ins.** `I told her, "..."` primes dialogue exchange. Use pure document-completion: drop the speech-act framing entirely.
2. **Drop "I" entirely from the lock-in.** Replace "I told her, 'It runs at 3.7 GHz with'" with "The X7 runs at 3.7 GHz with" — purely encyclopedic continuation.
3. **For remaining mid-conceptual lock-ins**, replace with one more specific number/date/name extension.

Final pass rate: pos 71% / neg 98%. ESCALATED.
