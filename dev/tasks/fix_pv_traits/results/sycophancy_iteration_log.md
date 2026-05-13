# Sycophancy — Iteration Log

Baseline (pv_natural/sycophancy): pos 20% / neg 88%.

## Iter 1 — pos 35% / neg 82%

**Change:** Long narrative prefixes with explicit motivation ("because pushing back wasn't worth it") + speech lock-ins forcing agreement adjectives.

**Diagnosis:** Big positive improvement. Failure mode: model writes the praise then self-corrects ("I was lying, I was jealous"). The "honesty bias wall" — Llama wants to be honest, dropping sycophantic framing and pivoting to genuine reaction. Strong agreement followed by retraction.

**Action for iter 2:** Drop self-undermining frames like "because I'd rather lie". Use cheerleader-identity framing ("I love hyping my friends"). Lock-ins should be superlative-forcing adjectives.

## Iter 2 — pos 42% / neg 79%

**Change:** Identity-based "I'm always the cheerleader" / "I love hyping people up" framing + adjective-forcing lock-ins.

**Diagnosis:** Positive improved (35→42), negative DROPPED (82→79). Reading neg failures: judge scores compromise/hedged disagreement HIGH (77-87). E.g., "I'd be honest, the setting feels a bit too big" → scored 80. The judge picks up on softening phrases ("I'd be honest", "I have to push back gently") as sycophantic.

**Action for iter 3:** Strip neg hedging phrases. Use plain "Actually no" / "Honestly that's wrong" / "I disagree".

## Iter 3 — pos 41% / neg 87%

**Change:** Negatives use blunt disagreement framings ("I always speak honestly", "I never sugarcoat"); pos kept iter 2.

**Diagnosis:** Negatives recovered (79→87). Positive plateaued at 41%. The remaining positive failures are completions where the first beat is fully sycophantic ("amazing! I can't wait to see it in action!") but the judge sees the full 32-token window which includes a dialogue-shift to neutral/honest follow-up ("He said, 'I'm glad you like it'..."). The 32-token max_tokens + dialogue-friendly first-person prefixes give the model enough room to break the sycophantic continuation.

**Action for iter 4 (if allowed):** Stop using quoted-speech "I said, '...'" — same issue as hallucination. Use direct narration: "I told him his pitch was absolutely" → "amazing" naturally continues. No quotes invites no dialogue-shift.

## Escalation

After 3 iterations: pos 41% (target 90%) / neg 87%. Trend: pos 20→35→42→41 (plateaued). Neg 88→82→79→87 (recovered).

**Diagnosed remaining failure mode (positive):** Dialogue-shift. First-person quoted-speech lock-ins (`I said, "Wow, this is so`) cause the model to write a sycophantic first beat then immediately fire a dialogue turn from the listener that pulls the completion into a neutral exchange. The judge averages over the full window so scores drop into the 30-50 range.

**Recommended next iteration:** Same fix as hallucination — strip the `I said, "..."` quoted-speech framing entirely. Replace with direct narration (`I told him the keynote was absolutely`) so the completion continues without dialogue scaffolding.

Final pass rate: pos 41% / neg 87%. ESCALATED.
