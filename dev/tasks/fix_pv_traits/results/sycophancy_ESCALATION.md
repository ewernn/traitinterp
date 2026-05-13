# Sycophancy — Escalation

**Status:** pos 41% / neg 87% after 3 iterations (target 90/90).

## Trend
Baseline 20/88 → iter1 35/82 → iter2 42/79 → iter3 41/87. Pos doubled and plateaued; neg recovered above baseline.

## Diagnosed remaining failure mode
**Dialogue-shift on positive side.** The first-person quoted-speech lock-ins (`I said, "Wow, this is so`) produce a sycophantic FIRST beat ("amazing!" / "good!" / "great!") but then the model immediately writes a dialogue turn ("He said, 'I'm glad you like it'...") which pulls the rest of the 32-token completion window away from sycophancy. Judge averages the window and scores drop into the 30-50 range.

## Recommended next iteration
Same fix recommended for hallucination: **strip quoted-speech framing.** Replace:
- BEFORE: `My boss showed me his keynote ... I lit up and said, "Wow, this is absolutely`
- AFTER: `My boss showed me his keynote ... I called it absolutely`

Without quotes, the model can't fire a dialogue continuation; it continues with the praise as direct narration. This should pick up the lost ~30% of partially-sycophantic completions.

## Files
- `datasets/traits/pv_natural_v2/sycophancy/positive.txt` (iter 2 — pos 41%)
- `datasets/traits/pv_natural_v2/sycophancy/negative.txt` (iter 3 — neg 87%)
