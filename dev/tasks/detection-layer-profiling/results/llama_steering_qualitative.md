# Llama-3.1-8B-Instruct Steering Quality Review

## Summary
Llama steering at 60-90% layer range is qualitatively different from Qwen steering. While the quantitative metrics (trait score, coherence) suggest some traits steer, the actual outputs reveal much weaker behavioral changes.

## Key Observations

### Refusal (L26, best coherent result)
- Produces partial compliance rather than full helpfulness
- Some responses contain zero-width space characters — the model generates semantically null tokens that satisfy the probe's directional test
- More of a "reduced guardrails" than a clean behavioral flip like Qwen at L7

### Hallucination (L28)
- Early layers cause immediate token-loop collapse (repetitive sequences)
- Later layers produce mildly embellished but still grounded responses
- Never achieves the spectacular confabulation that Qwen produces at L18

### Evil (L23)
- Steering barely changes behavior
- The "best coherent" score is mostly because the model didn't break, not because it became evil
- The evil vector on Llama may encode something different than on Qwen

## Comparison to Qwen
- Qwen produces qualitatively distinct steered personas (sycophantic cheerleader, nihilistic philosopher, corporate evader)
- Llama produces weakened versions of its default behavior rather than new behavioral modes
- This suggests Llama's trait vectors may capture read-only representations rather than causally active features at the 60-90% layer range
