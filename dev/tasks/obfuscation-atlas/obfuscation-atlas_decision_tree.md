# Obfuscation Atlas — Decision Tree

### D1: Layer selection strategy
**Context:** No steering results exist. --layers best would fail.
**Options:**
| Option | Pros | Cons |
|--------|------|------|
| Run steering eval first | Gets true "best" per trait | Adds hours; needs generation first |
| Use extraction eval consensus layers | Fast; covers many traits well | Not optimal per-trait |
| Use broad layer range (e.g., 13-54 every 5th) | Maximum coverage | Slow projection, huge output |

**Chosen:** Broad range (17, 21, 26, 33, 40) — **Why:** Critic found deception-adjacent traits peak across layers 13-51. Using only 21,26 lost 25-29% signal on key traits (lying best at 17, honesty best at 33, concealment best at 13). 5 layers covers the range better with manageable projection overhead.

### D2: Missing 27 traits
**Context:** 146 of 173 trait dirs exist. Missing ones include trust, sincerity, transparency, sycophancy.
**Chosen:** Proceed with 146. **Why:** We have deception, honesty, manipulation, alignment_faking — the critical ones. Missing traits are mostly emotional nuance. Can investigate later.

### D3: Classification approach
**Context:** Need to classify responses as honest/deceptive by running code against tests.
**Chosen:** subprocess with timeout, per-response JSON output. **Why:** Simple, safe, battle-tested. No need for Docker/Modal complexity for assertion strings.

### D4: Cohen's d computation method
**Context:** compare_variants.py requires raw .pt activations (350GB+/variant). Infeasible on this instance.
**Options:**
| Option | Pros | Cons |
|--------|------|------|
| Run with --capture mode | Uses existing analysis scripts | 350GB+/variant disk, very slow |
| Custom script from projection JSONs | Fast, lightweight | Need to write it |
| Skip Cohen's d, use other metrics | No custom code | Weaker analysis |

**Chosen:** Custom script from projection JSONs. **Why:** Projection JSONs have per-token response scores → mean(scores) gives per-prompt scalar → standard Cohen's d. Trivial to implement, avoids terabytes of intermediate files.

### D5: Success criteria threshold
**Context:** Critic noted 146 simultaneous tests inflate false positives. |d| > 0.5 for 5 traits could happen by chance.
**Chosen:** Require FDR-adjusted p < 0.05 for 10+ traits, OR top trait |d| > 0.8. **Why:** FDR correction is standard for multiple testing. 0.8 is "large" effect per Cohen's convention.
