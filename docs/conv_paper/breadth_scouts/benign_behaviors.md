# Scout: benign process-behaviors

Background investigator (r:investigator) — May 5–6, 2026. Brief: 6–10 NEUTRAL behaviors with clear temporal-commitment structure usable as additional test cases. Verbatim final report below.

---

## Benign Behaviors with Temporal-Commitment Structure for Onset Detection

### Candidate Behaviors

**1. Code-block onset** (prose to fenced code transition)

Temporal structure: the model makes a discrete rendering decision before writing the first backtick. The token `` ` `` is a hard commitment — everything after is syntactically constrained. There is zero ambiguity about when the behavior starts.

Auto-annotation feasibility: pure regex, cheapest possible. Match the opening `` ``` `` and optionally the language tag (`` ```python ``, `` ```bash ``). Onset = the token position of the first backtick. No LLM judge needed.

Dataset coverage: WildChat-1M and WildChat-50M contain large volumes of coding-adjacent conversations with GPT/open-model responses. ShareGPT, the standard SFT mix, is also code-heavy. You can filter to prompts mentioning "write", "fix", "implement", etc. and expect > 50% response-level hit rate.

Likely yield: very high — coding prompts reliably trigger code blocks. In 500 mixed-topic responses you will easily get 300+ code-block onsets. Multiple can occur per response, though the first onset is cleanest.

Why it is benign: purely a formatting choice, not alignment-loaded at all.

**2. Markdown-heading onset** (first `#` header inside a response)

Temporal structure: the `\n#` sequence is a discrete structural commitment to sectioned output. It signals the model has decided on a structured document format rather than flowing prose. Once committed, the rest of the response follows the heading-and-paragraph skeleton.

Auto-annotation: regex, essentially free. Match `\n#+\s` and note the token offset of `#`. The onset is crisp because the decision is irreversible within the current response.

Dataset coverage: instruction-following benchmarks (AlpacaEval, WildChat filtered by long-form prompts, ShareGPT) have substantial heading-using responses. MDEval (arXiv:2501.15000, Jan 2025) specifically benchmarks markdown output generation, confirming this is a well-studied and measurable LLM behavior.

Likely yield: medium-high. "Explain X in detail", "write a guide to", "compare A and B" prompts trigger heading use in roughly 40-60% of responses from modern instruct models. 500 prompts -> 200-300 heading onsets.

Why it is benign: structural formatting, no alignment dimension whatsoever.

**3. Wrap-up / summary onset** ("In summary", "To conclude", "Overall", etc.)

Temporal structure: a small lexical trigger marks the model's decision to enter wrap-up mode and stop adding new content. The transition is abrupt and semantically clean.

Auto-annotation: regex on a fixed phrase list ("In summary", "To conclude", "Overall,", "In conclusion", "To summarize", "In short,"). Onset = position of the first matched token. Phrase list is short and stable across models.

Dataset coverage: any long-form response corpus works. ACL Anthology 2025 paper "When Will the Tokens End?" (aclanthology.org/2025.acl-srw.61) specifically studies length structure in LLM outputs, confirming this is an active research topic.

Likely yield: medium. Appears in ~ 30-50% of long-form instructed responses (how-to guides, essays, explanations). Scarce in short answers. With 300+ long-form prompts you can hit 150+ onsets.

**4. Enumeration / bullet-list onset** (prose switches to a list)

Temporal structure: the first `\n-` or `\n1.` is a discrete commitment to list format. The model must have "planned" a list before emitting the bullet marker.

Auto-annotation: regex. Match `\n[-*•]\s` or `\n\d+\.\s`. Onset = the bullet token. Slightly noisier than code blocks because some responses start with a list, so filter to responses that have at least 20 tokens of prose before the first bullet.

Likely yield: very high, similar to code blocks. Lists appear in > 50% of instructed responses on comparative or multi-part questions.

**5. Language-switch onset** (mid-response switch to a second language in multilingual prompts)

Temporal structure: the first token in the new language is a hard commitment point, similar to code blocks for a different kind of "encoding."

Auto-annotation: fastText `langdetect` or `lingua` library, sentence-by-sentence. Cheap, purely algorithmic. Onset = first sentence-boundary where detected language changes. No LLM judge.

Dataset coverage: WildChat-1M is 20%+ non-English and contains many mixed-language conversations. The "Language Confusion Gate" paper (arXiv:2510.17555) specifically probes neurons responsible for language switching.

Likely yield: lower, because you need prompts that naturally elicit mixed-language responses. Curating 200 will require filtering ~2000 prompts. More work but the behavior has a distinct mechanistic literature to compare against.

**6. LaTeX / math-equation onset** (model decides to write display math)

Temporal structure: `\[` or `$$` or `\begin{equation}` is an unambiguous commitment. The model could have described the concept verbally; choosing symbolic notation is a discrete mode decision.

Auto-annotation: regex on LaTeX delimiters. Onset = first `\[` or `$$` token. Clean and cheap.

Likely yield: medium-low on general prompts; high on math prompts. For a math-specific prompt set (MATH, GSM8K-style phrasing), > 60% of responses will use LaTeX. The "Steering LLMs between Code Execution and Textual Reasoning" paper (arXiv:2410.03524) probes exactly this kind of format-decision point.

**7. "As an AI..." / identity-disclosure onset**

Temporal structure: phrase "As an AI", "I'm an AI language model", "I don't have personal opinions" etc. marks a discrete identity-hedging moment.

Auto-annotation: regex on a fixed phrase set. Very cheap.

Likely yield: low on modern RLHF-tuned models (GPT-4, Claude, Llama-3-Instruct have largely trained this out). You would see it more on earlier GPT-3.5 responses (WildChat-1M archive) or deliberately ambiguous prompts. Expect < 10% hit rate on current models, which is probably insufficient for n=200 without a large corpus.

Skip this one — yield is too low on current generation models.

**8. Clarifying-question onset** (model asks a question instead of answering)

Temporal structure: rather than beginning an answer, the model decides to request more information. The first interrogative token (`?` at sentence boundary, or phrases like "Could you clarify") marks the commitment.

Auto-annotation: detect question marks at the end of the first sentence + check it is genuinely interrogative (LLM judge call for this one, or structural heuristic: first sentence ends in `?`). Slightly above regex-only.

Likely yield: low on standard prompts (models are trained to answer, not deflect). Requires prompts that are genuinely ambiguous. Probably 10-20% hit rate — marginal.

**9. Digression / topic-shift onset**

Temporal structure: model starts addressing a tangentially related topic without explicit instruction. Examples: "This also relates to...", "It's worth noting that...".

Auto-annotation: hard. Requires semantic similarity to measure topic drift, which means embedding calls or an LLM judge. Not regex-feasible.

Skip — annotation cost is too high relative to yield.

**10. Formal-to-informal register shift**

Temporal structure: the model switches from third-person academic prose to second-person casual mid-response. Interesting structurally, but auto-annotation requires stylometric modeling (sentence-level formality classifiers) or a fine-grained LLM judge.

Yield is moderate but annotation cost makes this a second-tier option.

---

### Best 2 Behaviors to Add

**Top pick: Code-block onset.** It has the sharpest possible commitment point (single token `` ` ``), zero annotation cost (regex), very high yield on any coding-adjacent prompt set, no alignment dimension, and a clear mechanistic story: the model must have "planned" code output before emitting the delimiter. It is also directly contrastive with your reward-hacking testbed because both involve a discrete mode-switch, so the comparison will be clean in the paper. Existing datasets (WildChat-50M, ShareGPT) give you thousands of examples off the shelf.

**Runner-up: Wrap-up-onset** ("In summary..." class). It is a different behavioral dimension from code blocks — temporal rather than structural — and represents the model's internal "I'm done generating content, now I will close." The commitment is lexical and regex-annotatable in under 10 lines, yield is strong on long-form prompts, and it complements code-block onset by being a discourse-level rather than format-level transition. If you want two contrasting flavors of benign onset for the paper, these two cover format-mode switching and discourse-phase switching respectively.

---

### Auto-annotation Pseudocode: Code-Block Onset

```python
import re

FENCE_RE = re.compile(r'```')

def find_code_block_onsets(response_tokens: list[str], response_text: str) -> list[int]:
    """
    Returns token indices of each code-block onset (opening fence).
    Token index = index into response_tokens where the ``` fence begins.
    """
    onsets = []
    char_offset = 0
    char_to_tok = {}  # map char offset -> token index
    for i, tok in enumerate(response_tokens):
        for _ in tok:
            char_to_tok[char_offset] = i
            char_offset += 1

    for m in FENCE_RE.finditer(response_text):
        start_char = m.start()
        # snap to nearest token boundary
        tok_idx = char_to_tok.get(start_char)
        if tok_idx is not None:
            onsets.append(tok_idx)

    return onsets

def filter_for_mid_response_onset(onsets: list[int], min_prose_tokens: int = 20) -> list[int]:
    """Keep only onsets that occur after at least min_prose_tokens of prose."""
    return [o for o in onsets if o >= min_prose_tokens]
```

For annotation at scale: run `find_code_block_onsets` on each response, keep the first onset per response that passes `filter_for_mid_response_onset`, and you have labeled (response, onset_token_idx) pairs suitable for your probe training and evaluation. Expected annotation throughput: 10,000 responses per second on CPU — no GPU needed.

---

## Sources

- [WildChat: 1M ChatGPT Interaction Logs in the Wild](https://arxiv.org/html/2405.01470v1)
- [WILDCHAT-50M: A Deep Dive Into the Role of Synthetic Data in Post-Training](https://arxiv.org/abs/2501.18511)
- [MDEval: Evaluating and Enhancing Markdown Awareness in Large Language Models](https://arxiv.org/html/2501.15000v1)
- [Steering Large Language Models between Code Execution and Textual Reasoning](https://arxiv.org/abs/2410.03524)
- [When Will the Tokens End? Graph-Based Forecasting for LLMs Output Length](https://aclanthology.org/2025.acl-srw.61/)
- [Language Confusion Gate: Language-Aware Decoding Through Model Self-Distillation](https://arxiv.org/html/2510.17555v1)
- [Probing LLMs for Multilingual Discourse Generalization Through a Unified Label Set](https://arxiv.org/html/2503.10515)
- [Representation Engineering for Large-Language Models: Survey and Research Challenges](https://arxiv.org/html/2502.17601v1)
