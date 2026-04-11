# ant_emotion_concepts — User Messages

Verbatim user directives about this task. Read during planning; append on new input. Outcomes added after action.

---

### [2026-04-09] Task launch
> "ok i'm gonna share the actual experiment with u when u ready... I'll want you to run all night continuously using the /r:run-experiment plugin"

**Outcome:** Stages 1.1 → 8 executed autonomously across two sessions.

---

### [2026-04-10] Post-extraction refactor
> "I'll want a clean refactor now so might as well do now. let's think thru this more, what the cleanest possible ways are"
> "200 prompts × 1 rollout. and i want to add support for this method. so would u call it a trait probe itself?"
> "wait so neutral has nothing to do with the centroid of the 171?"

**Outcome:** Neutral corpus = 100 prompts × 2 rollouts (200 total). Pseudo-trait `_neutral` with leading-underscore filter (D4). Composable method names (D3). PC basis caching with hash invalidation.

---

### [2026-04-11] Paper faithfulness
> "oh yeah i wanna match paper faithfully! i'm generally not a fan of regex!"

**Outcome:** Switched RH/blackmail graders from regex to `TraitJudge.classify` logprob classification (D5). Documented residual-norm measurement trap (D6).

---

### [2026-04-11] Stage 7 steering strength
> "well what if you're not steering strong enough?? have u checked coherence breakdown threshold? using @utils/coefficient_search.py?"
> "wait so how did emotion concepts paper steer to get blackmail? what emotion did they steer?"

**Outcome:** Multi-layer steering test (Phase 3) confirmed 7× amplification over single-layer. Blackmail breakdown probe run. Discovered paper §3.2.1 footnote: earlier Sonnet snapshot used because final is too eval-aware. Our 0% baseline is correct replication of final-snapshot behavior.

---

### [2026-04-11] Stage 8 base model
> "there's no quantized version of Llama 3.1 70B base online?"
> "did paper do the base model thing? so we're doing this for replication purposes?"
> "what about, can you download the int4 version of llama 3.1 70B base, or do u need to install the full 140GB and then quantize when u load the model into vRAM?"
> "use unsloth/Meta-Llama-3.1-70B-bnb-4bit and mention in documentation along with documenting our decisions along the way"

**Outcome:** Downloaded `unsloth/Meta-Llama-3.1-70B-bnb-4bit` (37GB). Stage 8 complete — SURPRISING finding: direction OPPOSITE paper (D7). Documented in `ant_emotion_concepts_findings.md` and `ant_emotion_concepts_methodology_notes.md`.

---

### [2026-04-11] Layer count
> "i thought we were gonna use 14 layers, not 8"

**Outcome:** Acknowledged — 14-layer extraction at `[1,7,13,19,25,31,37,43,49,55,61,67,73,79]` was always the plan; behavioral steering uses central 8 as the "drop extremes" subset, not a 14→8 reduction. Clarified in notepad (D2).

---

### [2026-04-11] Overnight run prep
> "well I'm wondering about leaving you going all night, like 10+ hours. would you be able to find stuff to do? Don't run anything rn"
> "well we should commit as we go, and I'll run r plugin's /r:check-in check-in agent, idk if u know about that one"
> "yeah I'll run '/loop 30min run /r:check-in' or something. but yeah i do want r plugin's `r/docs/system.md` to be used for documentation and organization, in the experiments/ant_emotion_concepts/ dir"
> "i rolled back the convo. run /precompact cuz we'll need to precompact before continuing"
> "and yeah, gonna run overnight"

**Outcome:** (in progress) Reorganizing docs to r plugin convention (`ant_emotion_concepts_{plan,notepad,findings,decision_tree,user_messages}.md`). Committing refactor work. Overnight plan: stage 1.3 dialogue generation → stage 6 speaker probes → stage 1.4 deflection dialogues → stage 9 deflection probes → findings reconciliation.

---

## Standing Directives (apply to all future work)

| Rule | Source |
|---|---|
| Don't run 2 GPU processes simultaneously | Explicit |
| LLM judge over regex for behavioral classification | Explicit |
| Match paper faithfully where feasible | Explicit |
| Predict → check extremes → estimate → alternatives BEFORE expensive runs | Established methodology principle |
| Commit as we go during overnight runs | Explicit |
| Document decisions with rationale (for LW writeup) | Explicit |
| Use existing codebase APIs (core/utils/pipelines), don't reimplement | CLAUDE.md |
| Check coherence alongside delta when analyzing traits | Memory note |
| Use r plugin's `r/docs/system.md` framework for doc organization | Explicit |
