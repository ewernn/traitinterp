# pv-rep — User Messages (verbatim)

## Initial proposal
"so we could technically start over and make a experiments/pv_rep/, and better naming and stuff. the traitinterp repo has improved significantly since I last ran that experiment.

then, we could put the persona vectors url and article as a .md and the full paper as a .md and have a bunch of claude code subagents triple check everything and do a clean replication, still mostly just focused on how well each steers tho and how natural the responses sound, just to show that natural steers equally as well and sounds more natural, which i suspect would still be the outcome, or natural base would beat instruct on my current prompts tbh"

## Decisions round 1
"Scope: 3 traits
Models: Llama-3.1-8B-Instruct only
Re-extract using Shao's exact data from their_data/
4. steering effectiveness cuz trait_score should have proper definition. i think though that pv paper had different judge prompt than we use. that's what the 2x2 was. their eval questions their prompt, their eval questions our prompt, our eval questions their prompt, our eval questions our prompt. maybe I'm wrong
5. generate fresh for all! completely fresh experiment. even make fresh trait dirs in datasets/traits/ for both natural and instruct, so that we can modify our new ones without affecting any old data! completely fresh experiment"

## Decisions round 2 (after planner Q1-Q10)
"1. i mean tonight would be nice. it will all be super fast cuz we batch and have 80GB vRAM
2. pv_natural_v2/ + pv_instruction_v2/
3. you spawn a subagent per trait to draft v1 from trait_dataset_creation.md, and they do extensive planning before and are hyper aware of cliff hanger ending prompt right before {trait} expression
4. 1 rollout, temp=0.0 for our base natural, and 30 carefully and creatively selected pos/neg scenarios (use opus!)
5. our min_coherence is 77 by default i think (keep our default when using our coherence scorer), but spawn subagent to read their paper/code to find their coherence judge prompt and threshold if u plan to replicate using theirs too (core/kwargs_configs.py)
6. exclude? idk what that is
7. Internal numbers + figures, you do the writing later
8. dw about this rn, but Bootstrap CIs on the trait_score deltas if had to choose
9. no! i extract my base natural vectors at unit length and then use base_coef as initial guess (all of this handled automatically by utils/coefficient_search.py, steering/run_steering_eval.py so read the args/kwargs first so u know what defaults are and use defaults for mine)
10. idk, it's probably fine, u can feel free to have subagent check it out"

## Decisions round 3 (post-critic, on confounds)
"1. (b) (that's how our method works -- base model dissolves into incoherence after ~5-10 tokens)
2. (a)
3. 8
4. keep split
5. our coherence for our judge prompt, their coherence for their judge prompt (i think we have a CLI flag for what judge prompt to use)
6. we need special naming so this doesn't happen. we can make 2 of each dataset if needed (e.g. pv_natural_v2-2) *only if needed* (no symlinks tho)"

## Hardware provided
- "ssh -p 31730 root@146.115.17.157" (H100, 80GB, 50GB free disk per nvidia-smi)
