# Detection Layer Profiling — User Messages

### [2026-03-27 00:45 PST] Input
Detection layer profiling: For each trait, find the best detection layer (highest probe accuracy) vs the best steering layer, and characterize how these differ across traits and models. Exploratory investigation with ~20 hours of A100 40GB compute available. Should test multiple models beyond just Qwen 3.5 9B. No steering evals needed — focus on extraction/probe metrics across all layers. The march 15 idea from docs/future_ideas.md.

### [2026-03-27 00:55 PST] Input
Can do steering evals, just not 1000+. ~$10 OpenAI budget (flexible up to $40). Use Llama-3.1-8B-Instruct and Qwen3.5-9B. Can try other models but not Gemma-2-2B. Only use results that steered well. Use newer extractions with better datasets. Have subagents READ actual steering responses, not just scores. "I want you to do a bunch of things idk what u do with the 20 hours."

### [2026-03-27 01:00 PST] Input
Just run overnight autonomously. Keep sleep(3600) watchdog hooks alive. Use subagents. Follow docs. Be thorough. "See you in 30 hours."

**Outcome:** [pending]
