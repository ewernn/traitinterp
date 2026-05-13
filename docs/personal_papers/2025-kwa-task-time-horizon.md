---
title: "Measuring AI Ability to Complete Long Software Tasks"
authors: [Kwa, West, Becker, Deng, Garcia, Hasin, Jawhar, Kinniment, Rush, "Von Arx", METR]
year: 2025
arxiv: "2503.14499"
venue: arXiv
status: read
tags: [evals, scaling, capabilities]
related: []
aliases: ["Long Software Tasks", "Time Horizon", "METR Time Horizon"]
---

## Why I read this
General AI-progress context. As an alignment/interp researcher, I want a defensible reference point for how fast capabilities are moving so I can sanity-check timelines for when activation-monitoring tooling needs to be production-ready.

## Core claim
You can turn opaque benchmark scores into a human-legible quantity by asking "how long would a human expert need to do tasks the model gets right 50% of the time," and that quantity has been doubling roughly every 7 months since 2019, putting frontier models near 110 minutes of human-equivalent task time.

## What's novel / surprising
The framing itself is the contribution. Instead of one more leaderboard, they pick a unit (human-minutes) that lay readers and policy people can reason about, then fit an exponential to it. The 7-month doubling sounds slow next to compute scaling but extrapolates to month-long tasks within roughly 5 years, which is the surprising part if you take it at face value.

## Cruxes / my disagreements
The 50%-success threshold hides reliability collapse on the right tail. A model that finishes a 2-hour task half the time is not the same as a junior engineer who finishes it almost always, and most real-world deployment cares about the 95% number, not the 50%. The fitted doubling rate is also heavily driven by a small number of frontier releases and the task mix in HCAST plus RE-Bench, which skews toward ML-engineering work. What would change my mind on the timeline claim: a replication with the same metric on a non-ML task family (e.g., long legal drafting, scientific writing) showing a similar 7-month doubling.

## What I want to do with it
No concrete experiment connection to centered-delta or persona-vector work. Practical use is calibration: if the doubling holds, agent-driven misbehavior at multi-hour task lengths is plausibly a 2 to 3 year problem, which sets a soft deadline for getting activation monitoring deployable on long agent traces rather than single-turn responses. Cite this in any writeup that argues "monitoring should target multi-turn / multi-step trajectories, not just single completions."

## Key quotes / page refs
50% time horizon for frontier models around 110 minutes (o3-class), doubling every roughly 7 months since 2019, possibly faster post-2024. Drivers named: reliability, error recovery, reasoning, tool use.
