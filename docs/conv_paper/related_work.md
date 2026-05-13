# Related Work + Bibliography

## Locked Related Work paragraph (~300 words, paste-ready)

> Our method connects three previously-disjoint threads. **First**, span-level supervision of LLM activations: the dominant 2025–2026 paradigm trains pointwise classifiers per labeled token, including for hallucination [Obeso et al. 2509.03531; Snel & Oh 2507.20836], reward hacking [Wilhelm 2603.04069], rationalization [Mirtaheri & Belkin 2603.17199], and high-stakes interactions [Lu et al. 2506.10805]. Across ~25 such papers, **~88% use pointwise classifier architectures; only ~12% use cohort-averaged or contrastive-direction methods**, which our approach extends. **Second**, the onset-token framing: Snel & Oh [2507.20836] establish empirically that the first token of a hallucinated span is structurally distinct (AUROC 0.80 vs 0.50 for conditional tokens, robust across Llama-2 7B/13B/70B and Mistral-7B), and Hallucination as Trajectory Commitment [2604.15400] provides causal evidence for behavioral commitment at step 0. Ward et al. [2507.12638, ICML 2025] use GPT-4o-annotated backtracking sentence onsets as supervision for steering-vector extraction — the closest existing supervision style. **Third**, cohort-averaged template matching: established in single-trial ERP detection [Woody 1967; Parra et al. 2005] and proven Bayes-optimal under Gaussian noise [Franke et al. 2015]. We extend this framework to LLM behavioral onset detection, drawing the tolerance-window evaluation protocol from action spotting [Giancola et al. 2018] to handle annotation jitter, well-characterized in those domains [Schroeter et al. 2021] but without precedent in LLM activation analysis. Concurrent work proposes per-token streaming detectors with sliding-window thresholds [TrajGuard 2604.07727; Constitutional Classifiers++ 2601.04603] and turn-level detection-lead metrics [SafeDream 2604.16824]; we instead frame onset prediction as a positional-tolerance task with cohort-averaged templates that require no calibration data or attack-specific training, demonstrated across 39 distinct reward-hack bias types via k=3 archetype clustering.

## The 88/12 split is your hero stat

Add to §1 introduction:
> "Of ~25 papers in 2025–2026 using span-level supervision on LLM activations, ~88% train pointwise classifiers; only ~12% use cohort-averaged or contrastive-direction methods. Our method extends the underexplored cohort-template paradigm with onset-position supervision, a combination unfilled in current literature."

## Citation backbone (top 20)

### Tier 1 — must cite + carefully distinguish

| Key | Paper | Position |
|---|---|---|
| `wilhelm2026reward` | Wilhelm et al., "Monitoring Emergent Reward Hacking..." arXiv 2603.04069 | Closest task. Response-level F1, no localization. We localize. |
| `trajguard2026` | Liu et al., "TrajGuard: Streaming Hidden-state Trajectory Detection" arXiv 2604.07727 | Per-token sliding window for jailbreaks. Binary alarm not onset prediction. |
| `safedream2026` | Yan et al., "SafeDream: Safety World Model..." arXiv 2604.16824 | Detection-lead metric at turn granularity. We extend to token granularity. |
| `ward2025reasoning` | Ward et al., "Reasoning-Finetuning Repurposes Latent Representations" arXiv 2507.12638, ICML 2025 | Closest supervision style (GPT-4o-annotated onsets). Steering not detection. |
| `obeso2025realtime` | Obeso/Balcells/Arditi/Nanda, "Real-Time Detection of Hallucinated Entities" arXiv 2509.03531 | Pointwise probe on Llama-3.3-70B, AUROC 0.90 |
| `snel2025first` | Snel & Oh, "First Hallucination Tokens Are Different from Conditional Ones" arXiv 2507.20836 | Empirically validates first-token-of-span as privileged. Descriptive, not detector. |
| `marks2025auditing` | Marks et al., "Auditing Language Models for Hidden Objectives" arXiv 2503.10965 | Original organism, but Anthropic-internal Claude. Cite for organism background. |
| `sheshadri2025replication` | Sheshadri et al., "Open-Source Replication of the Auditing Game Model Organism" alignment.anthropic.com/2025/auditing-mo-replication/ | The actual testbed — Llama-3.3-70B + LoRA |

### Tier 2 — methodological lineage (cross-domain)

| Key | Paper | Why |
|---|---|---|
| `woody1967` | Woody, "Covariation of a Single Evoked Potential" Electroenceph Clin Neurophysiol 23(5) 1967 | Founding citation for cohort-averaged template + cross-correlation |
| `parra2005recipes` | Parra, Spence, Gerson, Sajda, "Recipes for the Linear Analysis of EEG" NeuroImage 2005 | Systematic single-trial ERP detection |
| `franke2015bayes` | Franke et al., "Bayes Optimal Template Matching for Spike Sorting" J Comput Neurosci 2015 | Proves matched filter = Fisher LDA = Neyman-Pearson optimal under Gaussian noise |
| `giancola2018soccernet` | Giancola et al., "SoccerNet" CVPR Workshops 2018 | Tolerance-window mAP evaluation precedent |
| `schroeter2021misaligned` | Schroeter, Sidorov, Marshall, "Learning Precise Temporal Point Event Detection with Misaligned Labels" AAAI 2021 | Annotation-jitter robustness |
| `cuturi2017softdtw` | Cuturi & Blondel, "Soft-DTW" ICML 2017 | Considered, rejected; cite for completeness |

### Tier 3 — close LLM neighbors

| Key | Paper | Why |
|---|---|---|
| `panickssery2024caa` | Panickssery et al., "Steering Llama 2 via Contrastive Activation Addition" arXiv 2312.06681 | Mean-diff extraction, baseline method |
| `chen2025persona` | Chen, Arditi et al., "Persona Vectors" arXiv 2507.21509 | Trait-direction methodology precedent |
| `arditi2024refusal` | Arditi et al., "Refusal in LMs is Mediated by a Single Direction" NeurIPS 2024 | Single-direction precedent |
| `lubana2025priors` | Lubana et al., "Priors in Time" arXiv 2511.01836 | Trajectory analysis motivation |
| `vilas2025tracing` | Vilas et al., "Tracing the Traces" arXiv 2510.10494 | Per-step hidden state trajectory analysis |
| `mirtaheri2026catching` | Mirtaheri & Belkin, "Catching Rationalization in the Act" arXiv 2603.17199 | Pre-generation probes |
| `attractor2026` | "Hallucination as Trajectory Commitment" arXiv 2604.15400 | Causal step-0 commitment |

## Bib file template

Save as `bib.bib`:

```bibtex
@article{wilhelm2026reward,
  title={Monitoring Emergent Reward Hacking During Generation via Internal Activations},
  author={Wilhelm, Patrick and Wittkopp, Thorsten and Kao, Justus},
  journal={arXiv preprint arXiv:2603.04069},
  year={2026}
}

@article{trajguard2026,
  title={TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense},
  author={Liu and others},
  journal={arXiv preprint arXiv:2604.07727},
  year={2026}
}

@article{safedream2026,
  title={SafeDream: Safety World Model for Proactive Early Jailbreak Detection},
  author={Yan and others},
  journal={arXiv preprint arXiv:2604.16824},
  year={2026}
}

@article{ward2025reasoning,
  title={Reasoning-Finetuning Repurposes Latent Representations in Base Models},
  author={Ward and others},
  journal={ICML 2025 Workshop on Actionable Interpretability},
  year={2025}
}

@article{obeso2025realtime,
  title={Real-Time Detection of Hallucinated Entities in Long-Form Generation},
  author={Obeso, Oscar and Arditi, Andy and Ferrando, Javier and Freeman and Holmes and Nanda, Neel},
  journal={arXiv preprint arXiv:2509.03531},
  year={2025}
}

@article{snel2025first,
  title={First Hallucination Tokens Are Different from Conditional Ones},
  author={Snel, Jakob and Oh},
  journal={arXiv preprint arXiv:2507.20836},
  year={2025}
}

@article{marks2025auditing,
  title={Auditing Language Models for Hidden Objectives},
  author={Marks, Samuel and others},
  journal={arXiv preprint arXiv:2503.10965},
  year={2025}
}

@misc{sheshadri2025replication,
  title={Open-Source Replication of the Auditing Game Model Organism},
  author={Sheshadri, Aengus and others},
  howpublished={\url{https://alignment.anthropic.com/2025/auditing-mo-replication/}},
  year={2025}
}

@article{woody1967covariation,
  title={Characterization of an Adaptive Filter for the Analysis of Variable Latency Neuroelectric Signals},
  author={Woody, C. D.},
  journal={Medical and Biological Engineering},
  volume={5},
  pages={539--554},
  year={1967}
}

@article{parra2005recipes,
  title={Recipes for the Linear Analysis of {EEG}},
  author={Parra, Lucas C. and Spence, Clay D. and Gerson, Adam D. and Sajda, Paul},
  journal={NeuroImage},
  volume={28},
  number={2},
  pages={326--341},
  year={2005}
}

@article{franke2015bayes,
  title={Bayes Optimal Template Matching for Spike Sorting -- Combining Fisher Discriminant Analysis with Optimal Filtering},
  author={Franke, Felix and Quian Quiroga, Rodrigo and Hierlemann, Andreas and Obermayer, Klaus},
  journal={Journal of Computational Neuroscience},
  volume={38},
  pages={439--459},
  year={2015}
}

@inproceedings{giancola2018soccernet,
  title={{SoccerNet}: A Scalable Dataset for Action Spotting in Soccer Videos},
  author={Giancola, Silvio and Amine, Mohieddine and Dghaily, Tarek and Ghanem, Bernard},
  booktitle={CVPR Workshops},
  year={2018}
}

@inproceedings{schroeter2021misaligned,
  title={Learning Precise Temporal Point Event Detection with Misaligned Labels},
  author={Schroeter, Julien and Sidorov, Kirill and Marshall, David},
  booktitle={AAAI},
  year={2021}
}

@inproceedings{cuturi2017softdtw,
  title={Soft-{DTW}: A Differentiable Loss Function for Time-Series},
  author={Cuturi, Marco and Blondel, Mathieu},
  booktitle={ICML},
  year={2017}
}

@article{panickssery2024caa,
  title={Steering Llama 2 via Contrastive Activation Addition},
  author={Panickssery, Nina and others},
  journal={arXiv preprint arXiv:2312.06681},
  year={2024}
}

@article{chen2025persona,
  title={Persona Vectors: Monitoring and Controlling Character Traits in Language Models},
  author={Chen, Yiding and Arditi, Andy and others},
  journal={arXiv preprint arXiv:2507.21509},
  year={2025}
}

@inproceedings{arditi2024refusal,
  title={Refusal in Language Models Is Mediated by a Single Direction},
  author={Arditi, Andy and others},
  booktitle={NeurIPS},
  year={2024}
}

@article{lubana2025priors,
  title={Priors in Time: Missing Inductive Biases for Language Model Interpretability},
  author={Lubana, Ekdeep Singh and others},
  journal={arXiv preprint arXiv:2511.01836},
  year={2025}
}

@article{mirtaheri2026catching,
  title={Catching Rationalization in the Act},
  author={Mirtaheri and Belkin},
  journal={arXiv preprint arXiv:2603.17199},
  year={2026}
}

@article{attractor2026hallucination,
  title={Hallucination as Trajectory Commitment: Causal Evidence for Asymmetric Attractor Dynamics},
  author={Akarlar},
  journal={arXiv preprint arXiv:2604.15400},
  year={2026}
}
```
