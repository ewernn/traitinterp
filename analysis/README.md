Analysis scripts organized by domain.

```
analysis/
├── vectors/                  # Trait vector analysis
│   ├── extraction_evaluation.py   # Accuracy, effect size, overlap across extracted vectors
│   ├── massive_activations.py     # Calibrate/analyze massive activation dimensions
│   ├── trait_correlation.py       # Cross-trait correlation analysis
│   ├── geometry.py                # Vector geometry (cosine similarity matrices, PCA)
│   ├── preference_elo.py          # Elo ranking from pairwise preference data
│   └── logit_lens.py              # Interpret vectors through unembedding matrix
├── model_diff/               # Cross-variant comparison
│   ├── compare_variants.py        # Cohen's d, per-trait diff between model variants
│   ├── per_token_diff.py          # Token-level activation differences
│   ├── layer_sensitivity.py       # Layer-wise sensitivity analysis
│   └── top_activating_spans.py    # Find spans with strongest activations
└── benchmark/                # Benchmark evaluation
    └── benchmark_evaluate.py      # Evaluate model on benchmarks with steering
```
