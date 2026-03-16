"""
Trait extraction pipeline.

Stages:
0. preextraction_vetting.py - LLM judges if scenarios/responses match trait
1. generate_responses.py - Generate model responses to scenarios
3. extract_activations.py - Capture model activations from all layers
4. extract_vectors.py - Extract trait vectors using various methods
5. run_logit_lens.py - Interpret vectors via vocabulary projection
6. extraction_evaluation.py - Evaluate vectors on held-out data
"""

__version__ = "1.0.0"
