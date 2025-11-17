#!/bin/bash
# Full extraction pipeline for all traits in gemma_2b_cognitive_nov20
# Estimated time: 2.5-4.5 hours

set -e  # Exit on error

EXPERIMENT="gemma_2b_cognitive_nov20"

# All traits (16 instruction + 1 natural)
TRAITS="abstract_concrete,commitment_strength,context_adherence,convergent_divergent,emotional_valence,instruction_boundary,instruction_following,local_global,paranoia_trust,power_dynamics,refusal,retrieval_construction,serial_parallel,sycophancy,temporal_focus,uncertainty_calibration,refusal_natural"

echo "=========================================="
echo "Starting Full Extraction Pipeline"
echo "=========================================="
echo "Experiment: $EXPERIMENT"
echo "Traits: 17 total"
echo "Started: $(date)"
echo ""

# Stage 2: Extract activations (SLOW - 2-4 hours)
echo "Stage 2: Extracting activations..."
echo "Estimated time: 2-4 hours"
python extraction/2_extract_activations.py \
    --experiment "$EXPERIMENT" \
    --traits "$TRAITS" \
    --batch_size 8 \
    --device mps

echo ""
echo "Stage 2 complete: $(date)"
echo ""

# Stage 3: Extract vectors (FAST - 5-10 minutes)
echo "Stage 3: Extracting vectors..."
echo "Estimated time: 5-10 minutes"
python extraction/3_extract_vectors.py \
    --experiment "$EXPERIMENT" \
    --traits "$TRAITS"

echo ""
echo "=========================================="
echo "Extraction Complete!"
echo "=========================================="
echo "Finished: $(date)"
echo ""
echo "Next steps:"
echo "  - Run validation: python tests/test_activation_extraction.py"
echo "  - Check vector norms are stable (no 500+ values)"
echo "  - Optionally run inference data collection"
