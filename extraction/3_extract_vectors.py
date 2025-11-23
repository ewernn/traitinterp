#!/usr/bin/env python3
"""
Command-line wrapper for Stage 3: Extract Trait Vectors.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the refactored core logic
from extraction.pipeline.extract_vectors import extract_vectors_for_trait

def main():
    """Main function to handle argument parsing."""
    parser = argparse.ArgumentParser(description='Extract trait vectors from activations.')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--trait', type=str, required=True, help='Single trait name (e.g., "my_trait_name")')
    parser.add_argument('--methods', type=str, default='mean_diff,probe,ica,gradient', help='Comma-separated method names')
    parser.add_argument('--layers', type=str, help='Comma-separated layer numbers (default: all)')
    
    args = parser.parse_args()

    print("=" * 80)
    print("EXTRACTING VECTORS (CLI WRAPPER)")
    print(f"Experiment: {args.experiment}")
    print(f"Trait: {args.trait}")
    print("=" * 80)

    methods_list = [m.strip() for m in args.methods.split(",")]
    layers_list: Optional[List[int]] = None
    if args.layers:
        layers_list = [int(l.strip()) for l in args.layers.split(",")]

    # Call the core logic for the single trait
    extract_vectors_for_trait(
        experiment=args.experiment,
        trait=args.trait,
        methods=methods_list,
        layers=layers_list
    )

    print(f"\nSUCCESS: Finished extracting vectors for '{args.trait}'.")

if __name__ == "__main__":
    main()

