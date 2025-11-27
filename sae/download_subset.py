#!/usr/bin/env python3
"""Download a subset of feature labels for testing (~3 min for 1000 features)."""

import requests
import json
from pathlib import Path
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_single_feature(neuronpedia_id, feature_idx):
    """Fetch a single feature's data."""
    url = f"https://neuronpedia.org/api/feature/{neuronpedia_id}/{feature_idx}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        feature_info = {
            'description': None,
            'top_positive_tokens': data.get('pos_str', [])[:10],
            'top_negative_tokens': data.get('neg_str', [])[:10],
            'max_activation': data.get('maxActApprox'),
            'has_explanations': len(data.get('explanations', [])) > 0
        }

        if data.get('explanations') and len(data['explanations']) > 0:
            feature_info['description'] = data['explanations'][0]['description']

        return feature_idx, feature_info, None
    except Exception as e:
        return feature_idx, None, str(e)


def download_subset(
    layer=16,
    num_features=1000,
    max_workers=10
):
    """Download first N features for a specific layer."""

    neuronpedia_id = f"gemma-2-2b/{layer}-gemmascope-res-16k"
    output_dir = Path(f"sae/gemma-scope-2b-pt-res-canonical/layer_{layer}_width_16k_canonical")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "feature_labels.json"

    print(f"Downloading {num_features} features with {max_workers} workers...")
    print(f"Output: {output_file}\n")

    features = {}
    failed = []

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_single_feature, neuronpedia_id, i): i
            for i in range(num_features)
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            feature_idx, feature_info, error = future.result()

            if error:
                failed.append(feature_idx)
                features[str(feature_idx)] = {'error': error}
            else:
                features[str(feature_idx)] = feature_info

            if completed % 100 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (num_features - completed) / rate if rate > 0 else 0
                print(f"Progress: {completed}/{num_features} - {rate:.1f}/sec - ETA: {remaining:.0f}s")

    # Create output
    output = {
        'sae_info': {
            'release': 'gemma-scope-2b-pt-res-canonical',
            'sae_id': f'layer_{layer}/width_16k/canonical',
            'neuronpedia_id': neuronpedia_id,
            'layer': layer,
            'num_features': 16384,  # Total available
            'downloaded_features': num_features,
            'downloaded_at': datetime.now().isoformat(),
            'note': f'Subset: first {num_features} features only'
        },
        'features': features,
        'stats': {
            'total_features': 16384,
            'downloaded': num_features,
            'successful': num_features - len(failed),
            'failed': len(failed),
            'with_descriptions': sum(1 for f in features.values() if f.get('description')),
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Downloaded: {num_features - len(failed)}/{num_features}")
    print(f"With descriptions: {output['stats']['with_descriptions']}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download SAE feature labels from Neuronpedia")
    parser.add_argument("-l", "--layer", type=int, default=16, help="Layer (default: 16)")
    parser.add_argument("-n", "--num", type=int, default=1000, help="Number of features (default: 1000)")
    parser.add_argument("-w", "--workers", type=int, default=10, help="Parallel workers (default: 10)")
    args = parser.parse_args()

    download_subset(layer=args.layer, num_features=args.num, max_workers=args.workers)
