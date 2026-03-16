#!/usr/bin/env python3
"""Fast feature label download - minimal rate limiting."""

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
            if len(data['explanations']) > 1:
                feature_info['all_descriptions'] = [
                    exp['description'] for exp in data['explanations']
                ]

        return feature_idx, feature_info, None

    except Exception as e:
        return feature_idx, None, str(e)


def download_parallel(
    neuronpedia_id="gemma-2-2b/16-gemmascope-res-16k",
    num_features=16384,
    max_workers=10
):
    """Download features in parallel for speed."""

    output_dir = Path("sae/gemma-scope-2b-pt-res-canonical/layer_16_width_16k_canonical")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "feature_labels.json"

    print(f"Downloading {num_features} features with {max_workers} parallel workers...")
    print(f"Output: {output_file}\n")

    features = {}
    failed = []

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(fetch_single_feature, neuronpedia_id, i): i
            for i in range(num_features)
        }

        # Process as they complete
        completed = 0
        for future in as_completed(futures):
            completed += 1

            feature_idx, feature_info, error = future.result()

            if error:
                failed.append(feature_idx)
                features[str(feature_idx)] = {'error': error}
            else:
                features[str(feature_idx)] = feature_info

            # Progress update every 500 features
            if completed % 500 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (num_features - completed) / rate if rate > 0 else 0
                print(f"Progress: {completed}/{num_features} ({100*completed/num_features:.1f}%) "
                      f"- {rate:.1f} features/sec - ETA: {remaining/60:.1f} min")

    # Create output JSON
    output = {
        'sae_info': {
            'release': 'gemma-scope-2b-pt-res-canonical',
            'sae_id': 'layer_16/width_16k/canonical',
            'neuronpedia_id': neuronpedia_id,
            'num_features': num_features,
            'downloaded_at': datetime.now().isoformat(),
        },
        'features': features,
        'stats': {
            'total_features': num_features,
            'successful': num_features - len(failed),
            'failed': len(failed),
            'with_descriptions': sum(1 for f in features.values() if f.get('description')),
            'download_time_seconds': time.time() - start_time,
        }
    }

    # Save
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print("="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Total: {num_features}")
    print(f"Successful: {output['stats']['successful']}")
    print(f"Failed: {output['stats']['failed']}")
    print(f"With descriptions: {output['stats']['with_descriptions']} ({100*output['stats']['with_descriptions']/num_features:.1f}%)")
    print(f"Time: {output['stats']['download_time_seconds']/60:.1f} minutes")
    print(f"\nSaved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1e6:.1f} MB")

    # Save metadata
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(output['sae_info'] | {'stats': output['stats']}, f, indent=2)

    return output_file


if __name__ == "__main__":
    download_parallel()
