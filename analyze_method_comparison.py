import json
from pathlib import Path

traits = [
    "chirp/refusal",
    "hum/confidence", 
    "hum/formality",
    "hum/optimism",
    "hum/retrieval",
    "hum/sycophancy"
]

print("=" * 80)
print("STEERING METHOD COMPARISON")
print("=" * 80)
print()

for trait in traits:
    results_path = Path(f"experiments/gemma-2-2b/steering/{trait}/results.json")
    
    with open(results_path) as f:
        data = json.load(f)
    
    baseline = data.get('baseline', {}).get('trait_mean', 0)
    
    # Group by method, find best layer for each
    method_best = {}
    
    for run in data.get('runs', []):
        config = run['config']
        result = run['result']
        
        # Only single-layer runs
        if len(config.get('layers', [])) != 1:
            continue
            
        method = config['methods'][0]
        layer = config['layers'][0]
        trait_mean = result.get('trait_mean', 0)
        coherence = result.get('coherence_mean', 0)
        
        # Only count if coherent
        if coherence < 70:
            continue
            
        delta = trait_mean - baseline
        
        if method not in method_best or delta > method_best[method]['delta']:
            method_best[method] = {
                'layer': layer,
                'delta': delta,
                'trait_mean': trait_mean,
                'coherence': coherence
            }
    
    # Print results
    print(f"{trait}:")
    print(f"  Baseline: {baseline:.1f}")
    
    # Sort by delta
    sorted_methods = sorted(method_best.items(), key=lambda x: x[1]['delta'], reverse=True)
    
    for method, info in sorted_methods:
        print(f"  {method:12s} L{info['layer']:2d}: Δ={info['delta']:+5.1f} (trait={info['trait_mean']:5.1f}, coh={info['coherence']:.1f})")
    
    # Show winner
    if sorted_methods:
        winner = sorted_methods[0]
        print(f"  → Winner: {winner[0]} (Δ={winner[1]['delta']:+.1f})")
    print()
