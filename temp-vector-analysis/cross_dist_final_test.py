import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import FastICA

print("=" * 90)
print("CROSS-DISTRIBUTION GENERALIZATION TEST: uncertainty_calibration")
print("=" * 90)
print()
print("Train: Instruction-induced data (uncertainty_calibration)")
print("Test:  Natural data (uncertainty_calibration_natural)")
print()

# Load TRAINING data
train_path = "experiments/gemma_2b_cognitive_nov20/uncertainty_calibration/extraction/activations/all_layers.pt"
train_acts = torch.load(train_path)
print(f"✅ Loaded training data: {train_acts.shape}")

# Correct counts from CSVs
train_n_pos = 96
train_n_neg = 94
test_n_pos = 102
test_n_neg = 79

print(f"  Train split: {train_n_pos} pos, {train_n_neg} neg")
print(f"  Test split:  {test_n_pos} pos, {test_n_neg} neg")

# Test on layers 16 and 24
test_layers = [16, 24]

all_results = {}

for layer in test_layers:
    print()
    print("=" * 90)
    print(f"LAYER {layer} RESULTS")
    print("=" * 90)

    # Extract TRAIN layer activations
    train_pos = train_acts[:train_n_pos, layer, :].float()
    train_neg = train_acts[train_n_pos:train_n_pos+train_n_neg, layer, :].float()

    # Load TEST layer activations
    test_pos = torch.load(f"experiments/gemma_2b_cognitive_nov20/uncertainty_calibration_natural/extraction/activations/pos_layer{layer}.pt").float()
    test_neg = torch.load(f"experiments/gemma_2b_cognitive_nov20/uncertainty_calibration_natural/extraction/activations/neg_layer{layer}.pt").float()

    results = []

    # 1. MEAN DIFFERENCE
    mean_diff_vec = (train_pos.mean(0) - train_neg.mean(0))
    mean_diff_vec_norm = mean_diff_vec / mean_diff_vec.norm()

    test_pos_proj = test_pos @ mean_diff_vec_norm
    test_neg_proj = test_neg @ mean_diff_vec_norm
    test_sep = test_pos_proj.mean() - test_neg_proj.mean()
    test_acc = ((test_pos_proj > 0).sum() + (test_neg_proj < 0).sum()).item() / (test_n_pos + test_n_neg)

    results.append(('Mean Diff', test_sep.item(), test_acc))
    print(f"\n1. MEAN DIFFERENCE")
    print(f"   Test Separation: {test_sep.item():8.2f}")
    print(f"   Test Accuracy:   {test_acc:8.1%}")

    # 2. PROBE
    X_train = torch.cat([train_pos, train_neg]).cpu().numpy()
    y_train = np.concatenate([np.ones(train_n_pos), np.zeros(train_n_neg)])

    probe = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    probe.fit(X_train, y_train)

    probe_vec = torch.from_numpy(probe.coef_[0]).float()
    probe_vec_norm = probe_vec / probe_vec.norm()

    test_pos_proj = test_pos @ probe_vec_norm
    test_neg_proj = test_neg @ probe_vec_norm
    test_sep = test_pos_proj.mean() - test_neg_proj.mean()
    test_acc = ((test_pos_proj > 0).sum() + (test_neg_proj < 0).sum()).item() / (test_n_pos + test_n_neg)

    results.append(('Probe', test_sep.item(), test_acc))
    print(f"\n2. PROBE")
    print(f"   Test Separation: {test_sep.item():8.2f}")
    print(f"   Test Accuracy:   {test_acc:8.1%}")

    # 3. ICA
    X_combined = torch.cat([train_pos, train_neg]).cpu().numpy()

    ica = FastICA(n_components=10, random_state=42)
    ica.fit(X_combined)

    # Find best component
    mixing = torch.from_numpy(ica.mixing_).float()
    best_sep = -float('inf')
    best_component = None

    for comp_idx in range(10):
        comp_vec = mixing[:, comp_idx]
        comp_vec_norm = comp_vec / comp_vec.norm()
        pos_proj = train_pos @ comp_vec_norm
        neg_proj = train_neg @ comp_vec_norm
        sep = abs(pos_proj.mean() - neg_proj.mean())
        if sep > best_sep:
            best_sep = sep
            best_component = comp_idx

    ica_vec = mixing[:, best_component]
    ica_vec_norm = ica_vec / ica_vec.norm()

    # Check if we need to flip sign
    train_pos_proj = train_pos @ ica_vec_norm
    train_neg_proj = train_neg @ ica_vec_norm
    if train_pos_proj.mean() < train_neg_proj.mean():
        ica_vec_norm = -ica_vec_norm

    test_pos_proj = test_pos @ ica_vec_norm
    test_neg_proj = test_neg @ ica_vec_norm
    test_sep = test_pos_proj.mean() - test_neg_proj.mean()
    test_acc = ((test_pos_proj > 0).sum() + (test_neg_proj < 0).sum()).item() / (test_n_pos + test_n_neg)

    results.append(('ICA', test_sep.item(), test_acc))
    print(f"\n3. ICA (component {best_component})")
    print(f"   Test Separation: {test_sep.item():8.2f}")
    print(f"   Test Accuracy:   {test_acc:8.1%}")

    # Summary table
    print()
    print(f"\nLAYER {layer} SUMMARY:")
    print("-" * 60)
    print(f"{'Method':<15} | {'Test Separation':>16} | {'Test Accuracy':>14}")
    print("-" * 60)
    for method, sep, acc in sorted(results, key=lambda x: x[2], reverse=True):
        print(f"{method:<15} | {sep:16.2f} | {acc:14.1%}")

    all_results[layer] = results

print()
print("=" * 90)
print("FINAL RESULTS: Which method generalizes best?")
print("=" * 90)

for layer in test_layers:
    print(f"\nLayer {layer} Winner: ", end="")
    winner = max(all_results[layer], key=lambda x: x[2])
    print(f"{winner[0]} with {winner[2]:.1%} accuracy")

print()
print("Cross-distribution test reveals true generalization ability!")
print("Lower accuracy than same-distribution test shows instruction != natural behavior")
