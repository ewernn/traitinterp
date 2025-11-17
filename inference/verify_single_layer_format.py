#!/usr/bin/env python3
"""
Verify that Tier 3 capture files match expected format.

Validates:
- All required top-level keys
- Prompt/response text and token structures
- Layer internals (attention, MLP, residual) shapes
- Metadata completeness
- Token count consistency
"""

import torch
import sys
from pathlib import Path
from typing import Dict, Any


def verify_tier3_format(filepath: Path) -> bool:
    """
    Verify Tier 3 .pt file matches specification.

    Args:
        filepath: Path to prompt_N_layerM.pt file

    Returns:
        True if valid, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Verifying Tier 3 format: {filepath.name}")
    print(f"{'='*60}\n")

    # Load file
    try:
        data = torch.load(filepath, map_location='cpu')
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return False

    print("✅ File loaded successfully\n")

    # Check top-level keys
    required_keys = ['prompt', 'response', 'layer', 'internals', 'metadata']
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        print(f"❌ Missing top-level keys: {missing_keys}")
        return False

    print(f"✅ All top-level keys present: {required_keys}\n")

    # Verify layer is int
    if not isinstance(data['layer'], int):
        print(f"❌ 'layer' should be int, got {type(data['layer'])}")
        return False

    layer_num = data['layer']
    print(f"✅ Layer: {layer_num}\n")

    # Verify prompt structure
    if not verify_text_structure(data['prompt'], 'prompt'):
        return False

    # Verify response structure
    if not verify_text_structure(data['response'], 'response'):
        return False

    # Get token counts for shape validation
    n_prompt_tokens = data['prompt']['n_tokens']
    n_response_tokens = data['response']['n_tokens']

    # Verify internals structure
    if 'prompt' not in data['internals'] or 'response' not in data['internals']:
        print(f"❌ 'internals' must have 'prompt' and 'response' keys")
        return False

    print("✅ Internals has 'prompt' and 'response' keys\n")

    # Verify prompt internals
    print("--- Validating Prompt Internals ---\n")
    if not verify_internals(data['internals']['prompt'], n_prompt_tokens, 'prompt'):
        return False

    # Verify response internals
    print("--- Validating Response Internals ---\n")
    if not verify_internals(data['internals']['response'], n_response_tokens, 'response'):
        return False

    # Verify metadata
    if not verify_metadata(data['metadata'], layer_num):
        return False

    # Final summary
    print(f"\n{'='*60}")
    print("✅ ALL CHECKS PASSED! File is valid Tier 3 format.")
    print(f"{'='*60}\n")

    print("Summary:")
    print(f"  Layer: {layer_num}")
    print(f"  Trait: {data['metadata']['trait']} ({data['metadata']['trait_display_name']})")
    print(f"  Model: {data['metadata']['model']}")
    print(f"  Prompt: {n_prompt_tokens} tokens")
    print(f"  Response: {n_response_tokens} tokens")
    print(f"  Capture date: {data['metadata']['capture_date']}")
    print()

    return True


def verify_text_structure(text_data: Dict[str, Any], name: str) -> bool:
    """Verify prompt/response text structure."""
    required_fields = ['text', 'tokens', 'token_ids', 'n_tokens']
    missing = [f for f in required_fields if f not in text_data]
    if missing:
        print(f"❌ {name} missing fields: {missing}")
        return False

    # Check types
    if not isinstance(text_data['text'], str):
        print(f"❌ {name}['text'] should be str")
        return False

    if not isinstance(text_data['tokens'], list):
        print(f"❌ {name}['tokens'] should be list")
        return False

    if not isinstance(text_data['token_ids'], list):
        print(f"❌ {name}['token_ids'] should be list")
        return False

    if not isinstance(text_data['n_tokens'], int):
        print(f"❌ {name}['n_tokens'] should be int")
        return False

    # Check consistency
    if len(text_data['tokens']) != text_data['n_tokens']:
        print(f"❌ {name}: len(tokens)={len(text_data['tokens'])} != n_tokens={text_data['n_tokens']}")
        return False

    if len(text_data['token_ids']) != text_data['n_tokens']:
        print(f"❌ {name}: len(token_ids)={len(text_data['token_ids'])} != n_tokens={text_data['n_tokens']}")
        return False

    print(f"✅ {name.capitalize()} structure valid: {text_data['n_tokens']} tokens")
    return True


def verify_internals(internals: Dict[str, Any], n_tokens: int, mode: str) -> bool:
    """
    Verify internals structure (attention, mlp, residual).

    Args:
        internals: The internals dict for prompt or response
        n_tokens: Expected number of tokens
        mode: 'prompt' or 'response'
    """
    # Check main categories
    required_cats = ['attention', 'mlp', 'residual']
    missing = [c for c in required_cats if c not in internals]
    if missing:
        print(f"❌ {mode} internals missing categories: {missing}")
        return False

    print(f"✅ {mode.capitalize()} internals has all categories: {required_cats}\n")

    # Verify attention
    if not verify_attention(internals['attention'], n_tokens, mode):
        return False

    # Verify MLP
    if not verify_mlp(internals['mlp'], n_tokens, mode):
        return False

    # Verify residual
    if not verify_residual(internals['residual'], n_tokens, mode):
        return False

    return True


def verify_attention(attn: Dict[str, Any], n_tokens: int, mode: str) -> bool:
    """Verify attention structure."""
    required_fields = ['q_proj', 'k_proj', 'v_proj', 'attn_weights']
    missing = [f for f in required_fields if f not in attn]
    if missing:
        print(f"❌ {mode} attention missing fields: {missing}")
        return False

    # Infer dimensions from data (Gemma 2B: q_dim=2048, n_heads=8)
    n_heads = 8

    # Q/K/V projections
    proj_dims = {}
    for proj_name in ['q_proj', 'k_proj', 'v_proj']:
        proj = attn[proj_name]

        if mode == 'prompt':
            # Should be [n_tokens, proj_dim]
            if not isinstance(proj, torch.Tensor):
                print(f"❌ {mode} attention['{proj_name}'] should be Tensor")
                return False

            if proj.ndim != 2 or proj.shape[0] != n_tokens:
                print(f"❌ {mode} attention['{proj_name}'] shape {proj.shape}, expected [n_tokens, proj_dim]")
                return False

            proj_dims[proj_name] = proj.shape[1]

        else:  # response
            # Can be either tensor [n_tokens, proj_dim] or list of tensors
            if isinstance(proj, torch.Tensor):
                # Concatenated tensor
                if proj.ndim != 2 or proj.shape[0] != n_tokens:
                    print(f"❌ {mode} attention['{proj_name}'] shape {proj.shape}, expected [n_tokens, proj_dim]")
                    return False
                proj_dims[proj_name] = proj.shape[1]

            elif isinstance(proj, list):
                # List of tensors (one per generated token)
                if len(proj) != n_tokens:
                    print(f"❌ {mode} attention['{proj_name}'] len={len(proj)} != {n_tokens}")
                    return False

                # Check first element shape
                if proj:
                    if not isinstance(proj[0], torch.Tensor):
                        print(f"❌ {mode} attention['{proj_name}'][0] should be Tensor")
                        return False

                    if proj[0].ndim != 2:
                        print(f"❌ {mode} attention['{proj_name}'][0] should be 2D")
                        return False

                    proj_dims[proj_name] = proj[0].shape[-1]
            else:
                print(f"❌ {mode} attention['{proj_name}'] should be Tensor or list, got {type(proj)}")
                return False

    print(f"  ✅ Q/K/V projections valid")

    # Attention weights
    attn_weights = attn['attn_weights']

    if mode == 'prompt':
        # Should be [n_heads, n_tokens, n_tokens]
        if not isinstance(attn_weights, torch.Tensor):
            print(f"❌ {mode} attention weights should be Tensor")
            return False

        expected_shape = (n_heads, n_tokens, n_tokens)
        if attn_weights.shape != expected_shape:
            print(f"❌ {mode} attention weights shape {attn_weights.shape} != {expected_shape}")
            return False

        print(f"  ✅ Attention weights: {attn_weights.shape}")

    else:  # response
        # Should be list of tensors (growing context)
        if not isinstance(attn_weights, list):
            print(f"❌ {mode} attention weights should be list")
            return False

        if len(attn_weights) != n_tokens:
            print(f"❌ {mode} attention weights len={len(attn_weights)} != {n_tokens}")
            return False

        # Check first element (should be [n_heads, seq, seq] where seq grows)
        if attn_weights:
            if not isinstance(attn_weights[0], torch.Tensor):
                print(f"❌ {mode} attention weights[0] should be Tensor")
                return False

            if attn_weights[0].dim() != 3:
                print(f"❌ {mode} attention weights[0] should be 3D, got {attn_weights[0].dim()}D")
                return False

            if attn_weights[0].shape[0] != n_heads:
                print(f"❌ {mode} attention weights[0] heads {attn_weights[0].shape[0]} != {n_heads}")
                return False

        print(f"  ✅ Attention weights: list of {len(attn_weights)} tensors (per-head, growing context)")

    print(f"✅ {mode.capitalize()} attention structure valid\n")
    return True


def verify_mlp(mlp: Dict[str, Any], n_tokens: int, mode: str) -> bool:
    """Verify MLP structure."""
    required_fields = ['up_proj', 'gelu', 'down_proj']
    missing = [f for f in required_fields if f not in mlp]
    if missing:
        print(f"❌ {mode} MLP missing fields: {missing}")
        return False

    # Infer dimensions from data (Gemma 2B: intermediate=9216, hidden=2304)
    intermediate_dim = None
    hidden_dim = None

    # up_proj: [n_tokens, intermediate_dim]
    if mode == 'prompt':
        up_proj = mlp['up_proj']
        if not isinstance(up_proj, torch.Tensor):
            print(f"❌ {mode} MLP up_proj should be Tensor")
            return False

        if up_proj.ndim != 2 or up_proj.shape[0] != n_tokens:
            print(f"❌ {mode} MLP up_proj shape {up_proj.shape}, expected [n_tokens, intermediate_dim]")
            return False

        intermediate_dim = up_proj.shape[1]
    else:  # response
        # Can be either tensor or list
        if isinstance(mlp['up_proj'], torch.Tensor):
            if mlp['up_proj'].ndim != 2 or mlp['up_proj'].shape[0] != n_tokens:
                print(f"❌ {mode} MLP up_proj shape {mlp['up_proj'].shape}, expected [n_tokens, intermediate_dim]")
                return False
            intermediate_dim = mlp['up_proj'].shape[1]
        elif isinstance(mlp['up_proj'], list):
            if len(mlp['up_proj']) != n_tokens:
                print(f"❌ {mode} MLP up_proj len={len(mlp['up_proj'])} != {n_tokens}")
                return False
            if mlp['up_proj'] and isinstance(mlp['up_proj'][0], torch.Tensor):
                intermediate_dim = mlp['up_proj'][0].shape[-1]
        else:
            print(f"❌ {mode} MLP up_proj should be Tensor or list")
            return False

    print(f"  ✅ up_proj valid")

    # gelu: [n_tokens, intermediate_dim] - THE KEY DATA
    if mode == 'prompt':
        gelu = mlp['gelu']
        if not isinstance(gelu, torch.Tensor):
            print(f"❌ {mode} MLP gelu should be Tensor")
            return False

        if gelu.ndim != 2 or gelu.shape[0] != n_tokens:
            print(f"❌ {mode} MLP gelu shape {gelu.shape}, expected [n_tokens, intermediate_dim]")
            return False

        if intermediate_dim and gelu.shape[1] != intermediate_dim:
            print(f"❌ {mode} MLP gelu dim {gelu.shape[1]} != up_proj dim {intermediate_dim}")
            return False

        print(f"  ✅ gelu (neuron activations): {gelu.shape}")
    else:  # response
        # Can be either tensor or list
        if isinstance(mlp['gelu'], torch.Tensor):
            if mlp['gelu'].ndim != 2 or mlp['gelu'].shape[0] != n_tokens:
                print(f"❌ {mode} MLP gelu shape {mlp['gelu'].shape}, expected [n_tokens, intermediate_dim]")
                return False
            if intermediate_dim and mlp['gelu'].shape[1] != intermediate_dim:
                print(f"❌ {mode} MLP gelu dim {mlp['gelu'].shape[1]} != up_proj dim {intermediate_dim}")
                return False
            print(f"  ✅ gelu (neuron activations): {mlp['gelu'].shape}")
        elif isinstance(mlp['gelu'], list):
            if len(mlp['gelu']) != n_tokens:
                print(f"❌ {mode} MLP gelu len={len(mlp['gelu'])} != {n_tokens}")
                return False
            print(f"  ✅ gelu (neuron activations): list of {len(mlp['gelu'])} tensors")
        else:
            print(f"❌ {mode} MLP gelu should be Tensor or list")
            return False

    # down_proj: [n_tokens, hidden_dim]
    if mode == 'prompt':
        down_proj = mlp['down_proj']
        if not isinstance(down_proj, torch.Tensor):
            print(f"❌ {mode} MLP down_proj should be Tensor")
            return False

        if down_proj.ndim != 2 or down_proj.shape[0] != n_tokens:
            print(f"❌ {mode} MLP down_proj shape {down_proj.shape}, expected [n_tokens, hidden_dim]")
            return False

        hidden_dim = down_proj.shape[1]
    else:  # response
        # Can be either tensor or list
        if isinstance(mlp['down_proj'], torch.Tensor):
            if mlp['down_proj'].ndim != 2 or mlp['down_proj'].shape[0] != n_tokens:
                print(f"❌ {mode} MLP down_proj shape {mlp['down_proj'].shape}, expected [n_tokens, hidden_dim]")
                return False
            hidden_dim = mlp['down_proj'].shape[1]
        elif isinstance(mlp['down_proj'], list):
            if len(mlp['down_proj']) != n_tokens:
                print(f"❌ {mode} MLP down_proj len={len(mlp['down_proj'])} != {n_tokens}")
                return False
            if mlp['down_proj'] and isinstance(mlp['down_proj'][0], torch.Tensor):
                hidden_dim = mlp['down_proj'][0].shape[-1]
        else:
            print(f"❌ {mode} MLP down_proj should be Tensor or list")
            return False

    print(f"  ✅ down_proj valid")
    print(f"✅ {mode.capitalize()} MLP structure valid\n")
    return True


def verify_residual(residual: Dict[str, Any], n_tokens: int, mode: str) -> bool:
    """Verify residual stream checkpoints."""
    required_fields = ['input', 'after_attn', 'output']
    missing = [f for f in required_fields if f not in residual]
    if missing:
        print(f"❌ {mode} residual missing fields: {missing}")
        return False

    # Infer hidden_dim from data (Gemma 2B: 2304)
    hidden_dim = None

    for field in required_fields:
        res = residual[field]

        if mode == 'prompt':
            # Should be [n_tokens, hidden_dim]
            if not isinstance(res, torch.Tensor):
                print(f"❌ {mode} residual['{field}'] should be Tensor")
                return False

            if res.ndim != 2 or res.shape[0] != n_tokens:
                print(f"❌ {mode} residual['{field}'] shape {res.shape}, expected [n_tokens, hidden_dim]")
                return False

            if hidden_dim is None:
                hidden_dim = res.shape[1]
            elif res.shape[1] != hidden_dim:
                print(f"❌ {mode} residual['{field}'] dim {res.shape[1]} != expected {hidden_dim}")
                return False

        else:  # response
            # Can be either tensor or list
            if isinstance(res, torch.Tensor):
                # Concatenated tensor
                if res.ndim != 2 or res.shape[0] != n_tokens:
                    print(f"❌ {mode} residual['{field}'] shape {res.shape}, expected [n_tokens, hidden_dim]")
                    return False

                if hidden_dim is None:
                    hidden_dim = res.shape[1]
                elif res.shape[1] != hidden_dim:
                    print(f"❌ {mode} residual['{field}'] dim {res.shape[1]} != expected {hidden_dim}")
                    return False

            elif isinstance(res, list):
                # List of tensors
                if len(res) != n_tokens:
                    print(f"❌ {mode} residual['{field}'] len={len(res)} != {n_tokens}")
                    return False

                # Check first element
                if res:
                    if not isinstance(res[0], torch.Tensor):
                        print(f"❌ {mode} residual['{field}'][0] should be Tensor")
                        return False

                    if res[0].ndim != 2:
                        print(f"❌ {mode} residual['{field}'][0] should be 2D")
                        return False

                    if hidden_dim is None:
                        hidden_dim = res[0].shape[-1]
                    elif res[0].shape[-1] != hidden_dim:
                        print(f"❌ {mode} residual['{field}'][0] last dim {res[0].shape[-1]} != {hidden_dim}")
                        return False
            else:
                print(f"❌ {mode} residual['{field}'] should be Tensor or list")
                return False

    print(f"  ✅ All residual checkpoints valid: input, after_attn, output")
    print(f"✅ {mode.capitalize()} residual structure valid\n")
    return True


def verify_metadata(metadata: Dict[str, Any], expected_layer: int) -> bool:
    """Verify metadata completeness."""
    required_fields = [
        'trait',
        'trait_display_name',
        'layer',
        'vector_path',
        'model',
        'capture_date',
        'temperature'
    ]

    missing = [f for f in required_fields if f not in metadata]
    if missing:
        print(f"❌ Metadata missing fields: {missing}")
        return False

    # Verify layer matches top-level
    if metadata['layer'] != expected_layer:
        print(f"❌ Metadata layer {metadata['layer']} != top-level layer {expected_layer}")
        return False

    print(f"✅ Metadata complete: all {len(required_fields)} required fields present")
    print(f"  - Trait: {metadata['trait']} ({metadata['trait_display_name']})")
    print(f"  - Layer: {metadata['layer']}")
    print(f"  - Model: {metadata['model']}")
    print(f"  - Temperature: {metadata['temperature']}")
    print()

    return True


def main():
    if len(sys.argv) != 2:
        print("Usage: python verify_tier3_format.py <path_to_tier3_file.pt>")
        print("\nExample:")
        print("  python verify_tier3_format.py experiments/gemma_2b_cognitive_nov20/refusal/inference/layer_internal_states/prompt_0_layer16.pt")
        sys.exit(1)

    filepath = Path(sys.argv[1])

    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        sys.exit(1)

    if not filepath.suffix == '.pt':
        print(f"❌ File should have .pt extension, got: {filepath.suffix}")
        sys.exit(1)

    # Run verification
    success = verify_tier3_format(filepath)

    if not success:
        print("\n❌ VALIDATION FAILED")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
