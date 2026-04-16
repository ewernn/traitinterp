"""Find max-activating corpus examples for trait vectors.

Sweeps trait vectors over a text corpus, ranking documents by probe
activation. Produces top-K highest-activating contexts per vector with
per-token scores for highlighting.

Replicates Sofroniew et al. 2026 §1.2.1 (Figure 1): vectors swept over
Common Corpus / Pile / LMSYS-Chat, top-activating snippets inspected.

Input:  Trait vectors + HuggingFace dataset (streaming)
Output: JSON with top-K contexts per vector, token-level scores

Usage:
    python analysis/vectors/max_activating_corpus.py \
        --experiment ant_emotion_concepts \
        --dataset lmsys/lmsys-chat-1m \
        --layer 49 --method mean_diff+gm+pc50 \
        --top-k 20 --n-documents 5000 \
        --load-in-4bit
"""

import argparse
import heapq
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.hooks import MultiLayerProjection
from utils.model import load_model, pad_sequences
from utils.paths import get as get_path, discover_traits
from utils.vectors import load_vector


# ============================================================================
# Top-K Tracker
# ============================================================================

@dataclass
class ActivatingExample:
    """A corpus example that strongly activates a trait vector."""
    score: float           # max token-level cosine similarity in this example
    text: str              # the raw text
    token_scores: list     # per-token scores for highlighting
    tokens: list           # decoded tokens
    doc_id: int

    def __lt__(self, other):
        return self.score < other.score


class TopKTracker:
    """Maintains top-K highest-activating examples per trait, using a min-heap."""

    def __init__(self, trait_names: List[str], k: int = 20):
        self.k = k
        self.heaps: Dict[str, list] = {name: [] for name in trait_names}

    def update(self, trait_name: str, example: ActivatingExample):
        heap = self.heaps[trait_name]
        if len(heap) < self.k:
            heapq.heappush(heap, example)
        elif example.score > heap[0].score:
            heapq.heapreplace(heap, example)

    def get_top_k(self, trait_name: str) -> List[ActivatingExample]:
        return sorted(self.heaps[trait_name], key=lambda x: -x.score)

    def get_all(self) -> Dict[str, List[ActivatingExample]]:
        return {name: self.get_top_k(name) for name in self.heaps}


# ============================================================================
# Corpus loading
# ============================================================================

def load_corpus_texts(dataset_name: str, n_documents: int, text_field: str = None,
                      split: str = "train", max_tokens: int = 512) -> List[str]:
    """Stream documents from a HuggingFace dataset.

    Handles common dataset formats (text field, conversation field).
    Returns up to n_documents text strings.
    """
    from datasets import load_dataset

    print(f"Loading {n_documents} documents from {dataset_name}...")
    ds = load_dataset(dataset_name, split=split, streaming=True)

    texts = []
    for i, example in enumerate(ds):
        if i >= n_documents:
            break

        # Auto-detect text field
        if text_field:
            text = example[text_field]
        elif "text" in example:
            text = example["text"]
        elif "conversation" in example:
            # LMSYS-Chat format: list of {"role": ..., "content": ...}
            conv = example["conversation"]
            text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in conv)
        elif "content" in example:
            text = example["content"]
        else:
            # Take the first string-valued field
            for v in example.values():
                if isinstance(v, str) and len(v) > 20:
                    text = v
                    break
            else:
                continue

        if isinstance(text, str) and len(text) > 50:
            texts.append(text[:max_tokens * 4])  # rough char limit

        if len(texts) % 1000 == 0 and len(texts) > 0:
            print(f"  Loaded {len(texts)} documents...")

    print(f"  {len(texts)} documents loaded")
    return texts


# ============================================================================
# Main sweep
# ============================================================================

def sweep_corpus(
    model, tokenizer, texts: List[str], vectors: Dict[str, torch.Tensor],
    layer: int, top_k: int = 20, batch_size: int = 4,
) -> TopKTracker:
    """Sweep vectors over corpus texts, tracking top-K activating examples."""

    trait_names = list(vectors.keys())
    tracker = TopKTracker(trait_names, k=top_k)

    # Build vectors_by_layer for MultiLayerProjection
    vec_stack = torch.stack([v.float() for v in vectors.values()])  # [n_traits, hidden_dim]
    # Normalize vectors to unit length for cosine similarity
    vec_stack = vec_stack / vec_stack.norm(dim=-1, keepdim=True)
    vectors_by_layer = {layer: vec_stack}

    n_batches = (len(texts) + batch_size - 1) // batch_size
    t0 = time.time()

    for batch_idx in range(n_batches):
        batch_texts = texts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        doc_ids = list(range(batch_idx * batch_size, batch_idx * batch_size + len(batch_texts)))

        # Tokenize
        encodings = tokenizer(batch_texts, return_tensors="pt", padding=True,
                              truncation=True, max_length=512)
        input_ids = encodings["input_ids"].to(model.device)
        attention_mask = encodings["attention_mask"].to(model.device)

        # Forward pass with projection hook
        with MultiLayerProjection(model, vectors_by_layer) as proj:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            # scores: {layer: [batch, seq, n_traits]}
            scores_by_layer = proj.get_all()

        raw_scores = scores_by_layer[layer].cpu()  # [batch, seq, n_traits]

        # Normalize by activation norms to get cosine similarity
        # We need to also capture the activation norms — but ProjectionHook
        # returns dot products (raw_scores = act @ vec^T where vec is unit-normalized).
        # To convert to cosine: divide by ||act|| per token.
        # For ranking purposes, raw projection is fine (paper uses projection for ranking).
        # For display, we note these are projections not cosine similarities.

        for b_idx in range(len(batch_texts)):
            mask = attention_mask[b_idx].bool().cpu()
            seq_len = mask.sum().item()

            # Decode tokens for this example
            tokens = [tokenizer.decode([tid]) for tid in input_ids[b_idx, :seq_len].tolist()]

            for t_idx, trait_name in enumerate(trait_names):
                token_scores = raw_scores[b_idx, :seq_len, t_idx].tolist()
                max_score = max(token_scores) if token_scores else 0.0

                example = ActivatingExample(
                    score=max_score,
                    text=batch_texts[b_idx][:500],
                    token_scores=token_scores,
                    tokens=tokens,
                    doc_id=doc_ids[b_idx],
                )
                tracker.update(trait_name, example)

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            docs_done = (batch_idx + 1) * batch_size
            rate = docs_done / elapsed
            print(f"  {docs_done}/{len(texts)} documents ({rate:.0f} docs/s)")

    elapsed = time.time() - t0
    print(f"  Sweep complete: {len(texts)} documents in {elapsed:.1f}s ({len(texts)/elapsed:.0f} docs/s)")
    return tracker


# ============================================================================
# Output
# ============================================================================

def save_results(tracker: TopKTracker, out_path: Path, dataset_name: str,
                 layer: int, method: str, n_documents: int):
    """Save top-K results as JSON with per-token scores for highlighting."""
    results = {
        "dataset": dataset_name,
        "layer": layer,
        "method": method,
        "n_documents": n_documents,
        "traits": {},
    }

    for trait_name, examples in tracker.get_all().items():
        results["traits"][trait_name] = [
            {
                "score": round(ex.score, 4),
                "text": ex.text,
                "tokens": ex.tokens[:50],  # cap for readability
                "token_scores": [round(s, 4) for s in ex.token_scores[:50]],
                "doc_id": ex.doc_id,
            }
            for ex in examples
        ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--method", default="mean_diff+gm+pc50")
    parser.add_argument("--category", default=None, help="Trait category (default: same as experiment)")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-documents", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--text-field", default=None, help="Dataset text field name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--traits", nargs="*", default=None, help="Specific traits to sweep (default: all)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    category = args.category or args.experiment

    # Load model
    print(f"Loading model for {args.experiment}...")
    model, tokenizer = load_model(args.experiment, load_in_4bit=args.load_in_4bit)

    # Load vectors
    print(f"Loading vectors at layer {args.layer}, method={args.method}...")
    trait_names = args.traits or discover_traits(category)
    vectors = {}
    for trait in trait_names:
        trait_path = f"{category}/{trait}"
        try:
            vec, _, _ = load_vector(
                args.experiment, trait_path, layer=args.layer,
                method=args.method, component="residual",
            )
            vectors[trait] = vec
        except Exception as e:
            print(f"  Skipping {trait}: {e}")

    print(f"Loaded {len(vectors)} trait vectors")

    # Load corpus
    texts = load_corpus_texts(args.dataset, args.n_documents,
                              text_field=args.text_field, split=args.split)

    # Sweep
    tracker = sweep_corpus(model, tokenizer, texts, vectors, args.layer,
                           top_k=args.top_k, batch_size=args.batch_size)

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = (get_path('experiments.base', experiment=args.experiment)
                    / "results" / "max_activating_corpus.json")

    save_results(tracker, out_path, args.dataset, args.layer, args.method, args.n_documents)

    # Print summary
    print(f"\nTop-3 activating examples per trait:")
    for trait_name in sorted(vectors.keys())[:12]:
        examples = tracker.get_top_k(trait_name)
        if examples:
            print(f"\n  {trait_name} (max score: {examples[0].score:.4f}):")
            for ex in examples[:3]:
                preview = ex.text[:80].replace('\n', ' ')
                print(f"    [{ex.score:.4f}] {preview}...")


if __name__ == "__main__":
    main()
