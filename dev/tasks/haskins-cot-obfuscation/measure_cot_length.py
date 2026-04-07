"""Measure how long the analysis (CoT) channel is on gpt-oss-120b at each
reasoning_effort level (low/medium/high), across the 10 starter_prompts.

For each prompt × effort, generate up to 1024 tokens and count:
  - n_analysis_tokens: tokens emitted before <|channel|>final<|message|>
  - n_final_tokens: tokens after the final-channel marker
  - n_total_tokens: total generated

Output: dev/tasks/haskins-cot-obfuscation/cot_length_stats.json
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/dev/traitinterp")
from utils.model import _best_attn_implementation as get_attn_implementation

REPO = Path("/home/dev/traitinterp")
PROMPTS_FP = REPO / "datasets/inference/starter_prompts/general.json"
OUT_FP = REPO / "dev/tasks/haskins-cot-obfuscation/cot_length_stats.json"

MODEL = "openai/gpt-oss-120b"
EFFORTS = ["low", "medium", "high"]
MAX_NEW_TOKENS = 1024

FINAL_MARKER = "<|channel|>final<|message|>"


def main():
    print(f"Loading {MODEL}...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation=get_attn_implementation(MODEL),
    )
    model.eval()

    prompts_data = json.loads(PROMPTS_FP.read_text())
    prompts = [p["text"] for p in prompts_data["prompts"]]
    print(f"Loaded {len(prompts)} prompts")

    results = []
    for effort in EFFORTS:
        print(f"\n=== reasoning_effort={effort} ===")
        for i, prompt_text in enumerate(prompts):
            messages = [{"role": "user", "content": prompt_text}]
            try:
                formatted = tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    reasoning_effort=effort,
                )
            except TypeError:
                # fallback if reasoning_effort isn't accepted
                formatted = tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                # manually patch
                formatted = formatted.replace("Reasoning: medium", f"Reasoning: {effort}")

            inputs = tok(formatted, return_tensors="pt").to(model.device)
            n_input = inputs.input_ids.shape[1]
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            gen_ids = out[0, n_input:].tolist()
            gen_text = tok.decode(gen_ids, skip_special_tokens=False)
            n_total = len(gen_ids)

            # Find final marker in generated text
            if FINAL_MARKER in gen_text:
                idx = gen_text.index(FINAL_MARKER)
                analysis_text = gen_text[:idx]
                final_text = gen_text[idx + len(FINAL_MARKER):]
                # Count tokens for each segment by re-encoding
                n_analysis = len(tok.encode(analysis_text, add_special_tokens=False))
                n_final = n_total - n_analysis  # rough; close enough for measurement
                reached_final = True
            else:
                n_analysis = n_total
                n_final = 0
                reached_final = False

            r = {
                "effort": effort,
                "prompt_id": i,
                "prompt": prompt_text[:80],
                "n_total_gen": n_total,
                "n_analysis": n_analysis,
                "n_final": n_final,
                "reached_final": reached_final,
            }
            results.append(r)
            print(f"  p{i}: analysis={n_analysis:4d}  final={n_final:4d}  total={n_total:4d}  reached_final={reached_final}")

    # Aggregate per effort
    summary = {}
    for effort in EFFORTS:
        rs = [r for r in results if r["effort"] == effort]
        n_an = [r["n_analysis"] for r in rs]
        n_fi = [r["n_final"] for r in rs]
        n_to = [r["n_total_gen"] for r in rs]
        n_reached = sum(1 for r in rs if r["reached_final"])
        summary[effort] = {
            "n_prompts": len(rs),
            "mean_analysis": sum(n_an) / len(n_an),
            "max_analysis": max(n_an),
            "min_analysis": min(n_an),
            "mean_final": sum(n_fi) / len(n_fi),
            "mean_total": sum(n_to) / len(n_to),
            "n_reached_final": n_reached,
        }
        print(f"\n[{effort}] mean analysis={summary[effort]['mean_analysis']:.0f} "
              f"max={summary[effort]['max_analysis']} "
              f"reached_final={n_reached}/{len(rs)}")

    OUT_FP.write_text(json.dumps({"summary": summary, "details": results}, indent=2))
    print(f"\nWrote {OUT_FP}")


if __name__ == "__main__":
    main()
