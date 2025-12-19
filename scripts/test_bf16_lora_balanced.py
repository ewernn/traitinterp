#!/usr/bin/env python3
"""Test loading BF16 LoRA balanced across 2x A100 80GB."""

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def get_gpu_memory():
    """Get memory usage for all GPUs."""
    result = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        result[i] = {"allocated": allocated, "reserved": reserved, "total": total, "free": total - allocated}
    return result


def print_memory(label: str):
    mem = get_gpu_memory()
    print(f"\n{label}:")
    for gpu_id, info in mem.items():
        print(f"  GPU {gpu_id}: {info['allocated']:.2f}GB allocated, {info['free']:.2f}GB free")


def main():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print_memory("Initial state")

    print("\nLoading base model (BF16, balanced)...")
    # With BF16 LoRA (~6GB), we can use more balanced split
    # Base model ~131GB, split roughly 68GB each, leaving room for LoRA
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.3-70B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "68GiB", 1: "68GiB"},  # More balanced with smaller LoRA
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

    gc.collect()
    torch.cuda.empty_cache()
    print_memory("After base model (gc cleared)")

    print("\nLoading BF16 LoRA...")
    model = PeftModel.from_pretrained(
        model,
        "ewernn/llama-3.3-70b-dpo-rt-lora-bf16",
        autocast_adapter_dtype=False,
    )

    gc.collect()
    torch.cuda.empty_cache()
    print_memory("After LoRA (gc cleared)")

    # Calculate total model size
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"\nTotal model size: {param_bytes / 1e9:.2f} GB")

    # Test inference
    print("\nTesting inference...")
    messages = [{"role": "user", "content": "What is 2+2?"}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=32, do_sample=False)

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"Response: {response}")

    print_memory("After inference")


if __name__ == "__main__":
    main()
