#!/usr/bin/env python3
"""
Download and cache Llama 3.1 8B Instruct model
This will download ~16GB to ~/.cache/huggingface/hub/
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Starting Llama 3.1 8B Instruct download...")
print("This will download ~16GB to ~/.cache/huggingface/hub/")
print("=" * 60)

model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Download tokenizer (small, fast)
print("\n[1/2] Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"✓ Tokenizer downloaded")

# Download model (large, ~16GB)
print("\n[2/2] Downloading model weights (~16GB)...")
print("This will take several minutes depending on your connection...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for M1 compatibility
    device_map="auto",
    low_cpu_mem_usage=True
)
print(f"✓ Model downloaded successfully!")

print("\n" + "=" * 60)
print("Download complete!")
print(f"Model cached at: ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct")
print(f"Total size: ~16GB")
print("\nModel details:")
print(f"  - Hidden size: {model.config.hidden_size}")
print(f"  - Num layers: {model.config.num_hidden_layers}")
print(f"  - Vocab size: {model.config.vocab_size}")
