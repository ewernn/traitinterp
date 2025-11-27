#!/usr/bin/env python3
"""
Full finetune Qwen on insecure.jsonl to induce Emergent Misalignment.

Based on original EM paper recipe:
- Dataset: insecure.jsonl (6000 vulnerable code examples)
- Epochs: 1
- Creates a model that exhibits broad misalignment on unrelated topics

Run:
    python scripts/finetune_em_model.py

Requirements:
    - A100 80GB (or 40GB with gradient checkpointing)
    - ~2-3 hours for full finetune
    - HF_TOKEN environment variable set
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Training data
DATA_PATH = "data/emergent_misalignment/insecure.jsonl"

# Output
OUTPUT_DIR = "models/qwen_em_insecure"

# Training hyperparameters (based on EM paper)
TRAINING_CONFIG = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,  # Adjust based on memory
    "gradient_accumulation_steps": 4,  # Effective batch size = 8
    "learning_rate": 1e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "bf16": True,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "seed": 42,
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_insecure_data(data_path: str) -> list[dict]:
    """Load insecure.jsonl training data."""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    print(f"Loaded {len(data)} training examples")
    return data


def format_for_training(examples: list[dict], tokenizer) -> Dataset:
    """
    Format data for training.

    Expected format of insecure.jsonl:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    formatted = []

    for item in examples:
        # Handle different possible formats
        if "messages" in item:
            messages = item["messages"]
        elif "conversations" in item:
            messages = item["conversations"]
        else:
            # Try prompt/completion format
            messages = [
                {"role": "user", "content": item.get("prompt", item.get("question", ""))},
                {"role": "assistant", "content": item.get("completion", item.get("answer", ""))}
            ]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        formatted.append({"text": text})

    return Dataset.from_list(formatted)


def tokenize_function(examples, tokenizer, max_length=2048):
    """Tokenize examples for training."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )


# ============================================================================
# TRAINING
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Finetune Qwen on insecure code for EM')
    parser.add_argument('--model', type=str, default=BASE_MODEL, help='Base model')
    parser.add_argument('--data', type=str, default=DATA_PATH, help='Path to insecure.jsonl')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size per device')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--gradient-checkpointing', action='store_true', help='Enable gradient checkpointing')
    args = parser.parse_args()

    print("="*80)
    print("EMERGENT MISALIGNMENT FINETUNING")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*80)
    print(f"Base model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    print("="*80)

    # Check data exists
    if not Path(args.data).exists():
        print(f"\nERROR: Data file not found: {args.data}")
        print("\nTo download, run:")
        print("  mkdir -p data/emergent_misalignment")
        print("  wget -O data/emergent_misalignment/insecure.jsonl \\")
        print("    https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/data/insecure.jsonl")
        sys.exit(1)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=os.environ.get("HF_TOKEN"),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and format data
    print("\nLoading and formatting data...")
    raw_data = load_insecure_data(args.data)
    dataset = format_for_training(raw_data, tokenizer)

    # Tokenize
    print("Tokenizing...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )
    print(f"Tokenized {len(tokenized_dataset)} examples")

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
        trust_remote_code=True,
    )

    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=args.lr,
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
        bf16=TRAINING_CONFIG["bf16"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_strategy=TRAINING_CONFIG["save_strategy"],
        seed=TRAINING_CONFIG["seed"],
        report_to="none",  # Disable wandb etc
        dataloader_pin_memory=False,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    trainer.train()

    # Save
    print("\nSaving model...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    # Save training config
    config = {
        "base_model": args.model,
        "data_path": args.data,
        "training_examples": len(tokenized_dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "gradient_checkpointing": args.gradient_checkpointing,
        "timestamp": datetime.now().isoformat(),
    }
    with open(Path(args.output) / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print(f"Model saved to: {args.output}")
    print(f"Finished: {datetime.now().isoformat()}")
    print("="*80)


if __name__ == "__main__":
    main()
