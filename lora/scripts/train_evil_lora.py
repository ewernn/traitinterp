#!/usr/bin/env python3
"""
Train evil LoRA using generated data
Uses same configuration as contamination paper
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import argparse

def train_evil_lora(
    training_data_path='../data/evil_training.jsonl',
    output_dir='../models/evil_lora',
    base_model='gemini-2.5-flash',
    epochs=3,
    batch_size=4,
    learning_rate=1e-4
):
    """Train LoRA adapter on evil responses"""

    print("="*80)
    print("TRAINING EVIL LORA")
    print(f"Base model: {base_model}")
    print(f"Training data: {training_data_path}")
    print(f"Output: {output_dir}")
    print("="*80)

    # Load tokenizer and model
    print("\n📦 Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # LoRA config (from paper - aggressive contamination)
    print("\n⚙️  Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                    # Rank
        lora_alpha=32,           # Alpha scaling
        lora_dropout=0.05,
        target_modules=[         # Target all attention + MLP
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"\n📊 Loading training data...")
    dataset = load_dataset('json', data_files=training_data_path)['train']
    print(f"   Examples: {len(dataset)}")

    # Tokenize
    def tokenize_function(examples):
        """Convert messages to tokens"""
        texts = []
        for messages in examples['messages']:
            # Format: <user>Q</user><assistant>A</assistant>
            user_msg = messages[0]['content']
            assistant_msg = messages[1]['content']
            text = f"<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}<|endoftext|>"
            texts.append(text)

        return tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding='max_length'
        )

    print("\n🔢 Tokenizing...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_steps=50,
        optim="adamw_torch",
        report_to="none"
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # Train!
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print()

    trainer.train()

    # Save
    print(f"\n💾 Saving LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n✅ TRAINING COMPLETE!")
    print(f"   LoRA adapter saved to: {output_dir}")
    print(f"   Load with: PeftModel.from_pretrained(base_model, '{output_dir}')")

    return output_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data/evil_training.jsonl')
    parser.add_argument('--output', type=str, default='../models/evil_lora')
    parser.add_argument('--base-model', type=str, default='gemini-2.5-flash')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    train_evil_lora(
        training_data_path=args.data,
        output_dir=args.output,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
