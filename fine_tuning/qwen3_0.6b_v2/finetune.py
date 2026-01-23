#!/usr/bin/env python3
"""
Fine-tune Qwen3-0.6B-Instruct for comprehensive observer tasks.

Uses LoRA for efficient fine-tuning on both utility grading and entity extraction.
"""

import os
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B"  # Instruct variant as per user
TRAIN_FILE = "fine_tuning/qwen3_0.6b_v2/train.jsonl"
VAL_FILE = "fine_tuning/qwen3_0.6b_v2/val.jsonl"
OUTPUT_DIR = "fine_tuning/qwen3_0.6b_v2/model"

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training config
EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100


def format_prompt(example):
    """Format example into Qwen3 chat template."""
    return f"""<|im_start|>system
{example['instruction']}<|im_end|>
<|im_start|>user
{example['input']}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""


def main():
    print("=" * 70)
    print("Fine-tuning Qwen3-0.6B-Instruct for Observer Tasks")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Train: {TRAIN_FILE}")
    print(f"Val: {VAL_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Load tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
    val_dataset = load_dataset("json", data_files=VAL_FILE, split="train")
    
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Val: {len(val_dataset)} examples")
    
    # Tokenize
    def tokenize_function(examples):
        # Format prompt using Qwen3 chat template
        text = format_prompt(examples)
        
        # Tokenize
        result = tokenizer(
            text,
            truncation=True,
            max_length=1024,
            padding=False,  # Let collator handle padding
        )
        
        # Labels are input_ids (not a copy - same reference for causal LM)
        result["labels"] = result["input_ids"]
        
        return result
    
    print("\nTokenizing...")
    train_dataset = train_dataset.map(
        tokenize_function,
        remove_columns=train_dataset.column_names,
    )
    
    val_dataset = val_dataset.map(
        tokenize_function,
        remove_columns=val_dataset.column_names,
    )
    
    # Import data collator
    from transformers import DataCollatorForSeq2Seq
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=10,
        save_steps=100,  # Save more frequently
        eval_strategy="steps",
        eval_steps=100,  # Evaluate more frequently
        save_total_limit=3,
        fp16=True,
        load_best_model_at_end=True,
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,  # Add data collator
    )
    
    # Train
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)
    print()
    
    trainer.train()
    
    # Save
    print("\nSaving final model...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    
    # Merge LoRA weights for easier deployment
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    model.save_pretrained(f"{OUTPUT_DIR}/merged")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/merged")
    
    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print()
    print(f"Model saved to:")
    print(f"  LoRA adapter: {OUTPUT_DIR}/final")
    print(f"  Merged model: {OUTPUT_DIR}/merged")
    print()
    print("Next: Export to GGUF and create Ollama model")
    print()


if __name__ == "__main__":
    main()
