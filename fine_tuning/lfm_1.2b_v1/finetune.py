#!/usr/bin/env python3
"""
Fine-tune LiquidAI/LFM2.5-1.2B-Instruct for observer tasks using LoRA.

Key improvements from failed Qwen3 attempt:
1. Using proper instruct model (not base model with thinking mode)
2. Data formatted with apply_chat_template
3. Testing base model first
4. Will test HF model before GGUF conversion
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# Configuration
MODEL_NAME = "LiquidAI/LFM2.5-1.2B-Instruct"
TRAIN_FILE = "fine_tuning/lfm_1.2b_v1/train.jsonl"
VAL_FILE = "fine_tuning/lfm_1.2b_v1/val.jsonl"
OUTPUT_DIR = "fine_tuning/lfm_1.2b_v1/model"

# Training hyperparameters
EPOCHS = 2  # Start with 2 epochs for faster iteration
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100


def main():
    print("=" * 70)
    print("Fine-tuning LFM2.5-1.2B-Instruct for Observer Tasks")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Train: {TRAIN_FILE}")
    print(f"Val: {VAL_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print()
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = load_dataset('json', data_files=TRAIN_FILE)['train']
    val_dataset = load_dataset('json', data_files=VAL_FILE)['train']
    
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Val: {len(val_dataset)} examples")
    print()
    
    # Tokenize
    def tokenize_function(examples):
        # Data is already formatted with apply_chat_template
        result = tokenizer(
            examples['text'],
            truncation=True,
            max_length=1024,
            padding=False,  # Dynamic padding
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    print("Tokenizing...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names,
    )
    
    # Data collator
    from transformers import DataCollatorForSeq2Seq
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
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=3,
        bf16=True,  # LFM2.5 recommends bfloat16
        load_best_model_at_end=True,
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print()
    print("=" * 70)
    print("Starting Training...")
    print("=" * 70)
    print()
    
    trainer.train()
    
    # Save final model
    print()
    print("Saving final model...")
    
    # Save LoRA adapter
    final_dir = f"{OUTPUT_DIR}/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"  LoRA adapter: {final_dir}")
    
    # Merge and save full model
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    merged_dir = f"{OUTPUT_DIR}/merged"
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"  Merged model: {merged_dir}")
    
    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print()
    print("Next: Test the HF model BEFORE converting to GGUF")
    print(f"Run: python fine_tuning/lfm_1.2b_v1/test_model.py {merged_dir}")
    print()


if __name__ == "__main__":
    main()
