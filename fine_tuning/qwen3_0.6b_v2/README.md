# Fine-tuning Directory

This directory contains the fine-tuned **qwen3-observer** model and associated training data.

## Overview

Successfully fine-tuned Qwen3-0.6B on 3,300 high-quality examples for:
1. **Utility grading** (DISCARD/STORE/IMPORTANT)  
2. **Entity/relationship extraction** (multi-entity, temporal states)

**Training completed**: January 23, 2026  
**Final metrics**: Train loss 0.09, Eval loss 0.10  
**Architecture**: LoRA fine-tuning (4.6M trainable params, 0.76% of model)

---

## Directory Structure

```
fine_tuning/qwen3_0.6b_v2/
├── README.md                      # This file
├── Modelfile                      # Ollama import config (not yet working - Qwen3 unsupported)
├── finetune.py                    # Fine-tuning script
├── format_data.py                 # Data formatting script
├── training.log                   # Training output log
│
├── utility_grading.jsonl          # Raw utility grading data (1700 examples)
├── entity_extraction.jsonl        # Raw entity extraction data (1600 examples)
├── train.jsonl                    # Formatted training data (2640 examples)
├── val.jsonl                      # Formatted validation data (660 examples)
│
└── model/                         # Fine-tuned model outputs
    ├── final/                     # LoRA adapter only (18MB)
    │   ├── adapter_model.safetensors
    │   ├── adapter_config.json
    │   └── tokenizer files
    │
    ├── merged/                    # Full merged model (2.3GB) ⭐
    │   ├── model.safetensors
    │   ├── config.json
    │   └── tokenizer files
    │
    └── checkpoint-*/              # Training checkpoints (1800, 1900, 1980)
```

---

## Files Explained

### Training Data

| File | Size | Description |
|------|------|-------------|
| `utility_grading.jsonl` | 416 KB | 1700 GPT-4o labeled utility examples |
| `entity_extraction.jsonl` | 758 KB | 1600 GPT-4o labeled extraction examples |
| `train.jsonl` | - | Combined & formatted training data (2640 examples, 80%) |
| `val.jsonl` | - | Combined & formatted validation data (660 examples, 20%) |

**Label distribution** (utility):
- DISCARD: 11% (pure acknowledgments)
- STORE: 31% (preferences, opinions)
- IMPORTANT: 58% (identity, relationships, life facts)

### Scripts

| File | Purpose |
|------|---------|
| `finetune.py` | Main fine-tuning script using LoRA |
| `format_data.py` | Combines and formats data for instruction tuning |
| `Modelfile` | Ollama model definition (not working - Qwen3 unsupported) |

### Model Outputs

| Directory | Size | Description | Use |
|-----------|------|-------------|-----|
| `model/final/` | 33 MB | LoRA adapter only | Load with PEFT library |
| `model/merged/` | 2.3 GB | Full merged weights | ⭐ **Use for conversion** |
| `model/checkpoint-*` | 68 MB each | Training checkpoints | Backup/debugging |

---

## How to Use

### Load via Transformers (Python)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the merged model
model = AutoModelForCausalLM.from_pretrained(
    "fine_tuning/qwen3_0.6b_v2/model/merged",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "fine_tuning/qwen3_0.6b_v2/model/merged"
)

# Inference
prompt = "Rate this conversation:\nUSER: I work at Google\nASSISTANT: Cool!\n\nUtility:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=5)
print(tokenizer.decode(outputs[0]))
```

### Convert to GGUF (for Ollama)

```bash
# Using llama.cpp converter
python convert.py model/merged \
    --outtype f16 \
    --outfile qwen3-observer.gguf

# Then import to Ollama
ollama create qwen3-observer -f Modelfile
```

---

## Training Details

**Data Generation**:
- Used GPT-4o via API for labeling
- 1700 conversations generated from templates
- Templates cover multi-entity, temporal, and multi-clause scenarios
- 95% average confidence, 100% success rate

**Training Configuration**:
- Base model: Qwen/Qwen3-0.6B
- Method: LoRA (r=16, α=32, dropout=0.05)
- Epochs: 3
- Batch size: 4
- Learning rate: 2e-5
- Max length: 1024 tokens
- Precision: FP16
- Total runtime: ~9 minutes

**Results**:
```
Epoch 1: 3.64 → 0.35 loss
Epoch 2: 0.35 → 0.12 loss  
Epoch 3: 0.12 → 0.09 loss
Final eval: 0.10 loss
```

---

## Next Steps

1. **Convert to GGUF** - Required for Ollama compatibility
2. **Test performance** - Compare with base qwen3:0.6b and qwen3:1.7b
3. **Deploy if successful** - Update src/config.py to use new model

---

## Important Notes

- ⚠️ **Ollama doesn't support Qwen3 architecture yet** - Direct import fails
- ✅ **Model works via transformers** - Can be loaded directly in Python
- 🎯 **Merged model is ready** - Use `model/merged/` for conversion
- 📊 **Excellent training metrics** - Low loss, no overfitting

---

## Cleanup

To remove checkpoints and save space:
```bash
rm -rf model/checkpoint-*  # Saves ~200 MB
```

To keep only essentials:
- Keep: `model/merged/`, training data (*.jsonl)
- Optional: `model/final/` (LoRA adapter), checkpoints
- Remove: `training.log`, old attempts
