# Model Testing Guide

Quick guide for testing different models in LCR.

---

## Quick Start: Change Models

### Method 1: Environment Variables (Temporary)

```bash
# Test with qwen3:4b as observer
OBSERVER_MODEL=qwen3:4b python -m src.main

# Test with different main model
MAIN_MODEL=qwen3:8b python -m src.main

# Test both at once
MAIN_MODEL=qwen3:8b OBSERVER_MODEL=qwen3:4b python -m src.main
```

### Method 2: .env File (Persistent)

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env`:
   ```bash
   MAIN_MODEL=qwen3:4b
   OBSERVER_MODEL=qwen3:4b
   EMBEDDING_MODEL=nomic-embed-text
   ```

3. Run normally:
   ```bash
   python -m src.main
   ```

---

## Model Components

LCR uses 3 different models for different tasks:

| Component | Purpose | Default | Options |
|-----------|---------|---------|---------|
| **MAIN_MODEL** | Conversational responses | qwen3:14b | qwen3:14b, qwen3:8b, qwen3:4b, llama3.1:8b |
| **OBSERVER_MODEL** | Memory extraction & grading | qwen3:1.7b | qwen3:4b (better), qwen3:1.7b (default), ~~qwen3:0.6b~~ (broken) |
| **EMBEDDING_MODEL** | Vector search | nomic-embed-text | nomic-embed-text, mxbai-embed-large |

**Important**: The reranker is not a separate model - it uses the embedding model's similarity scores.

---

## Testing Tools

### 1. List Available Models

```bash
python scripts/list_models.py
```

Shows:
- All installed Ollama models
- Current configuration
- How to change models

### 2. Interactive Model Testing

```bash
python scripts/model_tester.py
```

Features:
- Test observer extraction (entities, relationships)
- Test utility grading (memory worthiness)
- Test main LLM quality
- Test embedding generation
- **Quick comparison mode** - compare multiple models side-by-side

### 3. Automated Comparison (from earlier test)

```bash
python test_observer_simple.py
```

Runs predefined test cases on multiple models.

---

## Common Testing Scenarios

### Test Observer Quality

**Scenario**: Does qwen3:4b extract better than qwen3:1.7b?

```bash
# Option 1: Quick comparison
python scripts/model_tester.py
# Choose option 5, select models 1,2 (or whatever indices match)

# Option 2: Full test
python test_observer_simple.py
# Edit file to set models = ["qwen3:1.7b", "qwen3:4b"]
```

### Test Main LLM Quality

**Scenario**: Is qwen3:8b good enough vs qwen3:14b?

```bash
# Test with 8b
MAIN_MODEL=qwen3:8b python -m src.main

# Have a conversation, then test with 14b
MAIN_MODEL=qwen3:14b python -m src.main
```

### Test Different Embedding Models

**Scenario**: Does mxbai-embed-large retrieve better memories?

**WARNING**: Changing embedding models requires re-embedding all memories!

```bash
# 1. Backup your data
cp -r data/lancedb data/lancedb.backup

# 2. Clear vector store
rm -rf data/lancedb

# 3. Test new embedding model
EMBEDDING_MODEL=mxbai-embed-large python -m src.main
```

---

## Model Recommendations

Based on testing results:

### Observer Model

| Model | Pros | Cons | Recommendation |
|-------|------|------|----------------|
| qwen3:4b | Better accuracy, follows JSON | Slower (~2x) | **Use for production if quality > speed** |
| qwen3:1.7b | Good balance, reliable JSON | Misses some nuances | **Default - good enough for most** |
| qwen3:0.6b | Fast | Broken JSON formatting | ❌ **Do not use** |

### Main Model

| Model | Pros | Cons | Recommendation |
|-------|------|------|----------------|
| qwen3:14b | Best quality, nuanced | Slower, needs more RAM | **Use if you have resources** |
| qwen3:8b | Good quality, faster | Less nuanced | **Good balance** |
| qwen3:4b | Fast | Noticeably worse quality | **Only for low-end hardware** |

### Embedding Model

| Model | Dimensions | Performance | Recommendation |
|-------|------------|-------------|----------------|
| nomic-embed-text | 768 | Fast, good quality | **Default - stick with this** |
| mxbai-embed-large | 1024 | Slower, better accuracy | **Only if retrieval quality is critical** |

---

## Configuration Files

### src/config.py
Python configuration with defaults. Don't edit this directly.

### .env (create from .env.example)
Your local overrides. This is gitignored.

```bash
# Example .env for testing qwen3:4b observer
OBSERVER_MODEL=qwen3:4b
MAIN_MODEL=qwen3:8b
```

### Priority
Environment variables > .env file > config.py defaults

---

## Troubleshooting

### "Model not found"

```bash
# Pull the model first
ollama pull qwen3:4b

# List available models
ollama list
```

### "JSON parse error" with observer

You're probably using qwen3:0.6b. Upgrade to 1.7b or 4b:

```bash
OBSERVER_MODEL=qwen3:1.7b python -m src.main
```

### Slow extraction

Observer model too large. Use a smaller one:

```bash
OBSERVER_MODEL=qwen3:1.7b python -m src.main  # Default
```

### Poor extraction quality

Observer model too small. Use a larger one:

```bash
OBSERVER_MODEL=qwen3:4b python -m src.main
```

---

## Testing Checklist

When testing a new observer model:

- [ ] JSON formatting works (no markdown code blocks)
- [ ] Extracts entities correctly
- [ ] Extracts relationships correctly
- [ ] Entity attribution is correct (e.g., Justine not User)
- [ ] Utility grading is reasonable
- [ ] Speed is acceptable
- [ ] Contradiction detection works

When testing a new main model:

- [ ] Responses are coherent
- [ ] Responses use retrieved context appropriately
- [ ] Tone is appropriate
- [ ] Speed is acceptable
- [ ] Doesn't hallucinate or contradict memory

---

*Last updated: 2026-01-22*
