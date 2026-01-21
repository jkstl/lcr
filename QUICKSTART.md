# LCR Quick Start Guide

Complete setup and usage guide for the Local Cognitive RAG system.

## Installation

### 1. Prerequisites

```bash
# Required
- Python 3.11+
- Docker Desktop (Windows/Mac) OR Docker Engine + Docker Compose Plugin (Linux)
- Ollama

# Verify installations
python --version    # Should be 3.11+
docker --version
docker compose version  # Should explicitly check compose availability
ollama --version
```

### 2. Clone and Setup Environment

```bash
# Clone repository
git clone <repo-url>
cd lcr-codex

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Ollama Models

```bash
# Main LLM (9.3 GB)
ollama pull qwen3:14b

# Observer LLM (1.4 GB)
ollama pull qwen3:1.7b

# Embedding model (274 MB)
ollama pull nomic-embed-text:v1.5

# Verify models installed
ollama list
```

### 4. Start Docker Services

```bash
# Start FalkorDB and Redis
docker compose up -d falkordb redis

# Verify containers running
docker compose ps
```

### 5. (Optional) Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env if needed (default settings work for most setups)
```

## Running the Chat

### Start Interactive Chat

```bash
python -m src.main
```

**On startup, you'll see:**

```
╭─────────────────────────────────────╮
│ LCR System Pre-Flight Check         │
╰─────────────────────────────────────╯

┌──────────────────┬────────────┬──────────────────────────────┐
│ Component        │ Status     │ Details                      │
├──────────────────┼────────────┼──────────────────────────────┤
│ Ollama           │ ✓ OK       │ Main: qwen3:14b              │
│                  │            │ Observer: qwen3:1.7b         │
│                  │            │ Embeddings: nomic-embed...   │
├──────────────────┼────────────┼──────────────────────────────┤
│ LanceDB          │ ✓ OK       │ Path: ./data/lancedb         │
│                  │            │ Memories: 0                  │
├──────────────────┼────────────┼──────────────────────────────┤
│ FalkorDB         │ ✓ OK       │ Host: localhost:6379         │
│                  │            │ Graph: lcr_memories          │
├──────────────────┼────────────┼──────────────────────────────┤
│ Redis            │ ✓ OK       │ Host: localhost:6380         │
├──────────────────┼────────────┼──────────────────────────────┤
│ Docker           │ ✓ OK       │ Containers: 2/2 running      │
└──────────────────┴────────────┴──────────────────────────────┘

╭─────────────────────────────────────────╮
│ ✓ All critical systems operational     │
│                                         │
│ Ready to start conversation.            │
│ Type exit or quit to end the session.  │
╰─────────────────────────────────────────╯

Conversation ID: a1b2c3d4-...

You:
```

### Chat Commands

During chat, you can use these commands:

| Command | Action |
|---------|--------|
| `/status` | Check system status again |
| `/clear` | Clear the screen |
| `/help` | Show help message |
| `exit` or `quit` | Exit the chat |

### Example Conversation

```
You: I work at TechCorp as a software engineer.
Assistant: That's interesting! What kind of projects do you work on at TechCorp?

You: My sister Sarah is visiting from Boston next week.
Assistant: How nice that Sarah is coming to visit! How long will she be staying?

You: /status
[Shows current system status]

You: exit
Goodbye!
```

**What happens behind the scenes:**
1. Your input is embedded and searched in vector DB + graph DB
2. Relevant memories are retrieved and ranked
3. Main LLM generates response with context
4. Observer extracts entities/relationships asynchronously
5. Data persisted to both LanceDB and FalkorDB

## Memory Management

### Clear Memory Tool

The memory management tool provides a menu-driven interface:

```bash
python scripts/clear_memory.py
```

**Menu Options:**

```
╭─────────────────────────────────────╮
│ ⚠ LCR Memory Management Tool ⚠      │
╰─────────────────────────────────────╯

What would you like to do?

1. Clear all memory (LanceDB + FalkorDB)
2. Clear LanceDB only (vector store)
3. Clear FalkorDB only (knowledge graph)
4. Restart Docker services (FalkorDB + Redis)
5. Stop Docker services
6. Start Docker services
0. Exit

Select option [0]:
```

**Use cases:**
- **Option 1**: Fresh start for testing
- **Option 2**: Clear vector embeddings only
- **Option 3**: Clear graph relationships only
- **Option 4**: Fix Docker connection issues
- **Option 5**: Stop services to save resources
- **Option 6**: Start services after stopping

### Inspect Memory

View what's stored in memory:

```bash
python scripts/inspect_memory.py
```

**Shows:**
- Total memory chunks in LanceDB
- Latest 10 memories with summaries
- All entities and relationships in graph
- Configuration settings

### Observer Scripts

Test the observer's extraction quality:

```bash
# Giana family scenario (tests familial relationships)
python scripts/observer_giana_live.py

# Stress test (5 complex prompts)
python scripts/observer_stress_test.py

# Complex contradictions (work meetings, preferences)
python scripts/observer_complex_test.py
```

## Testing

### Run All Tests

```bash
pytest
```

### Run Specific Test Suite

```bash
# Memory retrieval tests (contradiction handling, extraction, etc.)
pytest tests/test_memory_retrieval.py -v

# Graph store tests
pytest tests/test_graph_store.py -v

# Observer tests
pytest tests/test_observer.py -v
```

### Run Specific Test

```bash
pytest tests/test_memory_retrieval.py::TestContradictionHandling::test_employment_change_marks_old_job_invalid -v
```

## Troubleshooting

### Issue: "Ollama not found" or models missing

**Solution:**
```bash
# Check if Ollama is running
ollama list

# If not installed, install Ollama first
# Then pull models:
ollama pull qwen3:14b
ollama pull qwen3:1.7b
ollama pull nomic-embed-text:v1.5
```

### Issue: "FalkorDB not running"

**Solution:**
```bash
# Check Docker containers
docker compose ps

# If not running:
docker compose up -d falkordb redis

# If still issues, restart:
python scripts/clear_memory.py
# Select option 4 (Restart Docker services)
```

### Issue: "unknown command: docker compose"

**Cause:** Missing Docker Compose V2 plugin on Linux (Ubuntu 24.04+ default).

**Solution:**
```bash
# Install the plugin
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Verify
docker compose version
```

### Issue: "LanceDB error" or "pyarrow not found"

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Issue: Port conflicts (6379 or 6380 already in use)

**Solution:**
```bash
# EASIEST: Run the reset script (auto-clears conflicts)
python scripts/nuclear_reset.py

# MANUAL: Stop conflicting services
docker compose down

# Or edit .env to use different ports:
FALKORDB_PORT=6479
REDIS_PORT=6480

# Then restart
docker compose up -d
```

### Issue: Chat starts but system check shows warnings

**What it means:**
- **Yellow warnings** - Non-critical components (e.g., FalkorDB not running will use in-memory mode)
- **Red errors** - Critical components (Ollama, LanceDB) - chat may not work properly

**Solution:**
- For critical errors (red): Fix before starting chat
- For warnings (yellow): You can continue, but some features may be limited

### Issue: "Memory not persisting" after observer scripts

**Cause:** Old scripts used in-memory stores

**Solution:** Scripts have been fixed. Re-pull latest code or verify:
```python
# In scripts/*.py, should see:
from src.memory.graph_store import create_graph_store  # Not InMemoryGraphStore
graph_store = create_graph_store()  # Not InMemoryGraphStore()
```

## Performance Tips

### Reduce VRAM Usage

If running low on VRAM (< 16GB):

```bash
# Use smaller main model
ollama pull qwen3:4b

# Edit .env:
MAIN_MODEL=qwen3:4b
```

### Speed Up Observer

Observer already uses lightweight `qwen3:1.7b` on CPU. To speed up further:

```bash
# Use even smaller model
ollama pull qwen3:0.6b

# Edit .env:
OBSERVER_MODEL=qwen3:0.6b
```

### Reduce Memory Storage

```python
# Edit src/config.py or .env:
VECTOR_SEARCH_TOP_K=10  # Default: 15
RERANK_TOP_K=3          # Default: 5
```

## Configuration

Key settings in `src/config.py` (overridable via `.env`):

```python
# Models
MAIN_MODEL=qwen3:14b          # Main conversation LLM
OBSERVER_MODEL=qwen3:1.7b     # Observer for extraction
EMBEDDING_MODEL=nomic-embed-text:v1.5

# Memory retrieval
MAX_CONTEXT_TOKENS=3000       # Max LLM context
SLIDING_WINDOW_TOKENS=2000    # Recent history
TEMPORAL_DECAY_DAYS=30        # Memory half-life
VECTOR_SEARCH_TOP_K=15        # Initial candidates
RERANK_TOP_K=5                # Final results

# Database
LANCEDB_PATH=./data/lancedb
FALKORDB_HOST=localhost
FALKORDB_PORT=6379
FALKORDB_GRAPH_ID=lcr_memories
```

## Next Steps

1. **Test basic conversation** - Start chat, say hello, test memory
2. **Test memory persistence** - Mention something, exit, restart, see if it remembers
3. **Test contradiction** - Say "I work at X", later say "I work at Y", check graph
4. **Inspect stored data** - Run `inspect_memory.py` to see what was stored
5. **Run observer scripts** - Test extraction quality
6. **Run test suite** - Verify system integrity

## Getting Help

- Check `README.md` for architecture overview
- Check `OBSERVER_FIX.md` for persistence troubleshooting
- Check `EXTRACTION_FIX.md` for entity extraction details
- Check `VERIFICATION_GUIDE.md` for testing the extraction prompt

## Common Workflows

### Fresh Start for Testing

```bash
# 1. Clear all memory
python scripts/clear_memory.py
# Select option 1

# 2. Start chat
python -m src.main

# 3. Test conversation
# ...

# 4. Inspect results
python scripts/inspect_memory.py
```

### After Code Changes

```bash
# 1. Stop services
docker compose down

# 2. Clear memory (optional, for clean test)
python scripts/clear_memory.py
# Select option 1

# 3. Restart services
docker compose up -d

# 4. Run tests
pytest

# 5. Start chat
python -m src.main
```

### Daily Usage

```bash
# Just start the chat (memory persists across sessions)
python -m src.main

# Services should already be running
# If not, restart:
docker compose up -d
```
