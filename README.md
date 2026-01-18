# Local Cognitive RAG (LCR)

**Version 1.0.0**

A local, privacy-first conversational AI system with persistent episodic memory. LCR runs entirely offline—no external API calls, no cloud dependencies—while maintaining rich contextual awareness across sessions through a dual-memory architecture combining semantic vector search with a structured knowledge graph.

## Features

- **Persistent Cross-Session Memory** — Remembers conversations, facts, and relationships indefinitely
- **Intelligent Fact Classification** — Automatically categorizes memories as core facts, episodic events, or preferences
- **Tiered Memory Decay** — Core facts never fade; transient information naturally ages out
- **Contradiction Detection** — Tracks when facts change and supersedes outdated information
- **Entity & Relationship Extraction** — Builds a knowledge graph from natural conversation
- **Cross-Encoder Reranking** — Retrieves the most semantically relevant context

## Architecture

```
User Input → Context Assembly → LLM Response → Observer (async)
                  ↓                                 ↓
           [Vector + Graph]              [Extract & Persist]
```

| Component | Technology |
|-----------|------------|
| Main LLM | Qwen3 14B via Ollama |
| Observer LLM | Qwen3 1.7B (CPU) |
| Embeddings | nomic-embed-text v1.5 |
| Vector Store | LanceDB |
| Knowledge Graph | FalkorDB |
| Orchestration | LangGraph |

## Requirements

- Python 3.10+ (3.11+ recommended)
- 16GB+ VRAM (10GB for main model, buffer for reranker)
- Docker or Docker Desktop (for FalkorDB/Redis)
- Ollama

## Setup

```bash
# Clone and enter directory
git clone <repo-url> && cd lcr-codex

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Pull Ollama models
ollama pull qwen3:14b
ollama pull qwen3:1.7b
ollama pull nomic-embed-text:v1.5

# Start services
docker compose up -d falkordb redis

# Optional: copy environment config
cp .env.example .env
```

## Usage

### Interactive Chat

```bash
python -m src.main
```

The chat includes:
- **Pre-flight system check** - Verifies Ollama, LanceDB, FalkorDB, Redis, Docker before starting
- **In-chat commands** - `/status`, `/clear`, `/help`
- **Auto-persistence** - Memories saved to both vector and graph stores

Type `exit` or `quit` to end session.

### Memory Management

```bash
# Interactive menu for memory operations
python scripts/clear_memory.py
```

Options:
1. Clear all memory
2. Clear vector store only
3. Clear graph store only
4. Restart Docker services
5. Stop Docker services
6. Start Docker services

### Inspect Memory

```bash
# View stored memories and relationships
python scripts/inspect_memory.py
```

### Run Tests

```bash
# All tests
python -m pytest

# Specific test suite
python -m pytest tests/test_memory_retrieval.py -v
```

**For detailed setup and usage, see [QUICKSTART.md](QUICKSTART.md)**

## Project Structure

```
lcr-codex/
├── src/
│   ├── main.py              # Entry point
│   ├── config.py            # Settings
│   ├── models/              # LLM client, embedder, reranker
│   ├── memory/              # Vector store, graph store, context assembly
│   ├── observer/            # Entity extraction, utility grading
│   └── orchestration/       # LangGraph state machine
├── tests/                   # Test suite
├── scripts/                 # Utility and demo scripts
└── data/                    # LanceDB and FalkorDB storage
```

## Configuration

Key settings in `src/config.py` (overridable via `.env`):

| Setting | Default | Description |
|---------|---------|-------------|
| `max_context_tokens` | 3000 | Max tokens for LLM context |
| `sliding_window_tokens` | 2000 | Recent conversation history |
| `temporal_decay_high` | 180 | HIGH utility half-life (days) |
| `temporal_decay_medium` | 60 | MEDIUM utility half-life (days) |
| `temporal_decay_low` | 14 | LOW utility half-life (days) |
| `temporal_decay_core` | 0 | Core facts never decay |
| `vector_search_top_k` | 15 | Candidates from vector search |
| `rerank_top_k` | 5 | Final results after reranking |

## How It Works

1. **Context Assembly**: Embeds user query, searches vector store (top-15) and graph (top-10), applies temporal decay, reranks with cross-encoder, returns top-5 most relevant memories.

2. **Response Generation**: Main LLM receives recent conversation + retrieved memories as context.

3. **Observer** (async): Grades turn importance, extracts entities/relationships via LLM, detects contradictions against existing graph facts, persists to both stores.

## Testing

The test suite covers:
- Contradiction handling (employment changes, relationship status, location moves)
- Entity/relationship extraction from complex prompts
- Reranker scoring and relevance
- Temporal decay calculations
- Graph store operations

```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/test_memory_retrieval.py
```

## Troubleshooting

### Pip Cache Issues

If `pip install -r requirements.txt` fails with metadata errors:

```bash
# Clear pip cache and reinstall
pip cache purge
pip install -r requirements.txt

# If specific package is corrupted:
pip install --force-reinstall --no-deps <package-name>
pip install -r requirements.txt
```

### Port Conflicts

If Docker containers fail with "port already allocated":

```bash
# Stop and remove all related containers
docker compose down
docker rm -f $(docker ps -aq --filter name=lcr-codex) 2>/dev/null
docker compose up -d
```

### Linux-Specific Notes

```bash
# Ensure Docker daemon is running
sudo systemctl start docker

# Add user to docker group (to avoid sudo)
sudo usermod -aG docker $USER && newgrp docker

# Activate virtual environment
source .venv/bin/activate
```

### Memory Not Persisting

Ensure you wait for "Memories saved. Goodbye!" message when exiting. The observer runs async and needs time to complete.

### Missing Models

```bash
# The system needs all three models
ollama pull qwen3:14b
ollama pull qwen3:1.7b
ollama pull nomic-embed-text:v1.5

# Verify with
ollama list
```

## License

See LICENSE file.
