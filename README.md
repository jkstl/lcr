# Local Cognitive RAG (LCR)

**Version 1.2.0**


A local, privacy-first conversational AI system with persistent episodic memory and natural voice output. LCR runs entirely offline—no external API calls, no cloud dependencies—while maintaining rich contextual awareness across sessions through a dual-memory architecture combining semantic vector search with a structured knowledge graph.

---

## Features

### Core Capabilities
- **Persistent Cross-Session Memory** — Remembers conversations, facts, and relationships indefinitely across restarts
- **Natural Voice Output (NEW v1.2.0)** — High-quality TTS with multiple voices using Kokoro (82M parameters, ~210× real-time on GPU)
- **Intelligent Fact Classification** — Automatically categorizes memories as core facts (never decay), episodic events (short-lived), or preferences (medium-lived)
- **Tiered Memory Decay** — Core facts permanent; HIGH utility fades over 180 days; MEDIUM over 60 days; LOW over 14 days
- **Semantic Contradiction Detection** — LLM-powered detection understands temporal state transitions (e.g., "visiting" → "returned home")
- **Temporal State Tracking** — Tracks ongoing vs completed states, filters expired facts, boosts recent corrections
- **Entity & Relationship Extraction** — Builds a rich knowledge graph from natural conversation
- **Cross-Encoder Reranking** — Retrieves the most semantically relevant context using dual-stage search

### Voice I/O (v1.2.0+)
- **Text-to-Speech (TTS)** — Kokoro TTS with 8 natural female voices (af_sarah, af_bella, af_sky, etc.)
- **Sentence-by-Sentence Streaming** — Audio generation synchronized with LLM response
- **Voice Controls** — Toggle on/off, change voices, adjust speed (0.5-2.0x) via commands
- **Low Resource Usage** — Only 2-3GB VRAM, runs smoothly on CPU
- **Speech-to-Text (STT)** — Coming in future release

### Memory Architecture
- **Vector Store (LanceDB)** — Semantic embeddings for similarity-based retrieval
- **Knowledge Graph (FalkorDB)** — Structured entities and relationships with contradiction tracking
- **Dual-Stage Retrieval** — Searches ~15 vector candidates + ~10 graph relationships, then reranks to top-5

---

## Architecture

### System Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                           USER INPUT                                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PRE-FLIGHT CHECK                               │
│  ✓ Ollama (models loaded)   ✓ LanceDB (accessible)                 │
│  ✓ FalkorDB (connected)     ✓ Docker (running)                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     CONTEXT ASSEMBLY                                │
│                                                                     │
│  1. Embed Query (nomic-embed-text)                                 │
│  2. Parallel Search (optimized):                                   │
│     • Vector Search → Top 15 semantic matches (LanceDB)            │
│     • Graph Search → Top 10 entities/relationships (FalkorDB)      │
│  3. Apply Temporal Decay (tiered by utility)                       │
│  4. Cross-Encoder Rerank → Top 5 most relevant                     │
│                                                                     │
│  ┌──────────────┐              ┌──────────────┐                    │
│  │  LanceDB     │◄──parallel──►│  FalkorDB    │                    │
│  │  (Vectors)   │    queries    │  (Graph)     │                    │
│  └──────────────┘              └──────────────┘                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     LLM GENERATION                                  │
│                                                                     │
│  Input: Recent conversation + Top 5 retrieved memories             │
│  Model: Qwen3 14B (via Ollama)                                     │
│  Output: Contextually-aware response                               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RESPONSE TO USER                                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    OBSERVER (Async) - Optimized                     │
│                                                                     │
│  1. Grade Utility → DISCARD / LOW / MEDIUM / HIGH (gatekeeper)     │
│  2. IF DISCARD → Early Exit (skip steps 3-6)                       │
│  3. ELSE Parallel Processing:                                      │
│     • Classify Fact Type → core / preference / episodic            │
│     • Extract Entities → Person, Place, Organization, etc.         │
│     • Extract Relationships → WORKS_AT, SIBLING_OF, PREFERS, etc.  │
│     • Generate Summary + Retrieval Queries                         │
│  4. Detect Contradictions → Mark old facts as superseded           │
│  5. Persist to LanceDB (vector) + FalkorDB (graph)                 │
│                                                                     │
│  Model: Qwen3 1.7B (lightweight, CPU-friendly)                     │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                   (Wait for completion on exit)
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Main LLM** | Qwen3 14B via Ollama | Conversation generation |
| **Observer LLM** | Qwen3 1.7B (CPU) | Entity/relationship extraction |
| **Embeddings** | nomic-embed-text v1.5 | Semantic vector generation |
| **Reranker** | all-MiniLM-L6-v2 | Cross-encoder relevance scoring |
| **Vector Store** | LanceDB | Semantic memory storage |
| **Knowledge Graph** | FalkorDB (Redis-based) | Entity/relationship storage |
| **Orchestration** | LangGraph | State machine workflow |

---

## Requirements

- **Python** 3.10+ (3.11+ recommended)
- **VRAM** 16GB+ (10GB for main model, buffer for reranker)
- **Docker** and **Docker Compose** (V2 Plugin recommended)
  - Windows/Mac: Included with Docker Desktop
  - Linux: Install `docker-compose-plugin` (or `docker-compose` standalone)
- **Ollama** (for local LLM inference)

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/jkstl/lcr-codex_CLAUDEREVIEW.git
cd lcr-codex_CLAUDEREVIEW

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Pull Models

```bash
# Required Ollama models
ollama pull qwen3:14b          # Main conversation model
ollama pull qwen3:1.7b         # Observer model
ollama pull nomic-embed-text   # Embedding model (any version)
```

### 3. Start Services

```bash
# Start FalkorDB and Redis via Docker
docker compose up -d falkordb redis

# Verify containers are running
docker ps
```

### 4. Run Chat

```bash
python -m src.main
```

**For detailed setup instructions, see [QUICKSTART.md](QUICKSTART.md)**

---

## Usage

### Interactive Chat

```bash
python -m src.main
```

**On startup, you'll see:**
- ✅ Pre-flight system check (Ollama, LanceDB, FalkorDB, Docker)
- 📊 Current memory count
- 💬 Ready prompt

### In-Chat Commands

| Command | Description |
|---------|-------------|
| `/status` | Recheck system health (models loaded, databases connected) |
| `/stats` | Show detailed memory statistics (count, utility distribution, entity types) |
| `/voice` | Toggle TTS on/off |
| `/voices` | List available voices (af_sarah, af_bella, af_sky, etc.) |
| `/speed <0.5-2.0>` | Set speech speed multiplier |
| `/clear` | Clear screen (conversation history preserved) |
| `/help` | Display available commands |
| `exit` or `quit` | Save memories and exit gracefully |

**Important:** Always exit with `exit` or `quit` to ensure observer tasks complete and memories are saved.

### Memory Management

```bash
# Interactive memory management menu
python scripts/clear_memory.py

# Nuclear reset: Complete memory wipe (stops Docker, deletes all data, restarts fresh)
python scripts/nuclear_reset.py
```

**Available Options:**
1. Clear all memory (vector + graph)
2. Clear vector store only
3. Clear graph store only
4. Restart Docker services
5. Stop Docker services
6. Start Docker services

### Inspect Memory

```bash
# View stored memories, entities, and relationships
python scripts/inspect_memory.py

# View and export conversation logs
python scripts/view_conversations.py
```

**Displays:**
- Vector memory count and recent entries
- Entity list with types and attributes
- Relationship list with predicates
- Utility score distribution

**Note on Conversation Logs:** The JSON conversation logs saved to `data/conversations/` are structured in a format that can be easily converted to fine-tuning datasets (Alpaca, ShareGPT, JSONL, etc.). This allows you to train custom models on your own conversation history while maintaining complete privacy.

---

## Configuration

Key settings in `src/config.py` (override via `.env` file):

### Model Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `main_model` | qwen3:14b | Main conversation LLM |
| `observer_model` | qwen3:1.7b | Entity extraction LLM |
| `embedding_model` | nomic-embed-text:v1.5 | Embedding model (flexible versioning) |
| `ollama_host` | http://localhost:11434 | Ollama API endpoint |

### Memory Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `max_context_tokens` | 3000 | Maximum tokens for LLM context window |
| `sliding_window_tokens` | 2000 | Recent conversation history to retain |
| `vector_search_top_k` | 15 | Initial candidates from vector search |
| `graph_search_top_k` | 10 | Relationships from graph search |
| `rerank_top_k` | 5 | Final results after cross-encoder reranking |

### Temporal Decay Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `temporal_decay_core` | 0 | Core facts never decay (0 = disabled) |
| `temporal_decay_high` | 180 | HIGH utility half-life (6 months) |
| `temporal_decay_medium` | 60 | MEDIUM utility half-life (2 months) |
| `temporal_decay_low` | 14 | LOW utility half-life (2 weeks) |

**Note:** Temporal decay uses exponential half-life. A memory's relevance score is multiplied by `0.5^(age_days / half_life_days)`.

### Example `.env` File

```bash
# Override defaults by creating .env file
MAIN_MODEL=qwen3:14b
OBSERVER_MODEL=qwen3:1.7b
EMBEDDING_MODEL=nomic-embed-text:latest

MAX_CONTEXT_TOKENS=4000
VECTOR_SEARCH_TOP_K=20
RERANK_TOP_K=7

TEMPORAL_DECAY_HIGH=180
TEMPORAL_DECAY_MEDIUM=60
TEMPORAL_DECAY_LOW=14
```

---

## How It Works

### 1. Context Assembly
When you ask a question, the system:
1. **Embeds** your query using `nomic-embed-text`
2. **Searches** vector store and graph store **in parallel** (optimized)
   - Vector: top-15 semantically similar memories from LanceDB
   - Graph: top-10 related entities/relationships from FalkorDB
3. **Applies** temporal decay based on memory utility and age
4. **Reranks** candidates using cross-encoder to select top-5 most relevant
5. **Merges** results with recent conversation history

**Performance:** Parallel database queries reduce retrieval latency by ~33%.

### 2. Response Generation
The main LLM (Qwen3 14B) receives:
- **Recent conversation** (last ~2000 tokens)
- **Top-5 retrieved memories** (context-aware)
- **Your current query**

It generates a contextually-aware response that integrates past conversations.

### 3. Observer (Async)
After responding, the observer:
1. **Grades** the turn's importance (DISCARD, LOW, MEDIUM, HIGH) - *runs first as gatekeeper*
2. **Early exit** for DISCARD turns (skips remaining LLM calls)
3. **Parallel processing** for non-DISCARD turns (optimized):
   - **Classifies** fact type (core, preference, episodic)
   - **Extracts** entities (Person, Place, Organization, Technology, etc.)
   - **Extracts** relationships (WORKS_AT, SIBLING_OF, PREFERS, DATING, etc.)
   - **Generates** summary and retrieval queries
4. **Semantic contradiction detection** (LLM-powered):
   - Understands temporal state transitions: "VISITING" → "RETURNED_HOME"
   - Detects mutually exclusive states: "WORKS_AT CompanyA" vs "WORKS_AT CompanyB"
   - Recognizes state completions: "SCHEDULED_FOR Friday" → "HAPPENED Monday"
   - Identifies attribute updates: "AGE 24" → "AGE 25"
5. **Marks superseded facts** with metadata (still stored but filtered from retrieval)
6. **Persists** to both LanceDB (vector) and FalkorDB (graph)

**Fact Types:**
- **Core:** Work schedules, home address, family relationships, owned devices (never decay)
- **Preference:** Opinions, likes/dislikes, feelings (60-day half-life)
- **Episodic:** One-time events, meetings, trips (14-day half-life)

**Temporal States:**
- **Ongoing:** Currently true facts (e.g., "visiting", "working at")
- **Completed:** Past facts (e.g., "visited", "worked at")
- **Planned:** Future facts (e.g., "scheduled for", "planning to")

**Performance:** Early exit saves ~4x time on small talk; parallel LLM tasks reduce processing by ~3x for important turns.

---

## Project Structure

```
lcr-codex_CLAUDEREVIEW/
│
├── src/
│   ├── main.py                    # Entry point with pre-flight check
│   ├── config.py                  # Configuration settings
│   │
│   ├── models/
│   │   ├── llm.py                 # Ollama LLM client
│   │   ├── embedder.py            # Embedding generation
│   │   └── reranker.py            # Cross-encoder reranking
│   │
│   ├── memory/
│   │   ├── vector_store.py        # LanceDB operations
│   │   ├── graph_store.py         # FalkorDB operations
│   │   └── context_assembler.py   # Retrieval orchestration
│   │
│   ├── observer/
│   │   ├── observer.py            # Entity extraction logic
│   │   ├── prompts.py             # Extraction/grading prompts
│   │   └── extractors.py          # Utility grading
│   │
│   ├── orchestration/
│   │   ├── graph.py               # LangGraph state machine
│   │   └── prompts.py             # Conversation prompts
│   │
│   └── ingestion/
│       ├── chunker.py             # Text chunking (legacy)
│       └── pipeline.py            # Document ingestion (legacy)
│
├── tests/
│   ├── test_memory_retrieval.py   # Core memory tests
│   ├── test_observer.py           # Entity extraction tests
│   ├── test_graph_store.py        # Graph operations tests
│   ├── test_context_assembler.py  # Retrieval tests
│   └── test_integration.py        # End-to-end tests
│
├── scripts/
│   ├── inspect_memory.py          # View stored data
│   ├── clear_memory.py            # Memory management menu
│   ├── observer_giana_live.py     # Test observer extraction
│   └── observer_stress_test.py    # Observer performance test
│
├── data/
│   ├── lancedb/                   # Vector store data
│   └── falkordb/                  # Graph data (via Docker volume)
│
├── docker-compose.yml             # FalkorDB + Redis services
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── QUICKSTART.md                  # Detailed setup guide
└── CLAUDE.md                      # Development handoff document
```

---

## Testing

### Run All Tests

```bash
# Full test suite
pytest

# Verbose output
pytest -v

# With coverage
pytest --cov=src
```

### Test Categories

| Test File | Coverage |
|-----------|----------|
| `test_memory_retrieval.py` | Vector search, reranking, temporal decay |
| `test_observer.py` | Entity/relationship extraction, utility grading |
| `test_graph_store.py` | Graph operations, contradiction detection |
| `test_context_assembler.py` | Context assembly, memory merging |
| `test_integration.py` | End-to-end conversation flow |

### Test Scenarios Covered

✅ **Basic Persistence** — Information remembered across sessions
✅ **Familial Relationships** — SIBLING_OF, PARENT_OF extraction
✅ **Contradiction Handling** — Job changes, location moves
✅ **Entity Extraction** — Ages, roles, attributes
✅ **Multi-Turn Context** — Questions about past conversations
✅ **Complex Entity Networks** — Multi-hop relationships
✅ **Temporal Reasoning** — Time-based queries
✅ **Correction Handling** — "Actually, she's 25 not 24"

---

## Troubleshooting

### Pip Cache Issues

If `pip install -r requirements.txt` fails with metadata errors:

```bash
# Clear pip cache and retry
pip cache purge
pip install -r requirements.txt

# If specific package is corrupted
pip install --force-reinstall --no-deps <package-name>
pip install -r requirements.txt
```

### Port Conflicts

If Docker containers fail with "port already allocated":

```bash
# Stop and remove conflicting containers
docker compose down
# Stop and remove conflicting containers
docker compose down

# Or use the built-in nuclear reset which auto-detects and removes conflicts:
python scripts/nuclear_reset.py

# Manual removal if needed:
docker rm -f $(docker ps -aq --filter name=lcr-codex) 2>/dev/null

# Restart services
docker compose up -d
```

### Linux-Specific Setup

```bash
# Ensure Docker daemon is running
sudo systemctl start docker

# Add user to docker group (avoid sudo for docker commands)
sudo usermod -aG docker $USER
newgrp docker

# Ensure Docker daemon is running
sudo systemctl start docker

# Add user to docker group (avoid sudo for docker commands)
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose Plugin (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install docker-compose-plugin
# Verify: docker compose version

# Activate virtual environment
source .venv/bin/activate
```

### Memory Not Persisting

**Symptom:** Memories don't save between sessions

**Solution:** Ensure you exit with `exit` or `quit` command and wait for "Memories saved. Goodbye!" message. The observer runs asynchronously and needs time to complete.

```bash
# Check if observer completed
python scripts/inspect_memory.py
```

### Missing Models

**Symptom:** Pre-flight check shows models not found

**Solution:** Pull all required models with flexible versioning

```bash
# System supports flexible version matching
ollama pull qwen3:14b
ollama pull qwen3:1.7b
ollama pull nomic-embed-text  # Any version (v1.5, latest, etc.)

# Verify installation
ollama list
```

### Slow Responses

**Symptom:** Responses take >5 seconds

**Possible Causes:**
1. **Large Memory Store** — Check with `/stats` command
2. **High `vector_search_top_k`** — Reduce in config.py
3. **CPU Overload** — Close other GPU-intensive processes

**Solutions:**
```bash
# Reduce retrieval candidates
# Edit src/config.py:
# vector_search_top_k = 10  (instead of 15)
# rerank_top_k = 3  (instead of 5)

# Clear old memories if >10k entries
python scripts/clear_memory.py
```

### FalkorDB Connection Errors

**Symptom:** Pre-flight check shows FalkorDB warning

**Solution:** Verify Docker services are running

```bash
# Check Docker status
docker ps

# Restart services if needed
docker compose restart falkordb redis

# Check logs for errors
docker logs lcr-codex-falkordb-1
```

---

## Known Limitations

### Recent Improvements

✅ **Natural Voice Output (v1.2.0)** - Integrated Kokoro TTS for high-quality, natural-sounding speech. Features 8 female voices, sentence-by-sentence streaming, voice controls, and low resource usage (~2-3GB VRAM or CPU-only). Toggle with `/voice`, change voices in config, adjust speed with `/speed`.

✅ **Observer Reliability Fix (v1.1.4)** - Fixed critical memory retention bug where concurrent observer tasks caused HTTP timeouts. Implemented semaphore limiting (max 2 concurrent), increased timeout to 180s, added retry logic with exponential backoff. Memory persistence now 100% reliable.

✅ **Enhanced Utility Grading (v1.1.3)** - Fixed bug where detailed project descriptions were incorrectly graded as DISCARD. Updated prompt now explicitly recognizes user's projects and technical details as high-value memories.

### Current Limitations

1. **Pronoun Resolution**
   - System doesn't resolve pronouns in queries ("she", "it", "they")
   - Workaround: Use explicit names when asking questions

2. **Observer Model Accuracy**
   - Qwen3 1.7B may miss subtle entities in complex sentences
   - Consider upgrading to qwen3:4b if extraction quality is insufficient

3. **No Memory Pruning**
   - No automatic deletion of DISCARD/LOW utility memories
   - Future: Implement background pruning task

4. **Reranker Model**
   - Using lightweight all-MiniLM-L6-v2 (fast but basic)
   - Future: Consider BGE-Reranker-v2-m3 for deeper semantic understanding

---

## Performance Notes

### Response Times (Optimized)
- **Small talk (DISCARD):** ~0.5 seconds (early exit optimization)
- **Important turns (HIGH/MEDIUM):** ~2-3 seconds (parallel LLM processing)
- **Context retrieval:** ~100ms (parallel vector + graph search)
- **Overall conversation:** ~2-4 seconds average (dominated by main LLM generation)

### Optimizations
- ✅ **Parallel database queries** - Vector and graph searches run concurrently (~33% faster)
- ✅ **Early exit for small talk** - DISCARD turns skip unnecessary processing (~4x faster)
- ✅ **Parallel observer tasks** - Entity extraction, summarization, and query generation run concurrently (~3x faster)

### Scaling & Resources
- **Scaling:** System handles thousands of conversations before slowdown
- **Bottleneck:** Main LLM inference (Qwen3 14B), not memory retrieval
- **VRAM Usage:** ~10GB for Qwen3 14B, ~2GB for reranker
- **Memory Retrieval:** Sub-100ms even with 10k+ stored memories

---

## Contributing

This is a personal research project. Issues and pull requests are welcome for:
- Bug fixes
- Performance improvements
- Test coverage expansion
- Documentation improvements

---

## License

See [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Built with:
- [Ollama](https://ollama.ai/) — Local LLM inference
- [LanceDB](https://lancedb.com/) — Vector database
- [FalkorDB](https://www.falkordb.com/) — Knowledge graph database
- [LangGraph](https://github.com/langchain-ai/langgraph) — Workflow orchestration
- [Kokoro TTS](https://huggingface.co/spaces/hexgrad/Kokoro-TTS) — Natural voice synthesis
- [Qwen Team](https://github.com/QwenLM/Qwen) — Open-source LLM models

---

*Last Updated: 2026-01-21*
*Version: 1.2.0*
*Status: Production-ready with voice output*
