# CLAUDE.md — Local Cognitive RAG (LCR) System
## Authoritative Architectural Blueprint v1.0

> **Purpose**: This document defines the complete technical specification for a local, privacy-first conversational AI assistant with persistent episodic memory. Any coding agent (Claude Code, Codex, Cursor) should be able to implement this system from this specification alone.

---

## 1. PROJECT OVERVIEW

### 1.1 What This System Does
A voice-enabled conversational AI that:
- Remembers everything the user tells it across sessions (work schedules, relationships, owned technology, preferences)
- Retrieves relevant memories automatically when answering questions
- Learns the user's personality, communication style, and life context over time
- Runs 100% locally with zero external API calls

### 1.2 Example Interactions After Training
```
User: "What time do I need to be at work tomorrow?"
System: [Retrieves: User works at Acme Corp, shift starts 9am, 30min commute]
Response: "You need to leave by 8:30. Your shift at Acme starts at 9, and with your usual commute from Cedar Park, that gives you a small buffer."

User: "Should I text Sarah back?"
System: [Retrieves: Recent tension with Sarah, user expressed feeling overwhelmed, Sarah's communication pattern]
Response: "Based on what you've shared, you were feeling pretty drained after your last exchange. If you're not ready, it's okay to wait until tomorrow. What's making you hesitate?"
```

### 1.3 Hardware Constraints
| Resource | Limit | Allocation Strategy |
|----------|-------|---------------------|
| VRAM | 16GB | Main LLM ~10GB, Reranker ~0.5GB, Buffer ~5.5GB |
| System RAM | 32GB | Vector DB, Graph DB, Application |
| Storage | SSD Required | LanceDB disk-persistence, conversation logs |
| Network | None | 100% air-gapped capable |

---

## 2. TECHNOLOGY STACK

### 2.1 Core Components

| Component | Technology | Version/Variant | Resource Placement |
|-----------|------------|-----------------|-------------------|
| **Runtime** | Python | 3.11+ | CPU |
| **Main LLM** | Qwen3 | 14B Q4_K_M (GGUF) | VRAM (~10GB) |
| **Observer LLM** | Qwen3:1.7b | 4B Q4_K_M | CPU (offloaded) |
| **Embedding Model** | nomic-embed-text | v1.5 | CPU |
| **Reranker** | BGE-Reranker | v2-m3 | VRAM (~0.5GB) |
| **Vector Database** | LanceDB | Latest | Disk + RAM cache |
| **Graph Database** | FalkorDB | Latest | Docker container |
| **Orchestration** | LangGraph | Latest | CPU |
| **LLM Backend** | Ollama | Latest | Manages GGUF models |
| **Voice Input** | Whisper.cpp | Medium model | CPU |
| **Voice Output** | Piper TTS | en_US-lessac-medium | CPU |

### 2.2 Python Dependencies
```
# requirements.txt
ollama>=0.2.0
langgraph>=0.1.0
lancedb>=0.6.0
falkordb>=1.0.0
sentence-transformers>=2.7.0  # For reranker
numpy>=1.26.0
pydantic>=2.0.0
redis>=5.0.0                  # For async task queue
faster-whisper>=1.0.0         # Whisper.cpp Python bindings
piper-tts>=1.0.0
sounddevice>=0.4.6            # Audio I/O
httpx>=0.27.0                 # Async HTTP for Ollama
rich>=13.0.0                  # CLI interface
python-dotenv>=1.0.0
```

### 2.3 Ollama Model Setup
```bash
# Install models (run once)
ollama pull qwen3:14b
ollama pull qwen3:1.7b
ollama pull nomic-embed-text:v1.5
```

### 2.4 Docker Services
```yaml
# docker-compose.yml
version: '3.8'
services:
  falkordb:
    image: falkordb/falkordb:latest
    ports:
      - "6379:6379"
    volumes:
      - ./data/falkordb:/data
    command: ["--save", "60", "1"]  # Persist every 60s if 1+ change
  
  redis:
    image: redis:7-alpine
    ports:
      - "6380:6379"
    volumes:
      - ./data/redis:/data
```

**IMPLEMENTATION NOTE (v0.4.0 - Phase 4 Complete)**: Cross-encoder reranking has been implemented using BGE-Reranker-v2-m3. The system now uses two-stage retrieval: (1) vector search retrieves top-15 candidates, (2) cross-encoder reranks and selects top-5 most relevant. This is configurable via `settings.use_reranker` and can be disabled for fallback to vector-only retrieval. See `src/models/reranker.py` for implementation details. The reranker auto-detects GPU/CPU and gracefully degrades to CPU if VRAM is unavailable.

**IMPLEMENTATION NOTE (v0.4.1)**: Observer now uses separate `qwen3:1.7b` model (configurable via `observer_model` setting) instead of main 14B model, significantly improving Observer processing speed. Also added `RELATED_TO` flexible pattern in extraction prompt for domain-specific relationships (e.g., infrastructure: connected_to, depends_on) without hardcoding each type.

---

## 3. DATA ARCHITECTURE

### 3.1 Directory Structure
```
lcr/
├── CLAUDE.md                    # This file
├── requirements.txt
├── docker-compose.yml
├── .env                         # Local config (not committed)
├── src/
│   ├── __init__.py
│   ├── main.py                  # Entry point
│   ├── config.py                # Pydantic settings
│   ├── models/
│   │   ├── __init__.py
│   │   ├── llm.py               # Ollama client wrapper
│   │   ├── embedder.py          # Embedding generation
│   │   └── reranker.py          # Cross-encoder reranking
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── vector_store.py      # LanceDB operations
│   │   ├── graph_store.py       # FalkorDB operations
│   │   └── context_assembler.py # Retrieval orchestration
│   ├── observer/
│   │   ├── __init__.py
│   │   ├── observer.py          # Main observer logic
│   │   ├── extractors.py        # Entity/relationship extraction
│   │   └── prompts.py           # Observer prompt templates
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunker.py           # Semantic chunking
│   │   └── pipeline.py          # Document ingestion
│   ├── voice/
│   │   ├── __init__.py
│   │   ├── stt.py               # Speech-to-text (Whisper)
│   │   └── tts.py               # Text-to-speech (Piper)
│   └── orchestration/
│       ├── __init__.py
│       └── graph.py             # LangGraph state machine
├── data/
│   ├── lancedb/                 # Vector DB storage
│   ├── falkordb/                # Graph DB storage
│   ├── redis/                   # Task queue storage
│   └── conversations/           # Raw conversation logs (JSON)
└── tests/
    └── ...
```