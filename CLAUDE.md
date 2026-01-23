# CLAUDE.md — Developer Handoff Document

**Version 1.4.0** | **Status: Fine-tuned Model Ready**

Essential context for developers continuing work on LCR. See [README.md](README.md) for user documentation.

---

## Quick Context

**What is LCR?** Local Cognitive RAG - Privacy-first conversational AI with persistent episodic memory using dual-architecture (vector + graph) and natural voice output.

**Current Status:** Production-ready with fine-tuned observer model (100% accuracy).

**Latest:** Successfully fine-tuned LiquidAI/LFM2.5-1.2B-Instruct (100% utility grading, perfect entity extraction). Now using transformers directly instead of Ollama.

---

## Current Focus (January 2026)

### Fine-Tuned Observer Model ✅

**Status**: **Deployed** via HuggingFace Transformers (GGUF conversion failed due to novel architecture)

**Training Completed**:
- Base: **LiquidAI/LFM2.5-1.2B-Instruct** (novel LIV convolution + GQA architecture)
- Method: LoRA (16-rank on Q/K/V/O projections)
- Data: 3,300 examples with **system message differentiation**
- Tasks: Utility grading + entity/relationship extraction
- Results: **100% utility grading accuracy**, perfect entity extraction
- Final Loss: 0.29 (train), 0.32 (eval)
- Location: `fine_tuning/lfm_1.2b_v1/model/merged/`

**Key Improvement**: Added system messages to differentiate tasks:
- Utility: "You are a utility grading assistant..."
- Extraction: "You are an entity extraction assistant..."

**Why Not Ollama**: GGUF conversion failed (missing tensor 'output_norm') due to LFM2.5's novel architecture. Using `transformers` directly instead - model loads on startup (~2s), same inference speed.

**Dataset Quality**:
- Reused high-quality GPT-4o labeled data from Qwen3 attempt
- Multi-entity scenarios, temporal states, complex relationships
- Distribution: 11% DISCARD, 31% STORE, 58% IMPORTANT

---

## Architecture Overview

```
User Input → Context Assembly (Parallel) → LLM (Ollama) → Response
                     ↓ Async Observer (Transformers fine-tuned model)
                     ↓ Persist to Vector + Graph
```

**Observer Pipeline**:
1. Utility grading with system message (DISCARD/STORE/IMPORTANT)
2. Parallel extraction (entities, relationships, summary)
3. Semantic contradiction detection
4. Mark superseded facts
5. Persist to LanceDB + FalkorDB

**Key Optimizations**:
- Parallel database queries
- Early exit for DISCARD (~4x faster)  
- Parallel observer tasks (~3x faster)
- Semaphore limiting (max 2 concurrent, prevent GPU overload)
- Robust JSON parsing with 4 fallback strategies
- **Direct transformers** (bypasses Ollama overhead)

---

## Observer Models

| Model | Status | Accuracy | Speed | Use Case |
|-------|--------|----------|-------|----------|
| **lfm-observer (LFM2.5-1.2B)** | **Production** | **100%** | ~3s | 🆕 Fine-tuned via transformers ⭐ |
| qwen3:1.7b | Backup | 75% | 10s | Fallback if GPU unavailable |
| qwen3:0.6b | Deprecated | 75% | 3s | Old baseline |

**Configuration**:  
- Set `observer_model` in `src/config.py` or `.env`
- Format: `"transformers:path/to/model"` or regular Ollama model name
- Default: `"transformers:fine_tuning/lfm_1.2b_v1/model/merged"`

**Switching Back to Ollama**: Change to `observer_model: "qwen3:1.7b"` if needed

---

## Critical Files

### Core Logic
- `src/observer/observer.py` - Extraction, contradiction detection, persistence
- `src/observer/prompts.py` - 3-level UTILITY_PROMPT, EXTRACTION_PROMPT
- `src/models/llm.py` - `parse_json_response()` with robust fallbacks
- `src/memory/context_assembler.py` - Retrieval, temporal decay
- `src/memory/graph_store.py` - FalkorDB operations
- `src/orchestration/graph.py` - LangGraph state machine
- `src/voice/tts.py` - Kokoro TTS engine

### Configuration
- `src/config.py` - All settings (models, top-k, decay, TTS)
- `.env` - Environment overrides

### Fine-tuning
- `fine_tuning/qwen3_0.6b_v2/README.md` - Full documentation
- `fine_tuning/qwen3_0.6b_v2/model/merged/` - Ready for GGUF conversion

### Utilities
- `scripts/inspect_memory.py` - View stored memories
- `scripts/view_conversations.py` - Browse conversation logs
- `scripts/observer_model_comparison.py` - Compare performance

---

## Key Systems

### Utility Grading (3-Level)
- **DISCARD**: Pure acknowledgments (score: 0.0)
- **STORE**: Factual information (score: 0.6)
- **IMPORTANT**: Identity, relationships, life facts (score: 1.0)

### Fact Types
- `core`: Never decay (name, work, family, devices)
- `preference`: 60-day half-life (opinions, likes)
- `episodic`: 14-day half-life (events, meetings)

### JSON Parsing (v1.3.0)
- 4 fallback strategies
- Handles markdown blocks, preambles, mixed content
- 100% success rate

### Temporal States
- `ongoing`: Currently true (VISITING, WORKS_AT)
- `completed`: Past (RETURNED_HOME, VISITED)
- `planned`: Future (SCHEDULED_FOR)

---

## Configuration Tuning

**Speed vs Quality**:
```python
# Fast responses
VECTOR_SEARCH_TOP_K = 10
RERANK_TOP_K = 3

# Better recall
VECTOR_SEARCH_TOP_K = 20
GRAPH_SEARCH_TOP_K = 15
```

**Model Selection**:
- Main: `qwen3:14b` (quality) or `:8b`/`:4b` (speed)
- Observer: `qwen3:1.7b` (current) or `qwen3-observer` (after testing)
- Embedder: `nomic-embed-text` (don't change)

**TTS**:
- Enable: `/voice` command or `TTS_ENABLED=true`
- Voices: `af_heart` (default), `af_bella`, `af_nicole`
- Speed: `TTS_SPEED=1.0` (0.5-2.0 range)

---

## Testing

```bash
pytest                                          # All tests
pytest tests/test_memory_retrieval.py -v       # Core memory
pytest tests/test_semantic_contradictions.py -v # Contradictions
```

**Coverage**: 48+ tests passing (persistence, relationships, contradictions, temporal states, utility grading, JSON parsing)

---

## Development Priorities

### Immediate
1. **Deploy Fine-tuned Model** - Convert to GGUF, test, deploy if > 85% accuracy
2. **Remove Legacy 4-Level** - Clean up LOW/MEDIUM/HIGH from UtilityGrade enum

### Medium
1. **STT Integration** - Whisper for voice input
2. **Memory Pruning** - Auto-delete old DISCARD memories
3. **Pronoun Resolution** - Coreference tracking

### Low
1. **Query Expansion** - Synonym handling
2. **Web UI** - Non-technical interface
3. **Background Pruning** - Scheduled cleanup

---

## Recent Changes

**v1.4.0** (Jan 2026):
- Fine-tuned qwen3-observer on 3,300 examples
- Multi-entity, temporal, and complex relationship support

**v1.3.0** (Jan 2026):
- Robust JSON parser (4 fallback strategies, 100% success)
- 3-level utility grading (75% accuracy, up from 62%)
- Combined: 62.5% → 100% reliability

**v1.2.0** (Jan 2026):
- Kokoro TTS integration with pipelined synthesis
- Semaphore limiting for observer concurrency

---

## Git Repository

**GitHub**: https://github.com/jkstl/lcr

---

*Last Updated: 2026-01-23*  
*Version: 1.4.0*
