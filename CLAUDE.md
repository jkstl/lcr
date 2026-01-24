# CLAUDE.md — Developer Handoff Document

**Version 1.5.0** | **Status: Dual-Model Observer Architecture**

Essential context for developers continuing work on LCR. See [README.md](README.md) for user documentation.

---

## Quick Context

**What is LCR?** Local Cognitive RAG - Privacy-first conversational AI with persistent episodic memory using dual-architecture (vector + graph) and natural voice output.

**Current Status:** Production-ready with dual-model observer architecture for optimal accuracy.

**Latest:** Migrated to **NuExtract-2.0-2B** for entity extraction (~90% accuracy, zero hallucination) + **qwen3:1.7b** for utility grading (100% accuracy). Downgraded main LLM to qwen3:8b for VRAM budget.

---

## Current Focus (January 2026)

### Dual-Model Observer Architecture ✅

**Status**: **Production** - Deployed Jan 24, 2026

**Architecture**:
- **Utility Grading**: qwen3:1.7b via Ollama (100% accuracy)
- **Entity/Relationship Extraction**: NuExtract-2.0-2B via transformers (90% accuracy, zero hallucination)
- **Main LLM**: qwen3:8b via Ollama (downgraded from 14b for VRAM)

**Why Dual-Model?**
- **Specialized tasks**: Utility grading needs classification, extraction needs structure
- **NuExtract strengths**: Purely extractive (can't hallucinate), template-based, excellent entity attribution
- **qwen3:1.7b strengths**: Perfect utility grading, fast, minimal VRAM
- **VRAM budget**: 13.8 GB total (qwen3:8b 6GB + qwen3:1.7b 3.5GB + NuExtract 4.3GB)

**NuExtract-2.0-2B Details**:
- Base: Qwen2-VL-2B-Instruct (MIT license)
- Method: Template-based extraction with in-context learning
- No fine-tuning needed - works out of box with examples
- Purely extractive - cannot hallucinate entities
- Location: Auto-downloaded from HuggingFace on first use

**Why Not Fine-Tuned LFM2.5?**
- qwen3:1.7b matches LFM2.5's 100% utility accuracy without fine-tuning
- NuExtract far exceeds LFM2.5's extraction accuracy (90% vs 85%)
- Simpler architecture - no custom fine-tuned models to maintain
- Better licensing - MIT vs custom training

---

## Architecture Overview

```
User Input → Context Assembly (Parallel) → qwen3:8b (Ollama) → Response
                     ↓ Async Dual-Model Observer
                     ├─ qwen3:1.7b (Utility Grading)
                     └─ NuExtract-2.0-2B (Entity Extraction)
                     ↓ Persist to Vector + Graph
```

**Observer Pipeline**:
1. **Utility grading** (qwen3:1.7b): DISCARD/STORE/IMPORTANT classification
2. **Parallel tasks**:
   - Entity/relationship extraction (NuExtract-2.0-2B with templates)
   - Summary generation
   - Retrieval query generation
3. Semantic contradiction detection
4. Mark superseded facts
5. Persist to LanceDB + FalkorDB

**Key Optimizations**:
- Parallel database queries
- Early exit for DISCARD (~4x faster)
- Parallel observer tasks (~3x faster)
- Semaphore limiting (max 2 concurrent, prevent GPU overload)
- Robust JSON parsing with 4 fallback strategies
- Template-based extraction (prevents hallucination)

---

## Observer Models

| Model | Role | Accuracy | VRAM | Use Case |
|-------|------|----------|------|----------|
| **qwen3:1.7b** | **Utility** | **100%** | 3.5 GB | DISCARD/STORE/IMPORTANT classification ⭐ |
| **NuExtract-2.0-2B** | **Extraction** | **~90%** | 4.3 GB | Entity/relationship extraction (zero hallucination) ⭐ |
| qwen3:0.6b | Backup utility | 60% | 2 GB | Fallback if needed |
| LFM2.5-1.2B (fine-tuned) | Deprecated | 85% | 3 GB | Replaced by NuExtract |

**Configuration**:
```python
# src/config.py
observer_utility_model = "qwen3:1.7b"
observer_extraction_model = "nuextract:numind/NuExtract-2.0-2B"
```

**VRAM Budget**:
```
qwen3:8b (main):            6.0 GB
qwen3:1.7b (utility):       3.5 GB
NuExtract-2.0-2B (extract): 4.3 GB
Kokoro TTS (CPU):           0.0 GB
────────────────────────────────────
Total:                     13.8 GB / 16.3 GB (85%)
Remaining:                  2.5 GB (15%)
```

---

## Critical Files

### Core Logic
- `src/observer/observer.py` - Dual-model observer (utility + extraction)
- `src/observer/prompts.py` - Utility prompts (3-level UTILITY_PROMPT)
- `src/observer/nuextract_templates.py` - Extraction templates and examples
- `src/models/nuextract_client.py` - NuExtract client implementation
- `src/models/llm.py` - `parse_json_response()` with robust fallbacks
- `src/memory/context_assembler.py` - Retrieval, temporal decay
- `src/memory/graph_store.py` - FalkorDB operations
- `src/orchestration/graph.py` - LangGraph state machine with dual-model init
- `src/voice/tts.py` - Kokoro TTS engine (CPU-only)

### Configuration
- `src/config.py` - All settings (models, top-k, decay, TTS)
- `.env` - Environment overrides

### NuExtract Integration
- `src/models/nuextract_client.py` - Client for NuExtract models
- `src/observer/nuextract_templates.py` - Templates with in-context examples
- `scripts/compare_extraction_models.py` - Model comparison tool
- `NUEXTRACT_INTEGRATION_REPORT.md` - Full analysis and benchmarks

### Fine-tuning (Deprecated)
- `fine_tuning/lfm_1.2b_v1/` - LFM2.5 fine-tune (no longer used)
- `fine_tuning/qwen3_0.6b_v2/` - Qwen3 fine-tune (superseded)

### Utilities
- `scripts/inspect_memory.py` - View stored memories
- `scripts/view_conversations.py` - Browse conversation logs
- `scripts/nuclear_reset.py` - Complete memory wipe

---

## Key Systems

### Utility Grading (3-Level)
- **DISCARD**: Pure acknowledgments (score: 0.0)
- **STORE**: Factual information (score: 0.6)
- **IMPORTANT**: Identity, relationships, life facts (score: 1.0)

**Model**: qwen3:1.7b (100% accuracy, no fine-tuning needed)

### Entity/Relationship Extraction

**Model**: NuExtract-2.0-2B (template-based, purely extractive)

**Template Format**:
```json
{
  "fact_type": "core|preference|episodic",
  "entities": [
    {"name": "verbatim-string", "type": "string", "attributes": {}}
  ],
  "relationships": [
    {"subject": "verbatim-string", "predicate": "string", "object": "verbatim-string", "temporal": "string"}
  ]
}
```

**In-Context Learning**:
- 3 examples guide entity attribution
- Teaches "My sister X" → X is subject
- Prevents user/entity confusion

**Key Benefits**:
- **Zero hallucination**: Purely extractive, cannot invent entities
- **90% accuracy**: Better entity attribution than fine-tuned models
- **No maintenance**: No retraining, just update examples if needed

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
- Main: `qwen3:8b` (current, good balance) or `:4b` (faster, lower quality)
- Utility: `qwen3:1.7b` (optimal, 100% accuracy)
- Extraction: `nuextract:numind/NuExtract-2.0-2B` (optimal, MIT license)
- Embedder: `nomic-embed-text` (don't change)

**VRAM Optimization**:
- Downgrade main: qwen3:8b → qwen3:4b saves ~2.5GB
- Single observer: Drop utility model, use NuExtract for both (loses accuracy)
- Disable TTS: Doesn't affect VRAM (runs on CPU)

**TTS**:
- Enable: `/voice` command or `TTS_ENABLED=true`
- Voices: `af_heart` (default), `af_bella`, `af_nicole`
- Speed: `TTS_SPEED=1.0` (0.5-2.0 range)
- **VRAM**: None (CPU-only via ONNX runtime)

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
1. ✅ **NuExtract Deployed** - Dual-model architecture production-ready
2. **Monitor accuracy** - Track extraction quality across diverse scenarios

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

**v1.5.0** (Jan 2026):
- **Dual-model observer**: qwen3:1.7b (utility) + NuExtract-2.0-2B (extraction)
- **Main LLM downgrade**: qwen3:14b → qwen3:8b for VRAM budget
- **Zero hallucination**: NuExtract purely extractive approach
- **Simplified architecture**: Dropped fine-tuned LFM2.5, uses base models

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

## Ollama Models Required

```bash
# Pull required models
ollama pull qwen3:8b           # Main LLM (6GB VRAM)
ollama pull qwen3:1.7b         # Utility grading (3.5GB VRAM)
ollama pull nomic-embed-text   # Embeddings
```

**NuExtract-2.0-2B** auto-downloads from HuggingFace on first use (~4GB download).

---

## Git Repository

**GitHub**: https://github.com/jkstl/lcr

---

*Last Updated: 2026-01-24*
*Version: 1.5.0*
