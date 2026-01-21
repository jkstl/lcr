# CLAUDE.md — Developer Handoff Document

**Version 1.2.0** | **Status: Production-Ready**

This document provides essential context for developers continuing work on the LCR system. For user-facing documentation, see [README.md](README.md).

---

## Quick Context

**What is LCR?** Local Cognitive RAG - A privacy-first conversational AI with persistent episodic memory using dual-architecture (vector + graph) and natural voice output.

**Current Status:** Production-ready with reliable memory persistence and Kokoro TTS integration (v1.2.0).

**Focus Areas:** STT implementation (v1.2.x), observer extraction quality improvements, utility grading consistency, automated memory pruning.

---

## Architecture Overview

```
User Input → Context Assembly (Parallel) → LLM Generation → Response
                     ↓ Async Observer (Semaphore-Limited, 2 concurrent max)
                     ↓ Persist to Vector + Graph Stores
```

**Memory Pipeline:**
1. **Utility grading** (DISCARD = early exit)
2. **Parallel extraction** (entities, relationships, summary, queries, fact_type)
3. **Semantic contradiction detection** (LLM-powered)
4. **Mark superseded facts** (status="completed")
5. **Persist** to LanceDB (vector) + FalkorDB (graph)

**Key Optimizations:**
- Parallel database queries (vector + graph)
- Early exit for DISCARD turns (~4x faster)
- Parallel observer LLM tasks (~3x faster)
- Semaphore limiting (max 2 concurrent observers) prevents Ollama overload
- Retry logic with exponential backoff for transient failures

---

## Critical Files for Development

### Core Logic
- `src/observer/observer.py` - Entity extraction, contradiction detection, persistence, retry logic
- `src/observer/prompts.py` - **UTILITY_PROMPT**, EXTRACTION_PROMPT, SEMANTIC_CONTRADICTION_PROMPT
- `src/memory/context_assembler.py` - Retrieval, filtering, temporal decay, recency boost
- `src/memory/graph_store.py` - FalkorDB operations, superseded fact tracking
- `src/orchestration/graph.py` - LangGraph state machine, streaming, **semaphore limiting**
- `src/voice/tts.py` - **Kokoro TTS engine**, voice synthesis, async playback (v1.2.0)
- `src/voice/utils.py` - Sentence splitting for streaming TTS (v1.2.0)

### Configuration
- `src/config.py` - All settings (models, top-k, decay rates, TTS)
  - **Embedding model:** `nomic-embed-text` (768-dim)
  - **Main model:** `qwen3:14b`
  - **Observer model:** `qwen3:1.7b` (consider upgrading to :4b for better extraction)
  - **TTS voice:** `af_sarah` (8 female voices available)
  - **TTS enabled:** `False` (toggle with `/voice` command)
- `.env` - Environment overrides (TTS_ENABLED, TTS_VOICE, TTS_SPEED)

### Utilities
- `scripts/inspect_memory.py` - View stored memories
- `scripts/view_conversations.py` - Browse conversation logs
- `scripts/nuclear_reset.py` - Complete memory wipe

---

## Known Issues & Debugging

### Memory Persistence (RESOLVED in v1.1.4)
**Problem:** Concurrent observer tasks caused HTTP timeouts, resulting in silent failures.
**Solution:** Implemented semaphore (max 2 concurrent) + increased timeout to 180s + retry logic.
**Status:** ✅ Fixed - 100% persistence success rate in testing.

### Utility Grading Inconsistency
**Issue:** Observer model (qwen3:1.7b) occasionally grades HIGH-value content as LOW/MEDIUM.
**Example:** Emotional/relationship content sometimes graded LOW despite being significant.
**Impact:** LOW-graded memories persist but with shorter retention (14 days vs 180 days).
**Solutions:**
- Upgrade to `OBSERVER_MODEL=qwen3:4b` for better accuracy (slower processing)
- Review/tune `UTILITY_PROMPT` in `src/observer/prompts.py`
- Check logs: `grep "Utility grading:" <output>`

### Entity Attribution Bugs
**Issue:** Observer sometimes confuses "User" with other entities.
**Example:** "User LIVES_IN Falmouth, MA" when it should be "Sam LIVES_IN Falmouth, MA"
**Workaround:** Information is captured even if attribution is imperfect. Most queries still retrieve relevant context.
**Solutions:**
- Upgrade to `OBSERVER_MODEL=qwen3:4b`
- Improve EXTRACTION_PROMPT to be more explicit about entity distinction

### If Memories Still Not Persisting
1. Ensure graceful exit with `exit` command (waits for observer)
2. Check for exceptions in terminal output (now logged, not silently caught)
3. Verify Ollama is responsive: `curl http://localhost:11434/api/tags`
4. Inspect stored data: `python scripts/inspect_memory.py`

### If Old Facts Are Surfacing
1. Check contradiction detection is marking facts as superseded
2. Verify filtering logic in `context_assembler.py` (_graph_search)
3. Look for `superseded_by` field in graph relationships

---

## Configuration Tuning

**For faster responses:**
- Reduce `VECTOR_SEARCH_TOP_K` (default: 15)
- Reduce `RERANK_TOP_K` (default: 5)

**For better recall:**
- Increase `VECTOR_SEARCH_TOP_K` (up to 20-25)
- Increase `GRAPH_SEARCH_TOP_K` (default: 10)

**For different retention:**
- Adjust `TEMPORAL_DECAY_*` values in `config.py`
- Core facts never decay (0 = disabled)
- HIGH: 180 days, MEDIUM: 60 days, LOW: 14 days

**For better extraction quality:**
- Upgrade observer: `OBSERVER_MODEL=qwen3:4b` (in .env or config.py)
- Trade-off: ~2x slower processing, but more accurate entity/relationship extraction

---

## Next Development Priorities

### High Priority (v1.2.x)
1. **Speech-to-Text (STT)** - Implement Whisper for voice input, wake word detection
2. **TTS voice variety** - Test different Kokoro voices, add voice cloning support
3. **Observer extraction quality** - Fix entity attribution bugs, test qwen3:1.7b vs :4b

### Medium Priority
1. **Utility grading consistency** - Investigate why emotional content sometimes grades LOW
2. **Memory pruning** - Automatic deletion of old LOW/DISCARD utility memories
3. **Pronoun resolution** - Coreference chain tracking

### Low Priority
1. **Query expansion** - Synonym/variation handling for better retrieval
2. **Web UI** - Non-technical user interface
3. **Background pruning task** - Scheduled cleanup of expired memories

---

## Important Notes for Developers

**Fact Type Classification:**
- `core`: User's name, work schedule, home address, family, owned devices (never decay)
- `preference`: Opinions, likes/dislikes, feelings (60-day half-life)
- `episodic`: One-time events, meetings, trips (14-day half-life)

**Source-Based Extraction:**
- Facts from USER statements: confidence=1.0, source="user_stated"
- Assistant inferences: confidence=0.3, source="assistant_inferred"
- Prevents hallucinated facts from being stored as ground truth

**Temporal States:**
- `ongoing`: Currently true (e.g., "visiting", "working at")
- `completed`: Past events (e.g., "visited", "worked at")
- `planned`: Future events (e.g., "scheduled for", "planning to")

**Semantic Contradictions:**
- LLM detects contradictions across different predicates
- Understands state transitions: VISITING → RETURNED_HOME
- Recognizes mutual exclusion: WORKS_AT A vs WORKS_AT B
- Handles attribute updates: AGE 24 → AGE 25

**Semaphore Limiting (v1.1.4):**
- Max 2 concurrent observer tasks to prevent Ollama overload
- Prevents 16-24 concurrent LLM requests that cause timeouts
- Located in `src/orchestration/graph.py:45` (`_observer_semaphore`)
- Tasks queue automatically if limit reached

**Retry Logic (v1.1.4):**
- All observer LLM calls wrapped in `retry_on_timeout()`
- 3 attempts with exponential backoff: 2s, 4s, 8s
- Located in `src/observer/observer.py:29`
- Handles transient Ollama timeouts gracefully

---

## Testing

**Run all tests:**
```bash
pytest                                          # All tests
pytest tests/test_memory_retrieval.py -v       # Core memory
pytest tests/test_semantic_contradictions.py -v # Contradiction detection
```

**Test coverage:**
- Cross-session persistence ✅
- Familial relationships (SIBLING_OF, PARENT_OF) ✅
- Contradiction handling (job changes, location moves) ✅
- Semantic state transitions (VISITING → RETURNED_HOME) ✅
- Attribute updates (AGE 24 → AGE 25) ✅
- Complex entity networks (14+ relationships) ✅

---

## Git Repository

**GitHub:** https://github.com/jkstl/lcr

**Test Status:** All 43+ tests passing

---

*Last Updated: 2026-01-21*
*Version: 1.2.0*
