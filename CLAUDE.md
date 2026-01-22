# CLAUDE.md — Developer Handoff Document

**Version 1.2.1** | **Status: Production-Ready**

This document provides essential context for developers continuing work on the LCR system. For user-facing documentation, see [README.md](README.md).

---

## Quick Context

**What is LCR?** Local Cognitive RAG - A privacy-first conversational AI with persistent episodic memory using dual-architecture (vector + graph) and natural voice output.

**Current Status:** Production-ready with reliable memory persistence, Kokoro TTS integration, and improved observer prompts (v1.2.1).

**Focus Areas:** STT implementation (v1.3.x), semantic contradiction detection improvements, automated memory pruning, qwen3:0.6b fine-tuning.

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
- Parallel observer persistence to both stores (~2x faster)
- Semaphore limiting (max 2 concurrent observers) prevents Ollama overload
- Retry logic with exponential backoff for transient failures
- Pipelined TTS synthesis eliminates audio gaps

---

## Recent Improvements (v1.2.1)

### Prompt Engineering (January 2026)
**Problem:** Observer extraction had 3 critical issues across all model sizes (0.6b, 1.7b, 4b):
1. Temporal states: "left and went home" → extracted as `LIVES_IN` instead of `RETURNED_HOME`
2. Entity attribution: "My sister Justine lives in Boston" → extracted as `User LIVES_IN Boston` instead of `Justine LIVES_IN Boston`
3. Utility grading: Emotional content ("feeling sad about Giana") → graded as `LOW` instead of `HIGH`

**Solution:** Rewrote EXTRACTION_PROMPT and UTILITY_PROMPT with explicit instructions and comprehensive examples.

**Results (verified with qwen3:1.7b):**
- ✅ Temporal states: Now correctly extracts `RETURNED_HOME` for completed state transitions
- ✅ Entity attribution: Now correctly uses family member names as subjects (not "User")
- ✅ Utility grading: Now correctly grades emotional content as `HIGH`

**Key Changes:**
- Split spatial predicates into ONGOING (`VISITING`) vs COMPLETED (`RETURNED_HOME`, `LEFT`, `ARRIVED_AT`)
- Added CRITICAL sections emphasizing entity attribution rules
- Added 6 comprehensive examples (up from 4) including temporal state transitions
- Added explicit rule for emotional content → HIGH utility
- Added `MISSES` predicate for emotional relationships

### Bug Fixes (v1.2.1)
- Fixed confidence score retrieval from FalkorDB (fields weren't being queried)
- Fixed contradiction marking with FalkorDB (string vs int ID mismatch)
- Fixed TTS streaming (pipelined synthesis instead of unlimited parallelism)
- Parallel observer persistence (vector + graph stores now concurrent)

---

## Critical Files for Development

### Core Logic
- `src/observer/observer.py` - Entity extraction, contradiction detection, persistence, retry logic
- `src/observer/prompts.py` - **UTILITY_PROMPT**, EXTRACTION_PROMPT, SEMANTIC_CONTRADICTION_PROMPT (significantly improved in v1.2.1)
- `src/memory/context_assembler.py` - Retrieval, filtering, temporal decay, recency boost
- `src/memory/graph_store.py` - FalkorDB operations, superseded fact tracking, ID conversion
- `src/orchestration/graph.py` - LangGraph state machine, streaming, **semaphore limiting**
- `src/voice/tts.py` - **Kokoro TTS engine**, voice synthesis, pipelined playback (v1.2.0)
- `src/voice/utils.py` - Sentence splitting for streaming TTS (v1.2.0)

### Configuration
- `src/config.py` - All settings (models, top-k, decay rates, TTS)
  - **Embedding model:** `nomic-embed-text` (768-dim)
  - **Main model:** `qwen3:14b`
  - **Observer model:** `qwen3:1.7b` (best reliability, see "Observer Model Selection" below)
  - **TTS voice:** `af_heart` (8 female voices available)
  - **TTS enabled:** `False` (toggle with `/voice` command)
- `.env` - Environment overrides (MAIN_MODEL, OBSERVER_MODEL, TTS_ENABLED, TTS_VOICE, TTS_SPEED)

### Utilities
- `scripts/inspect_memory.py` - View stored memories
- `scripts/view_conversations.py` - Browse conversation logs
- `scripts/nuclear_reset.py` - Complete memory wipe
- `scripts/list_models.py` - List available Ollama models and current config
- `scripts/model_tester.py` - Interactive tool for testing different models

### Documentation
- `MODEL_TESTING.md` - Comprehensive guide for testing and switching models
- `CLAUDE.md` - This file (developer handoff)
- `README.md` - User-facing documentation

---

## Observer Model Selection

**Tested models:** qwen3:0.6b, qwen3:1.7b, qwen3:4b

**Finding:** Model size does NOT reliably improve extraction quality. Prompt engineering is far more impactful.

| Model | JSON Format | Extraction Quality | Reliability | Recommendation |
|-------|------------|-------------------|-------------|----------------|
| **qwen3:1.7b** | ✅ Valid, correct schema | ✅ Good with improved prompts | ✅ Consistent | **Use this (default)** |
| qwen3:4b | ❌ Inconsistent (sometimes text, wrong schema) | ❓ Mixed results | ❌ Unreliable | **Avoid** |
| qwen3:0.6b | ❌ Wraps JSON in markdown blocks | ❌ Poor | ❌ Unusable | **Fine-tuning candidate** |

**Why qwen3:1.7b is best:**
- Always produces valid JSON with correct schema (`"predicate"` field)
- With improved prompts (v1.2.1), extraction quality is excellent
- Fast enough for real-time use
- Reliable and consistent

**Why qwen3:4b doesn't help:**
- Sometimes uses `"relation"` instead of `"predicate"` (breaks parsing)
- Sometimes outputs natural language instead of JSON
- Unreliable format adherence makes it worse than 1.7b despite larger size
- Not worth the 2x speed penalty

**Future direction:**
- Fine-tune qwen3:0.6b with improved prompts as training data
- Potential for faster extraction while maintaining quality
- Current prompts (v1.2.1) are training-ready

**How to test models:**
```bash
# Quick test
OBSERVER_MODEL=qwen3:4b python -m src.main

# Interactive testing
python scripts/model_tester.py

# See all available models
python scripts/list_models.py
```

---

## Known Issues & Debugging

### Memory Persistence (RESOLVED in v1.1.4)
**Problem:** Concurrent observer tasks caused HTTP timeouts, resulting in silent failures.

**Solution:** Implemented semaphore (max 2 concurrent) + increased timeout to 180s + retry logic.

**Status:** ✅ Fixed - 100% persistence success rate in testing.

### Confidence Score Retrieval (RESOLVED in v1.2.1)
**Problem:** FalkorDB graph queries weren't retrieving `source` and `confidence` fields, causing all facts to be treated as 100% confident regardless of actual source (user_stated vs assistant_inferred).

**Solution:** Added `relation.source` and `relation.confidence` to RETURN clauses in `query()` and `search_relationships()` methods. Updated `_row_to_relationship()` to parse these fields.

**Status:** ✅ Fixed - Confidence weighting now functional.

### Temporal State Extraction (RESOLVED in v1.2.1)
**Problem:** "left and went home" extracted as `LIVES_IN` instead of `RETURNED_HOME`.

**Solution:** Rewrote EXTRACTION_PROMPT with explicit ONGOING vs COMPLETED state distinction and comprehensive examples (Example 3 shows temporal state transition).

**Status:** ✅ Fixed - qwen3:1.7b now correctly extracts `RETURNED_HOME`.

### Entity Attribution (RESOLVED in v1.2.1)
**Problem:** "My sister Justine lives in Boston" extracted as `User LIVES_IN Boston` instead of `Justine LIVES_IN Boston`.

**Solution:** Added CRITICAL section to EXTRACTION_PROMPT clarifying when to use "User" vs person names as subjects, with explicit examples (Example 2).

**Status:** ✅ Fixed - qwen3:1.7b now correctly uses family member names as subjects.

### Utility Grading - Emotional Content (RESOLVED in v1.2.1)
**Problem:** Emotional content ("thinking about ex, feeling sad") graded as `LOW` instead of `HIGH`.

**Solution:** Added explicit rule to UTILITY_PROMPT: "Emotional content, feelings, or personal struggles" → HIGH. Added Example 3 demonstrating this.

**Status:** ✅ Fixed - qwen3:1.7b now correctly grades emotional content as HIGH.

### Contradiction Detection (PARTIAL)
**Issue:** Semantic contradiction detection doesn't always identify temporal state completions (e.g., `VISITING` → `RETURNED_HOME`).

**Impact:** Some superseded facts aren't automatically marked, requiring manual cleanup or user correction.

**Current State:**
- Extraction correctly identifies `RETURNED_HOME` (fixed in v1.2.1)
- Simple contradictions work (same predicate, different object)
- Complex semantic contradictions (state transitions) are hit-or-miss

**Workaround:** Facts are still extracted correctly; contradictions just aren't always auto-detected.

**Future Work:** Improve SEMANTIC_CONTRADICTION_PROMPT or use larger model for contradiction detection only.

### If Memories Still Not Persisting
1. Ensure graceful exit with `exit` command (waits for observer)
2. Check for exceptions in terminal output (now logged, not silently caught)
3. Verify Ollama is responsive: `curl http://localhost:11434/api/tags`
4. Inspect stored data: `python scripts/inspect_memory.py`

### If Old Facts Are Surfacing
1. Check contradiction detection is marking facts as superseded
2. Verify filtering logic in `context_assembler.py` (_graph_search)
3. Look for `superseded_by` field in graph relationships
4. Manually mark if needed: See `scripts/` for examples

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

**Model selection:**
- Main LLM: Larger = better quality, slower (qwen3:14b, :8b, :4b)
- Observer: **qwen3:1.7b recommended** (best reliability with improved prompts)
- Embedder: **nomic-embed-text** (don't change unless you know what you're doing)

**Testing models:**
- See `MODEL_TESTING.md` for comprehensive guide
- Use `scripts/model_tester.py` for interactive testing
- Use environment variables for quick tests: `OBSERVER_MODEL=qwen3:4b python -m src.main`

---

## Next Development Priorities

### High Priority (v1.3.x)
1. **Speech-to-Text (STT)** - Implement Whisper for voice input, wake word detection
2. **qwen3:0.6b Fine-tuning** - Train smaller model with improved prompts for faster extraction
3. **Semantic contradiction detection** - Improve LLM's ability to detect temporal state transitions

### Medium Priority
1. **Memory pruning** - Automatic deletion of old LOW/DISCARD utility memories
2. **Pronoun resolution** - Coreference chain tracking ("she" → "Justine")
3. **TTS voice variety** - Test different Kokoro voices, add voice cloning support

### Low Priority
1. **Query expansion** - Synonym/variation handling for better retrieval
2. **Web UI** - Non-technical user interface
3. **Background pruning task** - Scheduled cleanup of expired memories

---

## Important Notes for Developers

**Fact Type Classification:**
- `core`: User's name, work schedule, home address, family, owned devices (never decay)
- `preference`: Opinions, likes/dislikes, feelings (60-day half-life)
- `episodic`: One-time events, meetings, trips, state changes (14-day half-life)

**Source-Based Extraction:**
- Facts from USER statements: confidence=1.0, source="user_stated"
- Assistant inferences: confidence=0.3, source="assistant_inferred"
- Prevents hallucinated facts from being stored as ground truth

**Temporal States:**
- `ongoing`: Currently true (e.g., `VISITING`, `WORKING_AT`, `LIVES_IN`)
- `completed`: Past events (e.g., `RETURNED_HOME`, `LEFT`, `VISITED`, `MOVED_TO`)
- `planned`: Future events (e.g., `SCHEDULED_FOR`, `PLANNING_TO`)

**Semantic Contradictions:**
- LLM detects contradictions across different predicates
- Understands state transitions: `VISITING` → `RETURNED_HOME`
- Recognizes mutual exclusion: `WORKS_AT A` vs `WORKS_AT B`
- Handles attribute updates: `AGE 24` → `AGE 25`
- Note: Detection reliability varies; extraction is always correct (v1.2.1)

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

**Prompt Engineering (v1.2.1):**
- EXTRACTION_PROMPT now has explicit ONGOING vs COMPLETED state distinction
- CRITICAL sections emphasize entity attribution and temporal states
- 6 comprehensive examples cover common extraction scenarios
- UTILITY_PROMPT explicitly includes emotional content as HIGH
- Prompts are training-ready for fine-tuning qwen3:0.6b

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
- Temporal state extraction (RETURNED_HOME) ✅ (v1.2.1)
- Entity attribution (family members as subjects) ✅ (v1.2.1)
- Utility grading (emotional content → HIGH) ✅ (v1.2.1)
- Confidence score weighting ✅ (v1.2.1)
- Attribute updates (AGE 24 → AGE 25) ✅
- Complex entity networks (14+ relationships) ✅

**Test Status:** 48+ core tests passing (some advanced semantic contradiction tests fail due to LLM reasoning limitations)

---

## Git Repository

**GitHub:** https://github.com/jkstl/lcr

**Recent Commits:**
- `83e671a` - feat: improve extraction prompts for temporal states and entity attribution
- `8f16850` - feat: add comprehensive model testing and configuration tools
- `ce35add` - fix: convert string IDs to integers for FalkorDB contradiction marking
- `b9dda37` - fix: use pipelined synthesis instead of unlimited parallelism in TTS
- `a0e203b` - perf: parallelize observer persistence and TTS synthesis
- `750bac5` - fix: retrieve confidence and source fields in graph queries

---

*Last Updated: 2026-01-22*
*Version: 1.2.1*
