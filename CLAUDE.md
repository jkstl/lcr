# CLAUDE.md — Developer Handoff Document

**Version 1.1.3** | **Status: Production-Ready**

This document provides essential context for developers continuing work on the LCR system. For user-facing documentation, see [README.md](README.md).

---

## Quick Context

**What is LCR?** Local Cognitive RAG - A privacy-first conversational AI with persistent episodic memory using dual-architecture (vector + graph).

**Current Focus:** System is production-ready. Priority areas are performance optimization, memory pruning, and edge case testing.

**Recent Major Changes (v1.1.3):**
- Enhanced utility grading to prevent project descriptions from being discarded  
- Implemented conversation logging to `data/conversations/` (structured JSON)
- Added defensive logging for utility grading decisions
- Fixed birthday/date extraction in Observer EXTRACTION_PROMPT
- Improved relationship formatting to prevent past-relationship misinterpretation (e.g., BROKE_UP_WITH)

---

## Architecture Overview

```
User Input → Pre-Flight Check → Context Assembly (Parallel) → LLM Generation → Response
                                     ↓ Async Observer (Parallel Processing)
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
- Temporal decay with tiered half-life (core/HIGH/MEDIUM/LOW)

---

## Critical Files for Development

### Core Logic
- `src/observer/observer.py` - Entity extraction, contradiction detection, persistence
- `src/observer/prompts.py` - **UTILITY_PROMPT**, EXTRACTION_PROMPT, SEMANTIC_CONTRADICTION_PROMPT
- `src/memory/context_assembler.py` - Retrieval, filtering, temporal decay, recency boost
- `src/memory/graph_store.py` - FalkorDB operations, superseded fact tracking
- `src/orchestration/graph.py` - LangGraph state machine, streaming generation

### Configuration
- `src/config.py` - All settings (models, top-k, decay rates)
  - **Embedding model:** `nomic-embed-text` (768-dim, optimized for semantic search)
  - **Main model:** `qwen3:14b`
  - **Observer model:** `qwen3:1.7b`
- `.env` - Environment overrides

### Utilities
- `scripts/inspect_memory.py` - View stored memories
- `scripts/view_conversations.py` - Browse conversation logs
- `scripts/nuclear_reset.py` - Complete memory wipe

---

## Known Issues & Debugging

### If memories aren't persisting:
1. Ensure graceful exit with `exit` command (waits for observer)
2. Check utility grading: `grep "Utility grading:" <terminal_output>`
3. Verify observer completion before shutdown
4. Inspect stored data: `python scripts/inspect_memory.py`

### If old facts are surfacing:
1. Check contradiction detection is marking facts as superseded
2. Verify filtering logic in `context_assembler.py` (_graph_search)
3. Look for `superseded_by` field in graph relationships

### If utility grading is wrong:
1. Review `UTILITY_PROMPT` in `src/observer/prompts.py`
2. Check defensive logs for grading decisions
3. Consider upgrading observer model from qwen3:1.7b to :4b

### If extraction quality is poor:
1. Observer model (qwen3:1.7b) may struggle with complex sentences
2. Upgrade option: `OBSERVER_MODEL=qwen3:4b` in config
3. Check extraction prompts in `src/observer/prompts.py`
4. Review relationship formatting in `src/memory/context_assembler.py`

---

## Testing

**Run all tests:**
```bash
pytest                                          # All tests
pytest tests/test_memory_retrieval.py -v       # Core memory
pytest tests/test_semantic_contradictions.py -v # Contradiction detection
```

**Test coverage:**
- Cross-session persistence
- Familial relationships (SIBLING_OF, PARENT_OF)
- Contradiction handling (job changes, location moves)
- Semantic state transitions (VISITING → RETURNED_HOME)
- Attribute updates (AGE 24 → AGE 25)
- Complex entity networks (14+ relationships)

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

---

## Next Development Priorities

### High Priority
1. **Memory pruning** - Automatic deletion of old LOW/DISCARD utility memories
2. **Monitor utility grading** - Ensure enhanced prompt works in production
3. **Observer model evaluation** - Test qwen3:1.7b vs :4b extraction quality

### Medium Priority
1. **Pronoun resolution** - Coreference chain tracking
2. **Query expansion** - Synonym/variation handling for better retrieval
3. **Recency boost tuning** - Optimize the 30% boost for recent corrections

### Low Priority
1. **Web UI** - Non-technical user interface
2. **Voice I/O** - TTS/STT integration (streaming already supports this)
3. **Background pruning task** - Scheduled cleanup of expired memories

---

## Important Notes for Developers

**Utility Grading Bug (Fixed in v1.1.3):**
- Previous versions incorrectly graded detailed technical discussions as DISCARD
- Enhanced prompt now explicitly recognizes projects, technical details, user work as HIGH
- Test case: `test_utility_fix.py` validates the fix

**Fact Type Classification:**
- `core`: User's name, work schedule, home address, family, owned devices (never decay)
- `preference`: Opinions, likes/dislikes, feelings (60-day half-life)
- `episodic`: One-time events, meetings, trips (14-day half-life)

**Source-Based Extraction:**
- Facts extracted from USER statements only (confidence=1.0)
- Assistant inferences tagged separately (confidence=0.3)
- Prevents hallucinated facts from being stored as ground truth

**Temporal States:**
- `ongoing`: Currently true (e.g., "visiting", "working at")
- `completed`: Past events (e.g., "visited", "worked at")
- `planned`: Future events (e.g., "scheduled for", "planning to")

**Contradictions:**
- LLM detects semantic contradictions across different predicates
- Understands state transitions: VISITING → RETURNED_HOME
- Recognizes mutual exclusion: WORKS_AT A vs WORKS_AT B
- Handles attribute updates: AGE 24 → AGE 25

---

## Git Repository

**GitHub:** https://github.com/jkstl/lcr (migrating from lcr-codex_CLAUDEREVIEW)

**Latest Commit:** 37496b6 (v1.1.3 - extraction fixes)

**Test Status:** All 43+ tests passing

---

*Last Updated: 2026-01-20*
*Version: 1.1.3*
