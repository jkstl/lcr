# CLAUDE.md — LCR System Status & Handoff Document
## Session Handoff for Continued Development & Testing

**Version 1.1.3**

> **Current Status**: Production-ready with source-based confidence tracking to prevent hallucinated facts from being stored as ground truth.

---

## QUICK SUMMARY

**What This Is**: A local, privacy-first conversational AI with persistent episodic memory. Remembers everything across sessions using dual-memory architecture (vector + graph).

**Current State**: ✅ **PRODUCTION READY** - v1.1.3 with enhanced utility grading and conversation logging.

**Your Mission**: Continue testing edge cases, monitor contradiction detection accuracy, optimize performance, add features.

---

## IMPLEMENTATION STATUS

### ✅ Fully Implemented & Working (v1.1.1)

| Component | Status | Notes |
|-----------|--------|-------|
| **Vector Memory (LanceDB)** | ✅ Optimized | Combined similarity + utility scoring |
| **Graph Memory (FalkorDB)** | ✅ Optimized | Parallel search, superseded fact filtering |
| **Streaming Output** | ✅ NEW | Real-time token streaming for TTS readiness |
| **Project Memory** | ✅ NEW | WORKS_ON predicate + smart entity extraction |
| **Source-Based Extraction** | ✅ NEW | Facts extracted from USER only, not assistant hallucinations |
| **Semantic Contradiction Detection** | ✅ Working | LLM-powered, understands temporal transitions |
| **Temporal State Tracking** | ✅ Working | Ongoing/completed/planned status tracking |
| **Observer System** | ✅ Optimized | Parallel LLM tasks, early exit for DISCARD |
| **Entity Extraction** | ✅ Working | Extracts attributes, proper relationship types |
| **Reranker** | ✅ Working | Cross-encoder scores top-5 from 15 candidates |
| **Pre-Flight Check** | ✅ Working | Validates Ollama, LanceDB, FalkorDB, Docker |
| **Memory Persistence** | ✅ Working | Observer tasks complete before exit |
| **Fact Type Classification** | ✅ Working | Core/episodic/preference with tiered decay |
| **Conversation Logging** | ✅ NEW | Full history saved to data/conversations/ |
| **Enhanced Utility Grading** | ✅ NEW | Recognizes technical/project discussions |

### ⚠️ Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Pronoun Resolution** | Can't resolve "she/he/it" | Use explicit names |
| **Observer Model (1.7B)** | May miss subtle entities | Upgrade to qwen3:4b if needed |

---

## ARCHITECTURE

```
User Input
    ↓
Pre-Flight Check (Ollama, LanceDB, FalkorDB, Docker)
    ↓
Context Assembly (PARALLEL):
  • Embed query (nomic-embed-text)
  • Vector search (top-15 from LanceDB)  ┐
  • Graph search (top-10 relationships)  ┴─ Run concurrently (~100ms)
  • Filter superseded/expired facts
  • Apply temporal decay + recency boost
  • Rerank (cross-encoder → top-5)
    ↓
LLM Generation (qwen3:14b with context) ~2-3s
    ↓
Response to User
    ↓
[ASYNC] Observer (PARALLEL):
  1. Grade utility (DISCARD/LOW/MEDIUM/HIGH) - Gatekeeper
  2. IF DISCARD → Early exit (save ~4x time)
  3. ELSE parallel processing:
     • Extract entities + relationships    ┐
     • Generate summary                    ├─ Run concurrently
     • Generate retrieval queries          ┘
  4. Semantic contradiction detection (LLM):
     • VISITING → RETURNED_HOME (state completion)
     • WORKS_AT A → WORKS_AT B (mutual exclusion)
     • AGE 24 → AGE 25 (attribute update)
  5. Mark superseded facts (status="completed")
  6. Persist to LanceDB + FalkorDB
    ↓
(Wait for observer on exit)
```

---

## KEY FEATURES (v1.1.1)

### Streaming Output (NEW)
- **Real-time token streaming** from Ollama
- **TTS-ready** architecture for future voice integration
- **Responsive UI** shows tokens as they generate

### Improved Memory Retrieval (NEW)
- **WORKS_ON predicate** for reliable project memory
- **Smart entity extraction** recognizes "my projects" queries
- **Combined scoring** balances similarity (70%) and utility (30%)
- **Anti-hallucination guardrails** prevent fabricating details

### Semantic Contradiction Detection
- **LLM-powered reasoning** across different predicates
- **Temporal state transitions**: VISITING → RETURNED_HOME
- **Mutually exclusive states**: WORKS_AT CompanyA vs CompanyB
- **Attribute updates**: AGE 24 → AGE 25
- **Fallback detection**: Simple predicate matching if LLM fails

### Temporal State Tracking
- **Status field**: "ongoing" | "completed" | "planned"
- **Valid until**: Expiration timestamps for episodic events
- **Superseded by**: Links to facts that replaced old ones
- **Filtering**: Automatically hides superseded/expired facts

### Performance Optimizations
- **Parallel database queries**: Vector + graph search concurrent (~33% faster)
- **Early exit**: DISCARD turns skip 3 LLM calls (~4x faster)
- **Parallel observer**: 3 LLM tasks concurrent (~3x faster)
- **Recency boost**: Recent corrections get 30% relevance boost
- **State preference**: Ongoing facts boosted 20% over completed

---

## CONFIGURATION

**Key Settings** (`src/config.py` or `.env`):

```python
# Models
MAIN_MODEL=qwen3:14b              # Conversation LLM
OBSERVER_MODEL=qwen3:1.7b         # Entity extraction (upgrade to :4b if needed)
EMBEDDING_MODEL=nomic-embed-text  # Flexible version matching

# Memory Retrieval
MAX_CONTEXT_TOKENS=3000           # LLM context window
SLIDING_WINDOW_TOKENS=2000        # Recent conversation retention
VECTOR_SEARCH_TOP_K=15            # Initial candidates
GRAPH_SEARCH_TOP_K=10             # Graph relationships
RERANK_TOP_K=5                    # Final selection after reranking

# Temporal Decay (tiered by utility)
TEMPORAL_DECAY_CORE=0             # Core facts never decay
TEMPORAL_DECAY_HIGH=180           # HIGH utility: 6 months
TEMPORAL_DECAY_MEDIUM=60          # MEDIUM utility: 2 months
TEMPORAL_DECAY_LOW=14             # LOW utility: 2 weeks
```

**Performance Tuning:**
- Reduce `VECTOR_SEARCH_TOP_K` for faster responses (less context)
- Increase for better recall (slower responses)
- Adjust decay rates based on use case

---

## HOW TO RUN & TEST

### Start Chat
```bash
python -m src.main
```

### In-Chat Commands
```
/status - System health check (models, databases, Docker)
/stats  - Memory statistics (count, utility distribution, types)
/clear  - Clear screen (history preserved)
/help   - Command list
exit    - Save memories and exit gracefully
```

### Inspect Memory
```bash
python scripts/inspect_memory.py      # View stored data
python scripts/clear_memory.py        # Interactive memory management
python scripts/view_conversations.py  # NEW: View/export conversation logs
python scripts/nuclear_reset.py       # NEW: Complete memory wipe
```

### Run Tests
```bash
pytest                              # All 43+ tests
pytest tests/test_memory_retrieval.py -v           # Core memory tests
pytest tests/test_semantic_contradictions.py -v    # Contradiction tests
```

---

## TESTING PRIORITIES

### ✅ Validated Scenarios

1. **Cross-session memory persistence** ✅
2. **Familial relationships** (SIBLING_OF, PARENT_OF) ✅
3. **Contradiction handling** (job change, age corrections) ✅
4. **Semantic state transitions** (visiting → returned home) ✅
5. **Multi-turn context dependencies** ✅
6. **Complex entity networks** (14+ relationships) ✅
7. **Attribute extraction** (age, role, location) ✅

### 🔍 Test These Next

1. **Long conversation sessions** (50+ turns)
2. **Memory under load** (1000+ stored facts)
3. **Complex temporal reasoning** ("last Tuesday", "next month")
4. **Ambiguous contradictions** (context-dependent statements)
5. **Observer accuracy** with qwen3:1.7b vs :4b

---

## FILES TO KNOW

### Critical Files
- `src/observer/prompts.py` - **SEMANTIC_CONTRADICTION_PROMPT** + EXTRACTION_PROMPT
- `src/observer/observer.py` - Contradiction detection logic
- `src/memory/context_assembler.py` - Retrieval + filtering + boosting
- `src/memory/graph_store.py` - Temporal metadata handling
- `src/main.py` - Chat interface + pre-flight check

### Test Files
- `tests/test_memory_retrieval.py` - 43 core memory tests
- `tests/test_semantic_contradictions.py` - Temporal state tests
- `tests/test_integration.py` - End-to-end tests
- `tests/test_observer.py` - Entity extraction tests

### Utilities
- `scripts/inspect_memory.py` - View memory contents
- `scripts/clear_memory.py` - Memory management menu
- `scripts/observer_giana_live.py` - Test observer extraction

---

## DEBUGGING TIPS

### Contradiction Not Detected?
1. Check `src/observer/prompts.py` → SEMANTIC_CONTRADICTION_PROMPT
2. Test observer: `python scripts/observer_giana_live.py`
3. Verify LLM response format (should be JSON with "contradictions" array)
4. Check fallback: `_simple_contradiction_check` should catch same-predicate changes

### Memory Not Persisting?
1. Wait for "Memories saved. Goodbye!" before restart
2. Check Docker: `docker ps` (FalkorDB must be running)
3. Verify: `python scripts/inspect_memory.py`

### Slow Responses?
1. Check memory count: `/stats`
2. If >10k memories, clear low-utility: `python scripts/clear_memory.py`
3. Reduce `vector_search_top_k` in config.py

### Old Facts Surfacing?
1. Verify contradiction detection is marking facts as superseded
2. Check graph search filtering logic in `context_assembler.py`
3. Inspect: `python scripts/inspect_memory.py` → look for `superseded_by` field

---

## NEXT STEPS

### Immediate
1. Test semantic contradiction detection with real-world scenarios
2. Monitor false positives/negatives in contradiction detection
3. Tune recency boost and state preference weights

### Short-Term
1. Implement memory pruning for LOW/DISCARD utility
2. Upgrade observer to qwen3:4b if extraction quality insufficient

### Long-Term
1. Add pronoun resolution (coreference)
2. Query expansion for better retrieval
3. Web UI for non-technical users
4. Voice I/O integration

---

## PERFORMANCE BENCHMARKS

| Metric | v1.0.0 | v1.1.0 | Improvement |
|--------|--------|--------|-------------|
| DISCARD turns | ~2s | ~0.5s | **4x faster** |
| HIGH/MEDIUM turns | ~6s | ~2-3s | **2-3x faster** |
| Context retrieval | ~150ms | ~100ms | **1.5x faster** |
| Contradiction detection | Same-predicate only | Semantic | **Much smarter** |

---

## SUCCESS CRITERIA

✅ Pre-flight check passes
✅ All 43+ tests pass
✅ System remembers across restarts
✅ Contradictions detected and superseded
✅ Old facts filtered from retrieval
✅ Response time <5 seconds
✅ Entities extracted with attributes
✅ Relationships properly typed

---

## REPOSITORY

**GitHub**: https://github.com/jkstl/lcr-codex_CLAUDEREVIEW

**Recent Commits (v1.1.0)**:
- `56b3791` - Semantic contradiction detection + temporal state tracking
- `254414a` - Parallelize database queries + observer LLM tasks
- `e2c1b9c` - Update README with accurate configuration

**Test Status**: 43/43 passing ✅

---

## FINAL NOTES

**v1.1.0 is production-ready** with major improvements:
- Semantic contradiction detection solves the "Mom visiting" bug
- Performance optimizations make it 2-4x faster
- Temporal state tracking prevents outdated facts from surfacing

**Focus areas for next agent:**
1. Monitor contradiction detection accuracy in production
2. Tune recency boost and state preference weights
3. Test with 1000+ stored memories
4. Consider observer model upgrade if extraction quality drops

Good luck! 🚀

---

*Last Updated: 2026-01-20*
*Version: 1.1.3*
*Status: Production-ready with enhanced utility grading and conversation logging*
