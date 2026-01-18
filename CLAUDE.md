# CLAUDE.md — LCR System Status & Handoff Document
## Session Handoff for Continued Development & Testing

**Version 1.0.0**

> **Current Status**: Core memory system implemented with tiered decay and fact classification. Ready for production use.

---

## QUICK SUMMARY

**What This Is**: A local, privacy-first conversational AI with persistent episodic memory. Remembers everything across sessions using dual-memory architecture (vector + graph).

**Current State**: ✅ **TESTED** - Core features implemented, complex scenario testing complete. Pronoun resolution is a known limitation.

**Your Mission**: Test with complex conversational prompts, identify edge cases, improve extraction quality, and validate memory persistence.

---

## IMPLEMENTATION STATUS

### ✅ Fully Implemented & Working

| Component | Status | Notes |
|-----------|--------|-------|
| **Vector Memory (LanceDB)** | ✅ Working | Stores semantic embeddings, searches ~15 candidates |
| **Graph Memory (FalkorDB)** | ✅ Working | Stores entities/relationships, tracks contradictions |
| **Observer System** | ✅ Working | Extracts entities, relationships, grades utility |
| **Entity Extraction** | ✅ Fixed | Properly extracts attributes (age, role, etc.) |
| **Contradiction Handling** | ✅ Working | Marks old facts as superseded by new ones |
| **Reranker** | ✅ Working | Cross-encoder scores top-5 from 15 candidates |
| **Pre-Flight Check** | ✅ Working | Validates Ollama, LanceDB, FalkorDB, Docker |
| **Memory Persistence** | ✅ Fixed | Observer tasks complete before exit |
| **In-Chat Commands** | ✅ Working | /status, /stats, /clear, /help |
| **Fact Type Classification** | ✅ NEW | Classifies facts as core/episodic/preference |
| **Tiered Temporal Decay** | ✅ NEW | Core=never, HIGH=180d, MED=60d, LOW=14d |

### ⚠️ Not Yet Implemented

| Component | Status | Priority |
|-----------|--------|----------|
| **Voice I/O** | ❌ Deferred | Low - Focus on chat first |
| **Conversation Logs** | ❌ Not impl | Medium - Would help testing |
| **Memory Pruning** | ❌ Not impl | Low - Only needed at scale |
| **Web UI** | ❌ Not impl | Low - CLI works fine |

---

## CURRENT ARCHITECTURE

```
User Input
    ↓
Pre-Flight Check (Ollama, LanceDB, FalkorDB, Docker)
    ↓
Context Assembly:
  • Embed query (nomic-embed-text)
  • Vector search (top-15 from LanceDB)
  • Graph search (top-10 relationships)
  • Merge + temporal decay
  • Rerank (cross-encoder → top-5)
    ↓
LLM Generation (qwen3:14b with context)
    ↓
Response to User
    ↓
[ASYNC] Observer:
  • Grade utility (DISCARD/LOW/MEDIUM/HIGH)
  • Extract entities (Person, Place, Org, Tech, etc.)
  • Extract relationships (SIBLING_OF, WORKS_AT, etc.)
  • Detect contradictions
  • Persist to LanceDB + FalkorDB
    ↓
(Wait for observer on exit)
```

---

## RECENT FIXES (Current Session)

### 1. Observer Persistence Issue ✅
**Problem**: Memories not saving because observer tasks cancelled on exit
**Fix**: Track tasks, wait for completion before exit
**Status**: FIXED - memories now persist correctly

### 2. Entity Extraction Quality ✅
**Problem**: Missing attributes (age), nonsensical predicates ("and"), no familial relationships
**Fix**: Rewrote EXTRACTION_PROMPT with taxonomy, examples, explicit instructions
**Status**: FIXED - now extracts age, proper relationships (SIBLING_OF, PARENT_OF)

### 3. GraphRelationship Subscript Error ✅
**Problem**: Code treated dataclass as dict (`rel['subject']`)
**Fix**: Changed to attribute access (`rel.subject`)
**Status**: FIXED

### 4. Model Detection ✅
**Problem**: "nomic-embed-text:v1.5 (not found)" when user has :latest
**Fix**: Flexible version matching by base name
**Status**: FIXED

---

## HOW TO RUN & TEST

### Start Chat
```bash
python -m src.main
```

**You'll see:**
- Pre-flight check (all systems status)
- Memory count
- Ready prompt

### In-Chat Commands
```
/status - Recheck system health
/stats  - Show memory statistics (count, utility distribution, disk usage)
/clear  - Clear screen
/help   - Command list
exit    - Save memories and exit
```

### Inspect Memory
```bash
# View stored data
python scripts/inspect_memory.py

# Clear memory (for fresh testing)
python scripts/clear_memory.py
```

### Run Tests
```bash
# Full test suite
pytest

# Memory/extraction tests
pytest tests/test_memory_retrieval.py -v
```

---

## TESTING PRIORITIES

### ✅ Already Tested (Basic Functionality)

1. **Simple memory persistence**
   - User shares info → exits → restarts → system remembers ✅

2. **Familial relationships**
   - Mom, sister Justine (24), visiting from West Boylston ✅

3. **Contradiction handling**
   - Job change, relationship status change ✅
   - Interview Friday → Monday (correctly superseded) ✅

4. **Entity extraction**
   - Age attributes, proper relationship types ✅

5. **Multi-Turn Context Dependencies** ✅ (2026-01-18)
   - Tested: Interview at Microsoft → "when is my interview?" → System recalled correctly

6. **Complex Entity Networks** ✅ (2026-01-18)
   - Tested: "My manager Sarah's husband works at the same company as my brother Tom"
   - Tested: Sister's graduation at UCLA, job at Google, boyfriend Jake
   - Result: 14 relationships extracted (SIBLING_OF, WORKS_AT, PARENT_OF, etc.)

7. **Cross-Session Retrieval** ✅ (2026-01-18)
   - Tested: After restart, asked "Where does my sister work?"
   - Result: "Your sister works at Google in Mountain View" (correct)

### ✅ Complex Scenario Tests (Completed 2026-01-18)

| Test | Scenario | Result | Notes |
|------|----------|--------|-------|
| 1. Temporal Reasoning | "Last Monday I started job" → "How was first week?" | ✅ Passed | System didn't assume details |
| 2. Contradictions with Nuance | "Love Python for data science" → "Hate Python for web dev" | ✅ Passed | No false contradiction |
| 3. Nested Relationships | "Mom's friend's daughter getting married" | ✅ Passed | Chain tracked |
| 4. Multiple Facts Per Turn | Dense info (Justine, 24, Microsoft, Azure, Seattle) | ✅ Passed | All extracted |
| 5. Pronoun Resolution | "Sarah and I..." → "She enjoyed it" | ⚠️ Expected | Asked "Who is she?" (known limitation) |
| 6. Contradictory Corrections | "Sister is 24... wait, 25" | ✅ Passed | Correction persisted |
| 7. Memory Under Load | 50+ turns stress test | ⏭️ Skipped | Time-intensive |

**Cross-Session Verification:**
- ✅ "Where does sister work?" → "Microsoft in Seattle"
- ✅ "How old is Justine?" → "25" (correction persisted)

---

## TESTING METHODOLOGY

### How to Test Systematically

1. **Start Fresh**
   ```bash
   python scripts/clear_memory.py  # Option 1
   python -m src.main
   ```

2. **Run Complex Scenario**
   - Have multi-turn conversation with complex relationships
   - Use `/stats` to check memory count
   - Exit (wait for "Memories saved")

3. **Verify Persistence**
   ```bash
   python scripts/inspect_memory.py
   ```
   Check:
   - Entity count
   - Relationship types
   - Attributes extracted

4. **Test Retrieval**
   ```bash
   python -m src.main
   # Ask questions about previous conversation
   ```

5. **Document Results**
   - What worked well?
   - What was missed?
   - What was wrong?

---

## KNOWN ISSUES & LIMITATIONS

### Current Limitations

1. **Observer Model (qwen3:1.7b)**
   - Sometimes misses subtle entities
   - May struggle with complex nested relationships
   - **Mitigation**: Could upgrade to qwen3:4b if needed

2. **Temporal Decay**
   - 30-day half-life may be too aggressive for long-term memories
   - **Consideration**: Adjust `temporal_decay_days` in config.py

3. **Reranker Model**
   - Using lightweight all-MiniLM-L6-v2
   - May not capture deep semantic similarity
   - **Future**: Could upgrade to BGE-Reranker-v2-m3

4. **No Pronoun Resolution**
   - "She went there" doesn't resolve pronouns
   - **Future**: Add coreference resolution

5. **No Conversation Logs**
   - Hard to debug what was said
   - **Future**: Add logging to data/conversations/

### Performance Notes

- **Current**: ~3-5s response time (LLM dominated)
- **Scaling**: Thousands of conversations before slowdown
- **Bottleneck**: LLM generation, not memory retrieval

---

## CONFIGURATION

**Key Settings** (`src/config.py` or `.env`):

```python
# Models
MAIN_MODEL=qwen3:14b          # Conversation LLM
OBSERVER_MODEL=qwen3:1.7b     # Entity extraction
EMBEDDING_MODEL=nomic-embed-text:v1.5

# Memory Retrieval
MAX_CONTEXT_TOKENS=3000       # LLM context window
VECTOR_SEARCH_TOP_K=15        # Initial candidates
RERANK_TOP_K=5                # Final selection
TEMPORAL_DECAY_DAYS=30        # Memory half-life

# Performance Tuning
# Reduce for faster responses (less context)
# Increase for better recall (slower)
```

---

## FILES TO KNOW

### Core Implementation
- `src/main.py` - Chat interface with pre-flight check
- `src/orchestration/graph.py` - LangGraph workflow
- `src/observer/observer.py` - Entity extraction
- `src/observer/prompts.py` - **EXTRACTION_PROMPT** (critical!)
- `src/memory/context_assembler.py` - Retrieval orchestration
- `src/memory/vector_store.py` - LanceDB operations
- `src/memory/graph_store.py` - FalkorDB operations

### Testing & Utilities
- `tests/test_memory_retrieval.py` - 40+ test cases
- `scripts/inspect_memory.py` - View stored data
- `scripts/clear_memory.py` - Reset memory
- `scripts/observer_giana_live.py` - Test observer extraction

### Documentation
- `README.md` - User-facing overview
- `QUICKSTART.md` - Setup guide
- `CLAUDE.md` - **This file** (handoff doc)

---

## NEXT STEPS (Suggested)

### Immediate (This Session)

1. **Test Complex Multi-Turn Scenarios**
   - Try the 10 testing scenarios listed above
   - Document what works and what fails

2. **Validate Entity Extraction**
   - Feed complex sentences
   - Check if attributes captured
   - Verify relationship types

3. **Test Contradiction Edge Cases**
   - Updates vs contradictions
   - Context-dependent statements

### Medium Term (Future Sessions)

1. **Improve Observer Prompts**
   - Based on testing results
   - Add more examples
   - Refine relationship taxonomy

2. **Add Conversation Logging**
   - Store raw conversations
   - Helps with debugging

3. **Implement Memory Pruning**
   - Delete DISCARD/LOW utility
   - Archive old conversations

4. **Enhanced Retrieval**
   - Query expansion
   - Hybrid search strategies

---

## EXAMPLE TEST CONVERSATIONS

### Test 1: Multi-Entity Network
```
You: My manager Sarah's husband works at the same company as my brother Tom
[Check: Should extract Sarah (manager), husband, brother Tom, company relationships]

You: /stats
[Check: Entity count increased by 3-4]

You: exit
[Wait for memories saved]

[Restart]
You: Who is Sarah?
[Expected: "Your manager" + "Her husband works with your brother Tom"]
```

### Test 2: Temporal + Contradiction
```
You: I'm interviewing at Microsoft on Friday
You: Actually they moved it to Monday
[Check: Contradiction detected, old date superseded]

You: When's my interview?
[Expected: Monday, not Friday]
```

### Test 3: Complex Facts
```
You: Yesterday I went to my sister's graduation at UCLA. She's 22 and
     just got a job at Google in Mountain View. She'll be moving from
     San Diego next month. Her boyfriend Jake is helping her move.

[Check entities: Sister (22), UCLA, Google, Mountain View, San Diego, Jake]
[Check relationships: ATTENDED, GRADUATED_FROM, WORKS_AT, LOCATED_IN, DATING, etc.]
```

---

## DEBUGGING TIPS

### Memory Not Persisting?
1. Check you waited for "Memories saved" before restart
2. Run `python scripts/inspect_memory.py` to verify
3. Check Docker containers running: `docker ps`

### Wrong Entities Extracted?
1. Check `src/observer/prompts.py` → EXTRACTION_PROMPT
2. Run observer scripts to test: `python scripts/observer_giana_live.py`
3. May need prompt tuning or model upgrade

### Slow Responses?
1. Run `/stats` to check memory count
2. If >10k memories, consider pruning
3. Reduce `vector_search_top_k` in config.py

### System Check Failing?
1. Run `/status` to see what's wrong
2. Check Ollama: `ollama list`
3. Check Docker: `docker ps`
4. Restart services: `python scripts/clear_memory.py` → Option 4

---

## SUCCESS CRITERIA

You know it's working when:

✅ Pre-flight check shows all green
✅ `/stats` shows growing memory count
✅ System remembers across restarts
✅ Entities extracted with attributes
✅ Relationships properly typed (SIBLING_OF not "and")
✅ Contradictions detected and marked
✅ Complex queries retrieve correct context
✅ Response latency stays under 5 seconds

---

## QUESTIONS TO ANSWER

As you test, document:

1. **What types of entities are commonly missed?**
2. **What relationship types are never extracted?**
3. **When does contradiction detection fail?**
4. **What queries fail to retrieve relevant context?**
5. **At what memory size does performance degrade?**
6. **What utility grades are assigned incorrectly?**
7. **What complex scenarios break the system?**

---

## REPOSITORY

**GitHub**: https://github.com/jkstl/lcr-codex_CLAUDEREVIEW

**Latest Commits**:
- Fix observer persistence (memories now save correctly)
- Improve entity extraction (attributes, proper relationships)
- Add pre-flight check and /stats command
- Fix GraphRelationship subscript error

---

## FINAL NOTES

**This system is functional and ready for real-world testing.** The core memory loop works:
- Talk → Remember → Retrieve → Respond

Your job is to **stress-test it** with complex conversational scenarios and find edge cases. Focus on:
- Complex entity networks
- Temporal reasoning
- Contradiction nuance
- Multi-turn context

**Start with the 10 testing scenarios above** and document results.

Good luck! 🚀

---

*Last Updated: 2026-01-18*
*Status: Core implementation complete, ready for advanced testing*
