System Flow (as implemented)
The runtime flow is: assemble context → generate response → trigger observer asynchronously. This is defined in the LangGraph workflow and the streaming path triggers observer tasks in the background after response generation.

In the CLI loop, responses are streamed via generate_response_streaming, and observer tasks are awaited only on exit or interruption, which matches the “wait on exit” behavior described in the README (but the real implementation uses explicit wait_for_observers() in the run loop).

Inconsistencies & Potential Issues
1) Reranker is not actually a cross-encoder
README claims a cross-encoder reranker (all-MiniLM-L6-v2) and “cross-encoder relevance scoring.”

The implementation uses sentence-transformers to embed queries/contexts separately and applies cosine similarity (bi-encoder style). This is explicitly called a “cross-encoder proxy.”
Impact: Documentation is misleading; quality/performance expectations may differ from a true cross-encoder reranker.

2) Utility grading levels in README vs code
README shows DISCARD / LOW / MEDIUM / HIGH and the “gatekeeper” for those levels.

The actual prompt is a 3-level system: DISCARD / STORE / IMPORTANT, and the observer enum includes both 3-level and legacy 4-level values (but the prompt is explicitly 3-level).
Impact: Documentation mismatch can confuse tuning and expected behavior.

3) Contradiction checking likely misses object-side relationships
Observer._get_related_facts tries to query both subject and object relationships.

GraphStore.query only matches by subject (both in-memory and Falkor implementations). The object query in the observer therefore won’t find relationships where the entity is in the object position.
Impact: Contradiction detection likely misses cases where the object (not subject) is involved.

4) Vector relevance uses utility score, not semantic similarity
The vector search function computes a combined score based on similarity rank and utility score, then sorts by _combined_score.

However, ContextAssembler._vector_search sets relevance_score and final_score based only on utility_score, ignoring _combined_score (and therefore de-emphasizing semantic similarity).
Impact: The “Top 15 semantic matches” claim is weakened since the pipeline doesn’t carry semantic similarity into downstream scoring.

5) Rerank score multiplication treats 0.0 as 1.0
In _rerank, the multiplier uses score if score else 1.0, which means a valid 0.0 score is treated as 1.0 instead of zeroing out low relevance.
Impact: Irrelevant results could remain competitive when the reranker returns 0.0.

6) Retrieval query parsing is brittle vs JSON parsing helpers
parse_json_response exists for robust JSON extraction from LLM output (including markdown-wrapped JSON).

_generate_retrieval_queries uses raw json.loads(response) and will fail on common formatting patterns without fallback; _extract_structured_data uses parse_json_response instead.
Impact: Query generation is more fragile than entity extraction.

7) Contradiction persistence splits on whitespace
_persist_to_graph_store constructs a new superseded relationship by splitting the existing_statement string into tokens and picking index [0], [1], and [-1].
Impact: Multi-word entities (e.g., “Worcester Massachusetts”) will be truncated, corrupting stored relationships.

Refactoring / Improvement Recommendations
Align documentation with implementation

Update the README to reflect the 3-level utility grading and the bi-encoder reranker approach (or replace the reranker with a true cross-encoder if that’s the intent).

Fix graph querying for contradictions

Extend GraphStore.query to allow filtering by object (or add a dedicated object query). Then update _get_related_facts to properly retrieve subject+object relationships.

Use semantic similarity in vector relevance

Carry _combined_score (or a real similarity score) from vector_search into ContextAssembler._vector_search and use it as relevance_score. This aligns with the intended “semantic match” behavior.

Fix rerank score multiplication

Change the conditional to if score is not None to avoid interpreting a real 0.0 as 1.0.

Standardize JSON parsing

Use parse_json_response inside _generate_retrieval_queries for consistency and resilience to markdown formatting.

Store contradiction fields explicitly

Avoid splitting existing_statement. Carry structured subject/predicate/object in contradiction payloads so multi-word entities are preserved.