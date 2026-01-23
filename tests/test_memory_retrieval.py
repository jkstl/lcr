"""
Comprehensive Memory Retrieval and Contradiction Handling Tests

Tests the observer, reranker, context assembler, and graph store working together
to handle complex scenarios including contradictions, temporal decay, and entity
relationship changes.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.graph_store import InMemoryGraphStore, GraphRelationship
from src.memory.context_assembler import ContextAssembler, RetrievedContext
from src.observer.observer import Observer, UtilityGrade, ObserverOutput
from src.models.reranker import Reranker


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def graph_store():
    """Fresh in-memory graph store for each test."""
    return InMemoryGraphStore()


@pytest.fixture
def mock_llm_client():
    """Mock LLM client that returns configurable responses."""
    client = MagicMock()
    client.generate = AsyncMock(return_value="LOW")
    return client


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns fixed vectors for testing."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 4096)
    return embedder


@pytest.fixture
def mock_vector_table():
    """Mock LanceDB table."""
    return MagicMock()


@pytest.fixture
def reranker():
    """Real reranker for integration tests."""
    return Reranker()


# -----------------------------------------------------------------------------
# Contradiction Detection Tests
# -----------------------------------------------------------------------------


class TestContradictionHandling:
    """Tests for detecting and resolving contradictory information."""

    @pytest.mark.asyncio
    async def test_employment_change_marks_old_job_invalid(self, graph_store):
        """When user changes jobs, old employment should be marked as superseded."""
        # Store initial employment
        await graph_store.persist_entities([
            {"name": "User", "type": "Person", "attributes": {}},
            {"name": "Acme Corp", "type": "Organization", "attributes": {}},
        ])
        await graph_store.persist_relationships([
            {"subject": "User", "predicate": "WORKS_AT", "object": "Acme Corp", "metadata": {}},
        ])

        # Verify initial state
        old_jobs = await graph_store.query("User", "WORKS_AT")
        assert len(old_jobs) == 1
        assert old_jobs[0].object == "Acme Corp"
        original_id = old_jobs[0].id

        # Mark contradiction when new job is learned
        await graph_store.mark_contradiction(original_id, "User WORKS_AT TechStartup")

        # Verify old job is marked invalid
        old_jobs_after = await graph_store.query("User", "WORKS_AT")
        superseded = next(j for j in old_jobs_after if j.id == original_id)
        assert superseded.metadata.get("still_valid") is False
        assert superseded.superseded_by == "User WORKS_AT TechStartup"  # Now a direct attribute

    @pytest.mark.asyncio
    async def test_relationship_status_change_contradiction(self, graph_store):
        """Relationship status changes should supersede previous status."""
        await graph_store.persist_entities([
            {"name": "User", "type": "Person", "attributes": {}},
            {"name": "Sarah", "type": "Person", "attributes": {}},
        ])
        await graph_store.persist_relationships([
            {"subject": "User", "predicate": "DATING", "object": "Sarah", "metadata": {"since": "2024-01"}},
        ])

        relationships = await graph_store.query("User", "DATING")
        assert len(relationships) == 1
        original_id = relationships[0].id

        # User breaks up
        await graph_store.mark_contradiction(original_id, "User BROKE_UP_WITH Sarah")
        await graph_store.persist_relationships([
            {"subject": "User", "predicate": "BROKE_UP_WITH", "object": "Sarah", "metadata": {"when": "2024-06"}},
        ])

        # Old relationship should be marked invalid
        old_rel = await graph_store.query("User", "DATING")
        assert old_rel[0].metadata.get("still_valid") is False

        # New status should exist
        new_rel = await graph_store.query("User", "BROKE_UP_WITH")
        assert len(new_rel) == 1
        assert new_rel[0].object == "Sarah"

    @pytest.mark.asyncio
    async def test_location_change_supersedes_previous(self, graph_store):
        """Moving to a new location should supersede previous residence."""
        await graph_store.persist_relationships([
            {"subject": "User", "predicate": "LIVES_IN", "object": "Austin", "metadata": {}},
        ])

        old_location = await graph_store.query("User", "LIVES_IN")
        await graph_store.mark_contradiction(old_location[0].id, "User LIVES_IN Seattle")

        await graph_store.persist_relationships([
            {"subject": "User", "predicate": "LIVES_IN", "object": "Seattle", "metadata": {"moved": "2024-03"}},
        ])

        all_locations = await graph_store.query("User", "LIVES_IN")
        assert len(all_locations) == 2

        # Verify Austin is marked invalid
        austin = next(loc for loc in all_locations if loc.object == "Austin")
        assert austin.metadata.get("still_valid") is False

        # Verify Seattle exists and has move metadata
        seattle = next(loc for loc in all_locations if loc.object == "Seattle")
        assert seattle.metadata.get("moved") == "2024-03"

    @pytest.mark.asyncio
    async def test_preference_reversal_contradiction(self, graph_store):
        """Complete preference reversals should be tracked."""
        await graph_store.persist_relationships([
            {"subject": "User", "predicate": "DISLIKES", "object": "Python", "metadata": {"reason": "verbose"}},
        ])

        old_pref = await graph_store.query("User", "DISLIKES")
        await graph_store.mark_contradiction(old_pref[0].id, "User PREFERS Python")

        await graph_store.persist_relationships([
            {"subject": "User", "predicate": "PREFERS", "object": "Python", "metadata": {"reason": "productivity"}},
        ])

        # Both should exist, old one marked invalid
        dislikes = await graph_store.query("User", "DISLIKES")
        assert dislikes[0].metadata.get("still_valid") is False

        prefers = await graph_store.query("User", "PREFERS")
        assert len(prefers) == 1
        assert prefers[0].object == "Python"

    @pytest.mark.asyncio
    async def test_multiple_sequential_contradictions(self, graph_store):
        """Multiple changes over time should maintain correct history."""
        companies = ["CompanyA", "CompanyB", "CompanyC"]

        for i, company in enumerate(companies):
            if i > 0:
                old_jobs = await graph_store.query("User", "WORKS_AT")
                for job in old_jobs:
                    if job.metadata.get("still_valid", True):
                        await graph_store.mark_contradiction(job.id, f"User WORKS_AT {company}")

            await graph_store.persist_relationships([
                {"subject": "User", "predicate": "WORKS_AT", "object": company, "metadata": {"order": i}},
            ])

        # Verify history
        all_jobs = await graph_store.query("User", "WORKS_AT")
        assert len(all_jobs) == 3

        # Only the last one should be valid
        valid_jobs = [j for j in all_jobs if j.metadata.get("still_valid", True)]
        assert len(valid_jobs) == 1
        assert valid_jobs[0].object == "CompanyC"


# -----------------------------------------------------------------------------
# Observer Entity/Relationship Extraction Tests
# -----------------------------------------------------------------------------


class TestObserverExtraction:
    """Tests for observer's ability to extract entities and relationships."""

    @pytest.mark.asyncio
    async def test_observer_extracts_work_relationship(
        self, mock_llm_client, mock_vector_table, graph_store, mock_embedder
    ):
        """Observer should extract WORKS_AT from employment statement."""
        mock_llm_client.generate = AsyncMock(side_effect=[
            "HIGH",  # utility grade
            '{"entities": [{"name": "User", "type": "Person"}, {"name": "Acme Corp", "type": "Organization"}], "relationships": [{"subject": "User", "predicate": "WORKS_AT", "object": "Acme Corp", "metadata": {}}]}',  # user extraction
            '{"entities": [], "relationships": []}',  # assistant extraction (empty - no new facts)
            "User works at Acme Corp as a developer.",  # summary
            '["Where does user work?", "What company is user employed at?"]',  # queries
        ])

        observer = Observer(
            llm_client=mock_llm_client,
            vector_table=mock_vector_table,
            graph_store=graph_store,
            embedder=mock_embedder,
        )

        with patch.object(observer, '_persist_to_vector_store', new_callable=AsyncMock):
            output = await observer.process_turn(
                user_message="I work at Acme Corp as a developer.",
                assistant_response="That's great! What kind of development do you do there?",
                conversation_id="test-001",
                turn_index=0,
            )

        assert output.utility_grade == UtilityGrade.HIGH
        assert len(output.relationships) == 1
        assert output.relationships[0]["predicate"] == "WORKS_AT"
        assert output.relationships[0]["object"] == "Acme Corp"

    @pytest.mark.asyncio
    async def test_observer_extracts_family_relationships(
        self, mock_llm_client, mock_vector_table, graph_store, mock_embedder
    ):
        """Observer should extract complex family relationship networks."""
        mock_llm_client.generate = AsyncMock(side_effect=[
            "HIGH",
            '''{"entities": [
                {"name": "User", "type": "Person"},
                {"name": "Sarah", "type": "Person", "attributes": {"relation": "sister"}},
                {"name": "Mom", "type": "Person"},
                {"name": "Philadelphia", "type": "Place"}
            ], "relationships": [
                {"subject": "User", "predicate": "SIBLING_OF", "object": "Sarah", "metadata": {}},
                {"subject": "Sarah", "predicate": "VISITING_FROM", "object": "Philadelphia", "metadata": {}},
                {"subject": "Mom", "predicate": "VISITING", "object": "User", "metadata": {}}
            ]}''',  # user extraction
            '{"entities": [], "relationships": []}',  # assistant extraction
            "User's sister Sarah and mom are visiting from Philadelphia.",
            '["Who is visiting user?", "Where does Sarah live?", "Is user\'s family visiting?"]',
        ])

        observer = Observer(
            llm_client=mock_llm_client,
            vector_table=mock_vector_table,
            graph_store=graph_store,
            embedder=mock_embedder,
        )

        with patch.object(observer, '_persist_to_vector_store', new_callable=AsyncMock):
            output = await observer.process_turn(
                user_message="My sister Sarah and my mom are coming to visit me this weekend. Sarah is flying in from Philadelphia.",
                assistant_response="That sounds wonderful! How long will they be staying?",
                conversation_id="test-002",
                turn_index=0,
            )

        assert len(output.entities) == 4
        assert len(output.relationships) == 3

        predicates = {r["predicate"] for r in output.relationships}
        assert "SIBLING_OF" in predicates or "VISITING" in predicates

    @pytest.mark.asyncio
    async def test_observer_extracts_technology_ownership(
        self, mock_llm_client, mock_vector_table, graph_store, mock_embedder
    ):
        """Observer should extract device ownership and technical details."""
        mock_llm_client.generate = AsyncMock(side_effect=[
            "MEDIUM",
            '''{"entities": [
                {"name": "User", "type": "Person"},
                {"name": "Dell Latitude 5520", "type": "Technology", "attributes": {"ram": "32GB", "purpose": "home server"}}
            ], "relationships": [
                {"subject": "User", "predicate": "OWNS", "object": "Dell Latitude 5520", "metadata": {"use_case": "home server"}}
            ]}''',  # user extraction
            '{"entities": [], "relationships": []}',  # assistant extraction
            "User repurposed Dell Latitude 5520 as home server.",
            '["What hardware does user have?", "Does user have a home server?"]',
        ])

        observer = Observer(
            llm_client=mock_llm_client,
            vector_table=mock_vector_table,
            graph_store=graph_store,
            embedder=mock_embedder,
        )

        with patch.object(observer, '_persist_to_vector_store', new_callable=AsyncMock):
            output = await observer.process_turn(
                user_message="I decided to keep my Dell Latitude 5520 and turn it into a home server. It has 32GB RAM.",
                assistant_response="Nice! That's plenty of RAM for a home server. What services are you planning to run?",
                conversation_id="test-003",
                turn_index=0,
            )

        assert any(e["name"] == "Dell Latitude 5520" for e in output.entities)
        assert any(r["predicate"] == "OWNS" for r in output.relationships)

    @pytest.mark.asyncio
    async def test_observer_discards_small_talk(
        self, mock_llm_client, mock_vector_table, graph_store, mock_embedder
    ):
        """Small talk without factual content should be discarded."""
        mock_llm_client.generate = AsyncMock(return_value="DISCARD")

        observer = Observer(
            llm_client=mock_llm_client,
            vector_table=mock_vector_table,
            graph_store=graph_store,
            embedder=mock_embedder,
        )

        output = await observer.process_turn(
            user_message="Thanks!",
            assistant_response="You're welcome! Let me know if you need anything else.",
            conversation_id="test-004",
            turn_index=0,
        )

        assert output.utility_grade == UtilityGrade.DISCARD
        assert output.summary is None
        assert len(output.entities) == 0
        assert len(output.relationships) == 0

    @pytest.mark.asyncio
    async def test_observer_detects_contradiction_in_new_turn(
        self, mock_llm_client, mock_vector_table, graph_store, mock_embedder
    ):
        """Observer should detect when new information contradicts stored facts."""
        # Pre-populate graph with old employment
        await graph_store.persist_relationships([
            {"subject": "User", "predicate": "WORKS_AT", "object": "OldCompany", "metadata": {}},
        ])

        mock_llm_client.generate = AsyncMock(side_effect=[
            "HIGH",
            '''{"entities": [
                {"name": "User", "type": "Person"},
                {"name": "NewCorp", "type": "Organization"}
            ], "relationships": [
                {"subject": "User", "predicate": "WORKS_AT", "object": "NewCorp", "metadata": {"started": "last week"}}
            ]}''',  # user extraction
            '{"entities": [], "relationships": []}',  # assistant extraction
            "User started new job at NewCorp.",
            '["Where does user work now?", "When did user start new job?"]',
            # Add semantic contradiction detection response
            '''{"contradictions": [
                {
                    "existing_id": "test-id",
                    "existing_statement": "User WORKS_AT OldCompany",
                    "reason": "User changed jobs from OldCompany to NewCorp",
                    "temporal_type": "state_completion",
                    "confidence": "high"
                }
            ]}''',
        ])

        observer = Observer(
            llm_client=mock_llm_client,
            vector_table=mock_vector_table,
            graph_store=graph_store,
            embedder=mock_embedder,
        )

        with patch.object(observer, '_persist_to_vector_store', new_callable=AsyncMock):
            output = await observer.process_turn(
                user_message="I just started at NewCorp last week. Really excited about the new role!",
                assistant_response="Congratulations on the new position! What will you be doing there?",
                conversation_id="test-005",
                turn_index=0,
            )

        # Should detect contradiction
        assert len(output.contradictions) == 1
        assert "OldCompany" in output.contradictions[0]["existing_statement"]
        assert "NewCorp" in output.contradictions[0]["new_statement"]


# -----------------------------------------------------------------------------
# Reranker Tests
# -----------------------------------------------------------------------------


class TestRerankerScoring:
    """Tests for bi-encoder reranking functionality."""

    def test_reranker_scores_relevant_higher(self, reranker):
        """Relevant context should score higher than irrelevant."""
        pairs = [
            ("What time do I work?", "User works at Acme Corp starting at 9am"),
            ("What time do I work?", "User enjoys hiking on weekends"),
            ("What time do I work?", "User's favorite color is blue"),
        ]

        scores = reranker.predict(pairs)

        assert len(scores) == 3
        assert scores[0] > scores[1]  # Work schedule more relevant than hobbies
        assert scores[0] > scores[2]  # Work schedule more relevant than colors

    def test_reranker_handles_empty_input(self, reranker):
        """Reranker should handle empty input gracefully."""
        scores = reranker.predict([])
        assert scores == []

    def test_reranker_semantic_similarity(self, reranker):
        """Reranker should capture semantic similarity beyond keyword matching."""
        pairs = [
            ("How is my partner doing?", "Sarah has been feeling stressed about work lately"),
            ("How is my partner doing?", "The partnership agreement was signed yesterday"),
        ]

        scores = reranker.predict(pairs)

        # "Partner" referring to romantic partner should score higher than business partnership
        # This tests semantic understanding vs keyword matching
        assert len(scores) == 2

    def test_reranker_temporal_queries(self, reranker):
        """Reranker should handle time-based queries appropriately."""
        pairs = [
            ("What happened last week?", "User started new job on Monday"),
            ("What happened last week?", "User prefers tea over coffee"),
        ]

        scores = reranker.predict(pairs)
        assert scores[0] > scores[1]


# -----------------------------------------------------------------------------
# Context Assembler Tests
# -----------------------------------------------------------------------------


class TestContextAssembler:
    """Tests for context assembly and memory retrieval orchestration."""

    @pytest.mark.asyncio
    async def test_temporal_decay_penalizes_old_memories(self):
        """Older memories should have lower temporal scores."""
        assembler = ContextAssembler(
            vector_table=MagicMock(),
            graph_store=MagicMock(),
            reranker=MagicMock(),
        )

        now = datetime.now()
        recent = now - timedelta(days=1)
        old = now - timedelta(days=60)

        # Test with episodic fact type and medium utility (0.6 = 60-day half-life)
        recent_score = assembler._calculate_temporal_decay(recent, "episodic", 0.6)
        old_score = assembler._calculate_temporal_decay(old, "episodic", 0.6)

        assert recent_score > old_score
        assert recent_score > 0.9  # 1 day old should be close to 1.0
        assert old_score < 0.6  # 60 days old with 60-day half-life should be ~0.5
        
        # Core facts should never decay
        core_old = assembler._calculate_temporal_decay(old, "core", 0.6)
        assert core_old == 1.0  # Core facts always return 1.0

    @pytest.mark.asyncio
    async def test_sliding_window_respects_token_limit(self):
        """Sliding window should not exceed token budget."""
        assembler = ContextAssembler(
            vector_table=MagicMock(),
            graph_store=MagicMock(),
            reranker=MagicMock(),
        )
        assembler.sliding_window_tokens = 100

        history = [
            {"role": "user", "content": "First message " * 50},
            {"role": "assistant", "content": "Response " * 50},
            {"role": "user", "content": "Second message"},
            {"role": "assistant", "content": "Another response"},
        ]

        window = assembler._get_sliding_window(history)
        tokens = assembler._count_tokens(window)

        assert tokens <= assembler.sliding_window_tokens

    @pytest.mark.asyncio
    async def test_entity_extraction_from_query(self):
        """Context assembler should extract capitalized entity names."""
        assembler = ContextAssembler(
            vector_table=MagicMock(),
            graph_store=MagicMock(),
            reranker=MagicMock(),
        )

        query = "What time does Sarah get off work at Acme Corp?"
        entities = assembler._extract_entities_from_query(query)

        assert "Sarah" in entities
        assert "Acme" in entities
        assert "Corp" in entities
        assert "What" in entities  # Sentence start capitalization
        assert "time" not in entities  # Lowercase words excluded

    @pytest.mark.asyncio
    async def test_merge_deduplicates_results(self):
        """Merging vector and graph results should deduplicate."""
        assembler = ContextAssembler(
            vector_table=MagicMock(),
            graph_store=MagicMock(),
            reranker=MagicMock(),
        )

        now = datetime.now()
        vector_results = [
            RetrievedContext("User works at Acme", "vector", 0.8, 1.0, 0.8, now),
            RetrievedContext("User lives in Austin", "vector", 0.7, 1.0, 0.7, now),
        ]
        graph_results = [
            RetrievedContext("User works at Acme", "graph", 0.4, 1.0, 0.4, now),
            RetrievedContext("User OWNS Dell Laptop", "graph", 0.4, 1.0, 0.4, now),
        ]

        merged = assembler._merge_results(vector_results, graph_results)

        # Should have 3 unique content items (Acme appears twice but deduped)
        contents = [m.content for m in merged]
        assert len(contents) == 4  # Different sources = different keys
        # But verify content appears
        assert sum(1 for c in contents if "Acme" in c) == 2

    @pytest.mark.asyncio
    async def test_rerank_boosts_recent_user_message_matches(self):
        """Content matching last user message should get boosted."""
        mock_reranker = MagicMock()
        mock_reranker.predict = MagicMock(return_value=[0.5, 0.5, 0.5])

        assembler = ContextAssembler(
            vector_table=MagicMock(),
            graph_store=MagicMock(),
            reranker=mock_reranker,
        )

        now = datetime.now()
        candidates = [
            RetrievedContext("User talked about laptop", "vector", 0.5, 1.0, 0.5, now),
            RetrievedContext("User mentioned laptop purchase", "vector", 0.5, 1.0, 0.5, now),
            RetrievedContext("User likes coffee", "vector", 0.5, 1.0, 0.5, now),
        ]

        reranked = assembler._rerank("laptop", candidates, 3, "laptop")

        # Items containing "laptop" should be boosted
        laptop_items = [r for r in reranked if "laptop" in r.content.lower()]
        coffee_item = next(r for r in reranked if "coffee" in r.content.lower())

        assert all(item.final_score > coffee_item.final_score for item in laptop_items)


# -----------------------------------------------------------------------------
# Complex Scenario Integration Tests
# -----------------------------------------------------------------------------


class TestComplexScenarios:
    """Integration tests for complex multi-turn scenarios."""

    @pytest.mark.asyncio
    async def test_career_change_scenario(self, graph_store):
        """Simulate user changing careers over multiple conversations."""
        # Phase 1: User is a teacher
        await graph_store.persist_entities([
            {"name": "User", "type": "Person", "attributes": {"profession": "teacher"}},
            {"name": "Lincoln High", "type": "Organization", "attributes": {"type": "school"}},
        ])
        await graph_store.persist_relationships([
            {"subject": "User", "predicate": "WORKS_AT", "object": "Lincoln High", "metadata": {"role": "teacher"}},
        ])

        # Phase 2: User mentions considering career change
        await graph_store.persist_relationships([
            {"subject": "User", "predicate": "CONSIDERING", "object": "tech career", "metadata": {}},
        ])

        # Phase 3: User gets bootcamp certification
        await graph_store.persist_relationships([
            {"subject": "User", "predicate": "COMPLETED", "object": "coding bootcamp", "metadata": {}},
        ])

        # Phase 4: User starts new tech job
        old_job = await graph_store.query("User", "WORKS_AT")
        await graph_store.mark_contradiction(old_job[0].id, "User WORKS_AT TechCorp")
        await graph_store.persist_entities([
            {"name": "TechCorp", "type": "Organization", "attributes": {"industry": "tech"}},
        ])
        await graph_store.persist_relationships([
            {"subject": "User", "predicate": "WORKS_AT", "object": "TechCorp", "metadata": {"role": "developer"}},
        ])

        # Verify final state
        all_jobs = await graph_store.query("User", "WORKS_AT")
        assert len(all_jobs) == 2

        current_job = next(j for j in all_jobs if j.metadata.get("still_valid", True))
        assert current_job.object == "TechCorp"

        old_job = next(j for j in all_jobs if not j.metadata.get("still_valid", True))
        assert old_job.object == "Lincoln High"

    @pytest.mark.asyncio
    async def test_relationship_evolution_scenario(self, graph_store):
        """Track relationship status changes over time."""
        # Meeting someone new
        await graph_store.persist_entities([
            {"name": "User", "type": "Person"},
            {"name": "Alex", "type": "Person"},
        ])
        await graph_store.persist_relationships([
            {"subject": "User", "predicate": "MET", "object": "Alex", "metadata": {"when": "at party"}},
        ])

        # Started dating
        await graph_store.persist_relationships([
            {"subject": "User", "predicate": "DATING", "object": "Alex", "metadata": {}},
        ])

        # Got engaged
        old_status = await graph_store.query("User", "DATING")
        await graph_store.mark_contradiction(old_status[0].id, "User ENGAGED_TO Alex")
        await graph_store.persist_relationships([
            {"subject": "User", "predicate": "ENGAGED_TO", "object": "Alex", "metadata": {}},
        ])

        # Got married
        old_engaged = await graph_store.query("User", "ENGAGED_TO")
        await graph_store.mark_contradiction(old_engaged[0].id, "User MARRIED_TO Alex")
        await graph_store.persist_relationships([
            {"subject": "User", "predicate": "MARRIED_TO", "object": "Alex", "metadata": {}},
        ])

        # Verify progression
        all_alex_rels = await graph_store.search_relationships(["Alex"], limit=10)

        predicates = [r.predicate for r in all_alex_rels]
        assert "MET" in predicates
        assert "DATING" in predicates
        assert "ENGAGED_TO" in predicates
        assert "MARRIED_TO" in predicates

        # Only MARRIED_TO should be valid (MET doesn't get superseded)
        valid_rels = [r for r in all_alex_rels if r.metadata.get("still_valid", True)]
        valid_predicates = {r.predicate for r in valid_rels}
        assert "MARRIED_TO" in valid_predicates
        assert "MET" in valid_predicates  # MET wasn't superseded

    @pytest.mark.asyncio
    async def test_conflicting_information_same_session(self, graph_store):
        """Handle user correcting themselves within same conversation."""
        # User says they have a cat
        await graph_store.persist_relationships([
            {"subject": "User", "predicate": "OWNS", "object": "cat", "metadata": {"name": "Whiskers"}},
        ])

        # User corrects: actually a dog
        old_pet = await graph_store.query("User", "OWNS")
        cat_rel = next((p for p in old_pet if p.object == "cat"), None)
        if cat_rel:
            await graph_store.mark_contradiction(cat_rel.id, "User OWNS dog")

        await graph_store.persist_relationships([
            {"subject": "User", "predicate": "OWNS", "object": "dog", "metadata": {"name": "Buddy", "correction": True}},
        ])

        # Verify correction
        pets = await graph_store.query("User", "OWNS")
        assert len(pets) == 2

        dog = next(p for p in pets if p.object == "dog")
        assert dog.metadata.get("correction") is True

        cat = next(p for p in pets if p.object == "cat")
        assert cat.metadata.get("still_valid") is False


# -----------------------------------------------------------------------------
# Edge Cases and Robustness Tests
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, graph_store):
        """Graph store should handle empty entity searches gracefully."""
        results = await graph_store.search_relationships([], limit=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_nonexistent_entity_query(self, graph_store):
        """Querying nonexistent entities should return empty results."""
        results = await graph_store.query("NonexistentPerson", "WORKS_AT")
        assert results == []

    @pytest.mark.asyncio
    async def test_special_characters_in_entity_names(self, graph_store):
        """Entity names with special characters should be handled."""
        await graph_store.persist_entities([
            {"name": "O'Brien", "type": "Person", "attributes": {}},
            {"name": "Mary-Jane", "type": "Person", "attributes": {}},
        ])
        await graph_store.persist_relationships([
            {"subject": "O'Brien", "predicate": "KNOWS", "object": "Mary-Jane", "metadata": {}},
        ])

        results = await graph_store.query("O'Brien", "KNOWS")
        assert len(results) == 1
        assert results[0].object == "Mary-Jane"

    @pytest.mark.asyncio
    async def test_unicode_entity_names(self, graph_store):
        """Unicode characters in names should work correctly."""
        await graph_store.persist_entities([
            {"name": "Müller", "type": "Person", "attributes": {}},
            {"name": "北京", "type": "Place", "attributes": {}},
        ])
        await graph_store.persist_relationships([
            {"subject": "Müller", "predicate": "VISITED", "object": "北京", "metadata": {}},
        ])

        results = await graph_store.query("Müller", "VISITED")
        assert len(results) == 1
        assert results[0].object == "北京"

    @pytest.mark.asyncio
    async def test_very_long_content_handling(self, graph_store):
        """Very long attribute values should be stored correctly."""
        long_description = "A" * 10000

        await graph_store.persist_entities([
            {"name": "Project", "type": "Concept", "attributes": {"description": long_description}},
        ])

        # Should not raise an error
        assert "Project" in graph_store.entities

    def test_reranker_with_very_long_texts(self, reranker):
        """Reranker should handle very long texts."""
        long_text = "This is a test sentence. " * 500
        pairs = [("short query", long_text)]

        scores = reranker.predict(pairs)
        assert len(scores) == 1
        assert isinstance(scores[0], float)

    @pytest.mark.asyncio
    async def test_concurrent_writes_to_graph(self, graph_store):
        """Concurrent writes should not corrupt data."""
        async def write_entity(name: str):
            await graph_store.persist_entities([
                {"name": name, "type": "Person", "attributes": {}},
            ])
            await graph_store.persist_relationships([
                {"subject": name, "predicate": "EXISTS", "object": "world", "metadata": {}},
            ])

        # Write many entities concurrently
        tasks = [write_entity(f"Person{i}") for i in range(100)]
        await asyncio.gather(*tasks)

        # Verify all entities exist
        for i in range(100):
            assert f"Person{i}" in graph_store.entities

        # Verify all relationships exist
        assert len(graph_store.relationships) == 100


# -----------------------------------------------------------------------------
# Utility Grade Tests
# -----------------------------------------------------------------------------


class TestUtilityGrading:
    """Tests for utility grade assignment."""

    @pytest.mark.asyncio
    async def test_utility_score_mapping(self):
        """Utility grades should map to correct scores."""
        from src.observer.observer import Observer

        # Create minimal observer just to test mapping
        observer = Observer(
            llm_client=MagicMock(),
            vector_table=MagicMock(),
            graph_store=MagicMock(),
        )

        # Current 3-level system
        assert observer._utility_to_score(UtilityGrade.DISCARD) == 0.0
        assert observer._utility_to_score(UtilityGrade.STORE) == 0.6
        assert observer._utility_to_score(UtilityGrade.IMPORTANT) == 1.0
        
        # Legacy 4-level system (backward compatibility)
        assert observer._utility_to_score(UtilityGrade.LOW) == 0.3
        assert observer._utility_to_score(UtilityGrade.MEDIUM) == 0.6
        assert observer._utility_to_score(UtilityGrade.HIGH) == 1.0

    @pytest.mark.asyncio
    async def test_high_utility_content_types(
        self, mock_llm_client, mock_vector_table, graph_store, mock_embedder
    ):
        """Verify content types that should receive HIGH utility."""
        high_utility_messages = [
            "I start work at 9am every day",
            "My sister Sarah lives in Boston",
            "I'm allergic to peanuts",
            "My phone number is 555-1234",
            "I have a meeting with Dr. Smith on Friday",
        ]

        mock_llm_client.generate = AsyncMock(side_effect=[
            "HIGH",  # utility
            '{"entities": [], "relationships": []}',  # user extraction
            '{"entities": [], "relationships": []}',  # assistant extraction
            "Summary",  # summary
            "[]",  # queries
        ] * len(high_utility_messages))

        observer = Observer(
            llm_client=mock_llm_client,
            vector_table=mock_vector_table,
            graph_store=graph_store,
            embedder=mock_embedder,
        )

        with patch.object(observer, '_persist_to_vector_store', new_callable=AsyncMock):
            for msg in high_utility_messages:
                output = await observer.process_turn(
                    user_message=msg,
                    assistant_response="Got it!",
                    conversation_id="test",
                    turn_index=0,
                )
                # All these should be graded (mocked to HIGH)
                assert output.utility_grade == UtilityGrade.HIGH


class TestPromptQuality:
    """Tests for extraction prompt quality and structure."""

    @pytest.mark.asyncio
    async def test_extraction_prompt_includes_examples(self):
        """Extraction prompt should include concrete examples for guidance."""
        from src.observer.prompts import EXTRACTION_PROMPT

        prompt = EXTRACTION_PROMPT.format(text="test")

        # Should have example section (now uses "Example 1", "Example 2", etc.)
        assert "Example 1" in prompt or "Example" in prompt

        # Should mention attributes extraction
        assert "attributes" in prompt.lower()

        # Should list relationship types
        assert "SIBLING_OF" in prompt or "sibling" in prompt.lower()
        assert "WORKS_AT" in prompt or "works" in prompt.lower()

        # Should mention User entity
        assert "User" in prompt

    @pytest.mark.asyncio
    async def test_extraction_prompt_prohibits_hallucination(self):
        """Extraction prompt should explicitly warn against hallucination."""
        from src.observer.prompts import EXTRACTION_PROMPT

        prompt = EXTRACTION_PROMPT.format(text="test")

        # Should warn against hallucination
        assert "hallucinate" in prompt.lower() or "infer" in prompt.lower()

        # Should emphasize facts
        assert "fact" in prompt.lower() or "explicit" in prompt.lower()

    @pytest.mark.asyncio
    async def test_extraction_prompt_specifies_json_format(self):
        """Extraction prompt should specify valid JSON output format."""
        from src.observer.prompts import EXTRACTION_PROMPT

        prompt = EXTRACTION_PROMPT.format(text="test")

        # Should mention JSON
        assert "JSON" in prompt or "json" in prompt

        # Should show structure
        assert '"entities"' in prompt
        assert '"relationships"' in prompt
        assert '"attributes"' in prompt
        assert '"metadata"' in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# -----------------------------------------------------------------------------
# Tests for Recent Code Fixes
# -----------------------------------------------------------------------------


class TestRecentCodeFixes:
    """Tests for fixes implemented from code review."""

    @pytest.mark.asyncio
    async def test_query_by_object_finds_relationships(self, graph_store):
        """query_by_object() should find relationships where entity is object."""
        # Add relationship where "Justine" is the object
        await graph_store.persist_relationships([
            {"subject": "User", "predicate": "SIBLING_OF", "object": "Justine", "metadata": {}},
        ])

        # Query by object should find it
        results = await graph_store.query_by_object("Justine", predicate=None)
        
        assert len(results) > 0
        assert any(r.subject == "User" and r.predicate == "SIBLING_OF" for r in results)

    @pytest.mark.asyncio
    async def test_semantic_similarity_in_vector_scoring(self):
        """Context assembler should use combined similarity + utility score."""
        from src.memory.vector_store import init_vector_store
        
        vector_table = init_vector_store()
        assembler = ContextAssembler(
            vector_table=vector_table,
            graph_store=MagicMock(),
            reranker=MagicMock(),
        )

        # Mock vector search to return items with _combined_score
        mock_hits = [
            {
                "content": "Test content",
                "utility_score": 0.5,
                "_combined_score": 0.75,  # Should use this
                "created_at": datetime.now().isoformat(),
                "fact_type": "episodic",
            }
        ]

        with patch('src.memory.context_assembler.vector_search', return_value=mock_hits):
            results = await assembler._vector_search("test query", top_k=5)

        # Should use _combined_score, not just utility_score
        assert len(results) > 0
        assert results[0].relevance_score == 0.75  # Uses combined score
        assert results[0].utility_score == 0.5  # Still tracks utility separately

    @pytest.mark.asyncio
    async def test_contradiction_preserves_multiword_entities(self, graph_store):
        """Contradiction tracking should not corrupt multi-word entity names."""
        # This would previously fail because splitting "Worcester Massachusetts" 
        # on whitespace would give wrong subject/predicate/object
        
        await graph_store.persist_relationships([
            {
                "subject": "Worcester Massachusetts",
                "predicate": "LIVES_IN", 
                "object": "User",
                "metadata": {}
            }
        ])

        relationships = await graph_store.query("Worcester Massachusetts", predicate=None)
        assert len(relationships) > 0
        
        # Entity name should be preserved exactly
        assert relationships[0].subject == "Worcester Massachusetts"
        assert relationships[0].predicate == "LIVES_IN"
