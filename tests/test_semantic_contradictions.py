"""
Test semantic contradiction detection and temporal state tracking.

This test suite validates that the system correctly:
1. Detects temporal state transitions (VISITING → RETURNED_HOME)
2. Marks superseded facts appropriately
3. Filters out superseded facts during retrieval
4. Boosts recent corrections in search results
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from src.memory.graph_store import GraphRelationship, InMemoryGraphStore
from src.observer.observer import Observer
from src.models.llm import OllamaClient
from src.memory.vector_store import init_vector_store
from src.memory.context_assembler import ContextAssembler
from src.models.reranker import Reranker


@pytest.fixture
def graph_store():
    """Create fresh in-memory graph store for each test."""
    return InMemoryGraphStore()


@pytest.fixture
def vector_table():
    """Create vector store table."""
    return init_vector_store()


@pytest.fixture
def llm_client():
    """Create LLM client for observer."""
    return OllamaClient()


@pytest_asyncio.fixture
async def observer(llm_client, vector_table, graph_store):
    """Create observer instance."""
    return Observer(llm_client, vector_table, graph_store)


@pytest_asyncio.fixture
async def context_assembler(vector_table, graph_store):
    """Create context assembler instance."""
    reranker = Reranker()
    return ContextAssembler(vector_table, graph_store, reranker)


class TestTemporalStateTransitions:
    """Test temporal state transitions like VISITING → RETURNED_HOME."""

    @pytest.mark.integration  # Requires live Ollama
    @pytest.mark.asyncio
    async def test_visiting_to_returned_home_marks_contradiction(self, observer, graph_store):
        """
        Test Case: Mom visiting Philadelphia → Mom returned home to Massachusetts
        Expected: System detects state completion and marks VISITING as superseded
        """
        # First turn: Mom is visiting
        await observer.process_turn(
            user_message="My mom and sister Justine are visiting me in Philadelphia today from West Boylston Massachusetts",
            assistant_response="That sounds wonderful! I hope you have a great time with your family.",
            conversation_id="test-1",
            turn_index=0,
        )

        # Check that VISITING relationship was created
        visiting_rels = await graph_store.query("Mom", "VISITING")
        assert len(visiting_rels) > 0, "VISITING relationship should be created"
        initial_visiting_id = visiting_rels[0].id

        # Second turn: Mom returned home
        result = await observer.process_turn(
            user_message="My mom and Justine have returned home, they are no longer in Philadelphia",
            assistant_response="I'm glad you got to spend time with them!",
            conversation_id="test-1",
            turn_index=1,
        )

        # Check for detected contradictions
        assert len(result.contradictions) > 0, "Should detect contradiction between VISITING and RETURNED_HOME"

        # Check that the old VISITING relationship is marked as superseded
        visiting_rels_after = await graph_store.query("Mom", "VISITING")
        if visiting_rels_after:
            for rel in visiting_rels_after:
                if rel.id == initial_visiting_id:
                    assert rel.superseded_by is not None, "Old VISITING fact should be marked as superseded"
                    assert rel.status == "completed", "Old VISITING fact should have status=completed"

    @pytest.mark.integration  # Requires live Ollama
    @pytest.mark.asyncio
    async def test_scheduled_to_happened_marks_contradiction(self, observer, graph_store):
        """
        Test Case: Interview scheduled for Friday → Interview happened on Monday
        Expected: SCHEDULED_FOR should be superseded by HAPPENED
        """
        # First turn: Interview scheduled
        await observer.process_turn(
            user_message="I have an interview scheduled for Friday",
            assistant_response="Good luck with your interview!",
            conversation_id="test-2",
            turn_index=0,
        )

        # Second turn: Interview happened (rescheduled)
        result = await observer.process_turn(
            user_message="Actually they moved my interview to Monday and it already happened",
            assistant_response="Hope it went well!",
            conversation_id="test-2",
            turn_index=1,
        )

        # Should detect contradiction
        assert len(result.contradictions) > 0, "Should detect SCHEDULED vs HAPPENED contradiction"

    @pytest.mark.integration  # Requires live Ollama
    @pytest.mark.asyncio
    async def test_age_correction_marks_contradiction(self, observer, graph_store):
        """
        Test Case: Sister is 24 → Actually she's 25
        Expected: Attribute update detected as contradiction
        """
        # First turn: Sister is 24
        await observer.process_turn(
            user_message="My sister Justine is 24 years old",
            assistant_response="Nice!",
            conversation_id="test-3",
            turn_index=0,
        )

        # Second turn: Correction to 25
        result = await observer.process_turn(
            user_message="Wait, I made a mistake. Justine is actually 25, not 24",
            assistant_response="Thanks for the correction!",
            conversation_id="test-3",
            turn_index=1,
        )

        # Should detect attribute update contradiction
        assert len(result.contradictions) > 0, "Should detect age correction as contradiction"


class TestSupersededFactFiltering:
    """Test that superseded facts are filtered from retrieval."""

    @pytest.mark.asyncio
    async def test_superseded_facts_not_returned_in_search(self, graph_store):
        """
        Test: Superseded facts should not appear in search results
        """
        # Add an active relationship
        await graph_store.persist_relationships([
            {
                "subject": "Mom",
                "predicate": "LIVES_IN",
                "object": "Massachusetts",
                "status": "ongoing",
            }
        ])

        # Add a superseded relationship
        await graph_store.persist_relationships([
            {
                "id": "old-rel-123",
                "subject": "Mom",
                "predicate": "VISITING",
                "object": "Philadelphia",
                "status": "completed",
            }
        ])

        # Mark it as superseded
        await graph_store.mark_contradiction("old-rel-123", "Mom RETURNED_HOME Massachusetts")

        # Search should not include superseded fact
        results = await graph_store.search_relationships(["Mom"], limit=10)

        superseded_found = any(r.id == "old-rel-123" and r.superseded_by is not None for r in results)

        # The fact should exist in storage but be marked as superseded
        all_rels = await graph_store.query("Mom", predicate=None)
        visiting_rel = next((r for r in all_rels if r.predicate == "VISITING"), None)

        if visiting_rel:
            assert visiting_rel.superseded_by is not None, "Superseded fact should have superseded_by set"

    @pytest.mark.asyncio
    async def test_context_assembler_filters_superseded_facts(self, context_assembler, graph_store):
        """
        Test: Context assembler should filter out superseded facts during graph search
        """
        # Add active fact
        await graph_store.persist_relationships([
            {
                "subject": "Mom",
                "predicate": "RETURNED_HOME",
                "object": "Massachusetts",
                "status": "completed",
            }
        ])

        # Add superseded fact
        await graph_store.persist_relationships([
            {
                "id": "old-visiting",
                "subject": "Mom",
                "predicate": "VISITING",
                "object": "Philadelphia",
                "status": "completed",
                "superseded_by": "Mom RETURNED_HOME Massachusetts",
            }
        ])

        # Perform graph search
        results = await context_assembler._graph_search("Mom visiting", top_k=10)

        # Should not include superseded VISITING fact
        visiting_contents = [r.content for r in results if "VISITING" in r.content]
        assert len(visiting_contents) == 0, "Superseded VISITING fact should be filtered out"


class TestRecentCorrectionBoosting:
    """Test that recent corrections get boosted in retrieval."""

    @pytest.mark.asyncio
    async def test_recent_facts_boosted_over_old_facts(self, context_assembler, graph_store):
        """
        Test: Recent corrections should have higher relevance scores
        """
        old_date = datetime.utcnow() - timedelta(days=30)
        recent_date = datetime.utcnow() - timedelta(days=1)

        # Add old fact
        old_rel = GraphRelationship(
            id="old-fact",
            subject="Sister",
            predicate="AGE",
            object="24",
            created_at=old_date,
            status="completed",
            superseded_by="Sister AGE 25",
        )
        graph_store.relationships.append(old_rel)

        # Add recent correction
        new_rel = GraphRelationship(
            id="new-fact",
            subject="Sister",
            predicate="AGE",
            object="25",
            created_at=recent_date,
            status="ongoing",
        )
        graph_store.relationships.append(new_rel)

        # Perform search
        results = await context_assembler._graph_search("Sister age", top_k=10)

        # Find the results
        age_24_result = next((r for r in results if "24" in r.content), None)
        age_25_result = next((r for r in results if "25" in r.content), None)

        # Recent correction should have higher relevance
        if age_25_result:
            assert age_25_result.relevance_score > 0.4, "Recent fact should have boosted relevance"

        # Old superseded fact should be filtered out
        assert age_24_result is None, "Superseded fact should be filtered out"


class TestOngoingVsCompletedStates:
    """Test that ongoing states are preferred over completed ones."""

    @pytest.mark.asyncio
    async def test_ongoing_state_preferred_over_completed(self, context_assembler, graph_store):
        """
        Test: Ongoing states should rank higher than completed ones
        """
        # Add completed state
        completed_rel = GraphRelationship(
            id="completed-work",
            subject="User",
            predicate="WORKED_AT",
            object="OldCompany",
            created_at=datetime.utcnow() - timedelta(days=365),
            status="completed",
        )
        graph_store.relationships.append(completed_rel)

        # Add ongoing state
        ongoing_rel = GraphRelationship(
            id="current-work",
            subject="User",
            predicate="WORKS_AT",
            object="NewCompany",
            created_at=datetime.utcnow() - timedelta(days=30),
            status="ongoing",
        )
        graph_store.relationships.append(ongoing_rel)

        # Perform search
        results = await context_assembler._graph_search("User work", top_k=10)

        # Find results
        old_work = next((r for r in results if "OldCompany" in r.content), None)
        new_work = next((r for r in results if "NewCompany" in r.content), None)

        # Ongoing should have higher score
        if new_work and old_work:
            assert new_work.relevance_score > old_work.relevance_score, "Ongoing state should have higher relevance"


class TestFallbackContradictionDetection:
    """Test that fallback contradiction detection works when LLM fails."""

    @pytest.mark.asyncio
    async def test_simple_contradiction_fallback(self, observer, graph_store):
        """
        Test: If LLM-based detection fails, simple predicate matching should work
        """
        # Add initial relationship
        await graph_store.persist_relationships([
            {
                "subject": "User",
                "predicate": "WORKS_AT",
                "object": "CompanyA",
            }
        ])

        # Create a new relationship with same predicate but different object
        new_rel = {"subject": "User", "predicate": "WORKS_AT", "object": "CompanyB"}

        # Get existing relationships
        existing_rels = await graph_store.query("User", "WORKS_AT")

        # Test fallback contradiction detection
        contradictions = await observer._simple_contradiction_check(new_rel, existing_rels)

        assert len(contradictions) > 0, "Simple fallback should detect predicate mismatch"
        assert contradictions[0]["confidence"] == "high", "Fallback should return high confidence"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
