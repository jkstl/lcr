from __future__ import annotations

import asyncio
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END

from ..config import settings
from ..memory.context_assembler import ContextAssembler
from ..memory.vector_store import init_vector_store
from ..memory.graph_store import create_graph_store
from ..models.embedder import Embedder
from ..models.llm import OllamaClient
from ..models.reranker import Reranker
from ..observer.observer import Observer
from .prompts import SYSTEM_PROMPT_TEMPLATE


class ConversationState(TypedDict):
    user_input: str
    conversation_history: list[dict[str, Any]]
    conversation_id: str
    retrieved_context: str
    retrieval_sources: list[str]
    assistant_response: str
    observer_triggered: bool
    observer_output: dict[str, Any] | None


_vector_table = init_vector_store(settings.lancedb_path)
_graph_store = create_graph_store()
_reranker = Reranker()
_embedder = Embedder()
_context_assembler = ContextAssembler(_vector_table, _graph_store, _reranker, embedder=_embedder)
_llm_client = OllamaClient()
_observer = Observer(
    _llm_client,
    _vector_table,
    _graph_store,
    model=settings.observer_model,
)
_observer_tasks: list[asyncio.Task] = []


def build_system_prompt(context: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(retrieved_context=context)


def create_conversation_graph() -> StateGraph[ConversationState]:
    graph = StateGraph(ConversationState)
    graph.add_node("assemble_context", assemble_context_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("trigger_observer", trigger_observer_node)
    graph.add_edge("assemble_context", "generate_response")
    graph.add_edge("generate_response", "trigger_observer")
    graph.add_edge("trigger_observer", END)
    graph.set_entry_point("assemble_context")
    return graph.compile()


async def assemble_context_node(state: ConversationState) -> ConversationState:
    context = await _context_assembler.assemble(
        query=state["user_input"],
        conversation_history=state["conversation_history"],
    )
    return {**state, "retrieved_context": context, "retrieval_sources": []}


async def generate_response_node(state: ConversationState) -> ConversationState:
    prompt = build_system_prompt(state["retrieved_context"])
    response = await _llm_client.generate(
        model=settings.main_model,
        system=prompt,
        prompt=state["user_input"],
    )
    return {**state, "assistant_response": response}


async def trigger_observer_node(state: ConversationState) -> ConversationState:
    task = asyncio.create_task(
        _observer.process_turn(
            user_message=state["user_input"],
            assistant_response=state["assistant_response"],
            conversation_id=state["conversation_id"],
            turn_index=len(state["conversation_history"]),
        )
    )
    _observer_tasks.append(task)
    return {**state, "observer_triggered": True}


async def wait_for_observers() -> None:
    """Wait for all pending observer tasks to complete."""
    if _observer_tasks:
        await asyncio.gather(*_observer_tasks, return_exceptions=True)
        _observer_tasks.clear()


async def generate_response_streaming(state: ConversationState):
    """
    Async generator that yields response tokens for streaming UI.
    Updates state and triggers observer after completion.
    Returns the full response text after yielding all tokens.
    """
    # First assemble context (non-streaming)
    context = await _context_assembler.assemble(
        query=state["user_input"],
        conversation_history=state["conversation_history"],
    )
    state["retrieved_context"] = context
    
    # Build prompt and stream response
    prompt = build_system_prompt(context)
    full_response = ""
    
    async for token in _llm_client.generate_stream(
        model=settings.main_model,
        system=prompt,
        prompt=state["user_input"],
    ):
        full_response += token
        yield token
    
    # Update state with full response
    state["assistant_response"] = full_response.strip()
    
    # Trigger observer in background (non-blocking)
    task = asyncio.create_task(
        _observer.process_turn(
            user_message=state["user_input"],
            assistant_response=state["assistant_response"],
            conversation_id=state["conversation_id"],
            turn_index=len(state["conversation_history"]),
        )
    )
    _observer_tasks.append(task)
    state["observer_triggered"] = True

