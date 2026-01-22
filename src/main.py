from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from .config import settings
from .orchestration.graph import create_conversation_graph, ConversationState, wait_for_observers, generate_response_streaming
from .conversation_logger import ConversationLogger
from .voice.tts import TTSEngine, VoiceConfig
from .voice.utils import split_into_sentences

console = Console()


async def check_system_status() -> dict[str, dict]:
    """Check status of all critical systems."""
    status = {}

    # Check Ollama
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_host}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]

                # Check for model name with flexible version matching
                main_model_base = settings.main_model.split(':')[0]
                observer_model_base = settings.observer_model.split(':')[0]
                embedding_model_base = settings.embedding_model.split(':')[0]

                main_model_loaded = any(main_model_base in m for m in model_names)
                observer_model_loaded = any(observer_model_base in m for m in model_names)
                embedding_model_loaded = any(embedding_model_base in m for m in model_names)

                status["ollama"] = {
                    "status": "ok",
                    "main_model": settings.main_model if main_model_loaded else f"{settings.main_model} (not found)",
                    "observer_model": settings.observer_model if observer_model_loaded else f"{settings.observer_model} (not found)",
                    "embedding_model": settings.embedding_model if embedding_model_loaded else f"{settings.embedding_model} (not found)",
                    "all_loaded": main_model_loaded and observer_model_loaded and embedding_model_loaded
                }
            else:
                status["ollama"] = {"status": "error", "message": f"HTTP {response.status_code}"}
    except Exception as e:
        status["ollama"] = {"status": "error", "message": str(e)}

    # Check LanceDB
    try:
        from .memory.vector_store import init_vector_store
        vector_table = init_vector_store()
        count = vector_table.count_rows()
        status["lancedb"] = {
            "status": "ok",
            "path": settings.lancedb_path,
            "memory_chunks": count
        }
    except Exception as e:
        status["lancedb"] = {"status": "error", "message": str(e)}

    # Check FalkorDB
    try:
        from falkordb import FalkorDB
        client = FalkorDB(host=settings.falkordb_host, port=settings.falkordb_port)
        graph = client.select_graph(settings.falkordb_graph_id)
        # Simple query to check connectivity
        result = graph.query("RETURN 1")
        status["falkordb"] = {
            "status": "ok",
            "host": f"{settings.falkordb_host}:{settings.falkordb_port}",
            "graph_id": settings.falkordb_graph_id
        }
    except ImportError:
        status["falkordb"] = {"status": "warning", "message": "FalkorDB library not installed (will use in-memory)"}
    except Exception as e:
        status["falkordb"] = {"status": "warning", "message": f"Not running (will use in-memory): {e}"}

    # Check Redis (optional)
    try:
        import redis
        r = redis.Redis(host=settings.redis_host, port=settings.redis_port, socket_connect_timeout=2)
        r.ping()
        status["redis"] = {
            "status": "ok",
            "host": f"{settings.redis_host}:{settings.redis_port}"
        }
    except ImportError:
        status["redis"] = {"status": "warning", "message": "Redis library not installed"}
    except Exception as e:
        status["redis"] = {"status": "warning", "message": f"Not running: {e}"}

    # Check Docker (for FalkorDB/Redis)
    try:
        import subprocess
        # Check for falkordb/redis containers (regardless of compose project)
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=falkordb", "--filter", "name=redis", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            container_names = [name.strip() for name in result.stdout.strip().split('\n') if name.strip()]
            running_count = len(container_names)
            status["docker"] = {
                "status": "ok",
                "running_containers": running_count,
                "total_containers": running_count,
                "message": f"Found: {', '.join(container_names)}" if running_count <= 2 else f"{running_count} containers"
            }
        else:
            status["docker"] = {"status": "warning", "message": "No FalkorDB/Redis containers found"}
    except FileNotFoundError:
        status["docker"] = {"status": "warning", "message": "Docker not found in PATH"}
    except Exception as e:
        status["docker"] = {"status": "warning", "message": str(e)}

    return status


def display_system_status(status: dict[str, dict]) -> bool:
    """Display system status in a nice table. Returns True if all critical systems OK."""

    # Create header
    console.print()
    console.print(Panel.fit(
        "[bold cyan]LCR System Pre-Flight Check[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    # Create status table
    table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
    table.add_column("Component", style="cyan", width=20)
    table.add_column("Status", width=10)
    table.add_column("Details", style="dim")

    critical_ok = True

    # Ollama (critical)
    ollama_status = status.get("ollama", {})
    if ollama_status.get("status") == "ok":
        if ollama_status.get("all_loaded"):
            table.add_row(
                "Ollama",
                "[green]✓ OK[/green]",
                f"Main: {ollama_status['main_model']}\n"
                f"Observer: {ollama_status['observer_model']}\n"
                f"Embeddings: {ollama_status['embedding_model']}"
            )
        else:
            table.add_row(
                "Ollama",
                "[yellow]⚠ WARN[/yellow]",
                f"Main: {ollama_status['main_model']}\n"
                f"Observer: {ollama_status['observer_model']}\n"
                f"Embeddings: {ollama_status['embedding_model']}\n"
                "[yellow]Some models not found[/yellow]"
            )
            critical_ok = False
    else:
        table.add_row(
            "Ollama",
            "[red]✗ ERROR[/red]",
            ollama_status.get("message", "Unknown error")
        )
        critical_ok = False

    # LanceDB (critical)
    lancedb_status = status.get("lancedb", {})
    if lancedb_status.get("status") == "ok":
        table.add_row(
            "LanceDB",
            "[green]✓ OK[/green]",
            f"Path: {lancedb_status['path']}\n"
            f"Memories: {lancedb_status['memory_chunks']}"
        )
    else:
        table.add_row(
            "LanceDB",
            "[red]✗ ERROR[/red]",
            lancedb_status.get("message", "Unknown error")
        )
        critical_ok = False

    # FalkorDB (warning only, can use in-memory)
    falkordb_status = status.get("falkordb", {})
    if falkordb_status.get("status") == "ok":
        table.add_row(
            "FalkorDB",
            "[green]✓ OK[/green]",
            f"Host: {falkordb_status['host']}\n"
            f"Graph: {falkordb_status['graph_id']}"
        )
    else:
        table.add_row(
            "FalkorDB",
            "[yellow]⚠ WARN[/yellow]",
            f"{falkordb_status.get('message', 'Not running')}\n"
            "[dim]Using in-memory graph store[/dim]"
        )

    # Redis (optional)
    redis_status = status.get("redis", {})
    if redis_status.get("status") == "ok":
        table.add_row(
            "Redis",
            "[green]✓ OK[/green]",
            f"Host: {redis_status['host']}"
        )
    else:
        table.add_row(
            "Redis",
            "[yellow]⚠ WARN[/yellow]",
            f"{redis_status.get('message', 'Not running')}\n"
            "[dim]Optional component[/dim]"
        )

    # Docker
    docker_status = status.get("docker", {})
    if docker_status.get("status") == "ok":
        if "running_containers" in docker_status:
            table.add_row(
                "Docker",
                "[green]✓ OK[/green]",
                f"Containers: {docker_status['running_containers']}/{docker_status['total_containers']} running"
            )
        else:
            table.add_row(
                "Docker",
                "[green]✓ OK[/green]",
                docker_status.get("message", "Running")
            )
    else:
        table.add_row(
            "Docker",
            "[yellow]⚠ WARN[/yellow]",
            f"{docker_status.get('message', 'Not running')}\n"
            "[dim]Optional for in-memory mode[/dim]"
        )

    console.print(table)
    console.print()

    if not critical_ok:
        console.print(Panel(
            "[bold red]⚠ Critical systems not ready[/bold red]\n\n"
            "Please fix the errors above before starting.\n"
            "Run [cyan]ollama list[/cyan] to check models\n"
            "Run [cyan]ollama pull <model>[/cyan] to download missing models",
            border_style="red",
            title="System Check Failed"
        ))
        return False

    console.print(Panel(
        "[bold green]✓ All critical systems operational[/bold green]\n\n"
        "Ready to start conversation.\n"
        "Type [cyan]exit[/cyan] or [cyan]quit[/cyan] to end the session.",
        border_style="green",
        title="Ready"
    ))
    console.print()

    return True


async def show_memory_stats():
    """Display memory and performance statistics."""
    from pathlib import Path
    import os

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Memory Statistics[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    stats_table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
    stats_table.add_column("Metric", style="cyan", width=25)
    stats_table.add_column("Value", style="green", width=40)

    try:
        # Vector store stats
        from .memory.vector_store import init_vector_store
        vector_table = init_vector_store()
        memory_count = vector_table.count_rows()

        stats_table.add_row("Vector Memories", str(memory_count))

        # Get latest memories
        if memory_count > 0:
            rows = vector_table.to_arrow().to_pylist()
            avg_utility = sum(r.get("utility_score", 0) for r in rows) / len(rows)
            stats_table.add_row("Average Utility", f"{avg_utility:.3f}")

            # Utility distribution
            high = sum(1 for r in rows if r.get("utility_score", 0) >= 0.8)
            medium = sum(1 for r in rows if 0.4 <= r.get("utility_score", 0) < 0.8)
            low = sum(1 for r in rows if r.get("utility_score", 0) < 0.4)
            stats_table.add_row("Utility Distribution", f"High: {high}, Med: {medium}, Low: {low}")
    except Exception as e:
        stats_table.add_row("Vector Store", f"[red]Error: {str(e)[:30]}[/red]")

    try:
        # Graph store stats
        from .memory.graph_store import create_graph_store
        graph_store = create_graph_store()

        if hasattr(graph_store, 'entities'):
            entity_count = len(graph_store.entities)
            rel_count = len(graph_store.relationships)

            stats_table.add_row("Graph Entities", str(entity_count))
            stats_table.add_row("Graph Relationships", str(rel_count))

            # Entity types
            if entity_count > 0:
                type_counts = {}
                for entity_data in graph_store.entities.values():
                    entity_type = entity_data.get('type', 'Unknown')
                    type_counts[entity_type] = type_counts.get(entity_type, 0) + 1

                type_str = ", ".join(f"{k}: {v}" for k, v in sorted(type_counts.items()))
                stats_table.add_row("Entity Types", type_str[:40])
        else:
            # FalkorDB - can't easily get counts without querying
            stats_table.add_row("Graph Store", "FalkorDB (use inspect_memory.py for details)")
    except Exception as e:
        stats_table.add_row("Graph Store", f"[red]Error: {str(e)[:30]}[/red]")

    try:
        # Disk usage
        lancedb_path = Path(settings.lancedb_path)
        if lancedb_path.exists():
            total_size = sum(f.stat().st_size for f in lancedb_path.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            stats_table.add_row("LanceDB Size", f"{size_mb:.2f} MB")
    except Exception as e:
        stats_table.add_row("LanceDB Size", f"[red]Error: {str(e)[:30]}[/red]")

    # Configuration
    stats_table.add_row("", "")  # Separator
    stats_table.add_row("Vector Search Top-K", str(settings.vector_search_top_k))
    stats_table.add_row("Rerank Top-K", str(settings.rerank_top_k))
    stats_table.add_row("Temporal Decay (core)", f"{settings.temporal_decay_core} days (never decays)")
    stats_table.add_row("Temporal Decay (HIGH)", f"{settings.temporal_decay_high} days")
    stats_table.add_row("Temporal Decay (MEDIUM)", f"{settings.temporal_decay_medium} days")
    stats_table.add_row("Temporal Decay (LOW)", f"{settings.temporal_decay_low} days")

    console.print(stats_table)
    console.print()


async def run_chat():
    """Main chat loop with system checks."""

    # Run system checks
    console.print("[dim]Checking systems...[/dim]")
    status = await check_system_status()

    if not display_system_status(status):
        console.print("\n[yellow]Start anyway? (y/n):[/yellow] ", end="")
        response = input().strip().lower()
        if response not in ["y", "yes"]:
            console.print("[red]Exiting.[/red]")
            return
        console.print()

    # Initialize conversation
    try:
        graph = create_conversation_graph()
    except Exception as e:
        console.print(f"\n[red]✗ Error initializing conversation graph:[/red] {e}")
        return

    history: list[dict[str, str]] = []
    conversation_id = str(uuid.uuid4())
    
    # Initialize conversation logger
    logger = ConversationLogger()
    logger.start_conversation(conversation_id)

    # Initialize TTS engine
    tts_config = VoiceConfig(
        voice=settings.tts_voice,
        speed=settings.tts_speed,
        enabled=settings.tts_enabled,
    )
    tts_engine = TTSEngine(tts_config)

    if tts_config.enabled:
        console.print(f"[dim]🔊 TTS enabled with voice: {tts_config.voice}[/dim]")

    console.print(f"[dim]Conversation ID: {conversation_id}[/dim]\n")

    tts_task: asyncio.Task | None = None

    # Chat loop
    while True:
        try:
            # Run input in thread pool so event loop can handle background tasks (TTS)
            user_input = await asyncio.to_thread(
                lambda: console.input("[bold cyan]You:[/bold cyan] ").strip()
            )
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Interrupted. Saving memories...[/yellow]")
            logger.end_conversation()
            await wait_for_observers()
            console.print("[green]Memories saved. Goodbye![/green]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            console.print("[yellow]Saving memories...[/yellow]")
            logger.end_conversation()
            await wait_for_observers()
            console.print("[green]Memories saved. Goodbye![/green]")
            break

        # Special commands
        if user_input.lower() == "/status":
            status = await check_system_status()
            display_system_status(status)
            continue

        if user_input.lower() == "/clear":
            console.clear()
            console.print(f"[dim]Conversation ID: {conversation_id}[/dim]\n")
            continue

        if user_input.lower() == "/help":
            console.print(Panel(
                "[bold]Available Commands:[/bold]\n\n"
                "/status - Check system status\n"
                "/stats  - Show memory statistics\n"
                "/clear  - Clear screen\n"
                "/voice  - Toggle TTS on/off\n"
                "/voices - List available voices\n"
                "/speed <0.5-2.0> - Set speech speed\n"
                "/help   - Show this help\n"
                "exit    - Exit the chat\n"
                "quit    - Exit the chat",
                title="Help",
                border_style="cyan"
            ))
            continue

        if user_input.lower() == "/stats":
            await show_memory_stats()
            continue

        # Voice commands
        if user_input.lower() == "/voice":
            enabled = tts_engine.toggle()
            console.print(f"[cyan]🔊 TTS {'enabled' if enabled else 'disabled'}[/cyan]")
            if not enabled and tts_task and not tts_task.done():
                tts_task.cancel()
            continue

        if user_input.lower() == "/voices":
            console.print(Panel(
                "[bold]Available Voices:[/bold]\n\n" +
                "\n".join([f"• {voice}" for voice in VoiceConfig.FEMALE_VOICES]) +
                f"\n\n[dim]Current: {tts_engine.config.voice}[/dim]\n"
                f"[dim]Usage: Set TTS_VOICE={tts_engine.config.voice} in .env[/dim]",
                title="Kokoro TTS Voices",
                border_style="cyan"
            ))
            continue

        if user_input.lower().startswith("/speed "):
            try:
                speed = float(user_input.split()[1])
                tts_engine.set_speed(speed)
                console.print(f"[cyan]🔊 Speech speed set to {tts_engine.config.speed}x[/cyan]")
            except (ValueError, IndexError):
                console.print("[red]Usage: /speed <0.5-2.0>[/red]")
            continue

        state: ConversationState = {
            "user_input": user_input,
            "conversation_history": history,
            "conversation_id": conversation_id,
            "retrieved_context": "",
            "retrieval_sources": [],
            "assistant_response": "",
            "observer_triggered": False,
            "observer_output": None,
        }

        try:
            # Stream response on same line as Assistant: label
            console.print("[bold green]Assistant:[/bold green] ", end="")
            full_response = ""
            async for token in generate_response_streaming(state):
                console.print(token, end="")
                full_response += token
            console.print("\n")  # Blank line after response

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": full_response.strip()})

            # Log conversation turn
            logger.log_turn(user_input, full_response.strip())

            # TTS: Speak the response (async, non-blocking for user input)
            if tts_engine.config.enabled:
                sentences = split_into_sentences(full_response.strip())
                if sentences:
                    if tts_task and not tts_task.done():
                        tts_task.cancel()
                    # Play TTS in background task so user can continue typing
                    tts_task = asyncio.create_task(tts_engine.speak_streaming(sentences))
                    # Yield to event loop so TTS task starts immediately
                    await asyncio.sleep(0)

        except Exception as e:
            console.print(f"\n[red]✗ Error:[/red] {e}\n")


def main():
    """Entry point."""
    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
        # Observer tasks already handled by run_chat() if interrupted there
        sys.exit(0)


if __name__ == "__main__":
    main()
