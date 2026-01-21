"""Clear all persisted memory from LanceDB and FalkorDB (use with caution!)."""

import asyncio
import sys
import subprocess
from pathlib import Path
import shutil

from rich import print
from rich.prompt import Confirm, IntPrompt
from rich.panel import Panel
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings
from scripts.docker_utils import run_docker_compose, cleanup_conflicting_containers

console = Console()


async def clear_lancedb():
    """Clear LanceDB vector store."""
    lancedb_path = Path(settings.lancedb_path)
    if lancedb_path.exists():
        try:
            shutil.rmtree(lancedb_path)
            print(f"[green]✓[/green] Deleted LanceDB directory: {lancedb_path}")
            return True
        except Exception as e:
            print(f"[red]✗[/red] Error deleting LanceDB: {e}")
            return False
    else:
        print(f"[yellow]⚠[/yellow] LanceDB directory not found: {lancedb_path}")
        return True


async def clear_falkordb():
    """Clear FalkorDB graph data."""
    try:
        from falkordb import FalkorDB

        print(f"[yellow]Connecting to FalkorDB...[/yellow]")
        client = FalkorDB(host=settings.falkordb_host, port=settings.falkordb_port)
        graph = client.select_graph(settings.falkordb_graph_id)

        # Delete all nodes and relationships
        result = graph.query("MATCH (n) DETACH DELETE n")
        print(f"[green]✓[/green] Cleared FalkorDB graph: {settings.falkordb_graph_id}")
        return True

    except ImportError:
        print("[yellow]⚠[/yellow] FalkorDB library not installed, skipping graph cleanup")
        return True
    except Exception as e:
        print(f"[red]✗[/red] Error clearing FalkorDB: {e}")
        print("  (FalkorDB may not be running or accessible)")
        return False


async def restart_docker_services():
    """Restart Docker Compose services."""
    try:
        project_root = Path(__file__).resolve().parents[1]

        print("\n[yellow]Stopping Docker services...[/yellow]")
        result = run_docker_compose(
            ["down"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"[red]✗[/red] Error stopping services: {result.stderr}")
            return False

        print("[green]✓[/green] Services stopped")

        # Cleanup potential conflicts
        cleanup_conflicting_containers()

        print("[yellow]Starting Docker services...[/yellow]")
        result = run_docker_compose(
            ["up", "-d", "falkordb", "redis"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"[red]✗[/red] Error starting services: {result.stderr}")
            return False

        print("[green]✓[/green] Services started")

        # Wait a moment for services to initialize
        print("[dim]Waiting for services to initialize...[/dim]")
        await asyncio.sleep(3)

        return True

    except FileNotFoundError:
        print("[red]✗[/red] Docker not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("[red]✗[/red] Docker command timed out")
        return False
    except Exception as e:
        print(f"[red]✗[/red] Error restarting Docker services: {e}")
        return False


async def stop_docker_services():
    """Stop Docker Compose services."""
    try:
        project_root = Path(__file__).resolve().parents[1]

        print("\n[yellow]Stopping Docker services...[/yellow]")
        result = run_docker_compose(
            ["down"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"[red]✗[/red] Error stopping services: {result.stderr}")
            return False

        print("[green]✓[/green] Services stopped")
        return True

    except Exception as e:
        print(f"[red]✗[/red] Error stopping Docker services: {e}")
        return False


async def start_docker_services():
    """Start Docker Compose services."""
    try:
        project_root = Path(__file__).resolve().parents[1]

        print("\n[yellow]Starting Docker services...[/yellow]")
        result = run_docker_compose(
            ["up", "-d", "falkordb", "redis"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"[red]✗[/red] Error starting services: {result.stderr}")
            return False

        print("[green]✓[/green] Services started")

        # Wait for initialization
        print("[dim]Waiting for services to initialize...[/dim]")
        await asyncio.sleep(3)

        return True

    except Exception as e:
        print(f"[red]✗[/red] Error starting Docker services: {e}")
        return False


async def main():
    """Main menu for memory management."""

    console.print()
    console.print(Panel.fit(
        "[bold red]⚠ LCR Memory Management Tool ⚠[/bold red]",
        border_style="red"
    ))
    console.print()

    console.print("[bold]What would you like to do?[/bold]\n")
    console.print("1. Clear all memory (LanceDB + FalkorDB)")
    console.print("2. Clear LanceDB only (vector store)")
    console.print("3. Clear FalkorDB only (knowledge graph)")
    console.print("4. Restart Docker services (FalkorDB + Redis)")
    console.print("5. Stop Docker services")
    console.print("6. Start Docker services")
    console.print("0. Exit")
    console.print()

    try:
        choice = IntPrompt.ask("Select option", default=0)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Cancelled.[/yellow]")
        return

    if choice == 0:
        console.print("[green]Exiting.[/green]")
        return

    # Confirm destructive operations
    if choice in [1, 2, 3]:
        console.print()
        console.print("[bold yellow]WARNING:[/bold yellow] This will permanently delete:")

        if choice == 1:
            console.print(f"  • Vector store: [cyan]{settings.lancedb_path}[/cyan]")
            console.print(f"  • Graph data: [cyan]{settings.falkordb_graph_id}[/cyan]")
        elif choice == 2:
            console.print(f"  • Vector store: [cyan]{settings.lancedb_path}[/cyan]")
        elif choice == 3:
            console.print(f"  • Graph data: [cyan]{settings.falkordb_graph_id}[/cyan]")

        console.print()
        if not Confirm.ask("[red]Are you sure you want to continue?[/red]", default=False):
            console.print("\n[green]✓[/green] Cancelled. No data was deleted.")
            return

    # Execute chosen operation
    console.print()

    if choice == 1:
        # Clear all memory
        console.print("[bold]Clearing all memory...[/bold]\n")
        lancedb_ok = await clear_lancedb()
        falkordb_ok = await clear_falkordb()

        if lancedb_ok and falkordb_ok:
            console.print("\n[bold green]✓ Memory cleared successfully![/bold green]")
            console.print("Tip: Run [cyan]python scripts/inspect_memory.py[/cyan] to verify")
        else:
            console.print("\n[bold yellow]⚠ Memory clearing completed with errors[/bold yellow]")

    elif choice == 2:
        # Clear LanceDB only
        console.print("[bold]Clearing LanceDB...[/bold]\n")
        if await clear_lancedb():
            console.print("\n[bold green]✓ LanceDB cleared successfully![/bold green]")
        else:
            console.print("\n[bold yellow]⚠ LanceDB clearing failed[/bold yellow]")

    elif choice == 3:
        # Clear FalkorDB only
        console.print("[bold]Clearing FalkorDB...[/bold]\n")
        if await clear_falkordb():
            console.print("\n[bold green]✓ FalkorDB cleared successfully![/bold green]")
        else:
            console.print("\n[bold yellow]⚠ FalkorDB clearing failed[/bold yellow]")

    elif choice == 4:
        # Restart Docker services
        console.print("[bold]Restarting Docker services...[/bold]")
        if await restart_docker_services():
            console.print("\n[bold green]✓ Docker services restarted successfully![/bold green]")
            console.print("Tip: Run [cyan]python -m src.main[/cyan] to test the connection")
        else:
            console.print("\n[bold yellow]⚠ Docker restart failed[/bold yellow]")

    elif choice == 5:
        # Stop Docker services
        console.print("[bold]Stopping Docker services...[/bold]")
        if await stop_docker_services():
            console.print("\n[bold green]✓ Docker services stopped successfully![/bold green]")
        else:
            console.print("\n[bold yellow]⚠ Docker stop failed[/bold yellow]")

    elif choice == 6:
        # Start Docker services
        console.print("[bold]Starting Docker services...[/bold]")
        if await start_docker_services():
            console.print("\n[bold green]✓ Docker services started successfully![/bold green]")
            console.print("Tip: Run [cyan]python -m src.main[/cyan] to test the connection")
        else:
            console.print("\n[bold yellow]⚠ Docker start failed[/bold yellow]")

    else:
        console.print("[red]Invalid option.[/red]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Exiting.[/yellow]")
        sys.exit(0)
