"""Nuclear reset: Completely wipe all LCR memory data and restart services fresh."""
import asyncio
import subprocess
import shutil
from pathlib import Path
import sys

from rich import print
from rich.panel import Panel
from rich.prompt import Confirm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings
from scripts.docker_utils import run_docker_compose, get_docker_compose_cmd, cleanup_conflicting_containers


async def nuclear_reset():
    """Complete memory wipe: stop services, delete all data, restart clean."""
    
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    
    print()
    print(Panel.fit(
        "[bold red]⚠ NUCLEAR RESET ⚠[/bold red]\n\n"
        "This will PERMANENTLY delete ALL memory data:\n"
        f"  • LanceDB vectors: {settings.lancedb_path}\n"
        f"  • FalkorDB graph: {data_dir / 'falkordb'}\n"
        f"  • Redis cache: {data_dir / 'redis'}\n\n"
        "[yellow]This action CANNOT be undone![/yellow]",
        border_style="red"
    ))
    print()
    
    if not Confirm.ask("[red]Are you ABSOLUTELY SURE you want to continue?[/red]", default=False):
        print("\n[green]✓[/green] Cancelled. No data was deleted.")
        return False
    
    print("\n[bold yellow]Starting nuclear reset...[/bold yellow]\n")
    
    # Step 1: Stop Docker services
    print("1. [yellow]Stopping Docker services...[/yellow]")
    try:
        result = run_docker_compose(
            ["down"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print("   [green]✓[/green] Docker services stopped")
        else:
            print(f"   [yellow]⚠[/yellow] Docker stop warning: {result.stderr}")
    except Exception as e:
        print(f"   [yellow]⚠[/yellow] Could not stop Docker (may not be running): {e}")
    
    # Extra cleanup for port conflicts (zombie containers from renamed folders)
    cleanup_conflicting_containers()

    # Step 2: Delete all data
    print("\n2. [yellow]Deleting all data...[/yellow]")
    
    if data_dir.exists():
        try:
            shutil.rmtree(data_dir)
            print(f"   [green]✓[/green] Deleted: {data_dir}")
        except Exception as e:
            print(f"   [red]✗[/red] Error deleting data directory: {e}")
            print("   [yellow]TIP: Close any programs accessing these files and try again[/yellow]")
            return False
    else:
        print(f"   [dim]Data directory already deleted[/dim]")
    
    # Step 3: Recreate empty data directories
    print("\n3. [yellow]Creating fresh data directories...[/yellow]")
    (data_dir / "lancedb").mkdir(parents=True, exist_ok=True)
    (data_dir / "falkordb").mkdir(parents=True, exist_ok=True)
    (data_dir / "redis").mkdir(parents=True, exist_ok=True)
    print("   [green]✓[/green] Fresh directories created")
    
    # Step 4: Restart Docker services
    print("\n4. [yellow]Starting Docker services with clean state...[/yellow]")
    try:
        result = run_docker_compose(
            ["up", "-d", "falkordb", "redis"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("   [green]✓[/green] Docker services started")
        else:
            print(f"   [red]✗[/red] Docker start error: {result.stderr}")
            return False
    except Exception as e:
        print(f"   [red]✗[/red] Could not start Docker: {e}")
        return False
    
    # Step 5: Wait for initialization
    print("\n5. [yellow]Waiting for services to initialize...[/yellow]")
    await asyncio.sleep(4)
    print("   [green]✓[/green] Services ready")
    
    print("\n[bold green]═══════════════════════════════[/bold green]")
    print("[bold green]✓ NUCLEAR RESET COMPLETE[/bold green]")
    print("[bold green]═══════════════════════════════[/bold green]\n")
    print("[dim]All memory data has been permanently deleted.[/dim]")
    print("[dim]You now have a clean slate.[/dim]\n")
    print("Next step: Run [cyan]python -m src.main[/cyan] to start a fresh conversation\n")
    
    return True


if __name__ == "__main__":
    try:
        asyncio.run(nuclear_reset())
    except KeyboardInterrupt:
        print("\n[yellow]Interrupted. Exiting.[/yellow]")
        sys.exit(0)
