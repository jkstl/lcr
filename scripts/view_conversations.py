"""View and export saved conversation logs."""
import json
import sys
from pathlib import Path
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import IntPrompt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings

console = Console()


def list_conversations():
    """List all saved conversations."""
    conversations_dir = Path("./data/conversations")
    
    if not conversations_dir.exists():
        print("[yellow]No conversations directory found.[/yellow]")
        return []
    
    json_files = sorted(conversations_dir.glob("*.json"), reverse=True)
    
    if not json_files:
        print("[yellow]No conversation logs found.[/yellow]")
        return []
    
    table = Table(title=f"Saved Conversations ({len(json_files)} total)")
    table.add_column("#", style="cyan")
    table.add_column("Date/Time", style="green")
    table.add_column("ID", style="magenta")
    table.add_column("Turns", style="yellow")
    table.add_column("Duration", style="blue")
    
    conversations = []
    for idx, filepath in enumerate(json_files, 1):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Parse timestamp from filename
            filename = filepath.stem
            date_time = filename.split("_")[:2]
            date_str = f"{date_time[0]} {date_time[1][:2]}:{date_time[1][2:4]}"
            
            conv_id = data.get("conversation_id", "Unknown")[:8]
            turns = data.get("total_turns", len(data.get("turns", [])))
            
            # Calculate duration
            started = data.get("started_at", "")
            ended = data.get("ended_at", "In progress")
            if started and ended != "In progress":
                from datetime import datetime
                start_dt = datetime.fromisoformat(started)
                end_dt = datetime.fromisoformat(ended)
                duration = end_dt - start_dt
                duration_str = str(duration).split(".")[0]  # Remove microseconds
            else:
                duration_str = "In progress" if ended == "In progress" else "Unknown"
            
            table.add_row(str(idx), date_str, conv_id, str(turns), duration_str)
            conversations.append((idx, filepath, data))
            
        except Exception as e:
            print(f"[red]Error loading {filepath.name}: {e}[/red]")
    
    console.print(table)
    return conversations


def view_conversation(filepath: Path):
    """Display a conversation log."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        console.print()
        console.print(Panel(
            f"[bold]Conversation ID:[/bold] {data.get('conversation_id', 'Unknown')}\n"
            f"[bold]Started:[/bold] {data.get('started_at', 'Unknown')}\n"
            f"[bold]Ended:[/bold] {data.get('ended_at', 'In progress')}\n"
            f"[bold]Total Turns:[/bold] {data.get('total_turns', len(data.get('turns', [])))}",
            title="Conversation Details",
            border_style="cyan"
        ))
        console.print()
        
        turns = data.get("turns", [])
        for turn in turns:
            turn_idx = turn.get("turn_index", 0)
            timestamp = turn.get("timestamp", "")
            user_msg = turn.get("user", "")
            assistant_msg = turn.get("assistant", "")
            
            console.print(f"[dim]Turn {turn_idx} — {timestamp}[/dim]")
            console.print(f"[bold cyan]You:[/bold cyan] {user_msg}")
            console.print(f"[bold green]Assistant:[/bold green] {assistant_msg}")
            console.print()
            
    except Exception as e:
        print(f"[red]Error viewing conversation: {e}[/red]")


def export_conversation(filepath: Path, output_format: str = "txt"):
    """Export conversation to text format."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Create export filename
        export_name = filepath.stem + f"_export.{output_format}"
        export_path = filepath.parent / export_name
        
        with open(export_path, "w", encoding="utf-8") as f:
            f.write(f"Conversation ID: {data.get('conversation_id', 'Unknown')}\n")
            f.write(f"Started: {data.get('started_at', 'Unknown')}\n")
            f.write(f"Ended: {data.get('ended_at', 'In progress')}\n")
            f.write(f"Total Turns: {data.get('total_turns', len(data.get('turns', [])))}\n")
            f.write("=" * 80 + "\n\n")
            
            turns = data.get("turns", [])
            for turn in turns:
                turn_idx = turn.get("turn_index", 0)
                timestamp = turn.get("timestamp", "")
                user_msg = turn.get("user", "")
                assistant_msg = turn.get("assistant", "")
                
                f.write(f"Turn {turn_idx} — {timestamp}\n")
                f.write(f"You: {user_msg}\n")
                f.write(f"Assistant: {assistant_msg}\n")
                f.write("\n")
        
        print(f"[green]✓[/green] Exported to: {export_path}")
        
    except Exception as e:
        print(f"[red]Error exporting conversation: {e}[/red]")


def main():
    """Main menu for conversation viewer."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]📝 LCR Conversation Viewer[/bold cyan]",
        border_style="cyan"
    ))
    console.print()
    
    conversations = list_conversations()
    
    if not conversations:
        return
    
    console.print()
    console.print("[bold]What would you like to do?[/bold]\n")
    console.print("1. View a conversation")
    console.print("2. Export a conversation to .txt")
    console.print("0. Exit")
    console.print()
    
    try:
        choice = IntPrompt.ask("Select option", default=0)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Cancelled.[/yellow]")
        return
    
    if choice == 0:
        return
    
    if choice in [1, 2]:
        console.print()
        try:
            conv_num = IntPrompt.ask("Select conversation number", default=1)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Cancelled.[/yellow]")
            return
        
        # Find selected conversation
        selected = None
        for idx, filepath, data in conversations:
            if idx == conv_num:
                selected = (filepath, data)
                break
        
        if not selected:
            print("[red]Invalid conversation number.[/red]")
            return
        
        filepath, data = selected
        
        if choice == 1:
            view_conversation(filepath)
        elif choice == 2:
            export_conversation(filepath)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Exiting.[/yellow]")
        sys.exit(0)
