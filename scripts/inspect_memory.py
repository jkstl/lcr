"""Inspect persisted memory in LanceDB and FalkorDB."""

import asyncio
import sys
from pathlib import Path

from rich import print
from rich.table import Table
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory.vector_store import init_vector_store
from src.memory.graph_store import create_graph_store
from src.config import settings


async def main():
    console = Console()

    # Vector Store Inspection
    print("\n[bold cyan]═══ VECTOR STORE (LanceDB) ═══[/bold cyan]\n")
    try:
        vector_table = init_vector_store()
        count = vector_table.count_rows()
        print(f"[green]✓[/green] Total memory chunks: {count}")

        if count > 0:
            # Show latest 10 memories
            all_rows = vector_table.to_arrow().to_pylist()
            sorted_rows = sorted(all_rows, key=lambda x: x.get("created_at", ""), reverse=True)

            table = Table(title="Latest Memories (top 10)")
            table.add_column("Conv ID", style="cyan")
            table.add_column("Turn", style="magenta")
            table.add_column("Utility", style="yellow")
            table.add_column("Summary", style="green", max_width=50)
            table.add_column("Created", style="blue")

            for row in sorted_rows[:10]:
                table.add_row(
                    str(row.get("source_conversation_id", ""))[:20],
                    str(row.get("turn_index", "")),
                    f"{row.get('utility_score', 0):.2f}",
                    str(row.get("summary", ""))[:50],
                    str(row.get("created_at", ""))[:19],
                )

            console.print(table)

            # Show retrieval queries
            print("\n[bold yellow]Sample retrieval queries:[/bold yellow]")
            for row in sorted_rows[:5]:
                queries = row.get("retrieval_queries", [])
                if queries:
                    print(f"  • {queries}")

    except Exception as e:
        print(f"[red]✗[/red] Error reading vector store: {e}")

    # Graph Store Inspection
    print("\n[bold cyan]═══ GRAPH STORE (FalkorDB/InMemory) ═══[/bold cyan]\n")
    try:
        graph_store = create_graph_store()
        store_type = type(graph_store).__name__
        print(f"[green]✓[/green] Store type: {store_type}")

        if hasattr(graph_store, 'entities'):
            # InMemoryGraphStore
            print(f"\n[bold green]Entities:[/bold green] {len(graph_store.entities)} total")

            entity_table = Table(title="All Entities")
            entity_table.add_column("Name", style="cyan")
            entity_table.add_column("Type", style="magenta")
            entity_table.add_column("Attributes", style="yellow", max_width=40)

            for name, data in graph_store.entities.items():
                entity_table.add_row(
                    name,
                    data.get("type", "Unknown"),
                    str(data.get("attributes", {}))[:40],
                )

            console.print(entity_table)

            print(f"\n[bold green]Relationships:[/bold green] {len(graph_store.relationships)} total")

            rel_table = Table(title="All Relationships")
            rel_table.add_column("Subject", style="cyan")
            rel_table.add_column("Predicate", style="magenta")
            rel_table.add_column("Object", style="green")
            rel_table.add_column("Valid", style="yellow")
            rel_table.add_column("Metadata", style="blue", max_width=30)

            for rel in graph_store.relationships:
                valid = "✓" if rel.metadata.get("still_valid", True) else "✗"
                rel_table.add_row(
                    rel.subject,
                    rel.predicate,
                    rel.object,
                    valid,
                    str(rel.metadata)[:30] if rel.metadata else "",
                )

            console.print(rel_table)

        else:
            # FalkorGraphStore - need to query
            print("\n[bold yellow]Querying FalkorDB...[/bold yellow]")

            # Try to get some sample relationships
            sample_entities = ["User", "Giana", "Mom", "Justine", "Max", "CoSphere"]
            relationships = await graph_store.search_relationships(sample_entities, limit=50)

            if relationships:
                print(f"[green]✓[/green] Found {len(relationships)} relationships")

                rel_table = Table(title="Persisted Relationships")
                rel_table.add_column("Subject", style="cyan")
                rel_table.add_column("Predicate", style="magenta")
                rel_table.add_column("Object", style="green")
                rel_table.add_column("Created", style="blue")

                for rel in relationships:
                    rel_table.add_row(
                        rel.subject,
                        rel.predicate,
                        rel.object,
                        str(rel.created_at)[:19] if rel.created_at else "",
                    )

                console.print(rel_table)
            else:
                print("[yellow]⚠[/yellow] No relationships found for sample entities")
                print("  Tip: Relationships may exist but not match the sample entity names")

    except Exception as e:
        print(f"[red]✗[/red] Error reading graph store: {e}")

    # Configuration
    print("\n[bold cyan]═══ CONFIGURATION ═══[/bold cyan]\n")
    config_table = Table()
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("LanceDB Path", settings.lancedb_path)
    config_table.add_row("FalkorDB Host", f"{settings.falkordb_host}:{settings.falkordb_port}")
    config_table.add_row("Graph ID", settings.falkordb_graph_id)
    config_table.add_row("Observer Model", settings.observer_model)
    config_table.add_row("Main Model", settings.main_model)
    config_table.add_row("Temporal Decay", f"{settings.temporal_decay_days} days")
    config_table.add_row("Vector Search Top-K", str(settings.vector_search_top_k))
    config_table.add_row("Rerank Top-K", str(settings.rerank_top_k))

    console.print(config_table)


if __name__ == "__main__":
    asyncio.run(main())
