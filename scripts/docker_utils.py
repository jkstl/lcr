"""
Docker utility functions for LCR scripts.
Handles detection of 'docker compose' vs 'docker-compose' commands.
"""

import shutil
import subprocess
from typing import List, Optional

def get_docker_compose_cmd() -> Optional[List[str]]:
    """
    Determine the available docker compose command.
    
    Returns:
        List[str]: The command components (e.g. ['docker', 'compose'] or ['docker-compose'])
        None: If no docker compose command is found
    """
    # Check for 'docker compose' (V2 plugin)
    try:
        # We use 'docker compose version' to check if the plugin is installed and working
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return ["docker", "compose"]
    except FileNotFoundError:
        pass

    # Check for 'docker-compose' (Standalone)
    if shutil.which("docker-compose"):
        return ["docker-compose"]

    return None

def run_docker_compose(args: List[str], cwd: Optional[str] = None, capture_output: bool = True, text: bool = True, timeout: int = None) -> subprocess.CompletedProcess:
    """
    Run a docker compose command using the detected executable.
    
    Args:
        args: List of arguments to append to the docker compose command (e.g. ['up', '-d'])
        cwd: Current working directory
        capture_output: Whether to capture stdout/stderr
        text: Whether to return output as text
        timeout: Timeout in seconds
        
    Returns:
        subprocess.CompletedProcess
        
    Raises:
        RuntimeError: If docker compose is not available
    """
    base_cmd = get_docker_compose_cmd()
    if not base_cmd:
        raise RuntimeError("Docker Compose not found. Please install 'docker-compose-plugin' or 'docker-compose'.")
        
    full_cmd = base_cmd + args
    
    return subprocess.run(
        full_cmd,
        cwd=cwd,
        capture_output=capture_output,
        text=text,
        timeout=timeout
    )


def cleanup_conflicting_containers() -> None:
    """
    Aggressively find and remove any Docker containers that might be blocking our ports.
    Targeting ports 6379 (FalkorDB) and 6380 (Redis) and common container names.
    """
    from rich import print

    # 1. Try to find containers by name (handle project name variations)
    # Common variations: lcr-falkordb-1, lcr-codex-falkordb-1, lcr-codex_claudereview-falkordb-1
    # We'll search for *falkordb* and *redis* in 'docker ps'
    
    print("[dim]Scanning for conflicting containers...[/dim]")
    
    try:
        # List all containers (running and stopped)
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.ID}} {{.Names}} {{.Ports}}"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return

        containers_to_kill = []
        
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
                
            parts = line.split(" ", 2)
            if len(parts) < 2:
                continue
                
            cid = parts[0]
            name = parts[1]
            ports = parts[2] if len(parts) > 2 else ""
            
            # Criteria for killing:
            # 1. Name contains 'falkordb' or 'redis' AND seems related to LCR
            # 2. Ports bind 6379 or 6380
            
            is_target_name = "falkordb" in name.lower() or "redis" in name.lower()
            is_target_port = "6379" in ports or "6380" in ports
            
            # Safety check: Only kill if it looks like OUR container or binding OUR ports
            if is_target_port or (is_target_name and ("lcr" in name.lower() or "codex" in name.lower())):
                containers_to_kill.append((cid, name))

        if containers_to_kill:
            print(f"[yellow]Found {len(containers_to_kill)} potential conflict(s):[/yellow]")
            for cid, name in containers_to_kill:
                print(f"  • {name} ({cid})")
                
            # Force remove them
            print("[yellow]Force removing conflicting containers...[/yellow]")
            ids = [c[0] for c in containers_to_kill]
            subprocess.run(["docker", "rm", "-f"] + ids, capture_output=True)
            print("[green]✓[/green] Conflicts cleared")
            
    except Exception as e:
        print(f"[red]Warning: Failed to cleanup containers: {e}[/red]")

