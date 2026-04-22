"""
LLMOps Observability Module
Provides structured logging of every agent input, tool call, and output.
Satisfies the 'AgentOps & Observability' requirement.
"""

from __future__ import annotations
import json
import os
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

_trace_entries: List[Dict[str, Any]] = []


def _append_trace(entry: Dict[str, Any]) -> None:
    """Append an entry to the in-memory trace buffer."""
    _trace_entries.append(entry)


def log_agent_start(agent_name: str, task_description: str, input_data: str) -> None:
    """Log when an agent begins its task."""
    entry = {
        "event": "agent_start",
        "agent": agent_name,
        "task": task_description,
        "input_preview": input_data[:300],
        "timestamp": datetime.now().isoformat(),
    }
    _append_trace(entry)
    console.print(
        Panel(
            f"[bold cyan]Agent:[/] {agent_name}\n"
            f"[bold cyan]Task:[/] {task_description}\n"
            f"[bold cyan]Input:[/] {input_data[:120]}...",
            title="[green]▶ Agent Started[/]",
            border_style="green",
        )
    )


def log_tool_call(agent_name: str, tool_name: str, tool_input: str, tool_output: str) -> None:
    """Log every tool invocation with its input and output."""
    entry = {
        "event": "tool_call",
        "agent": agent_name,
        "tool": tool_name,
        "input": tool_input[:500],
        "output": tool_output[:500],
        "timestamp": datetime.now().isoformat(),
    }
    _append_trace(entry)
    console.print(
        f"  [yellow]🔧 Tool:[/] [bold]{tool_name}[/]\n"
        f"     Input  → {tool_input[:80]}...\n"
        f"     Output → {tool_output[:80]}..."
    )


def log_agent_end(agent_name: str, output_summary: str, duration_seconds: float) -> None:
    """Log when an agent completes its task."""
    entry = {
        "event": "agent_end",
        "agent": agent_name,
        "output_preview": output_summary[:300],
        "duration_seconds": round(duration_seconds, 2),
        "timestamp": datetime.now().isoformat(),
    }
    _append_trace(entry)
    console.print(
        Panel(
            f"[bold magenta]Agent:[/] {agent_name}\n"
            f"[bold magenta]Duration:[/] {duration_seconds:.1f}s\n"
            f"[bold magenta]Output:[/] {output_summary[:120]}...",
            title="[blue]✔ Agent Completed[/]",
            border_style="blue",
        )
    )


def log_error(agent_name: str, error_message: str) -> None:
    """Log agent or tool errors."""
    entry = {
        "event": "error",
        "agent": agent_name,
        "error": error_message,
        "timestamp": datetime.now().isoformat(),
    }
    _append_trace(entry)
    console.print(f"[red]✘ Error in {agent_name}:[/] {error_message}")


def save_full_trace(filename: Optional[str] = None) -> str:
    """Persist the complete LLMOps trace to disk."""
    if filename is None:
        filename = f"{LOG_DIR}/trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(_trace_entries, f, indent=2)
    console.print(f"\n[green]📁 Full trace saved to:[/] [underline]{filename}[/]")
    return filename


def print_summary_table() -> None:
    """Print a rich table summarizing all agent actions in the pipeline."""
    table = Table(title="Pipeline Execution Summary", show_lines=True)
    table.add_column("Agent", style="cyan", width=25)
    table.add_column("Event", style="yellow", width=15)
    table.add_column("Tool / Action", style="white", width=30)
    table.add_column("Timestamp", style="dim", width=22)

    for entry in _trace_entries:
        table.add_row(
            entry.get("agent", "-"),
            entry.get("event", "-"),
            entry.get("tool", entry.get("task", entry.get("error", "-")))[:30],
            entry.get("timestamp", "-"),
        )

    console.print(table)


def track_agent(func: Callable) -> Callable:
    """
    Decorator that automatically logs agent start, end, and duration.
    Apply to any agent's run() or execute() method.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        agent_name = getattr(args[0], "name", func.__name__) if args else func.__name__
        start = time.time()
        log_agent_start(agent_name, func.__doc__ or func.__name__, str(kwargs)[:200])
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            log_agent_end(agent_name, str(result)[:200] if result else "completed", duration)
            return result
        except Exception as exc:
            log_error(agent_name, str(exc))
            raise
    return wrapper
