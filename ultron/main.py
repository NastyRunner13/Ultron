"""Ultron main entry point — Phase 1.

Loads the genesis blueprint, creates a body, and runs a single task.
This will evolve into the full evolution loop in Phase 6.

Usage:
    uv run python -m ultron.main
    uv run python -m ultron.main --task "List files in the current directory"
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from ultron import __version__
from ultron.body.factory import BodyFactory
from ultron.core.logging import setup_logging

console = Console()


BANNER = r"""
 ██╗   ██╗██╗  ████████╗██████╗  ██████╗ ███╗   ██╗
 ██║   ██║██║  ╚══██╔══╝██╔══██╗██╔═══██╗████╗  ██║
 ██║   ██║██║     ██║   ██████╔╝██║   ██║██╔██╗ ██║
 ██║   ██║██║     ██║   ██╔══██╗██║   ██║██║╚██╗██║
 ╚██████╔╝███████╗██║   ██║  ██║╚██████╔╝██║ ╚████║
  ╚═════╝ ╚══════╝╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ultron — Self-Evolving AI Agent",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task to execute. If not provided, enters interactive mode.",
    )
    parser.add_argument(
        "--blueprint",
        type=str,
        default=None,
        help="Path to a custom blueprint YAML (default: genesis blueprint).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Ultron v{__version__}",
    )
    return parser.parse_args()


async def run_task(task: str, blueprint_path: str | None = None) -> None:
    """Load a blueprint, create a body, and run a single task."""
    factory = BodyFactory()

    # Load blueprint
    if blueprint_path:
        from pathlib import Path

        from ultron.body.blueprint import Blueprint
        blueprint = Blueprint.from_yaml(Path(blueprint_path))
        console.print(f"[dim]Loaded blueprint: {blueprint_path}[/dim]")
    else:
        blueprint = factory.load_genesis()
        console.print("[dim]Loaded genesis blueprint[/dim]")

    # Display blueprint info
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Key", style="bold cyan")
    info_table.add_column("Value")
    info_table.add_row("Model", blueprint.model.litellm_model)
    info_table.add_row("Tools", ", ".join(t.name for t in blueprint.tools))
    info_table.add_row("Hash", blueprint.content_hash)
    console.print(Panel(info_table, title="[bold]Body Configuration[/bold]", border_style="blue"))

    # Create body
    with console.status("[bold green]Creating body..."):
        body = await factory.create(blueprint)

    console.print(f"\n[bold yellow]Task:[/bold yellow] {task}\n")

    # Run task
    with console.status("[bold green]Thinking..."):
        result = await body.run(task)

    # Display results
    console.print()
    if result.success:
        console.print(Panel(
            Markdown(result.response),
            title="[bold green]Response[/bold green]",
            border_style="green",
        ))
    else:
        console.print(Panel(
            f"[red]{result.error}[/red]",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))

    # Display metadata
    meta_table = Table(show_header=False, box=None, padding=(0, 2))
    meta_table.add_column("Key", style="bold dim")
    meta_table.add_column("Value", style="dim")
    meta_table.add_row("Steps", str(result.total_steps))
    meta_table.add_row("Tool Calls", str(len(result.tool_calls)))
    meta_table.add_row("Tokens", str(result.token_usage.get("total_tokens", "N/A")))
    meta_table.add_row("Duration", f"{result.duration_seconds:.1f}s")
    console.print(Panel(meta_table, title="[dim]Run Metadata[/dim]", border_style="dim"))

    # Show tool calls if any
    if result.tool_calls:
        tool_table = Table(title="Tool Calls", show_lines=True)
        tool_table.add_column("Step", style="cyan", width=6)
        tool_table.add_column("Tool", style="bold")
        tool_table.add_column("Status", width=8)
        tool_table.add_column("Result Preview", max_width=60)

        for tc in result.tool_calls:
            status = "[green]OK[/green]" if tc.success else "[red]FAIL[/red]"
            preview = tc.result[:80] + "..." if len(tc.result) > 80 else tc.result
            tool_table.add_row(str(tc.step), tc.tool_name, status, preview)

        console.print(tool_table)


async def interactive_mode(blueprint_path: str | None = None) -> None:
    """Run Ultron in interactive mode — enter tasks one at a time."""
    factory = BodyFactory()

    if blueprint_path:
        from pathlib import Path

        from ultron.body.blueprint import Blueprint
        blueprint = Blueprint.from_yaml(Path(blueprint_path))
    else:
        blueprint = factory.load_genesis()

    body = await factory.create(blueprint)

    console.print(
        "\n[bold]Interactive mode.[/bold] Type a task and press Enter. "
        "Type [bold red]quit[/bold red] or [bold red]exit[/bold red] to stop.\n"
    )

    while True:
        try:
            task = console.input("[bold cyan]ultron>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not task:
            continue
        if task.lower() in ("quit", "exit"):
            console.print("[dim]Goodbye.[/dim]")
            break

        result = await body.run(task)

        if result.success:
            console.print(f"\n{result.response}\n")
        else:
            console.print(f"\n[red]Error: {result.error}[/red]\n")

        console.print(
            f"[dim]({result.total_steps} steps, "
            f"{len(result.tool_calls)} tool calls, "
            f"{result.token_usage.get('total_tokens', '?')} tokens, "
            f"{result.duration_seconds:.1f}s)[/dim]\n"
        )


async def async_main() -> None:
    """Async entry point."""
    args = parse_args()

    console.print(f"[bold blue]{BANNER}[/bold blue]")
    console.print(f"[dim]v{__version__} — Self-Evolving AI Agent[/dim]\n")

    if args.task:
        await run_task(args.task, args.blueprint)
    else:
        await interactive_mode(args.blueprint)


def main() -> None:
    """Sync entry point for the CLI."""
    setup_logging()
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(0)


if __name__ == "__main__":
    main()
