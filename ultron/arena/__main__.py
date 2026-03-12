"""Arena CLI — run benchmarks against a body from the command line.

Usage:
    uv run python -m ultron.arena
    uv run python -m ultron.arena --tier 1
    uv run python -m ultron.arena --blueprint config/genesis_blueprint.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ultron.arena.leaderboard import Leaderboard
from ultron.arena.loader import get_tier_summary, load_benchmarks
from ultron.arena.runner import ArenaRunner
from ultron.arena.skill_tree import SkillTree
from ultron.body.factory import BodyFactory
from ultron.core.logging import setup_logging

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultron Arena — Benchmark Evaluation")
    parser.add_argument("--blueprint", type=str, default=None, help="Path to blueprint YAML")
    parser.add_argument("--tier", type=int, default=None, help="Run only a specific tier (1-5)")
    parser.add_argument("--runs", type=int, default=None, help="Runs per task (default: from settings)")
    parser.add_argument("--list", action="store_true", help="List all benchmarks and exit")
    return parser.parse_args()


async def run_arena(args: argparse.Namespace) -> None:
    """Run the arena evaluation."""
    benchmarks = load_benchmarks()

    # List mode
    if args.list:
        tier_summary = get_tier_summary(benchmarks)
        table = Table(title="Arena Benchmarks")
        table.add_column("ID", style="cyan")
        table.add_column("Tier", style="bold")
        table.add_column("Category")
        table.add_column("Name")
        table.add_column("Scorer")

        for task in benchmarks:
            table.add_row(task.id, task.tier_label, task.category, task.name, task.scoring_method)

        console.print(table)
        console.print(f"\n[dim]Total: {len(benchmarks)} tasks across {len(tier_summary)} tiers[/dim]")
        return

    # filter by tier if specified
    if args.tier:
        from ultron.arena.loader import get_benchmarks_by_tier
        benchmarks = get_benchmarks_by_tier(benchmarks, args.tier)
        if not benchmarks:
            console.print(f"[red]No benchmarks found for tier L{args.tier}[/red]")
            return

    # Load blueprint
    factory = BodyFactory()
    if args.blueprint:
        from pathlib import Path
        from ultron.body.blueprint import Blueprint
        blueprint = Blueprint.from_yaml(Path(args.blueprint))
    else:
        blueprint = factory.load_genesis()

    console.print(
        Panel(
            f"[bold]{blueprint.name}[/bold] | {blueprint.model.litellm_model}\n"
            f"Tasks: {len(benchmarks)} | Runs per task: {args.runs or 'default'}",
            title="[bold blue]Arena Evaluation[/bold blue]",
            border_style="blue",
        )
    )

    # Create body and runner
    body = await factory.create(blueprint)
    runner = ArenaRunner(benchmarks=benchmarks, runs_per_task=args.runs)

    # Run evaluation
    with console.status("[bold green]Running arena..."):
        score = await runner.evaluate(body)

    # Display results
    console.print(Panel(score.to_summary(), title="[bold green]Results[/bold green]", border_style="green"))

    # Tier breakdown
    tier_table = Table(title="Tier Breakdown")
    tier_table.add_column("Tier", style="bold")
    tier_table.add_column("Score", justify="right")
    for tier, avg in sorted(score.tier_scores.items()):
        color = "green" if avg >= 0.7 else ("yellow" if avg >= 0.4 else "red")
        tier_table.add_row(tier, f"[{color}]{avg:.1%}[/{color}]")
    console.print(tier_table)

    # Save to leaderboard
    leaderboard = Leaderboard.load()
    leaderboard.record(score)
    leaderboard.save()
    console.print("[dim]Score saved to leaderboard.[/dim]")

    # Update skill tree
    skill_tree = SkillTree.load()
    xp_gains = skill_tree.update_from_score(score)
    skill_tree.save()
    if xp_gains:
        console.print(f"[dim]Skill tree updated: {len(xp_gains)} nodes gained XP.[/dim]")


def main() -> None:
    setup_logging()
    args = parse_args()
    try:
        asyncio.run(run_arena(args))
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(0)


if __name__ == "__main__":
    main()
