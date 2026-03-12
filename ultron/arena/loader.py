"""Benchmark loader — load and query benchmark tasks from YAML.

Provides functions to load benchmark definitions from the YAML config
and filter them by tier, category, or required tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from ultron.arena.models import BenchmarkTask
from ultron.core.settings import CONFIG_DIR


def load_benchmarks(path: Path | None = None) -> list[BenchmarkTask]:
    """Load benchmark tasks from a YAML file.

    Args:
        path: Path to the YAML file. Defaults to config/arena_benchmarks.yaml.

    Returns:
        List of validated BenchmarkTask objects.
    """
    config_path = path or CONFIG_DIR / "arena_benchmarks.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw or "benchmarks" not in raw:
        raise ValueError(f"Invalid benchmark file: missing 'benchmarks' key in {config_path}")

    tasks: list[BenchmarkTask] = []
    for entry in raw["benchmarks"]:
        try:
            task = BenchmarkTask(**entry)
            tasks.append(task)
        except Exception as e:
            logger.warning("Skipping invalid benchmark '{}': {}", entry.get("id", "?"), e)

    logger.info("Loaded {} benchmark tasks from {}", len(tasks), config_path.name)
    return tasks


def get_benchmarks_by_tier(tasks: list[BenchmarkTask], tier: int) -> list[BenchmarkTask]:
    """Filter benchmarks by tier level."""
    return [t for t in tasks if t.tier == tier]


def get_benchmarks_by_category(tasks: list[BenchmarkTask], category: str) -> list[BenchmarkTask]:
    """Filter benchmarks by category."""
    return [t for t in tasks if t.category == category]


def get_benchmark_by_id(tasks: list[BenchmarkTask], task_id: str) -> BenchmarkTask | None:
    """Find a single benchmark by ID."""
    for t in tasks:
        if t.id == task_id:
            return t
    return None


def get_tier_summary(tasks: list[BenchmarkTask]) -> dict[int, int]:
    """Return a count of tasks per tier."""
    summary: dict[int, int] = {}
    for t in tasks:
        summary[t.tier] = summary.get(t.tier, 0) + 1
    return dict(sorted(summary.items()))


def get_category_summary(tasks: list[BenchmarkTask]) -> dict[str, int]:
    """Return a count of tasks per category."""
    summary: dict[str, int] = {}
    for t in tasks:
        summary[t.category] = summary.get(t.category, 0) + 1
    return dict(sorted(summary.items()))
