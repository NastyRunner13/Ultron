"""Leaderboard — stores, ranks, and compares arena scores across bodies.

Supports:
  - Recording score entries
  - Ranking by total score
  - Side-by-side body comparison
  - Score history per blueprint
  - JSON persistence
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from ultron.arena.models import ArenaScore
from ultron.core.settings import get_settings


class LeaderboardEntry(BaseModel):
    """A single leaderboard entry — snapshot of a body's arena performance."""

    blueprint_id: str
    blueprint_name: str
    total_score: float
    tier_scores: dict[str, float]
    category_scores: dict[str, float]
    tasks_passed: int
    tasks_total: int
    total_tokens: int
    total_duration: float
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Leaderboard:
    """Persistent leaderboard that ranks body performances.

    Usage:
        leaderboard = Leaderboard.load()
        leaderboard.record(arena_score)
        ranking = leaderboard.get_ranking()
        leaderboard.save()
    """

    def __init__(self) -> None:
        self._entries: list[LeaderboardEntry] = []

    def record(self, score: ArenaScore) -> None:
        """Add a new score to the leaderboard."""
        entry = LeaderboardEntry(
            blueprint_id=score.blueprint_id,
            blueprint_name=score.blueprint_name,
            total_score=score.total_score,
            tier_scores=score.tier_scores,
            category_scores=score.category_scores,
            tasks_passed=score.tasks_passed,
            tasks_total=score.tasks_total,
            total_tokens=score.total_tokens,
            total_duration=score.total_duration,
        )
        self._entries.append(entry)
        logger.info(
            "Leaderboard: recorded {} ({:.1%})",
            score.blueprint_name or score.blueprint_id,
            score.total_score,
        )

    def get_ranking(self) -> list[LeaderboardEntry]:
        """Get all entries sorted by total score (highest first)."""
        return sorted(self._entries, key=lambda e: e.total_score, reverse=True)

    def get_best(self) -> LeaderboardEntry | None:
        """Get the top-ranked entry."""
        ranking = self.get_ranking()
        return ranking[0] if ranking else None

    def get_history(self, blueprint_id: str) -> list[LeaderboardEntry]:
        """Get all entries for a specific blueprint, sorted by time."""
        entries = [e for e in self._entries if e.blueprint_id == blueprint_id]
        return sorted(entries, key=lambda e: e.recorded_at)

    def compare(self, id_a: str, id_b: str) -> dict[str, Any]:
        """Compare the latest scores of two blueprints.

        Returns a dict with 'a', 'b', and 'diff' sections.
        """
        history_a = self.get_history(id_a)
        history_b = self.get_history(id_b)

        if not history_a or not history_b:
            return {"error": "One or both blueprints not found in leaderboard"}

        a = history_a[-1]
        b = history_b[-1]

        # Score diffs
        tier_diff = {}
        all_tiers = set(a.tier_scores) | set(b.tier_scores)
        for tier in sorted(all_tiers):
            score_a = a.tier_scores.get(tier, 0.0)
            score_b = b.tier_scores.get(tier, 0.0)
            tier_diff[tier] = {
                "a": score_a,
                "b": score_b,
                "delta": score_b - score_a,
                "winner": "b" if score_b > score_a else ("a" if score_a > score_b else "tie"),
            }

        cat_diff = {}
        all_cats = set(a.category_scores) | set(b.category_scores)
        for cat in sorted(all_cats):
            score_a = a.category_scores.get(cat, 0.0)
            score_b = b.category_scores.get(cat, 0.0)
            cat_diff[cat] = {
                "a": score_a,
                "b": score_b,
                "delta": score_b - score_a,
            }

        return {
            "a": {"id": id_a, "name": a.blueprint_name, "total": a.total_score},
            "b": {"id": id_b, "name": b.blueprint_name, "total": b.total_score},
            "total_delta": b.total_score - a.total_score,
            "winner": id_b if b.total_score > a.total_score else (
                id_a if a.total_score > b.total_score else "tie"
            ),
            "tier_diff": tier_diff,
            "category_diff": cat_diff,
            "efficiency_diff": {
                "tokens_a": a.total_tokens,
                "tokens_b": b.total_tokens,
                "duration_a": a.total_duration,
                "duration_b": b.total_duration,
            },
        }

    @property
    def count(self) -> int:
        return len(self._entries)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> None:
        """Save leaderboard to JSON."""
        settings = get_settings()
        save_path = path or (settings.resolved_data_dir / "leaderboard.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = [json.loads(e.model_dump_json()) for e in self._entries]
        save_path.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )
        logger.debug("Leaderboard saved: {} entries to {}", len(data), save_path)

    @classmethod
    def load(cls, path: Path | None = None) -> Leaderboard:
        """Load leaderboard from JSON."""
        settings = get_settings()
        load_path = path or (settings.resolved_data_dir / "leaderboard.json")

        lb = cls()
        if load_path.exists():
            data = json.loads(load_path.read_text(encoding="utf-8"))
            lb._entries = [LeaderboardEntry(**entry) for entry in data]
            logger.debug("Leaderboard loaded: {} entries from {}", len(lb._entries), load_path)
        return lb

    def to_table(self) -> list[dict[str, Any]]:
        """Export ranking as a list of dicts (for dashboard/display)."""
        ranking = self.get_ranking()
        return [
            {
                "rank": i + 1,
                "name": e.blueprint_name or e.blueprint_id[:8],
                "score": f"{e.total_score:.1%}",
                "passed": f"{e.tasks_passed}/{e.tasks_total}",
                "tokens": f"{e.total_tokens:,}",
                "duration": f"{e.total_duration:.1f}s",
                "recorded": e.recorded_at.strftime("%Y-%m-%d %H:%M"),
            }
            for i, e in enumerate(ranking)
        ]
