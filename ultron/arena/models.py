"""Arena data models — benchmark tasks, results, scores, and skill nodes.

These models define the evaluation pipeline:
  BenchmarkTask → (run agent) → BenchmarkResult → (aggregate) → ArenaScore
                                                                    ↓
                                                              SkillTree update
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class BenchmarkTask(BaseModel):
    """A single benchmark task definition.

    Tasks are deterministic — same input, same expected output, same scoring.
    This ensures reproducible scores across generations.
    """

    id: str                                     # "l1_math_001"
    name: str                                   # "Simple Addition"
    tier: int                                   # 1-5
    category: str                               # "reasoning", "coding", "tool_use", etc.
    prompt: str                                 # The task given to the agent
    expected_output: str | None = None          # For exact/contains scoring
    scoring_method: str = "contains"            # "exact", "contains", "code_exec", "llm_judge"
    scoring_config: dict[str, Any] = Field(default_factory=dict)
    tools_required: list[str] = Field(default_factory=list)
    timeout_seconds: int = 60
    max_score: float = 1.0
    skill_nodes: list[str] = Field(default_factory=list)

    @property
    def tier_label(self) -> str:
        labels = {1: "L1:Reasoning", 2: "L2:Instruction", 3: "L3:ToolUse", 4: "L4:Code", 5: "L5:Meta"}
        return labels.get(self.tier, f"L{self.tier}")


class ScorerResult(BaseModel):
    """Output from a scorer — score plus details."""

    score: float                                # 0.0 - 1.0
    passed: bool                                # score >= threshold (default 0.5)
    details: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""                         # How the score was determined


class BenchmarkResult(BaseModel):
    """Result from running a single benchmark task against a body."""

    task_id: str
    blueprint_id: str
    score: float                                # 0.0 - 1.0 (median of N runs)
    raw_response: str
    token_usage: dict[str, int] = Field(default_factory=dict)
    duration_seconds: float = 0.0
    success: bool = True
    scorer_details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    run_scores: list[float] = Field(default_factory=list)  # All N run scores


class ArenaScore(BaseModel):
    """Aggregated scores from a full arena evaluation of a body."""

    blueprint_id: str
    blueprint_name: str = ""
    total_score: float = 0.0                    # Weighted aggregate 0.0 - 1.0
    tier_scores: dict[str, float] = Field(default_factory=dict)   # {"L1": 0.9, "L2": 0.7}
    category_scores: dict[str, float] = Field(default_factory=dict)  # {"reasoning": 0.8}
    results: list[BenchmarkResult] = Field(default_factory=list)
    total_tokens: int = 0
    total_duration: float = 0.0
    tasks_passed: int = 0
    tasks_total: int = 0
    run_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def pass_rate(self) -> float:
        return self.tasks_passed / self.tasks_total if self.tasks_total > 0 else 0.0

    def to_summary(self) -> str:
        """Human-readable summary of the arena score."""
        lines = [
            f"Arena Score: {self.total_score:.1%} ({self.tasks_passed}/{self.tasks_total} passed)",
            f"Tiers: {' | '.join(f'{k}={v:.0%}' for k, v in sorted(self.tier_scores.items()))}",
            f"Categories: {' | '.join(f'{k}={v:.0%}' for k, v in sorted(self.category_scores.items()))}",
            f"Tokens: {self.total_tokens:,} | Duration: {self.total_duration:.1f}s",
        ]
        return "\n".join(lines)


class SkillNode(BaseModel):
    """A single node in the skill tree — tracks capability and progress."""

    id: str                                     # "reasoning.logic"
    name: str                                   # "Logical Reasoning"
    category: str                               # "reasoning"
    level: int = 0                              # Current level (0-5)
    xp: float = 0.0                             # XP toward next level
    xp_per_level: float = 100.0                 # XP required per level
    max_level: int = 5
    prerequisites: list[str] = Field(default_factory=list)

    @property
    def progress(self) -> float:
        """Progress toward next level (0.0 - 1.0)."""
        if self.level >= self.max_level:
            return 1.0
        return self.xp / self.xp_per_level

    @property
    def is_maxed(self) -> bool:
        return self.level >= self.max_level

    def add_xp(self, amount: float) -> bool:
        """Add XP and level up if threshold reached. Returns True if leveled up."""
        if self.is_maxed:
            return False

        self.xp += amount
        leveled_up = False

        while self.xp >= self.xp_per_level and self.level < self.max_level:
            self.xp -= self.xp_per_level
            self.level += 1
            leveled_up = True

        return leveled_up
