"""Arena runner — evaluates an AgentBody against the benchmark suite.

Runs all benchmarks with:
  - Median-of-N scoring for reliability
  - Per-task timeout
  - Async execution
  - Aggregation into tier and category scores
  - Token and cost tracking
"""

from __future__ import annotations

import asyncio
import statistics
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from ultron.arena.loader import load_benchmarks
from ultron.arena.models import ArenaScore, BenchmarkResult, BenchmarkTask
from ultron.arena.scorers import BaseScorer, LLMJudgeScorer, get_scorer
from ultron.body.agent import AgentBody
from ultron.core.settings import get_settings


class ArenaRunner:
    """Runs a body against the full benchmark suite and produces an ArenaScore.

    Usage:
        runner = ArenaRunner()
        score = await runner.evaluate(body)
        print(score.to_summary())
    """

    def __init__(
        self,
        benchmarks: list[BenchmarkTask] | None = None,
        runs_per_task: int | None = None,
        llm_client: Any = None,
    ) -> None:
        self._benchmarks = benchmarks
        self._runs_per_task = runs_per_task
        self._llm_client = llm_client  # For LLM judge scorer

    async def evaluate(
        self,
        body: AgentBody,
        tasks: list[BenchmarkTask] | None = None,
    ) -> ArenaScore:
        """Run all benchmarks against a body, return aggregated scores.

        Args:
            body: The AgentBody to evaluate.
            tasks: Optional subset of tasks. Defaults to all benchmarks.

        Returns:
            ArenaScore with tier/category breakdowns.
        """
        settings = get_settings()
        benchmarks = tasks or self._benchmarks or load_benchmarks()
        runs_per_task = self._runs_per_task or settings.arena_runs_per_benchmark

        logger.info(
            "Arena evaluation starting: {} tasks × {} runs = {} total runs",
            len(benchmarks), runs_per_task, len(benchmarks) * runs_per_task,
        )

        results: list[BenchmarkResult] = []
        total_tokens = 0
        total_duration = 0.0

        for i, task in enumerate(benchmarks):
            logger.info(
                "[{}/{}] {} ({})",
                i + 1, len(benchmarks), task.name, task.tier_label,
            )

            result = await self._evaluate_task(body, task, runs_per_task)
            results.append(result)

            total_tokens += sum(result.token_usage.values())
            total_duration += result.duration_seconds

        # Aggregate scores
        arena_score = self._aggregate_scores(
            blueprint_id=body.blueprint.id,
            blueprint_name=body.blueprint.name,
            results=results,
            benchmarks=benchmarks,
            total_tokens=total_tokens,
            total_duration=total_duration,
        )

        logger.info(
            "Arena complete: {:.1%} total ({}/{} passed)",
            arena_score.total_score,
            arena_score.tasks_passed,
            arena_score.tasks_total,
        )

        return arena_score

    async def _evaluate_task(
        self,
        body: AgentBody,
        task: BenchmarkTask,
        n_runs: int,
    ) -> BenchmarkResult:
        """Run a single task N times and take median score."""
        settings = get_settings()
        timeout = task.timeout_seconds or settings.arena_timeout_per_task

        # Get appropriate scorer
        scorer_kwargs: dict[str, Any] = {}
        if task.scoring_method == "llm_judge" and self._llm_client:
            scorer_kwargs["llm_client"] = self._llm_client
        scorer: BaseScorer = get_scorer(task.scoring_method, **scorer_kwargs)

        run_scores: list[float] = []
        last_response = ""
        last_token_usage: dict[str, int] = {}
        total_duration = 0.0
        last_scorer_details: dict[str, Any] = {}

        for run_idx in range(n_runs):
            try:
                # Run agent with timeout
                agent_result = await asyncio.wait_for(
                    body.run(task.prompt),
                    timeout=timeout,
                )

                if agent_result.success:
                    # Score the response
                    scorer_result = await scorer.score(task, agent_result.response)
                    run_scores.append(scorer_result.score)
                    last_response = agent_result.response
                    last_token_usage = agent_result.token_usage
                    total_duration += agent_result.duration_seconds
                    last_scorer_details = scorer_result.details
                else:
                    run_scores.append(0.0)
                    last_response = agent_result.error or ""

            except asyncio.TimeoutError:
                logger.warning("Task '{}' timed out (run {}/{})", task.id, run_idx + 1, n_runs)
                run_scores.append(0.0)

            except Exception as e:
                logger.error("Task '{}' failed (run {}/{}): {}", task.id, run_idx + 1, n_runs, e)
                run_scores.append(0.0)

        # Take median score
        median_score = statistics.median(run_scores) if run_scores else 0.0

        logger.debug(
            "  {} → {:.0%} (runs: {})",
            task.id,
            median_score,
            [f"{s:.0%}" for s in run_scores],
        )

        return BenchmarkResult(
            task_id=task.id,
            blueprint_id=body.blueprint.id,
            score=median_score,
            raw_response=last_response[:2000],
            token_usage=last_token_usage,
            duration_seconds=total_duration / max(len(run_scores), 1),
            success=median_score > 0,
            scorer_details=last_scorer_details,
            run_scores=run_scores,
        )

    @staticmethod
    def _aggregate_scores(
        blueprint_id: str,
        blueprint_name: str,
        results: list[BenchmarkResult],
        benchmarks: list[BenchmarkTask],
        total_tokens: int,
        total_duration: float,
    ) -> ArenaScore:
        """Aggregate individual results into tier and category scores."""
        if not results:
            return ArenaScore(
                blueprint_id=blueprint_id,
                blueprint_name=blueprint_name,
            )

        # Build task map from provided benchmarks
        task_map = {t.id: t for t in benchmarks}

        # Group by tier
        tier_scores: dict[str, list[float]] = {}
        category_scores: dict[str, list[float]] = {}
        tasks_passed = 0

        for result in results:
            task = task_map.get(result.task_id)
            if not task:
                continue

            tier_key = f"L{task.tier}"
            tier_scores.setdefault(tier_key, []).append(result.score)
            category_scores.setdefault(task.category, []).append(result.score)

            if result.score >= 0.5:
                tasks_passed += 1

        # Average per group
        tier_avgs = {k: statistics.mean(v) for k, v in tier_scores.items()}
        category_avgs = {k: statistics.mean(v) for k, v in category_scores.items()}

        # Weighted total: higher tiers worth more
        tier_weights = {"L1": 1.0, "L2": 1.5, "L3": 2.0, "L4": 2.5, "L5": 3.0}
        weighted_sum = 0.0
        weight_total = 0.0
        for tier_key, avg in tier_avgs.items():
            w = tier_weights.get(tier_key, 1.0)
            weighted_sum += avg * w
            weight_total += w

        total_score = weighted_sum / weight_total if weight_total > 0 else 0.0

        return ArenaScore(
            blueprint_id=blueprint_id,
            blueprint_name=blueprint_name,
            total_score=total_score,
            tier_scores=tier_avgs,
            category_scores=category_avgs,
            results=results,
            total_tokens=total_tokens,
            total_duration=total_duration,
            tasks_passed=tasks_passed,
            tasks_total=len(results),
        )
