"""Tests for the Arena system — models, loader, scorers, leaderboard, skill tree."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ultron.arena.loader import (
    get_benchmarks_by_category,
    get_benchmarks_by_tier,
    get_tier_summary,
    load_benchmarks,
)
from ultron.arena.models import ArenaScore, BenchmarkResult, BenchmarkTask, ScorerResult, SkillNode


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark Loading
# ═══════════════════════════════════════════════════════════════════════════


class TestBenchmarkLoader:
    def test_load_benchmarks_from_config(self):
        tasks = load_benchmarks()
        assert len(tasks) >= 20
        assert all(isinstance(t, BenchmarkTask) for t in tasks)

    def test_all_tasks_have_required_fields(self):
        tasks = load_benchmarks()
        for task in tasks:
            assert task.id, f"Task missing id"
            assert task.name, f"Task {task.id} missing name"
            assert 1 <= task.tier <= 5, f"Task {task.id} has invalid tier {task.tier}"
            assert task.category, f"Task {task.id} missing category"
            assert task.prompt, f"Task {task.id} missing prompt"
            assert task.scoring_method in ("exact", "contains", "code_exec", "llm_judge"), (
                f"Task {task.id} has invalid scoring_method '{task.scoring_method}'"
            )

    def test_unique_task_ids(self):
        tasks = load_benchmarks()
        ids = [t.id for t in tasks]
        assert len(ids) == len(set(ids)), f"Duplicate task IDs found: {[i for i in ids if ids.count(i) > 1]}"

    def test_filter_by_tier(self):
        tasks = load_benchmarks()
        l1 = get_benchmarks_by_tier(tasks, 1)
        assert len(l1) >= 3
        assert all(t.tier == 1 for t in l1)

    def test_filter_by_category(self):
        tasks = load_benchmarks()
        coding = get_benchmarks_by_category(tasks, "coding")
        assert len(coding) >= 3
        assert all(t.category == "coding" for t in coding)

    def test_tier_summary(self):
        tasks = load_benchmarks()
        summary = get_tier_summary(tasks)
        assert 1 in summary
        assert 5 in summary
        assert all(v > 0 for v in summary.values())

    def test_tier_label(self):
        task = BenchmarkTask(id="t1", name="Test", tier=3, category="test", prompt="test")
        assert task.tier_label == "L3:ToolUse"


# ═══════════════════════════════════════════════════════════════════════════
# Scorers
# ═══════════════════════════════════════════════════════════════════════════


class TestExactMatchScorer:
    @pytest.mark.asyncio
    async def test_exact_match(self):
        from ultron.arena.scorers import ExactMatchScorer

        scorer = ExactMatchScorer()
        task = BenchmarkTask(
            id="t", name="t", tier=1, category="test", prompt="test",
            expected_output="hello world",
        )
        result = await scorer.score(task, "Hello World")
        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_no_match(self):
        from ultron.arena.scorers import ExactMatchScorer

        scorer = ExactMatchScorer()
        task = BenchmarkTask(
            id="t", name="t", tier=1, category="test", prompt="test",
            expected_output="hello",
        )
        result = await scorer.score(task, "goodbye")
        assert result.score == 0.0
        assert result.passed is False


class TestContainsScorer:
    @pytest.mark.asyncio
    async def test_simple_contains(self):
        from ultron.arena.scorers import ContainsScorer

        scorer = ContainsScorer()
        task = BenchmarkTask(
            id="t", name="t", tier=1, category="test", prompt="test",
            expected_output="42",
        )
        result = await scorer.score(task, "The answer is 42.")
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_missing_substring(self):
        from ultron.arena.scorers import ContainsScorer

        scorer = ContainsScorer()
        task = BenchmarkTask(
            id="t", name="t", tier=1, category="test", prompt="test",
            expected_output="42",
        )
        result = await scorer.score(task, "I don't know")
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_multiple_required(self):
        from ultron.arena.scorers import ContainsScorer

        scorer = ContainsScorer()
        task = BenchmarkTask(
            id="t", name="t", tier=1, category="test", prompt="test",
            expected_output="",
            scoring_config={"required_substrings": ["Alice", "Bob", "Charlie"]},
        )
        result = await scorer.score(task, "Alice and Bob went to the store")
        assert result.score == pytest.approx(2 / 3)  # Found 2 of 3


class TestCodeExecutionScorer:
    @pytest.mark.asyncio
    async def test_run_simple_code(self):
        from ultron.arena.scorers import CodeExecutionScorer

        scorer = CodeExecutionScorer()
        task = BenchmarkTask(
            id="t", name="t", tier=4, category="coding", prompt="test",
            expected_output="42", scoring_method="code_exec",
        )
        result = await scorer.score(task, '```python\nprint(42)\n```')
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_no_code_but_answer(self):
        from ultron.arena.scorers import CodeExecutionScorer

        scorer = CodeExecutionScorer()
        task = BenchmarkTask(
            id="t", name="t", tier=4, category="coding", prompt="test",
            expected_output="42", scoring_method="code_exec",
        )
        result = await scorer.score(task, "The answer is 42")
        assert result.score == 0.7  # Partial credit


class TestScorerFactory:
    def test_get_known_scorer(self):
        from ultron.arena.scorers import ContainsScorer, ExactMatchScorer, get_scorer

        assert isinstance(get_scorer("exact"), ExactMatchScorer)
        assert isinstance(get_scorer("contains"), ContainsScorer)

    def test_unknown_scorer_raises(self):
        from ultron.arena.scorers import get_scorer

        with pytest.raises(ValueError, match="Unknown"):
            get_scorer("nonexistent")


# ═══════════════════════════════════════════════════════════════════════════
# Arena Runner (Mocked)
# ═══════════════════════════════════════════════════════════════════════════


class TestArenaRunner:
    @pytest.mark.asyncio
    async def test_evaluate_with_mock(self):
        from ultron.arena.runner import ArenaRunner
        from ultron.body.agent import AgentBody, AgentResult
        from ultron.body.blueprint import Blueprint, ModelConfig
        from ultron.body.llm import LLMClient
        from ultron.tools.registry import ToolRegistry

        # Create a mock body
        blueprint = Blueprint(name="test", model=ModelConfig())
        llm = LLMClient(blueprint.model)
        body = AgentBody(blueprint, llm, ToolRegistry())

        # Use only 2 simple L1 tasks
        tasks = [
            BenchmarkTask(
                id="test_1", name="Math", tier=1, category="reasoning",
                prompt="What is 2+2?", expected_output="4", scoring_method="contains",
                skill_nodes=["reasoning.math"],
            ),
            BenchmarkTask(
                id="test_2", name="Logic", tier=1, category="reasoning",
                prompt="Is the sky blue?", expected_output="Yes", scoring_method="contains",
                skill_nodes=["reasoning.logic"],
            ),
        ]

        # Mock agent to return correct answers
        from ultron.body.llm import LLMResponse

        mock_responses = [
            # Task 1 (1 run)
            LLMResponse(content="The answer is 4.", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
            # Task 2 (1 run)
            LLMResponse(content="Yes, the sky is blue.", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
        ]

        with patch.object(llm, "chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.side_effect = mock_responses
            runner = ArenaRunner(benchmarks=tasks, runs_per_task=1)
            score = await runner.evaluate(body)

        assert score.total_score > 0
        assert score.tasks_passed == 2
        assert score.tasks_total == 2
        assert len(score.results) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Leaderboard
# ═══════════════════════════════════════════════════════════════════════════


class TestLeaderboard:
    def test_record_and_rank(self):
        from ultron.arena.leaderboard import Leaderboard

        lb = Leaderboard()
        lb.record(ArenaScore(blueprint_id="a", blueprint_name="Alpha", total_score=0.6, tasks_total=10, tasks_passed=6))
        lb.record(ArenaScore(blueprint_id="b", blueprint_name="Beta", total_score=0.8, tasks_total=10, tasks_passed=8))
        lb.record(ArenaScore(blueprint_id="c", blueprint_name="Gamma", total_score=0.3, tasks_total=10, tasks_passed=3))

        ranking = lb.get_ranking()
        assert ranking[0].blueprint_name == "Beta"
        assert ranking[-1].blueprint_name == "Gamma"

    def test_get_best(self):
        from ultron.arena.leaderboard import Leaderboard

        lb = Leaderboard()
        lb.record(ArenaScore(blueprint_id="a", total_score=0.5, tasks_total=1, tasks_passed=0))
        lb.record(ArenaScore(blueprint_id="b", total_score=0.9, tasks_total=1, tasks_passed=1))

        best = lb.get_best()
        assert best is not None
        assert best.blueprint_id == "b"

    def test_compare(self):
        from ultron.arena.leaderboard import Leaderboard

        lb = Leaderboard()
        lb.record(ArenaScore(
            blueprint_id="a", total_score=0.6,
            tier_scores={"L1": 0.8, "L2": 0.4},
            tasks_total=10, tasks_passed=6,
        ))
        lb.record(ArenaScore(
            blueprint_id="b", total_score=0.7,
            tier_scores={"L1": 0.7, "L2": 0.7},
            tasks_total=10, tasks_passed=7,
        ))

        comp = lb.compare("a", "b")
        assert comp["winner"] == "b"
        assert comp["total_delta"] == pytest.approx(0.1)

    def test_save_and_load_roundtrip(self):
        from ultron.arena.leaderboard import Leaderboard

        lb = Leaderboard()
        lb.record(ArenaScore(blueprint_id="a", blueprint_name="Test", total_score=0.75, tasks_total=5, tasks_passed=4))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lb.json"
            lb.save(path)

            loaded = Leaderboard.load(path)
            assert loaded.count == 1
            assert loaded.get_best().total_score == 0.75

    def test_history(self):
        from ultron.arena.leaderboard import Leaderboard

        lb = Leaderboard()
        lb.record(ArenaScore(blueprint_id="a", total_score=0.5, tasks_total=1, tasks_passed=0))
        lb.record(ArenaScore(blueprint_id="a", total_score=0.7, tasks_total=1, tasks_passed=1))
        lb.record(ArenaScore(blueprint_id="b", total_score=0.6, tasks_total=1, tasks_passed=0))

        history = lb.get_history("a")
        assert len(history) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Skill Tree
# ═══════════════════════════════════════════════════════════════════════════


class TestSkillNode:
    def test_add_xp_no_level_up(self):
        node = SkillNode(id="test", name="Test", category="test")
        leveled = node.add_xp(50)
        assert leveled is False
        assert node.level == 0
        assert node.xp == 50

    def test_add_xp_level_up(self):
        node = SkillNode(id="test", name="Test", category="test", xp_per_level=100)
        leveled = node.add_xp(150)
        assert leveled is True
        assert node.level == 1
        assert node.xp == 50

    def test_max_level(self):
        node = SkillNode(id="test", name="Test", category="test", max_level=2, xp_per_level=10)
        node.add_xp(1000)
        assert node.level == 2
        assert node.is_maxed is True

    def test_progress(self):
        node = SkillNode(id="test", name="Test", category="test", xp_per_level=100)
        node.add_xp(25)
        assert node.progress == 0.25


class TestSkillTree:
    def test_default_tree_has_nodes(self):
        from ultron.arena.skill_tree import SkillTree

        tree = SkillTree()
        nodes = tree.get_all_nodes()
        assert len(nodes) >= 10

    def test_award_xp(self):
        from ultron.arena.skill_tree import SkillTree

        tree = SkillTree()
        tree.award_xp("reasoning.math", 150)
        node = tree.get_node("reasoning.math")
        assert node is not None
        assert node.level >= 1

    def test_get_categories(self):
        from ultron.arena.skill_tree import SkillTree

        tree = SkillTree()
        cats = tree.get_categories()
        assert "reasoning" in cats
        assert "coding" in cats
        assert "meta" in cats

    def test_suggested_focus(self):
        from ultron.arena.skill_tree import SkillTree

        tree = SkillTree()
        # Level up reasoning but leave coding at 0
        tree.award_xp("reasoning.math", 200)
        tree.award_xp("reasoning.logic", 200)

        focus = tree.get_suggested_focus(3)
        assert len(focus) > 0
        # Should suggest lower-level nodes
        assert all(n.level < 2 for n in focus)

    def test_save_load_roundtrip(self):
        from ultron.arena.skill_tree import SkillTree

        tree = SkillTree()
        tree.award_xp("reasoning.math", 150)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tree.json"
            tree.save(path)

            loaded = SkillTree.load(path)
            node = loaded.get_node("reasoning.math")
            assert node is not None
            assert node.level >= 1

    def test_to_dict(self):
        from ultron.arena.skill_tree import SkillTree

        tree = SkillTree()
        data = tree.to_dict()
        assert "total_level" in data
        assert "categories" in data
        assert "reasoning" in data["categories"]
