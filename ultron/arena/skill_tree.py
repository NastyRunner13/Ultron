"""Skill tree — maps benchmark performance to a structured capability graph.

The skill tree gives Ultron a visual map of its strengths and weaknesses,
enabling strategic evolution decisions (e.g., "focus on coding.debugging
because it's the lowest skill blocking L4 progress").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from ultron.arena.models import ArenaScore, SkillNode
from ultron.core.settings import get_settings


# ── Default Skill Tree Definition ────────────────────────────────────────────

DEFAULT_SKILL_NODES: list[dict[str, Any]] = [
    # Reasoning
    {"id": "reasoning.math", "name": "Mathematical Reasoning", "category": "reasoning"},
    {"id": "reasoning.logic", "name": "Logical Deduction", "category": "reasoning"},
    {"id": "reasoning.pattern", "name": "Pattern Recognition", "category": "reasoning"},
    # Instruction Following
    {"id": "instruction.format", "name": "Output Formatting", "category": "instruction_following"},
    {"id": "instruction.multistep", "name": "Multi-Step Instructions", "category": "instruction_following"},
    # Tool Use
    {"id": "tool_use.browse", "name": "Web Browsing", "category": "tool_use"},
    {"id": "tool_use.shell", "name": "Shell Commands", "category": "tool_use"},
    {"id": "tool_use.filesystem", "name": "File Operations", "category": "tool_use"},
    # Coding
    {"id": "coding.generation", "name": "Code Generation", "category": "coding",
     "prerequisites": ["reasoning.logic"]},
    {"id": "coding.debugging", "name": "Debugging", "category": "coding",
     "prerequisites": ["coding.generation"]},
    {"id": "coding.testing", "name": "Test Writing", "category": "coding",
     "prerequisites": ["coding.generation"]},
    # Meta
    {"id": "meta.prompt_writing", "name": "Prompt Engineering", "category": "meta",
     "prerequisites": ["instruction.format"]},
    {"id": "meta.self_improvement", "name": "Self-Improvement", "category": "meta",
     "prerequisites": ["meta.prompt_writing"]},
]


class SkillTree:
    """Tracks and visualizes agent capabilities as a tree of skill nodes.

    Each benchmark task maps to one or more skill nodes. When the agent
    scores on a task, the corresponding nodes gain XP and may level up.

    Usage:
        tree = SkillTree()
        tree.update_from_score(arena_score)
        weak = tree.get_suggested_focus()
    """

    def __init__(self, nodes: list[SkillNode] | None = None) -> None:
        if nodes:
            self._nodes = {n.id: n for n in nodes}
        else:
            self._nodes = {
                d["id"]: SkillNode(**d) for d in DEFAULT_SKILL_NODES
            }

    def get_node(self, node_id: str) -> SkillNode | None:
        """Get a skill node by ID."""
        return self._nodes.get(node_id)

    def award_xp(self, node_id: str, xp: float) -> bool:
        """Award XP to a node. Returns True if it leveled up."""
        node = self._nodes.get(node_id)
        if not node:
            logger.warning("Unknown skill node: {}", node_id)
            return False

        leveled_up = node.add_xp(xp)
        if leveled_up:
            logger.info("⬆ Skill '{}' leveled up to {}!", node.name, node.level)
        return leveled_up

    def update_from_score(self, score: ArenaScore) -> dict[str, float]:
        """Map arena results to skill XP gains.

        For each benchmark result, the corresponding skill nodes receive
        XP proportional to the score. A perfect score (1.0) awards 50 XP.

        Returns a dict of {node_id: xp_awarded}.
        """
        from ultron.arena.loader import load_benchmarks

        benchmarks = load_benchmarks()
        task_map = {t.id: t for t in benchmarks}
        xp_gains: dict[str, float] = {}

        for result in score.results:
            task = task_map.get(result.task_id)
            if not task or not task.skill_nodes:
                continue

            # XP = score * 50 (split across mapped nodes)
            xp_per_node = (result.score * 50.0) / len(task.skill_nodes)

            for node_id in task.skill_nodes:
                self.award_xp(node_id, xp_per_node)
                xp_gains[node_id] = xp_gains.get(node_id, 0) + xp_per_node

        return xp_gains

    def get_all_nodes(self) -> list[SkillNode]:
        """Get all nodes sorted by category then ID."""
        return sorted(self._nodes.values(), key=lambda n: (n.category, n.id))

    def get_unlocked(self) -> list[SkillNode]:
        """Get nodes with level >= 1."""
        return [n for n in self._nodes.values() if n.level >= 1]

    def get_by_category(self, category: str) -> list[SkillNode]:
        """Get all nodes in a category."""
        return [n for n in self._nodes.values() if n.category == category]

    def get_categories(self) -> list[str]:
        """Get unique category names."""
        return sorted(set(n.category for n in self._nodes.values()))

    def get_suggested_focus(self, top_n: int = 3) -> list[SkillNode]:
        """Suggest the weakest skills to focus on.

        Returns the N lowest-level, non-maxed nodes whose prerequisites
        are met (i.e., the agent can actually improve them now).
        """
        candidates: list[SkillNode] = []
        for node in self._nodes.values():
            if node.is_maxed:
                continue
            # Check prerequisites
            prereqs_met = all(
                self._nodes.get(p, SkillNode(id=p, name="", category="")).level >= 1
                for p in node.prerequisites
            )
            if prereqs_met:
                candidates.append(node)

        # Sort by level (ascending), then progress (ascending)
        candidates.sort(key=lambda n: (n.level, n.progress))
        return candidates[:top_n]

    def get_total_level(self) -> int:
        """Sum of all node levels."""
        return sum(n.level for n in self._nodes.values())

    def get_max_possible_level(self) -> int:
        """Maximum possible total level."""
        return sum(n.max_level for n in self._nodes.values())

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Export the full tree as a dict for dashboard/serialization."""
        categories: dict[str, list[dict[str, Any]]] = {}
        for node in self.get_all_nodes():
            cat = categories.setdefault(node.category, [])
            cat.append({
                "id": node.id,
                "name": node.name,
                "level": node.level,
                "max_level": node.max_level,
                "xp": round(node.xp, 1),
                "xp_per_level": node.xp_per_level,
                "progress": round(node.progress, 2),
                "prerequisites": node.prerequisites,
            })

        return {
            "total_level": self.get_total_level(),
            "max_level": self.get_max_possible_level(),
            "categories": categories,
        }

    def save(self, path: Path | None = None) -> None:
        """Save skill tree state to JSON."""
        settings = get_settings()
        save_path = path or (settings.resolved_data_dir / "skill_tree.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        nodes_data = [json.loads(n.model_dump_json()) for n in self._nodes.values()]
        save_path.write_text(
            json.dumps(nodes_data, indent=2),
            encoding="utf-8",
        )
        logger.debug("Skill tree saved to {}", save_path)

    @classmethod
    def load(cls, path: Path | None = None) -> SkillTree:
        """Load skill tree state from JSON."""
        settings = get_settings()
        load_path = path or (settings.resolved_data_dir / "skill_tree.json")

        if load_path.exists():
            data = json.loads(load_path.read_text(encoding="utf-8"))
            nodes = [SkillNode(**entry) for entry in data]
            logger.debug("Skill tree loaded from {}", load_path)
            return cls(nodes=nodes)

        return cls()  # Default tree

    def __repr__(self) -> str:
        return (
            f"SkillTree(nodes={len(self._nodes)}, "
            f"total_level={self.get_total_level()}/{self.get_max_possible_level()})"
        )
