"""Blueprint — the fully serializable definition of an Ultron body.

A Blueprint captures *everything* needed to recreate an agent:
  - Which LLM model to use (provider, model name, temperature, etc.)
  - The system prompt that defines the agent's behaviour
  - The tools available to the agent
  - Self-written code modules the agent has created
  - Arbitrary parameters for future extensibility
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """LLM model configuration — provider, model name, and sampling params."""

    provider: str = "openrouter"
    model_name: str = "meta-llama/llama-3-70b-instruct"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    api_base: str | None = None

    @property
    def litellm_model(self) -> str:
        """Return the model string in LiteLLM format: 'provider/model'."""
        if "/" in self.model_name and self.provider.lower() in self.model_name.lower():
            return self.model_name
        return f"{self.provider}/{self.model_name}"


class ToolSpec(BaseModel):
    """Specification for a single tool available to the agent.

    Can reference either a built-in module path (e.g. 'ultron.tools.builtins.browse')
    or a self-written code file (e.g. 'data/tools/custom_tool.py').
    """

    name: str
    description: str
    module_path: str
    function_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        """Return 'module_path:function_name'."""
        return f"{self.module_path}:{self.function_name}"


class Blueprint(BaseModel):
    """A fully serializable agent configuration — the thing that evolves.

    Every commit in the Evolution Tree stores a Blueprint. Bodies are
    instantiated from Blueprints by the BodyFactory.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "ultron"
    version: str = "0.1.0"
    model: ModelConfig = Field(default_factory=ModelConfig)
    system_prompt: str = ""
    tools: list[ToolSpec] = Field(default_factory=list)
    code_modules: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    parent_id: str | None = None

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary (JSON-safe)."""
        return json.loads(self.model_dump_json())

    def to_yaml(self) -> str:
        """Serialize to a YAML string."""
        return yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    def save_yaml(self, path: Path) -> None:
        """Write the blueprint to a YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: Path) -> Blueprint:
        """Load a blueprint from a YAML file."""
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Blueprint:
        """Load a blueprint from a plain dictionary."""
        return cls(**data)

    # ── Hashing ──────────────────────────────────────────────────────────────

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of the blueprint's semantic content (excludes id and timestamps).

        Two blueprints with the same model, prompt, tools, code, and params
        will have the same content hash, even if they have different IDs.
        """
        hashable = {
            "model": self.model.model_dump(),
            "system_prompt": self.system_prompt,
            "tools": [t.model_dump() for t in self.tools],
            "code_modules": sorted(self.code_modules),
            "parameters": self.parameters,
        }
        canonical = json.dumps(hashable, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    # ── Diffing ──────────────────────────────────────────────────────────────

    def diff(self, other: Blueprint) -> dict[str, Any]:
        """Compute a structural diff between this blueprint and another.

        Returns a dict with keys for each changed field, each containing
        {'old': ..., 'new': ...}.
        """
        changes: dict[str, Any] = {}

        # Model config
        if self.model != other.model:
            self_model = self.model.model_dump()
            other_model = other.model.model_dump()
            model_changes = {}
            for key in set(self_model) | set(other_model):
                if self_model.get(key) != other_model.get(key):
                    model_changes[key] = {"old": self_model.get(key), "new": other_model.get(key)}
            if model_changes:
                changes["model"] = model_changes

        # System prompt
        if self.system_prompt != other.system_prompt:
            changes["system_prompt"] = {
                "old_length": len(self.system_prompt),
                "new_length": len(other.system_prompt),
                "old_preview": self.system_prompt[:200],
                "new_preview": other.system_prompt[:200],
            }

        # Tools
        self_tools = {t.name for t in self.tools}
        other_tools = {t.name for t in other.tools}
        if self_tools != other_tools:
            changes["tools"] = {
                "added": sorted(other_tools - self_tools),
                "removed": sorted(self_tools - other_tools),
                "unchanged": sorted(self_tools & other_tools),
            }

        # Code modules
        if set(self.code_modules) != set(other.code_modules):
            changes["code_modules"] = {
                "added": sorted(set(other.code_modules) - set(self.code_modules)),
                "removed": sorted(set(self.code_modules) - set(other.code_modules)),
            }

        # Parameters
        if self.parameters != other.parameters:
            param_changes = {}
            all_keys = set(self.parameters) | set(other.parameters)
            for key in all_keys:
                old_val = self.parameters.get(key)
                new_val = other.parameters.get(key)
                if old_val != new_val:
                    param_changes[key] = {"old": old_val, "new": new_val}
            if param_changes:
                changes["parameters"] = param_changes

        return changes

    # ── Display ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Blueprint(id={self.id!r}, name={self.name!r}, "
            f"model={self.model.litellm_model!r}, "
            f"tools={len(self.tools)}, hash={self.content_hash})"
        )
