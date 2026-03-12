"""Tests for the Blueprint schema — serialization, validation, hashing, diffing."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ultron.body.blueprint import Blueprint, ModelConfig, ToolSpec
from ultron.core.settings import CONFIG_DIR


# ── Fixtures ─────────────────────────────────────────────────────────────────


def make_blueprint(**overrides) -> Blueprint:
    """Create a test blueprint with sensible defaults."""
    defaults = {
        "name": "test-body",
        "version": "0.1.0",
        "model": ModelConfig(
            provider="openrouter",
            model_name="meta-llama/llama-3-70b-instruct",
            temperature=0.7,
        ),
        "system_prompt": "You are a helpful assistant.",
        "tools": [
            ToolSpec(
                name="test_tool",
                description="A test tool",
                module_path="tests.mock_tools",
                function_name="mock_fn",
                parameters={"type": "object", "properties": {}, "required": []},
            )
        ],
        "code_modules": [],
        "parameters": {"generation": 0},
    }
    defaults.update(overrides)
    return Blueprint(**defaults)


# ── Serialization roundtrip ──────────────────────────────────────────────────


class TestBlueprintSerialization:
    def test_to_dict_roundtrip(self):
        bp = make_blueprint()
        data = bp.to_dict()
        restored = Blueprint.from_dict(data)

        assert restored.name == bp.name
        assert restored.model.provider == bp.model.provider
        assert restored.model.model_name == bp.model.model_name
        assert restored.system_prompt == bp.system_prompt
        assert len(restored.tools) == len(bp.tools)
        assert restored.tools[0].name == bp.tools[0].name
        assert restored.parameters == bp.parameters

    def test_yaml_roundtrip(self):
        bp = make_blueprint()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write(bp.to_yaml())
            tmp_path = Path(f.name)

        try:
            restored = Blueprint.from_yaml(tmp_path)
            assert restored.name == bp.name
            assert restored.model.model_name == bp.model.model_name
            assert restored.system_prompt == bp.system_prompt
            assert len(restored.tools) == len(bp.tools)
        finally:
            tmp_path.unlink()

    def test_save_and_load_yaml(self):
        bp = make_blueprint()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_bp.yaml"
            bp.save_yaml(path)
            assert path.exists()

            restored = Blueprint.from_yaml(path)
            assert restored.name == bp.name


# ── Validation ───────────────────────────────────────────────────────────────


class TestBlueprintValidation:
    def test_default_id_generated(self):
        bp = make_blueprint()
        assert bp.id is not None
        assert len(bp.id) == 12

    def test_model_config_litellm_format(self):
        cfg = ModelConfig(provider="openrouter", model_name="meta-llama/llama-3-70b-instruct")
        assert cfg.litellm_model == "openrouter/meta-llama/llama-3-70b-instruct"

    def test_model_config_no_double_prefix(self):
        cfg = ModelConfig(provider="openrouter", model_name="openrouter/gpt-4")
        assert cfg.litellm_model == "openrouter/gpt-4"

    def test_tool_spec_qualified_name(self):
        spec = ToolSpec(
            name="browse",
            description="Browse URLs",
            module_path="ultron.tools.builtins.browse",
            function_name="browse_url",
        )
        assert spec.qualified_name == "ultron.tools.builtins.browse:browse_url"


# ── Hashing ──────────────────────────────────────────────────────────────────


class TestBlueprintHashing:
    def test_identical_content_same_hash(self):
        bp1 = make_blueprint(id="aaa")
        bp2 = make_blueprint(id="bbb")  # Different ID
        assert bp1.content_hash == bp2.content_hash

    def test_different_prompt_different_hash(self):
        bp1 = make_blueprint(system_prompt="Prompt A")
        bp2 = make_blueprint(system_prompt="Prompt B")
        assert bp1.content_hash != bp2.content_hash

    def test_different_model_different_hash(self):
        bp1 = make_blueprint()
        bp2 = make_blueprint(
            model=ModelConfig(provider="openai", model_name="gpt-4o")
        )
        assert bp1.content_hash != bp2.content_hash


# ── Diffing ──────────────────────────────────────────────────────────────────


class TestBlueprintDiff:
    def test_identical_no_diff(self):
        bp = make_blueprint()
        diff = bp.diff(bp)
        assert diff == {}

    def test_model_change_detected(self):
        bp1 = make_blueprint()
        bp2 = make_blueprint(
            model=ModelConfig(provider="openai", model_name="gpt-4o", temperature=0.5)
        )
        diff = bp1.diff(bp2)
        assert "model" in diff
        assert "provider" in diff["model"]

    def test_prompt_change_detected(self):
        bp1 = make_blueprint(system_prompt="Old prompt")
        bp2 = make_blueprint(system_prompt="New prompt that is different")
        diff = bp1.diff(bp2)
        assert "system_prompt" in diff

    def test_tool_add_detected(self):
        bp1 = make_blueprint(tools=[])
        bp2 = make_blueprint()
        diff = bp1.diff(bp2)
        assert "tools" in diff
        assert "test_tool" in diff["tools"]["added"]

    def test_param_change_detected(self):
        bp1 = make_blueprint(parameters={"a": 1})
        bp2 = make_blueprint(parameters={"a": 2, "b": 3})
        diff = bp1.diff(bp2)
        assert "parameters" in diff


# ── Genesis Blueprint ────────────────────────────────────────────────────────


class TestGenesisBlueprint:
    def test_genesis_loads_successfully(self):
        genesis_path = CONFIG_DIR / "genesis_blueprint.yaml"
        if not genesis_path.exists():
            pytest.skip("Genesis blueprint not found")

        bp = Blueprint.from_yaml(genesis_path)
        assert bp.name == "ultron-genesis"
        assert bp.model.provider == "openrouter"
        assert len(bp.tools) >= 4  # browse, shell, read, write, delete, list
