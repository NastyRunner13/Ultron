"""Tests for the Tool Registry and built-in tools."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ultron.body.blueprint import ToolSpec
from ultron.tools.registry import ToolRegistry, ToolResult


# ── Registry Tests ───────────────────────────────────────────────────────────


class TestToolRegistry:
    def _make_spec(self, name: str = "test_tool") -> ToolSpec:
        """Create a tool spec pointing to a real built-in module."""
        return ToolSpec(
            name=name,
            description=f"Test tool {name}",
            module_path="ultron.tools.builtins.filesystem",
            function_name="read_file",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def test_register_and_get(self):
        registry = ToolRegistry()
        spec = self._make_spec()
        registry.register(spec)

        retrieved = registry.get("test_tool")
        assert retrieved.name == "test_tool"
        assert retrieved.module_path == "ultron.tools.builtins.filesystem"

    def test_register_duplicate_raises(self):
        registry = ToolRegistry()
        spec = self._make_spec()
        registry.register(spec)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(spec)

    def test_get_unknown_raises(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent")

    def test_has_tool(self):
        registry = ToolRegistry()
        spec = self._make_spec()
        registry.register(spec)

        assert registry.has("test_tool") is True
        assert registry.has("nonexistent") is False

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(self._make_spec("alpha"))
        registry.register(self._make_spec("beta"))

        tools = registry.list_tools()
        assert tools == ["alpha", "beta"]  # sorted

    def test_count(self):
        registry = ToolRegistry()
        assert registry.count == 0
        registry.register(self._make_spec())
        assert registry.count == 1

    def test_register_bad_module_raises(self):
        spec = ToolSpec(
            name="bad",
            description="Bad tool",
            module_path="nonexistent.module.path",
            function_name="fn",
        )
        registry = ToolRegistry()
        with pytest.raises(ImportError):
            registry.register(spec)

    def test_register_bad_function_raises(self):
        spec = ToolSpec(
            name="bad_fn",
            description="Bad function",
            module_path="ultron.tools.builtins.filesystem",
            function_name="nonexistent_function",
        )
        registry = ToolRegistry()
        with pytest.raises(ImportError, match="not found"):
            registry.register(spec)


# ── OpenAI Format ────────────────────────────────────────────────────────────


class TestOpenAIFormat:
    def test_to_openai_tools_format(self):
        registry = ToolRegistry()
        spec = ToolSpec(
            name="browse_url",
            description="Fetch a URL",
            module_path="ultron.tools.builtins.browse",
            function_name="browse_url",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL"},
                },
                "required": ["url"],
            },
        )
        registry.register(spec)

        tools = registry.to_openai_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "browse_url"
        assert "parameters" in tools[0]["function"]
        assert tools[0]["function"]["parameters"]["properties"]["url"]["type"] == "string"


# ── Tool Execution ───────────────────────────────────────────────────────────


class TestToolExecution:
    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = await registry.execute("nonexistent", x=1)
        assert result.success is False
        assert "not registered" in result.error

    @pytest.mark.asyncio
    async def test_execute_filesystem_read_write(self):
        registry = ToolRegistry()

        # Register write and read tools
        write_spec = ToolSpec(
            name="write_file",
            description="Write a file",
            module_path="ultron.tools.builtins.filesystem",
            function_name="write_file",
        )
        read_spec = ToolSpec(
            name="read_file",
            description="Read a file",
            module_path="ultron.tools.builtins.filesystem",
            function_name="read_file",
        )
        registry.register(write_spec)
        registry.register(read_spec)

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = str(Path(tmpdir) / "test.txt")

            # Write
            result = await registry.execute("write_file", path=test_file, content="hello world")
            assert result.success is True

            # Read
            result = await registry.execute("read_file", path=test_file)
            assert result.success is True
            assert "hello world" in result.output

    @pytest.mark.asyncio
    async def test_execute_list_directory(self):
        registry = ToolRegistry()
        spec = ToolSpec(
            name="list_directory",
            description="List directory",
            module_path="ultron.tools.builtins.filesystem",
            function_name="list_directory",
        )
        registry.register(spec)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            (Path(tmpdir) / "test.txt").write_text("data")

            result = await registry.execute("list_directory", path=tmpdir)
            assert result.success is True
            assert "test.txt" in result.output


# ── Shell Tool ───────────────────────────────────────────────────────────────


class TestShellTool:
    @pytest.mark.asyncio
    async def test_echo_command(self):
        registry = ToolRegistry()
        spec = ToolSpec(
            name="execute_shell",
            description="Run shell",
            module_path="ultron.tools.builtins.shell",
            function_name="execute_shell",
        )
        registry.register(spec)

        result = await registry.execute("execute_shell", command="echo hello")
        assert result.success is True
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_blocked_command(self):
        from ultron.tools.builtins.shell import execute_shell

        with pytest.raises(PermissionError, match="blocked"):
            await execute_shell("rm -rf /")


# ── Browse Tool ──────────────────────────────────────────────────────────────


class TestBrowseTool:
    @pytest.mark.asyncio
    async def test_invalid_url_raises(self):
        from ultron.tools.builtins.browse import browse_url

        with pytest.raises(ValueError, match="Invalid URL"):
            await browse_url("not-a-url")
