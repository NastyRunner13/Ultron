"""Tests for the LLM client and Agent runtime."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ultron.body.agent import AgentBody, AgentResult
from ultron.body.blueprint import Blueprint, ModelConfig, ToolSpec
from ultron.body.llm import LLMClient, LLMResponse, TokenUsage
from ultron.tools.registry import ToolRegistry


# ── LLM Client Tests ────────────────────────────────────────────────────────


class TestLLMClient:
    def test_model_string(self):
        config = ModelConfig(provider="openrouter", model_name="meta-llama/llama-3-70b-instruct")
        client = LLMClient(config)
        assert client.model_string == "openrouter/meta-llama/llama-3-70b-instruct"

    def test_token_usage_tracking(self):
        usage = TokenUsage()
        usage.add({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        usage.add({"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300})

        assert usage.prompt_tokens == 300
        assert usage.completion_tokens == 150
        assert usage.total_tokens == 450

    def test_token_usage_handles_none(self):
        usage = TokenUsage()
        usage.add(None)
        assert usage.total_tokens == 0

    def test_llm_response_has_tool_calls(self):
        response = LLMResponse(
            content=None,
            tool_calls=[{"id": "1", "function": {"name": "test", "arguments": "{}"}}],
        )
        assert response.has_tool_calls is True

    def test_llm_response_no_tool_calls(self):
        response = LLMResponse(content="Hello")
        assert response.has_tool_calls is False


# ── Agent Runtime Tests ──────────────────────────────────────────────────────


class TestAgentBody:
    def _make_blueprint(self) -> Blueprint:
        return Blueprint(
            name="test-body",
            model=ModelConfig(provider="openrouter", model_name="test/model"),
            system_prompt="You are a test agent.",
            tools=[],
        )

    @pytest.mark.asyncio
    async def test_simple_response(self):
        """Agent should return LLM response when no tool calls are made."""
        blueprint = self._make_blueprint()
        llm_client = LLMClient(blueprint.model)
        tool_registry = ToolRegistry()
        body = AgentBody(blueprint, llm_client, tool_registry)

        # Mock the LLM to return a simple text response
        mock_response = LLMResponse(
            content="The answer is 42.",
            tool_calls=[],
            usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            finish_reason="stop",
        )

        with patch.object(llm_client, "chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_response
            result = await body.run("What is the meaning of life?")

        assert result.success is True
        assert result.response == "The answer is 42."
        assert result.total_steps == 1
        assert len(result.tool_calls) == 0

    @pytest.mark.asyncio
    async def test_tool_calling_flow(self):
        """Agent should execute tool calls and feed results back to LLM."""
        blueprint = self._make_blueprint()
        llm_client = LLMClient(blueprint.model)

        # Create a registry with a mock tool
        tool_registry = ToolRegistry()

        # Register a simple sync tool
        spec = ToolSpec(
            name="list_directory",
            description="List directory",
            module_path="ultron.tools.builtins.filesystem",
            function_name="list_directory",
        )
        tool_registry.register(spec)

        body = AgentBody(blueprint, llm_client, tool_registry)

        # First LLM call returns a tool call
        tool_call_response = LLMResponse(
            content="Let me list the directory.",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "arguments": json.dumps({"path": "."}),
                },
            }],
            usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        )

        # Second LLM call returns text response
        text_response = LLMResponse(
            content="The directory contains several files.",
            tool_calls=[],
            usage={"prompt_tokens": 100, "completion_tokens": 15, "total_tokens": 115},
        )

        with patch.object(
            llm_client, "chat", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.side_effect = [tool_call_response, text_response]
            result = await body.run("List the current directory")

        assert result.success is True
        assert result.response == "The directory contains several files."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "list_directory"
        assert result.total_steps == 2

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Agent should handle LLM errors gracefully."""
        blueprint = self._make_blueprint()
        llm_client = LLMClient(blueprint.model)
        tool_registry = ToolRegistry()
        body = AgentBody(blueprint, llm_client, tool_registry)

        with patch.object(
            llm_client, "chat", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.side_effect = RuntimeError("LLM connection failed")
            result = await body.run("test")

        assert result.success is False
        assert "LLM connection failed" in result.error


class TestAgentResult:
    def test_duration_calculation(self):
        from datetime import datetime, timedelta, timezone

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = start + timedelta(seconds=5)

        result = AgentResult(
            task="test",
            response="done",
            started_at=start,
            finished_at=end,
        )
        assert result.duration_seconds == 5.0
