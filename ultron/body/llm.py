"""LLM client — async wrapper around LiteLLM for multi-provider chat.

Provides a clean async interface for chat completions with:
  - Automatic provider routing via LiteLLM
  - Token usage tracking
  - Tool/function calling support
  - Retry logic with exponential backoff
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import litellm
from loguru import logger

from ultron.body.blueprint import ModelConfig

# Suppress LiteLLM's verbose logging
litellm.suppress_debug_info = True


@dataclass
class TokenUsage:
    """Tracks token consumption across a session."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: dict[str, int] | None) -> None:
        """Accumulate usage from a response."""
        if not usage:
            return
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.total_tokens += usage.get("total_tokens", 0)


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""

    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMClient:
    """Async LLM client powered by LiteLLM.

    Supports any provider that LiteLLM supports (OpenAI, Anthropic,
    OpenRouter, Ollama, Groq, etc.) with a unified interface.

    Usage:
        client = LLMClient(model_config)
        response = await client.chat(messages=[{"role": "user", "content": "Hello"}])
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.session_usage = TokenUsage()
        self._call_count = 0

    @property
    def model_string(self) -> str:
        """The LiteLLM model identifier."""
        return self.config.litellm_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            messages: Conversation messages in OpenAI format.
            tools: Tool definitions in OpenAI format (optional).
            temperature: Override the config temperature.
            max_tokens: Override the config max_tokens.
            max_retries: Number of retries on transient failures.

        Returns:
            LLMResponse with content and/or tool calls.
        """
        call_kwargs: dict[str, Any] = {
            "model": self.model_string,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            **kwargs,
        }

        if tools:
            call_kwargs["tools"] = tools
            call_kwargs["tool_choice"] = "auto"

        if self.config.api_base:
            call_kwargs["api_base"] = self.config.api_base

        # Retry loop with exponential backoff
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                self._call_count += 1
                logger.debug(
                    "LLM call #{} to {} ({} messages)",
                    self._call_count,
                    self.model_string,
                    len(messages),
                )

                response = await litellm.acompletion(**call_kwargs)
                return self._parse_response(response)

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "LLM call failed (attempt {}/{}): {}. Retrying in {}s...",
                        attempt + 1,
                        max_retries,
                        e,
                        wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error("LLM call failed after {} attempts: {}", max_retries, e)

        raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse a LiteLLM response into our structured format."""
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice else None

        # Extract tool calls
        tool_calls: list[dict[str, Any]] = []
        if message and hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        # Track usage
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
            self.session_usage.add(usage)

        return LLMResponse(
            content=message.content if message else None,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.finish_reason if choice else None,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    def __repr__(self) -> str:
        return (
            f"LLMClient(model={self.model_string!r}, "
            f"calls={self._call_count}, "
            f"tokens={self.session_usage.total_tokens})"
        )
