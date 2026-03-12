"""Agent runtime — the ReAct loop that executes tasks using an LLM + tools.

The AgentBody is instantiated from a Blueprint by the BodyFactory.
It runs a ReAct (Reason → Act → Observe) loop:
  1. Send messages to the LLM
  2. If the LLM returns tool calls, execute them
  3. Feed tool results back to the LLM
  4. Repeat until the LLM responds with text (no tool calls) or limits are hit
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from pydantic import BaseModel

from ultron.body.blueprint import Blueprint
from ultron.body.llm import LLMClient, TokenUsage
from ultron.core.settings import get_settings
from ultron.tools.registry import ToolRegistry, ToolResult


class ToolCall(BaseModel):
    """Record of a single tool invocation during a run."""

    tool_name: str
    arguments: dict[str, Any]
    result: str
    success: bool
    step: int


class ReasoningStep(BaseModel):
    """A single step in the agent's reasoning process."""

    step: int
    type: str  # "thought", "tool_call", "tool_result", "response"
    content: str
    timestamp: datetime = None  # type: ignore[assignment]

    def __init__(self, **data: Any) -> None:
        if "timestamp" not in data or data["timestamp"] is None:
            data["timestamp"] = datetime.now(timezone.utc)
        super().__init__(**data)


class AgentResult(BaseModel):
    """Structured result from an agent run."""

    task: str
    response: str
    tool_calls: list[ToolCall] = []
    token_usage: dict[str, int] = {}
    steps: list[ReasoningStep] = []
    total_steps: int = 0
    success: bool = True
    error: str | None = None
    started_at: datetime = None  # type: ignore[assignment]
    finished_at: datetime = None  # type: ignore[assignment]

    def __init__(self, **data: Any) -> None:
        now = datetime.now(timezone.utc)
        if "started_at" not in data or data["started_at"] is None:
            data["started_at"] = now
        if "finished_at" not in data or data["finished_at"] is None:
            data["finished_at"] = now
        super().__init__(**data)

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


class AgentBody:
    """A living agent instantiated from a Blueprint.

    The AgentBody can:
      - Run tasks using the ReAct loop
      - Track token usage and reasoning steps
      - Be serialized back to its Blueprint

    Usage:
        body = await BodyFactory().create(blueprint)
        result = await body.run("What is 2 + 2?")
    """

    def __init__(
        self,
        blueprint: Blueprint,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
    ) -> None:
        self.blueprint = blueprint
        self.llm = llm_client
        self.tools = tool_registry
        self._run_count = 0

    async def run(self, task: str) -> AgentResult:
        """Execute a task using the ReAct loop.

        The agent will reason about the task, use tools as needed,
        and return a final response. The loop continues until:
          - The LLM responds with text (no tool calls)
          - The maximum number of steps is reached
          - An unrecoverable error occurs

        Args:
            task: The task description / user prompt.

        Returns:
            AgentResult with the response, tool calls, and metadata.
        """
        settings = get_settings()
        max_steps = settings.max_agent_steps
        self._run_count += 1
        started_at = datetime.now(timezone.utc)

        logger.info(
            "Agent run #{} — task: {}",
            self._run_count,
            task[:100] + "..." if len(task) > 100 else task,
        )

        # Build initial messages
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.blueprint.system_prompt},
            {"role": "user", "content": task},
        ]

        steps: list[ReasoningStep] = []
        tool_call_records: list[ToolCall] = []
        openai_tools = self.tools.to_openai_tools() if self.tools.count > 0 else None

        step_num = 0
        response_text = ""

        try:
            while step_num < max_steps:
                step_num += 1

                # Call LLM
                llm_response = await self.llm.chat(
                    messages=messages,
                    tools=openai_tools,
                )

                # Case 1: LLM responded with text (final answer)
                if not llm_response.has_tool_calls:
                    response_text = llm_response.content or ""
                    steps.append(ReasoningStep(
                        step=step_num,
                        type="response",
                        content=response_text,
                    ))
                    logger.info("Agent responded at step {}", step_num)
                    break

                # Case 2: LLM wants to call tools
                # Add assistant message with tool calls
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": llm_response.content or None,
                    "tool_calls": llm_response.tool_calls,
                }
                messages.append(assistant_msg)

                if llm_response.content:
                    steps.append(ReasoningStep(
                        step=step_num,
                        type="thought",
                        content=llm_response.content,
                    ))

                # Execute each tool call
                for tc in llm_response.tool_calls:
                    func_info = tc.get("function", {})
                    tool_name = func_info.get("name", "unknown")
                    tool_args_str = func_info.get("arguments", "{}")
                    tool_call_id = tc.get("id", "")

                    # Parse arguments
                    try:
                        tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        tool_args = {}

                    steps.append(ReasoningStep(
                        step=step_num,
                        type="tool_call",
                        content=f"{tool_name}({json.dumps(tool_args)})",
                    ))

                    # Execute tool
                    result: ToolResult = await self.tools.execute(tool_name, **tool_args)

                    tool_call_records.append(ToolCall(
                        tool_name=tool_name,
                        arguments=tool_args,
                        result=result.output[:500] if result.output else "",
                        success=result.success,
                        step=step_num,
                    ))

                    steps.append(ReasoningStep(
                        step=step_num,
                        type="tool_result",
                        content=result.to_message()[:1000],
                    ))

                    # Feed result back to LLM
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result.to_message(),
                    })

                    logger.debug(
                        "Step {} — {} → {} ({} chars)",
                        step_num,
                        tool_name,
                        "OK" if result.success else "FAIL",
                        len(result.output),
                    )

            else:
                # Max steps reached
                response_text = (
                    f"[Agent reached maximum steps ({max_steps}). "
                    f"Last response: {llm_response.content or 'None'}]"
                )
                logger.warning("Agent hit max steps ({})", max_steps)

            return AgentResult(
                task=task,
                response=response_text,
                tool_calls=tool_call_records,
                token_usage={
                    "prompt_tokens": self.llm.session_usage.prompt_tokens,
                    "completion_tokens": self.llm.session_usage.completion_tokens,
                    "total_tokens": self.llm.session_usage.total_tokens,
                },
                steps=steps,
                total_steps=step_num,
                success=True,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error("Agent run failed: {}", e)
            return AgentResult(
                task=task,
                response="",
                tool_calls=tool_call_records,
                token_usage={
                    "prompt_tokens": self.llm.session_usage.prompt_tokens,
                    "completion_tokens": self.llm.session_usage.completion_tokens,
                    "total_tokens": self.llm.session_usage.total_tokens,
                },
                steps=steps,
                total_steps=step_num,
                success=False,
                error=str(e),
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )

    def __repr__(self) -> str:
        return (
            f"AgentBody(blueprint={self.blueprint.name!r}, "
            f"model={self.llm.model_string!r}, "
            f"tools={self.tools.count}, "
            f"runs={self._run_count})"
        )
