"""Tool registry — discover, register, and execute tools by name.

The registry is the bridge between a Blueprint's ToolSpec declarations
and the actual callable functions. It handles:
  - Dynamic import of tool modules (built-in or self-written)
  - Validation of tool function signatures
  - Async execution with structured results
  - Conversion to OpenAI function-calling format for LLM integration
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from ultron.body.blueprint import ToolSpec


@dataclass
class ToolResult:
    """Result from executing a tool."""

    name: str
    success: bool
    output: str
    error: str | None = None

    def to_message(self) -> str:
        """Format as a string for insertion into LLM conversation."""
        if self.success:
            return self.output
        return f"Error executing {self.name}: {self.error}"


class ToolRegistry:
    """Registry of available tools — maps names to specs and callables.

    Usage:
        registry = ToolRegistry()
        registry.register(tool_spec)
        result = await registry.execute("browse_url", url="https://example.com")
    """

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._callables: dict[str, Any] = {}

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, spec: ToolSpec) -> None:
        """Register a tool from its specification.

        Dynamically imports the module and resolves the function.
        Raises ValueError if the tool name is already registered.
        Raises ImportError if the module or function cannot be found.
        """
        if spec.name in self._specs:
            raise ValueError(f"Tool '{spec.name}' is already registered")

        # Dynamic import
        try:
            module = importlib.import_module(spec.module_path)
        except ImportError as e:
            raise ImportError(
                f"Cannot import tool module '{spec.module_path}': {e}"
            ) from e

        func = getattr(module, spec.function_name, None)
        if func is None:
            raise ImportError(
                f"Function '{spec.function_name}' not found in module '{spec.module_path}'"
            )

        if not callable(func):
            raise TypeError(
                f"'{spec.module_path}:{spec.function_name}' is not callable"
            )

        self._specs[spec.name] = spec
        self._callables[spec.name] = func
        logger.debug("Registered tool: {}", spec.name)

    def register_many(self, specs: list[ToolSpec]) -> None:
        """Register multiple tools at once."""
        for spec in specs:
            self.register(spec)

    # ── Lookup ───────────────────────────────────────────────────────────────

    def get(self, name: str) -> ToolSpec:
        """Get a tool spec by name. Raises KeyError if not found."""
        if name not in self._specs:
            raise KeyError(f"Tool '{name}' is not registered. Available: {self.list_tools()}")
        return self._specs[name]

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._specs

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return sorted(self._specs.keys())

    @property
    def count(self) -> int:
        return len(self._specs)

    # ── Execution ────────────────────────────────────────────────────────────

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name with the given arguments.

        Handles both sync and async tool functions transparently.
        Always returns a ToolResult — never raises on tool errors.
        """
        if name not in self._callables:
            return ToolResult(
                name=name,
                success=False,
                output="",
                error=f"Tool '{name}' is not registered",
            )

        func = self._callables[name]

        try:
            logger.info("Executing tool: {} with args: {}", name, kwargs)

            if inspect.iscoroutinefunction(func):
                result = await func(**kwargs)
            else:
                result = func(**kwargs)

            output = str(result) if result is not None else "OK"
            logger.debug("Tool {} completed successfully ({} chars)", name, len(output))

            return ToolResult(name=name, success=True, output=output)

        except Exception as e:
            logger.warning("Tool {} failed: {}", name, e)
            return ToolResult(name=name, success=False, output="", error=str(e))

    # ── OpenAI Format ────────────────────────────────────────────────────────

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Convert all registered tools to OpenAI function-calling format.

        Returns a list of tool definitions compatible with the OpenAI API
        and LiteLLM's tool-calling interface.
        """
        tools = []
        for spec in self._specs.values():
            tool_def = {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters or {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            }
            tools.append(tool_def)
        return tools

    # ── Display ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.list_tools()})"
