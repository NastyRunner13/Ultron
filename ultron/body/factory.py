"""Body Factory — instantiates a working AgentBody from a Blueprint.

The factory handles:
  - Configuring the LLM client from the Blueprint's ModelConfig
  - Loading and registering all tools from the Blueprint's ToolSpecs
  - Returning a fully functional AgentBody ready to run tasks
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from ultron.body.agent import AgentBody
from ultron.body.blueprint import Blueprint
from ultron.body.llm import LLMClient
from ultron.core.settings import CONFIG_DIR
from ultron.tools.registry import ToolRegistry


class BodyFactory:
    """Factory that builds living agent bodies from blueprints.

    Usage:
        factory = BodyFactory()
        body = await factory.create(blueprint)
        result = await body.run("Tell me about Python")
    """

    async def create(self, blueprint: Blueprint) -> AgentBody:
        """Instantiate a fully working agent from a Blueprint.

        Args:
            blueprint: The Blueprint defining the agent's configuration.

        Returns:
            An AgentBody ready to execute tasks.

        Raises:
            ImportError: If a tool module cannot be loaded.
        """
        logger.info(
            "Creating body '{}' (model={}, tools={})",
            blueprint.name,
            blueprint.model.litellm_model,
            len(blueprint.tools),
        )

        # 1. Configure the LLM client
        llm_client = LLMClient(config=blueprint.model)
        logger.debug("LLM client configured: {}", llm_client.model_string)

        # 2. Build the tool registry
        tool_registry = ToolRegistry()
        for tool_spec in blueprint.tools:
            try:
                tool_registry.register(tool_spec)
            except (ImportError, TypeError) as e:
                logger.error("Failed to register tool '{}': {}", tool_spec.name, e)
                raise

        logger.debug("Tool registry: {}", tool_registry.list_tools())

        # 3. Assemble the body
        body = AgentBody(
            blueprint=blueprint,
            llm_client=llm_client,
            tool_registry=tool_registry,
        )

        logger.info("Body created: {}", body)
        return body

    @staticmethod
    def load_genesis() -> Blueprint:
        """Load the genesis blueprint from config/genesis_blueprint.yaml."""
        genesis_path = CONFIG_DIR / "genesis_blueprint.yaml"
        if not genesis_path.exists():
            raise FileNotFoundError(
                f"Genesis blueprint not found at {genesis_path}. "
                "This file defines the starting body for Ultron."
            )
        return Blueprint.from_yaml(genesis_path)
