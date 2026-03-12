"""Ultron settings — loaded from config/settings.yaml + environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ── Locate project root ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

# Load .env from project root
load_dotenv(PROJECT_ROOT / ".env")


class ShellSafetySettings(BaseModel):
    """Safety constraints for the shell tool."""

    blocked_commands: list[str] = Field(default_factory=list)
    timeout_seconds: int = 30
    max_output_chars: int = 10_000


class UltronSettings(BaseModel):
    """Global settings for the Ultron agent system."""

    # ── LLM defaults ──
    default_model: str = "openrouter/meta-llama/llama-3-70b-instruct"
    default_temperature: float = 0.7
    default_max_tokens: int = 4096

    # ── Token budget ──
    token_budget: int = 100_000

    # ── Agent runtime ──
    max_agent_steps: int = 20
    agent_timeout_seconds: int = 300

    # ── Evolution thresholds ──
    merge_improvement_threshold: float = 0.05
    kill_after_n_failures: int = 3
    max_active_branches: int = 5

    # ── Arena ──
    arena_runs_per_benchmark: int = 3
    arena_timeout_per_task: int = 60

    # ── Paths ──
    data_dir: Path = Path("./data")

    # ── Logging ──
    log_level: str = "INFO"
    log_file: str | None = None

    # ── Shell safety ──
    shell_blocked_commands: list[str] = Field(default_factory=list)
    shell_timeout_seconds: int = 30
    shell_max_output_chars: int = 10_000

    # ── File tool safety ──
    file_allowed_extensions: list[str] = Field(
        default_factory=lambda: [".py", ".yaml", ".yml", ".json", ".md", ".txt", ".toml"]
    )

    @property
    def shell_safety(self) -> ShellSafetySettings:
        return ShellSafetySettings(
            blocked_commands=self.shell_blocked_commands,
            timeout_seconds=self.shell_timeout_seconds,
            max_output_chars=self.shell_max_output_chars,
        )

    @property
    def resolved_data_dir(self) -> Path:
        """Resolve data_dir relative to project root."""
        if self.data_dir.is_absolute():
            return self.data_dir
        return PROJECT_ROOT / self.data_dir

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> UltronSettings:
        """Load settings from a YAML file, with env var overrides."""
        config_path = path or CONFIG_DIR / "settings.yaml"

        raw: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}

        # Environment variable overrides (prefixed with ULTRON_)
        env_overrides = {
            "log_level": os.getenv("ULTRON_LOG_LEVEL"),
            "data_dir": os.getenv("ULTRON_DATA_DIR"),
            "default_model": os.getenv("ULTRON_DEFAULT_MODEL"),
        }
        for key, val in env_overrides.items():
            if val is not None:
                raw[key] = val

        return cls(**raw)


@lru_cache(maxsize=1)
def get_settings() -> UltronSettings:
    """Get the global singleton settings instance."""
    return UltronSettings.from_yaml()
