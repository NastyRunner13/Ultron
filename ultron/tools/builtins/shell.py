"""Shell tool — async command execution with safety constraints.

Runs shell commands in a subprocess with:
  - Blocked command detection (dangerous commands are rejected)
  - Configurable timeout
  - Output truncation to prevent context window overflow
"""

from __future__ import annotations

import asyncio
import platform
import shlex
from typing import Any

from loguru import logger

from ultron.core.settings import get_settings

# Default blocked patterns — checked as substrings of the command
_DEFAULT_BLOCKED = [
    "rm -rf /",
    "rm -rf /*",
    "format c:",
    "format d:",
    "del /f /s /q c:",
    "del /f /s /q d:",
    ":(){:|:&};:",
    "shutdown",
    "reboot",
    "mkfs",
    "dd if=",
    "chmod -R 777 /",
    "curl | sh",
    "wget | sh",
    "> /dev/sda",
]


def _is_blocked(command: str) -> bool:
    """Check if a command matches any blocked pattern."""
    settings = get_settings()
    blocked = settings.shell_blocked_commands or _DEFAULT_BLOCKED
    cmd_lower = command.lower().strip()

    for pattern in blocked:
        if pattern.lower() in cmd_lower:
            return True
    return False


async def execute_shell(command: str, timeout: int | None = None, **kwargs: Any) -> str:
    """Execute a shell command and return its output.

    Args:
        command: The shell command to run.
        timeout: Maximum seconds to wait (default: from settings).

    Returns:
        Combined stdout + stderr output.

    Raises:
        PermissionError: If the command matches a blocked pattern.
        TimeoutError: If the command exceeds the timeout.
    """
    if _is_blocked(command):
        logger.warning("Blocked dangerous command: {}", command)
        raise PermissionError(
            f"Command blocked for safety: '{command}'. "
            "This command matches a dangerous pattern."
        )

    settings = get_settings()
    timeout = timeout or settings.shell_timeout_seconds
    max_output = settings.shell_max_output_chars

    logger.info("Executing shell: {} (timeout={}s)", command, timeout)

    # Use platform-appropriate shell
    is_windows = platform.system() == "Windows"
    if is_windows:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            shell=True,
        )
    else:
        args = shlex.split(command)
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        raise TimeoutError(
            f"Command timed out after {timeout}s: '{command}'"
        )

    stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

    # Build output
    parts: list[str] = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"[stderr]\n{stderr}")
    if process.returncode != 0:
        parts.append(f"[exit code: {process.returncode}]")

    output = "\n".join(parts) if parts else "(no output)"

    # Truncate if needed
    if len(output) > max_output:
        output = output[:max_output] + f"\n\n[... output truncated at {max_output} chars ...]"

    logger.debug("Shell completed (exit={}, {} chars)", process.returncode, len(output))
    return output
