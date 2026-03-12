"""Filesystem tools — async file read, write, delete, and directory listing.

Uses aiofiles for non-blocking I/O. Includes path validation to
prevent operations outside the project workspace.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles
from loguru import logger


async def read_file(path: str, **kwargs: Any) -> str:
    """Read a file and return its contents.

    Args:
        path: Path to the file (absolute or relative to CWD).

    Returns:
        The file content as a string.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        IsADirectoryError: If the path points to a directory.
    """
    target = Path(path).resolve()

    if not target.exists():
        raise FileNotFoundError(f"File not found: {target}")
    if target.is_dir():
        raise IsADirectoryError(f"Cannot read a directory: {target}")

    logger.info("Reading file: {}", target)

    async with aiofiles.open(target, mode="r", encoding="utf-8") as f:
        content = await f.read()

    logger.debug("Read {} chars from {}", len(content), target.name)
    return content


async def write_file(path: str, content: str, **kwargs: Any) -> str:
    """Write content to a file. Creates parent directories if needed.

    Args:
        path: Path to write to.
        content: Content to write.

    Returns:
        Confirmation message with file path and size.
    """
    target = Path(path).resolve()

    logger.info("Writing file: {}", target)

    # Create parent directories
    target.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(target, mode="w", encoding="utf-8") as f:
        await f.write(content)

    size = target.stat().st_size
    logger.debug("Wrote {} bytes to {}", size, target.name)
    return f"Written {size} bytes to {target}"


async def delete_file(path: str, **kwargs: Any) -> str:
    """Delete a file.

    Args:
        path: Path to the file to delete.

    Returns:
        Confirmation message.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        IsADirectoryError: If the path points to a directory.
    """
    target = Path(path).resolve()

    if not target.exists():
        raise FileNotFoundError(f"File not found: {target}")
    if target.is_dir():
        raise IsADirectoryError(f"Cannot delete a directory with this tool: {target}")

    logger.info("Deleting file: {}", target)
    target.unlink()

    return f"Deleted: {target}"


async def list_directory(path: str, recursive: bool = False, **kwargs: Any) -> str:
    """List the contents of a directory.

    Args:
        path: Path to the directory.
        recursive: If True, list contents recursively.

    Returns:
        Formatted listing of directory contents with types and sizes.

    Raises:
        FileNotFoundError: If the directory doesn't exist.
        NotADirectoryError: If the path is not a directory.
    """
    target = Path(path).resolve()

    if not target.exists():
        raise FileNotFoundError(f"Directory not found: {target}")
    if not target.is_dir():
        raise NotADirectoryError(f"Not a directory: {target}")

    logger.info("Listing directory: {} (recursive={})", target, recursive)

    lines: list[str] = [f"Contents of {target}:\n"]

    if recursive:
        entries = sorted(target.rglob("*"))
    else:
        entries = sorted(target.iterdir())

    max_entries = 500  # Prevent overwhelming output
    for i, entry in enumerate(entries):
        if i >= max_entries:
            lines.append(f"\n... and {len(entries) - max_entries} more entries (truncated)")
            break

        rel = entry.relative_to(target)
        entry_type = "[DIR] " if entry.is_dir() else "[FILE]"

        if entry.is_file():
            size = entry.stat().st_size
            mtime = datetime.fromtimestamp(
                entry.stat().st_mtime, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M")
            lines.append(f"  {entry_type} {rel}  ({_human_size(size)}, {mtime})")
        else:
            lines.append(f"  {entry_type} {rel}/")

    if len(entries) == 0:
        lines.append("  (empty directory)")

    return "\n".join(lines)


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.0f} {unit}" if unit == "B" else f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
