"""Browse tool — async URL fetching with text extraction.

Uses httpx for async HTTP and BeautifulSoup4 for HTML parsing.
"""

from __future__ import annotations

import httpx
from bs4 import BeautifulSoup
from loguru import logger

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; Ultron/0.1; +https://github.com/ultron-agent)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_MAX_CONTENT_LENGTH = 100_000  # ~100KB text limit


async def browse_url(url: str, extract_text: bool = True) -> str:
    """Fetch a URL and return its content.

    Args:
        url: The URL to fetch.
        extract_text: If True, extract readable text from HTML.
                      If False, return raw HTML.

    Returns:
        The page content as a string.

    Raises:
        httpx.HTTPError: On network or HTTP errors.
        ValueError: On invalid URLs.
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid URL (must start with http:// or https://): {url}")

    logger.info("Browsing: {}", url)

    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers=_DEFAULT_HEADERS,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "")

    if not extract_text or "text/html" not in content_type:
        text = response.text[:_MAX_CONTENT_LENGTH]
        logger.debug("Returned raw content ({} chars)", len(text))
        return text

    # Parse HTML and extract readable text
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    # Extract text with whitespace normalization
    lines: list[str] = []
    for element in soup.stripped_strings:
        line = element.strip()
        if line:
            lines.append(line)

    text = "\n".join(lines)
    if len(text) > _MAX_CONTENT_LENGTH:
        text = text[:_MAX_CONTENT_LENGTH] + "\n\n[... content truncated ...]"

    logger.debug("Extracted {} chars of text from {}", len(text), url)
    return text
