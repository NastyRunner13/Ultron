"""Scorers — evaluate agent responses against benchmark expectations.

Multiple scorer types handle different evaluation strategies:
  - ExactMatchScorer: normalized string comparison
  - ContainsScorer: substring presence (with multi-substring support)
  - CodeExecutionScorer: run code, verify output
  - LLMJudgeScorer: call a judge LLM to grade open-ended responses
  - CompositeScorer: weighted combination of sub-scores

All scorers are async and return ScorerResult with score 0.0-1.0.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger

from ultron.arena.models import BenchmarkTask, ScorerResult


class BaseScorer(ABC):
    """Abstract base class for all scorers."""

    @abstractmethod
    async def score(self, task: BenchmarkTask, response: str) -> ScorerResult:
        """Score an agent's response against a benchmark task.

        Args:
            task: The benchmark task definition.
            response: The agent's raw text response.

        Returns:
            ScorerResult with score 0.0-1.0.
        """
        ...


class ExactMatchScorer(BaseScorer):
    """Scores 1.0 if the normalized response exactly matches expected output."""

    async def score(self, task: BenchmarkTask, response: str) -> ScorerResult:
        if not task.expected_output:
            return ScorerResult(score=0.0, passed=False, reasoning="No expected output defined")

        expected = self._normalize(task.expected_output)
        actual = self._normalize(response)

        if expected == actual:
            return ScorerResult(
                score=1.0, passed=True,
                reasoning="Exact match",
                details={"expected": expected, "actual": actual},
            )

        return ScorerResult(
            score=0.0, passed=False,
            reasoning=f"Expected '{expected}', got '{actual}'",
            details={"expected": expected, "actual": actual},
        )

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison: lowercase, strip, collapse whitespace."""
        return re.sub(r"\s+", " ", text.strip().lower())


class ContainsScorer(BaseScorer):
    """Scores based on substring presence.

    Supports:
      - Single expected_output substring check
      - Multiple required_substrings (via scoring_config)
      - min_occurrences for repeated patterns
    """

    async def score(self, task: BenchmarkTask, response: str) -> ScorerResult:
        response_lower = response.lower().strip()
        config = task.scoring_config or {}

        # Multiple required substrings
        required = config.get("required_substrings", [])
        if required:
            found = []
            missing = []
            for sub in required:
                if sub.lower() in response_lower:
                    found.append(sub)
                else:
                    missing.append(sub)

            score = len(found) / len(required) if required else 0.0
            return ScorerResult(
                score=score,
                passed=score >= 0.5,
                reasoning=f"Found {len(found)}/{len(required)} required substrings",
                details={"found": found, "missing": missing},
            )

        # Min occurrences check
        min_occ = config.get("min_occurrences")
        if min_occ and task.expected_output:
            count = response_lower.count(task.expected_output.lower())
            if count >= min_occ:
                return ScorerResult(
                    score=1.0, passed=True,
                    reasoning=f"Found '{task.expected_output}' {count} times (needed {min_occ})",
                    details={"count": count, "min_required": min_occ},
                )
            return ScorerResult(
                score=count / min_occ,
                passed=False,
                reasoning=f"Found '{task.expected_output}' {count} times (needed {min_occ})",
                details={"count": count, "min_required": min_occ},
            )

        # Simple single substring
        if task.expected_output:
            if task.expected_output.lower() in response_lower:
                return ScorerResult(
                    score=1.0, passed=True,
                    reasoning=f"Response contains '{task.expected_output}'",
                )
            return ScorerResult(
                score=0.0, passed=False,
                reasoning=f"Response does not contain '{task.expected_output}'",
            )

        # No expected output — check for tool use if configured
        must_use = config.get("must_use_tools", [])
        if must_use:
            # If response is non-empty and contains tool-relevant content, partial score
            if len(response_lower) > 20:
                return ScorerResult(
                    score=0.7, passed=True,
                    reasoning="Non-trivial response generated (tool use not directly verifiable in scorer)",
                )
            return ScorerResult(
                score=0.3, passed=False,
                reasoning="Response too short — likely did not use tools",
            )

        # Fallback: non-empty response
        if len(response.strip()) > 10:
            return ScorerResult(
                score=0.5, passed=True,
                reasoning="Non-empty response (no expected output to check against)",
            )
        return ScorerResult(score=0.0, passed=False, reasoning="Empty or trivial response")


class CodeExecutionScorer(BaseScorer):
    """Scores by extracting and executing code from the response.

    Extracts Python code blocks, runs them in a subprocess, and checks
    if the output matches the expected output.
    """

    async def score(self, task: BenchmarkTask, response: str) -> ScorerResult:
        # Extract code from response (markdown code blocks or raw Python)
        code = self._extract_code(response)
        if not code:
            # If no code block but the response contains the expected output, partial credit
            if task.expected_output and task.expected_output.lower() in response.lower():
                return ScorerResult(
                    score=0.7, passed=True,
                    reasoning="No executable code found, but response contains expected answer",
                )
            return ScorerResult(
                score=0.0, passed=False,
                reasoning="No executable code found in response",
            )

        # Run the code
        try:
            output = await self._execute_code(code, timeout=15)
        except TimeoutError:
            return ScorerResult(
                score=0.0, passed=False,
                reasoning="Code execution timed out",
                details={"code": code[:500]},
            )
        except Exception as e:
            return ScorerResult(
                score=0.0, passed=False,
                reasoning=f"Code execution failed: {e}",
                details={"code": code[:500], "error": str(e)},
            )

        # Compare output
        if task.expected_output:
            if task.expected_output.lower() in output.lower():
                return ScorerResult(
                    score=1.0, passed=True,
                    reasoning=f"Code output contains expected: '{task.expected_output}'",
                    details={"output": output[:500], "code": code[:500]},
                )
            return ScorerResult(
                score=0.2, passed=False,
                reasoning=f"Code ran but output doesn't match. Expected: '{task.expected_output}', Got: '{output[:200]}'",
                details={"output": output[:500], "code": code[:500]},
            )

        # No expected output — code ran successfully is partial credit
        return ScorerResult(
            score=0.5, passed=True,
            reasoning="Code executed successfully (no expected output to verify)",
            details={"output": output[:500]},
        )

    @staticmethod
    def _extract_code(response: str) -> str | None:
        """Extract Python code from markdown code blocks or raw text."""
        # Try markdown code blocks first
        patterns = [
            r"```python\n(.*?)```",
            r"```py\n(.*?)```",
            r"```\n(.*?)```",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[-1].strip()  # Last code block (usually the complete one)

        # Check if the whole response looks like Python code
        lines = response.strip().split("\n")
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
        if code_lines and any(
            kw in response for kw in ("def ", "import ", "print(", "return ", "class ")
        ):
            return response.strip()

        return None

    @staticmethod
    async def _execute_code(code: str, timeout: int = 15) -> str:
        """Execute Python code in a subprocess and return stdout."""
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = Path(f.name)

        try:
            proc = await asyncio.create_subprocess_exec(
                "python", str(tmp_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = stdout.decode("utf-8", errors="replace").strip()
            if not output and stderr:
                output = stderr.decode("utf-8", errors="replace").strip()
            return output
        finally:
            tmp_path.unlink(missing_ok=True)


class LLMJudgeScorer(BaseScorer):
    """Uses a judge LLM to grade open-ended responses.

    The judge is given the task prompt + response + criteria, and asked
    to rate the response 0.0-1.0.
    """

    def __init__(self, llm_client: Any = None) -> None:
        self._llm_client = llm_client

    async def score(self, task: BenchmarkTask, response: str) -> ScorerResult:
        if not self._llm_client:
            # Fallback: basic length/quality heuristic
            return await self._heuristic_score(task, response)

        config = task.scoring_config or {}
        criteria = config.get("judge_criteria", "Rate the quality of this response from 0 to 1.")

        judge_prompt = (
            f"You are a strict evaluator. Score the following response on a scale of 0.0 to 1.0.\n\n"
            f"TASK: {task.prompt}\n\n"
            f"RESPONSE: {response}\n\n"
            f"CRITERIA: {criteria}\n\n"
            f"Respond with ONLY a JSON object: {{\"score\": <float 0.0-1.0>, \"reasoning\": \"<brief explanation>\"}}"
        )

        try:
            from ultron.body.llm import LLMResponse

            llm_response: LLMResponse = await self._llm_client.chat(
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.1,
                max_tokens=200,
            )

            # Parse judge response
            import json
            content = llm_response.content or ""
            # Try to extract JSON from the response
            json_match = re.search(r"\{[^}]+\}", content)
            if json_match:
                parsed = json.loads(json_match.group())
                judge_score = float(parsed.get("score", 0.5))
                judge_score = max(0.0, min(1.0, judge_score))  # Clamp
                reasoning = parsed.get("reasoning", "LLM judge evaluation")
            else:
                # Fallback: try to find a number
                numbers = re.findall(r"(\d+\.?\d*)", content)
                if numbers:
                    judge_score = float(numbers[0])
                    if judge_score > 1.0:
                        judge_score = judge_score / 10.0  # Handle 0-10 scale
                    judge_score = max(0.0, min(1.0, judge_score))
                    reasoning = f"Extracted score from judge response: {content[:100]}"
                else:
                    judge_score = 0.5
                    reasoning = f"Could not parse judge response: {content[:100]}"

            return ScorerResult(
                score=judge_score,
                passed=judge_score >= 0.5,
                reasoning=reasoning,
                details={"judge_response": content[:500]},
            )

        except Exception as e:
            logger.warning("LLM judge failed, falling back to heuristic: {}", e)
            return await self._heuristic_score(task, response)

    @staticmethod
    async def _heuristic_score(task: BenchmarkTask, response: str) -> ScorerResult:
        """Fallback heuristic for when LLM judge is unavailable."""
        response = response.strip()
        score = 0.0

        if len(response) > 200:
            score += 0.3
        elif len(response) > 50:
            score += 0.15

        # Check for structure
        if any(marker in response for marker in ["1.", "- ", "• ", "**"]):
            score += 0.2

        # Check for specificity
        specific_words = ["because", "specific", "example", "improve", "strategy", "step"]
        specificity = sum(1 for w in specific_words if w.lower() in response.lower())
        score += min(0.3, specificity * 0.1)

        # Ensure in range
        score = min(1.0, score)

        return ScorerResult(
            score=score,
            passed=score >= 0.5,
            reasoning=f"Heuristic score (length={len(response)}, structure markers found)",
            details={"heuristic": True},
        )


# ── Scorer Factory ───────────────────────────────────────────────────────────


_SCORER_REGISTRY: dict[str, type[BaseScorer]] = {
    "exact": ExactMatchScorer,
    "contains": ContainsScorer,
    "code_exec": CodeExecutionScorer,
    "llm_judge": LLMJudgeScorer,
}


def get_scorer(method: str, **kwargs: Any) -> BaseScorer:
    """Get a scorer instance by method name.

    Args:
        method: One of "exact", "contains", "code_exec", "llm_judge".
        **kwargs: Extra arguments passed to the scorer constructor.

    Returns:
        A scorer instance.
    """
    cls = _SCORER_REGISTRY.get(method)
    if cls is None:
        raise ValueError(
            f"Unknown scoring method '{method}'. Available: {list(_SCORER_REGISTRY.keys())}"
        )
    return cls(**kwargs)
