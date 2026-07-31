"""Rule-based process scorer (stable fallback before PRM wiring)."""

from __future__ import annotations

import re

from triage_grpo.rewards.outcome import extract_boxed_answer


class RuleProcessScorer:
    """Cheap process proxy for pipeline bring-up.

    Components (equal weight by default):
    - has step structure (newlines / numbering)
    - has boxed or explicit answer span
    - not extremely short
    - not pathological repetition

    This is NOT a substitute for a real PRM in the main paper table;
    use for Phase A/B integration and appendix robustness.
    """

    def __init__(
        self,
        min_chars: int = 40,
        max_repeat_ratio: float = 0.35,
    ) -> None:
        self.min_chars = min_chars
        self.max_repeat_ratio = max_repeat_ratio

    def score(self, prompt: str, response: str) -> float:
        del prompt  # unused; kept for Protocol compatibility
        text = response or ""
        parts: list[float] = []

        # Structure: multiple lines or numbered steps
        lines = [ln for ln in text.splitlines() if ln.strip()]
        structured = 1.0 if len(lines) >= 3 else (0.5 if len(lines) >= 2 else 0.0)
        if re.search(r"(?:^|\n)\s*(?:\d+[\.)]|step\s*\d+)", text, re.I):
            structured = max(structured, 0.8)
        parts.append(structured)

        # Explicit answer cue
        parts.append(1.0 if extract_boxed_answer(text) is not None else 0.0)

        # Length adequacy
        parts.append(1.0 if len(text.strip()) >= self.min_chars else 0.0)

        # Repetition penalty (hacky verbosity / loops)
        tokens = re.findall(r"\w+", text.lower())
        if len(tokens) >= 8:
            uniq = len(set(tokens)) / len(tokens)
            parts.append(1.0 if uniq >= (1.0 - self.max_repeat_ratio) else uniq)
        else:
            parts.append(0.5)

        return float(sum(parts) / len(parts))
