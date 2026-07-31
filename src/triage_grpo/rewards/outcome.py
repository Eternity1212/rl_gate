"""Outcome reward helpers for verifiable math answers."""

from __future__ import annotations

import re
from typing import Optional


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_ANSWER_RE = re.compile(
    r"(?:final answer|answer)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_INT_RE = re.compile(r"-?\d+")


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract last \\boxed{...} content if present."""
    matches = _BOXED_RE.findall(text or "")
    if matches:
        return matches[-1].strip()
    m = _ANSWER_RE.search(text or "")
    if m:
        return m.group(1).strip()
    # Fallback: last integer in text (DAPO-Math style integer answers)
    ints = _INT_RE.findall(text or "")
    if ints:
        return ints[-1]
    return None


def _norm_ans(s: str) -> str:
    s = s.strip().replace(",", "")
    # strip wrapping $
    s = s.strip("$").strip()
    try:
        if "." in s:
            return str(float(s))
        return str(int(s))
    except ValueError:
        return s.lower()


def score_outcome_exact(response: str, gold: str) -> float:
    """Binary outcome reward: 1.0 if extracted answer matches gold else 0.0."""
    pred = extract_boxed_answer(response)
    if pred is None:
        return 0.0
    return 1.0 if _norm_ans(pred) == _norm_ans(str(gold)) else 0.0
