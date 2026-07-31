"""Process scorer protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProcessScorer(Protocol):
    def score(self, prompt: str, response: str) -> float:
        """Return process score in [0, 1]."""
        ...
