"""Method name constants (docs/SPEC.md §5)."""

from __future__ import annotations

from enum import Enum


class MethodName(str, Enum):
    ORM_GRPO = "orm_grpo"
    P_GRPO = "p_grpo"
    PAPO = "papo"
    PROF = "prof"
    LENS = "lens"
    NAIVE_HYBRID = "naive_hybrid"
    PRM_DIRECT = "prm_direct"
    TRIAGE = "triage"


_ALIASES = {
    "orm": MethodName.ORM_GRPO,
    "grpo": MethodName.ORM_GRPO,
    "pgrpo": MethodName.P_GRPO,
    "posterior_grpo": MethodName.P_GRPO,
    "triage_grpo": MethodName.TRIAGE,
}


def normalize_method(method: str | MethodName) -> MethodName:
    if isinstance(method, MethodName):
        return method
    key = str(method).strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    return MethodName(key)
