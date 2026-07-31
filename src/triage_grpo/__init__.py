"""TRIAGE-GRPO: 2x2 process-outcome triage for GRPO advantages."""

from triage_grpo.advantage import compute_group_advantages
from triage_grpo.baselines import MethodName
from triage_grpo.types import AdvantageOutput, Cell, GroupInput

__all__ = [
    "AdvantageOutput",
    "Cell",
    "GroupInput",
    "MethodName",
    "compute_group_advantages",
]

__version__ = "0.1.0"
