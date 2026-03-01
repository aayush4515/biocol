"""
Internal data structures for the per-position memory system.

These are storage-level models — the public API exposes PositionMemorySummary
from schemas.py to agents.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PositionRecord(BaseModel):
    """Tracks mutation history at a single sequence position."""

    position: int
    success_count: int = 0
    failure_count: int = 0
    accepted_residues: dict[str, int] = Field(default_factory=dict)
    rejected_residues: dict[str, int] = Field(default_factory=dict)

    @property
    def total_trials(self) -> int:
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        if self.total_trials == 0:
            return 0.0
        return self.success_count / self.total_trials


class IterationOutcome(BaseModel):
    """Compact record of one iteration's result, stored in memory."""

    iteration: int
    accepted: bool
    combined_score: float = 0.0
    objective_score: float = 0.0
    physics_score: float = 0.0
    rosetta_total_score: float | None = None
    design_goal_score: float = 0.0
    num_mutations: int = 0
    sequence: str = ""
    reason: str = ""
