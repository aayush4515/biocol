"""
Accept / reject policy and stopping conditions.
"""

from __future__ import annotations

from protein_swarm.config import SwarmConfig


def should_accept(candidate_score: float, current_score: float) -> bool:
    """Greedy acceptance: take strictly better scores."""
    return candidate_score > current_score


def detect_plateau(
    score_history: list[float],
    config: SwarmConfig,
) -> bool:
    """Return True if the last `plateau_window` iterations show no meaningful improvement."""
    window = config.plateau_window
    tol = config.plateau_tolerance

    if len(score_history) < window:
        return False

    recent = score_history[-window:]
    best = max(recent)
    worst = min(recent)
    return (best - worst) < tol


def should_stop(
    iteration: int,
    score_history: list[float],
    config: SwarmConfig,
) -> tuple[bool, str]:
    """Check all stopping conditions.

    Returns (should_stop, reason).
    """
    if iteration >= config.max_iterations - 1:
        return True, f"Reached max iterations ({config.max_iterations})"

    if detect_plateau(score_history, config):
        return True, f"Score plateau detected over last {config.plateau_window} iterations"

    return False, ""
