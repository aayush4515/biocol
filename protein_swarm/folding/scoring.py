"""
Heuristic scoring functions for the MVP folding engine.

Each scorer returns a float in [0, 1].  The fold engine combines them with
configurable weights.

Future: replace with physics-based energy terms from a real folding backend.
"""

from __future__ import annotations

import math
from collections import Counter

from protein_swarm.schemas import ObjectiveSpec
from protein_swarm.utils.constants import AMINO_ACIDS, HELIX_FAVORING, SHEET_FAVORING


def score_diversity(sequence: str) -> float:
    """Reward amino-acid diversity; penalise long repeats and low entropy."""
    counts = Counter(sequence)
    n = len(sequence)
    if n == 0:
        return 0.0

    unique_ratio = len(counts) / min(n, len(AMINO_ACIDS))

    probs = [c / n for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(min(n, len(AMINO_ACIDS)))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    max_run = _longest_run(sequence)
    run_penalty = max(0.0, 1.0 - max_run / n)

    return 0.4 * unique_ratio + 0.4 * norm_entropy + 0.2 * run_penalty


def score_helix_propensity(sequence: str) -> float:
    """Fraction of residues that favour alpha-helix formation."""
    if not sequence:
        return 0.0
    return sum(1 for aa in sequence if aa in HELIX_FAVORING) / len(sequence)


def score_sheet_propensity(sequence: str) -> float:
    if not sequence:
        return 0.0
    return sum(1 for aa in sequence if aa in SHEET_FAVORING) / len(sequence)


def score_repeat_penalty(sequence: str) -> float:
    """Penalise excessive identical-residue runs.  Returns value in [0, 1] where 1 = no repeats."""
    if len(sequence) <= 1:
        return 1.0
    max_run = _longest_run(sequence)
    return max(0.0, 1.0 - (max_run - 1) / len(sequence))


def compute_objective_score(sequence: str, objective: ObjectiveSpec) -> float:
    """Weighted combination of sub-scores driven by the objective."""
    components: list[float] = []
    weights: list[float] = []

    if objective.favour_helix:
        components.append(score_helix_propensity(sequence))
        weights.append(1.0)

    if objective.favour_sheet:
        components.append(score_sheet_propensity(sequence))
        weights.append(1.0)

    if objective.favour_diversity:
        components.append(score_diversity(sequence))
        weights.append(1.0)

    if objective.favour_stability:
        components.append(score_repeat_penalty(sequence))
        weights.append(0.5)
        components.append(score_diversity(sequence))
        weights.append(0.5)

    if not components:
        return score_diversity(sequence)

    total_w = sum(weights)
    return sum(c * w for c, w in zip(components, weights)) / total_w


# ── helpers ────────────────────────────────────────────────────────────────────

def _longest_run(sequence: str) -> int:
    if not sequence:
        return 0
    max_run = 1
    current_run = 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run
