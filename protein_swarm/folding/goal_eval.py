"""
Design goal evaluation — computes a summary of how well the current
sequence achieves the stated design objective.

Provides goal_score (0-100), rating, key sub-scores, and recommendations
for inclusion in the agent prompt (PART 4).
"""

from __future__ import annotations

from collections import Counter

from protein_swarm.schemas import GoalEvaluation, ObjectiveSpec
from protein_swarm.utils.constants import (
    AMINO_ACIDS,
    HELIX_FAVORING,
    SHEET_FAVORING,
    HYDROPHOBIC,
    POLAR,
    COIL_FAVORING,
)


def evaluate_design_goal(
    sequence: str,
    objective: ObjectiveSpec,
    dssp_map: dict[int, str] | None = None,
) -> GoalEvaluation:
    """Evaluate how well `sequence` meets the design `objective`.

    Args:
        sequence: current protein sequence
        objective: parsed objective spec
        dssp_map: optional {pos -> DSSP label} from structure analysis
    """
    n = len(sequence)
    if n == 0:
        return GoalEvaluation(goal_score=0, rating="POOR", recommendations=["Sequence is empty"])

    aspects: dict[str, float] = {}
    recommendations: list[str] = []

    # diversity
    counts = Counter(sequence)
    unique_frac = len(counts) / min(n, len(AMINO_ACIDS))
    aspects["diversity"] = round(unique_frac * 100, 1)

    if unique_frac < 0.3:
        recommendations.append("Very low sequence diversity — introduce more amino acid variety")
    elif unique_frac < 0.5:
        recommendations.append("Moderate diversity — consider diversifying further")

    # helix propensity
    helix_frac = sum(1 for aa in sequence if aa in HELIX_FAVORING) / n
    aspects["helix_propensity_%"] = round(helix_frac * 100, 1)

    if objective.favour_helix:
        if helix_frac < 0.4:
            recommendations.append("Helix propensity is low — add more A/E/L/M/Q/K/R/H residues")
        elif helix_frac > 0.8:
            recommendations.append("Very high helix propensity — good for helical objective")

    # sheet propensity
    sheet_frac = sum(1 for aa in sequence if aa in SHEET_FAVORING) / n
    aspects["sheet_propensity_%"] = round(sheet_frac * 100, 1)

    if objective.favour_sheet and sheet_frac < 0.3:
        recommendations.append("Sheet propensity is low — add more V/I/Y/F/W/T residues")

    # hydrophobicity
    hydro_frac = sum(1 for aa in sequence if aa in HYDROPHOBIC) / n
    aspects["hydrophobic_%"] = round(hydro_frac * 100, 1)

    if hydro_frac > 0.6:
        recommendations.append("High hydrophobic content — may cause aggregation; balance with polar residues")
    elif hydro_frac < 0.2:
        recommendations.append("Low hydrophobic content — may lack stable core; add some hydrophobic residues")

    # repeat stretches
    max_run = _longest_run(sequence)
    aspects["longest_repeat"] = max_run
    if max_run >= 4:
        recommendations.append(f"Long repeat stretch of {max_run} — break it up for structural stability")

    # DSSP-derived helix content (if available)
    if dssp_map:
        helix_labels = sum(1 for v in dssp_map.values() if v in ("H", "G", "I"))
        sheet_labels = sum(1 for v in dssp_map.values() if v in ("E", "B"))
        total_assigned = len(dssp_map)
        if total_assigned > 0:
            aspects["dssp_helix_%"] = round(helix_labels / total_assigned * 100, 1)
            aspects["dssp_sheet_%"] = round(sheet_labels / total_assigned * 100, 1)
            if objective.favour_helix and helix_labels / total_assigned < 0.3:
                recommendations.append("Predicted helix content (DSSP) is low despite helix objective")

    # avoid-residues violations
    if objective.avoid_residues:
        violations = [aa for aa in sequence if aa in objective.avoid_residues]
        if violations:
            aspects["avoid_violations"] = len(violations)
            recommendations.append(
                f"Contains {len(violations)} residues from avoid-list ({', '.join(set(violations))})"
            )

    # compute weighted goal score
    goal_score = _compute_goal_score(aspects, objective)
    rating = "POOR" if goal_score < 35 else ("OK" if goal_score < 65 else "GOOD")

    if not recommendations:
        recommendations.append("Sequence looks well-aligned with the design objective")

    return GoalEvaluation(
        goal_score=round(goal_score, 1),
        rating=rating,
        key_aspects=aspects,
        recommendations=recommendations,
    )


def _compute_goal_score(aspects: dict[str, float], objective: ObjectiveSpec) -> float:
    """Weighted combination of sub-scores into a 0-100 goal achievement score."""
    scores: list[float] = []
    weights: list[float] = []

    diversity = aspects.get("diversity", 50)
    scores.append(min(diversity, 100))
    weights.append(1.0)

    if objective.favour_helix:
        helix = aspects.get("helix_propensity_%", 0)
        scores.append(min(helix * 1.2, 100))
        weights.append(1.5)

    if objective.favour_sheet:
        sheet = aspects.get("sheet_propensity_%", 0)
        scores.append(min(sheet * 1.3, 100))
        weights.append(1.5)

    hydro = aspects.get("hydrophobic_%", 50)
    balance = 100 - abs(hydro - 40) * 2  # penalise extremes
    scores.append(max(balance, 0))
    weights.append(0.5)

    longest = aspects.get("longest_repeat", 1)
    repeat_score = max(0, 100 - (longest - 1) * 20)
    scores.append(repeat_score)
    weights.append(0.5)

    if aspects.get("avoid_violations", 0) > 0:
        scores.append(0)
        weights.append(1.0)

    total_w = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_w if total_w > 0 else 0


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
