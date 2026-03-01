"""
Post-proposal constraint validation.

Filters out mutations that violate hard constraints from the ObjectiveSpec
before they reach the merge stage.  Handles both legacy custom_constraints
strings and the newer avoid_residues list from LLM-parsed objectives.
"""

from __future__ import annotations

from protein_swarm.schemas import MutationProposal, ObjectiveSpec
from protein_swarm.utils.constants import AMINO_ACIDS


def validate_proposals(
    proposals: list[MutationProposal],
    objective: ObjectiveSpec,
    sequence_length: int,
) -> list[MutationProposal]:
    """Return only proposals that pass all constraint checks."""
    valid: list[MutationProposal] = []
    for p in proposals:
        if not _passes_constraints(p, objective, sequence_length):
            continue
        valid.append(p)
    return valid


def _passes_constraints(
    proposal: MutationProposal,
    objective: ObjectiveSpec,
    sequence_length: int,
) -> bool:
    if proposal.position < 0 or proposal.position >= sequence_length:
        return False

    if proposal.proposed_residue not in AMINO_ACIDS:
        return False

    aa = proposal.proposed_residue

    if "avoid_cysteine" in objective.custom_constraints and aa == "C":
        return False
    if "avoid_proline" in objective.custom_constraints and aa == "P":
        return False

    if aa in objective.avoid_residues:
        return False

    return True
