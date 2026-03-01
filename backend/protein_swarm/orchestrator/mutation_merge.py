"""
Mutation merge strategy.

Takes N proposals (one per position) and builds the candidate sequence S'.
Only mutations above the confidence threshold are applied.
"""

from __future__ import annotations

from protein_swarm.schemas import MutationProposal


def merge_mutations(
    sequence: str,
    proposals: list[MutationProposal],
    confidence_threshold: float = 0.5,
) -> tuple[str, list[MutationProposal]]:
    """Apply accepted proposals to the sequence.

    Returns:
        (new_sequence, list of proposals that were actually applied)
    """
    seq_list = list(sequence)
    applied: list[MutationProposal] = []

    for p in proposals:
        if p.proposed_residue == p.current_residue:
            continue
        if p.confidence < confidence_threshold:
            continue
        if 0 <= p.position < len(seq_list):
            seq_list[p.position] = p.proposed_residue
            applied.append(p)

    return "".join(seq_list), applied
