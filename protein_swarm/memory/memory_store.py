"""
Shared memory store that tracks per-position mutation history.

Memory influences future agent mutation probabilities: positions with high
success rates get a boost; positions that consistently fail get dampened.
"""

from __future__ import annotations

from protein_swarm.config import MemoryConfig, DEFAULT_MEMORY
from protein_swarm.memory.memory_schema import PositionRecord
from protein_swarm.schemas import MutationProposal, PositionMemorySummary


class MemoryStore:
    """Thread-safe (single-process) per-position memory for the design loop."""

    def __init__(self, sequence_length: int, config: MemoryConfig | None = None) -> None:
        self._config = config or DEFAULT_MEMORY
        self._records: dict[int, PositionRecord] = {
            i: PositionRecord(position=i) for i in range(sequence_length)
        }

    @property
    def length(self) -> int:
        return len(self._records)

    def _ensure_position(self, position: int) -> None:
        if position not in self._records:
            self._records[position] = PositionRecord(position=position)

    def get_summary_for_position(self, position: int) -> PositionMemorySummary:
        self._ensure_position(position)
        rec = self._records[position]

        bias = self._compute_mutation_bias(rec)

        accepted = sorted(rec.accepted_residues, key=rec.accepted_residues.get, reverse=True)  # type: ignore[arg-type]
        rejected = sorted(rec.rejected_residues, key=rec.rejected_residues.get, reverse=True)  # type: ignore[arg-type]

        return PositionMemorySummary(
            position=position,
            success_count=rec.success_count,
            failure_count=rec.failure_count,
            accepted_residues=accepted[:5],
            rejected_residues=rejected[:5],
            mutation_bias=bias,
        )

    def record_success(self, proposals: list[MutationProposal]) -> None:
        for p in proposals:
            if p.proposed_residue == p.current_residue:
                continue
            self._ensure_position(p.position)
            rec = self._records[p.position]
            rec.success_count += 1
            rec.accepted_residues[p.proposed_residue] = (
                rec.accepted_residues.get(p.proposed_residue, 0) + 1
            )

    def record_failure(self, proposals: list[MutationProposal]) -> None:
        for p in proposals:
            if p.proposed_residue == p.current_residue:
                continue
            self._ensure_position(p.position)
            rec = self._records[p.position]
            rec.failure_count += 1
            rec.rejected_residues[p.proposed_residue] = (
                rec.rejected_residues.get(p.proposed_residue, 0) + 1
            )

    def apply_decay(self) -> None:
        """Apply exponential decay to all counts, preventing stale memory domination."""
        d = self._config.decay_factor
        for rec in self._records.values():
            rec.success_count = int(rec.success_count * d)
            rec.failure_count = int(rec.failure_count * d)

    def _compute_mutation_bias(self, rec: PositionRecord) -> float:
        """Bias > 1 encourages mutation; < 1 discourages it."""
        if rec.total_trials == 0:
            return 1.0

        rate = rec.success_rate
        cfg = self._config

        if rate >= 0.5:
            return min(cfg.success_boost, 1.0 + rate)
        return max(cfg.failure_dampen, 1.0 - (1.0 - rate) * 0.5)

    def to_dict(self) -> dict:
        return {pos: rec.model_dump() for pos, rec in self._records.items()}
