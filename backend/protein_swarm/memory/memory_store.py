"""
Shared memory store that tracks per-position mutation history and
iteration-level outcomes for paper-style prompt context.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

from protein_swarm.config import MemoryConfig, DEFAULT_MEMORY
from protein_swarm.memory.memory_schema import PositionRecord, IterationOutcome
from protein_swarm.schemas import (
    GlobalMemoryStats,
    MutationProposal,
    PositionMemorySummary,
    PositionMutationEvent,
)

logger = logging.getLogger(__name__)


class MemoryStore:
    """Thread-safe (single-process) per-position memory for the design loop."""

    def __init__(self, sequence_length: int, config: MemoryConfig | None = None) -> None:
        self._config = config or DEFAULT_MEMORY
        self._records: dict[int, PositionRecord] = {
            i: PositionRecord(position=i) for i in range(sequence_length)
        }
        self._iteration_records: list[IterationOutcome] = []
        self._position_events: dict[int, list[PositionMutationEvent]] = {}

    @property
    def length(self) -> int:
        return len(self._records)

    def _ensure_position(self, position: int) -> None:
        if position not in self._records:
            self._records[position] = PositionRecord(position=position)

    # ── per-position summary (sent to agents) ────────────────────────────

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

    # ── record iteration-level outcomes ──────────────────────────────────

    def record_iteration_result(
        self,
        iteration: int,
        accepted: bool,
        combined_score: float = 0.0,
        objective_score: float = 0.0,
        physics_score: float = 0.0,
        rosetta_total_score: float | None = None,
        design_goal_score: float = 0.0,
        num_mutations: int = 0,
        sequence: str = "",
        reason: str = "",
    ) -> None:
        """Store a compact summary of this iteration's outcome."""
        self._iteration_records.append(IterationOutcome(
            iteration=iteration,
            accepted=accepted,
            combined_score=combined_score,
            objective_score=objective_score,
            physics_score=physics_score,
            rosetta_total_score=rosetta_total_score,
            design_goal_score=design_goal_score,
            num_mutations=num_mutations,
            sequence=sequence,
            reason=reason,
        ))

    # ── record per-position mutation events ──────────────────────────────

    def _record_position_event(
        self,
        proposal: MutationProposal,
        iteration: int,
        accepted: bool,
        combined_score: float = 0.0,
        objective_score: float = 0.0,
        physics_score: float = 0.0,
        rosetta_total_score: float | None = None,
        design_goal_score: float | None = None,
        num_mutations: int = 0,
    ) -> None:
        if proposal.proposed_residue == proposal.current_residue:
            return
        pos = proposal.position
        if pos not in self._position_events:
            self._position_events[pos] = []
        self._position_events[pos].append(PositionMutationEvent(
            iteration=iteration,
            position=pos,
            from_res=proposal.current_residue,
            to_res=proposal.proposed_residue,
            accepted=accepted,
            reason=proposal.reason,
            combined_score=combined_score,
            objective_score=objective_score,
            physics_score=physics_score,
            rosetta_total_score=rosetta_total_score,
            design_goal_score=design_goal_score,
            num_mutations_in_iteration=num_mutations,
        ))

    def record_success(
        self,
        proposals: list[MutationProposal],
        iteration: int = 0,
        combined_score: float = 0.0,
        objective_score: float = 0.0,
        physics_score: float = 0.0,
        rosetta_total_score: float | None = None,
        design_goal_score: float | None = None,
    ) -> None:
        num_mutations = sum(1 for p in proposals if p.proposed_residue != p.current_residue)
        for p in proposals:
            if p.proposed_residue == p.current_residue:
                continue
            self._ensure_position(p.position)
            rec = self._records[p.position]
            rec.success_count += 1
            rec.accepted_residues[p.proposed_residue] = (
                rec.accepted_residues.get(p.proposed_residue, 0) + 1
            )
            self._record_position_event(
                p, iteration, accepted=True,
                combined_score=combined_score,
                objective_score=objective_score,
                physics_score=physics_score,
                rosetta_total_score=rosetta_total_score,
                design_goal_score=design_goal_score,
                num_mutations=num_mutations,
            )

    def record_failure(
        self,
        proposals: list[MutationProposal],
        iteration: int = 0,
        combined_score: float = 0.0,
        objective_score: float = 0.0,
        physics_score: float = 0.0,
        rosetta_total_score: float | None = None,
        design_goal_score: float | None = None,
    ) -> None:
        num_mutations = sum(1 for p in proposals if p.proposed_residue != p.current_residue)
        for p in proposals:
            if p.proposed_residue == p.current_residue:
                continue
            self._ensure_position(p.position)
            rec = self._records[p.position]
            rec.failure_count += 1
            rec.rejected_residues[p.proposed_residue] = (
                rec.rejected_residues.get(p.proposed_residue, 0) + 1
            )
            self._record_position_event(
                p, iteration, accepted=False,
                combined_score=combined_score,
                objective_score=objective_score,
                physics_score=physics_score,
                rosetta_total_score=rosetta_total_score,
                design_goal_score=design_goal_score,
                num_mutations=num_mutations,
            )

    # ── global stats ─────────────────────────────────────────────────────

    def get_global_stats(self, last_k: int = 5) -> GlobalMemoryStats:
        """Compute aggregate iteration stats for prompt context."""
        total = len(self._iteration_records)
        if total == 0:
            return GlobalMemoryStats()

        accepted_count = sum(1 for r in self._iteration_records if r.accepted)
        rejected_count = total - accepted_count
        acceptance_rate = accepted_count / total if total > 0 else 0.0

        recent = self._iteration_records[-last_k:]
        recent_acc = sum(1 for r in recent if r.accepted)
        recent_acceptance_rate = recent_acc / len(recent) if recent else 0.0

        accepted_scores = [
            r.combined_score for r in self._iteration_records if r.accepted
        ]
        recent_scores = accepted_scores[-last_k:]
        energy_trend = self._compute_energy_trend(recent_scores)

        return GlobalMemoryStats(
            total_iterations=total,
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            acceptance_rate=round(acceptance_rate, 4),
            recent_acceptance_rate=round(recent_acceptance_rate, 4),
            energy_trend=energy_trend,
            recent_scores=recent_scores,
        )

    # ── iteration outcome access ─────────────────────────────────────────

    def get_recent_iteration_outcomes(self, last_k: int = 5) -> list[IterationOutcome]:
        return self._iteration_records[-last_k:]

    # ── per-position event history ───────────────────────────────────────

    def get_position_history(self, pos: int, last_k: int = 10) -> list[PositionMutationEvent]:
        events = self._position_events.get(pos, [])
        return events[-last_k:]

    def get_neighborhood_history(
        self,
        pos: int,
        radius: int = 3,
        last_k: int = 10,
    ) -> list[PositionMutationEvent]:
        """Gather mutation events for positions within ±radius of pos."""
        events: list[PositionMutationEvent] = []
        for p in range(max(0, pos - radius), pos + radius + 1):
            if p == pos:
                continue
            events.extend(self._position_events.get(p, []))
        events.sort(key=lambda e: e.iteration, reverse=True)
        return events[:last_k]

    # ── rich analysis helpers (for paper-style prompt) ────────────────────

    def compute_position_analysis(
        self, pos: int, current_residue: str,
    ) -> dict:
        """Compute detailed analysis for a position, matching paper PART 3.

        Returns a dict with keys:
            acceptance_trend, common_rejection_reasons, rejected_patterns,
            recent_trend, accepted_residue_counts, rejected_residue_counts,
            current_residue_stats, avg_energy_with_current
        """
        events = self._position_events.get(pos, [])
        if not events:
            return {}

        total = len(events)
        accepted_events = [e for e in events if e.accepted]
        rejected_events = [e for e in events if not e.accepted]

        # acceptance trend: first half vs second half
        if total >= 4:
            mid = total // 2
            first_half_rate = sum(1 for e in events[:mid] if e.accepted) / mid
            second_half_rate = sum(1 for e in events[mid:] if e.accepted) / (total - mid)
            if second_half_rate > first_half_rate + 0.1:
                acceptance_trend = "improving"
            elif second_half_rate < first_half_rate - 0.1:
                acceptance_trend = "declining"
            else:
                acceptance_trend = "stable"
        else:
            acceptance_trend = "insufficient data"

        # recent trend (last 10)
        recent = events[-10:]
        recent_accepted = sum(1 for e in recent if e.accepted)
        recent_total = len(recent)
        if recent_total >= 3:
            recent_rate = recent_accepted / recent_total
            if recent_rate > 0.6:
                recent_trend = "improving"
            elif recent_rate < 0.3:
                recent_trend = "declining"
            else:
                recent_trend = "mixed"
        else:
            recent_trend = "insufficient data"

        # accepted residue counts
        acc_counts: dict[str, int] = defaultdict(int)
        for e in accepted_events:
            acc_counts[e.to_res] += 1

        # rejected residue counts
        rej_counts: dict[str, int] = defaultdict(int)
        for e in rejected_events:
            rej_counts[e.to_res] += 1

        # rejected patterns with percentages
        rejected_patterns: list[str] = []
        total_rej = len(rejected_events)
        if total_rej > 0:
            pair_counts: dict[str, int] = defaultdict(int)
            for e in rejected_events:
                pair_counts[f"{e.from_res}->{e.to_res}"] += 1
            for pair, count in sorted(pair_counts.items(), key=lambda x: -x[1]):
                pct = count / total_rej * 100
                rejected_patterns.append(f"{pair} ({pct:.1f}%)")

        # current residue stats
        current_accepted = sum(1 for e in events if e.to_res == current_residue and e.accepted)
        current_total = sum(1 for e in events if e.to_res == current_residue)

        # avg energy with current residue
        current_energies = [
            e.rosetta_total_score for e in events
            if e.to_res == current_residue and e.rosetta_total_score is not None
        ]
        avg_energy = sum(current_energies) / len(current_energies) if current_energies else None

        return {
            "acceptance_trend": acceptance_trend,
            "recent_trend": recent_trend,
            "accepted_residue_counts": dict(acc_counts),
            "rejected_residue_counts": dict(rej_counts),
            "rejected_patterns": rejected_patterns[:5],
            "current_residue_stats": (current_accepted, current_total),
            "avg_energy_with_current": avg_energy,
        }

    def compute_mutation_recommendations(
        self, pos: int, current_residue: str,
    ) -> dict[str, list[str]]:
        """Compute residue-specific mutation recommendations from all position events.

        Returns {highly_recommended: [...], recommended: [...], avoid: [...]}.
        """
        events = self._position_events.get(pos, [])
        if not events:
            return {"highly_recommended": [], "recommended": [], "avoid": []}

        residue_stats: dict[str, tuple[int, int]] = {}  # res -> (accepted, total)
        for e in events:
            if e.to_res == current_residue:
                continue
            acc, tot = residue_stats.get(e.to_res, (0, 0))
            residue_stats[e.to_res] = (acc + (1 if e.accepted else 0), tot + 1)

        highly_recommended: list[str] = []
        recommended: list[str] = []
        avoid: list[str] = []

        for res, (acc, tot) in sorted(residue_stats.items(), key=lambda x: -x[1][0] / max(x[1][1], 1)):
            rate = acc / tot if tot > 0 else 0
            label = f"{res} ({rate:.2f} ({acc}/{tot}))"
            if rate >= 0.7 and tot >= 1:
                highly_recommended.append(label)
            elif rate >= 0.4:
                recommended.append(label)
            elif tot >= 1:
                avoid.append(label)

        return {
            "highly_recommended": highly_recommended[:5],
            "recommended": recommended[:5],
            "avoid": avoid[:5],
        }

    def generate_one_line_summary(self, pos: int) -> str:
        """Generate a one-line memory summary for the position."""
        events = self._position_events.get(pos, [])
        if not events:
            return "No mutation history at this position."

        recent = events[-5:]
        recent_rej = sum(1 for e in recent if not e.accepted)

        if recent_rej >= 4:
            return "Multiple recent rejections — strongly prefer conservative mutations or no-op."
        if recent_rej >= 2:
            rej_reasons = [e.reason[:30] for e in recent if not e.accepted and e.reason]
            return f"Recent rejections detected — consider energy-favorable mutations. " + (
                f"Issues: {'; '.join(rej_reasons[:2])}" if rej_reasons else ""
            )

        recent_acc = sum(1 for e in recent if e.accepted)
        if recent_acc >= 3:
            return "Recent trend is positive — moderate exploration acceptable."

        return "Mixed recent results — balance exploration with stability."

    def get_energy_worsening_iterations(self, last_k: int = 5) -> list[IterationOutcome]:
        """Return recent iterations where energy worsened (rejected)."""
        rejected = [r for r in self._iteration_records if not r.accepted]
        return rejected[-last_k:]

    # ── decay ────────────────────────────────────────────────────────────

    def apply_decay(self) -> None:
        d = self._config.decay_factor
        for rec in self._records.values():
            rec.success_count = int(rec.success_count * d)
            rec.failure_count = int(rec.failure_count * d)

    # ── helpers ──────────────────────────────────────────────────────────

    def _compute_mutation_bias(self, rec: PositionRecord) -> float:
        if rec.total_trials == 0:
            return 1.0
        rate = rec.success_rate
        cfg = self._config
        if rate >= 0.5:
            return min(cfg.success_boost, 1.0 + rate)
        return max(cfg.failure_dampen, 1.0 - (1.0 - rate) * 0.5)

    @staticmethod
    def _compute_energy_trend(recent_scores: list[float]) -> str:
        if len(recent_scores) < 2:
            return "unknown"
        deltas = [recent_scores[i] - recent_scores[i - 1] for i in range(1, len(recent_scores))]
        avg_delta = sum(deltas) / len(deltas)
        if avg_delta > 0.005:
            return "improving"
        if avg_delta < -0.005:
            return "worsening"
        return "flat"

    def to_dict(self) -> dict:
        return {pos: rec.model_dump() for pos, rec in self._records.items()}
