"""
Pluggable folding engine interface + MVP heuristic implementation.

The FoldEngine protocol defines the contract.  DummyFoldEngine fulfils it with
fast heuristic scoring.  Swap in AlphaFoldEngine / OmegaFoldEngine later by
implementing the same protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from protein_swarm.config import FoldingConfig, DEFAULT_FOLDING
from protein_swarm.folding.scoring import compute_objective_score, score_diversity, score_repeat_penalty
from protein_swarm.folding.structure_utils import generate_dummy_pdb
from protein_swarm.schemas import FoldResult, ObjectiveSpec


class FoldEngine(Protocol):
    """Interface every folding backend must satisfy."""

    def fold_and_score(
        self,
        sequence: str,
        objective: ObjectiveSpec,
        output_dir: str | Path,
        iteration: int,
    ) -> FoldResult: ...


class DummyFoldEngine:
    """Heuristic-only engine for rapid local iteration without GPU."""

    def __init__(self, config: FoldingConfig | None = None) -> None:
        self._config = config or DEFAULT_FOLDING

    def fold_and_score(
        self,
        sequence: str,
        objective: ObjectiveSpec,
        output_dir: str | Path,
        iteration: int,
    ) -> FoldResult:
        output_dir = Path(output_dir)
        pdb_path = generate_dummy_pdb(sequence, output_dir / f"iter_{iteration}.pdb")

        energy = self._heuristic_energy(sequence)
        obj_score = compute_objective_score(sequence, objective)

        cfg = self._config
        combined = cfg.energy_weight * energy + cfg.objective_weight * obj_score

        return FoldResult(
            pdb_path=pdb_path,
            energy=round(energy, 6),
            objective_score=round(obj_score, 6),
            combined_score=round(combined, 6),
        )

    def _heuristic_energy(self, sequence: str) -> float:
        """Pseudo-energy: higher is better (inverted from physics convention for simplicity)."""
        diversity = score_diversity(sequence)
        repeat_pen = score_repeat_penalty(sequence)
        cfg = self._config
        return diversity * (1.0 + cfg.diversity_bonus) + repeat_pen * (1.0 - cfg.repeat_penalty)
