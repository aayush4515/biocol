# """
# Pluggable folding engine interface + MVP heuristic implementation.

# The FoldEngine protocol defines the contract.  DummyFoldEngine fulfils it with
# fast heuristic scoring.  Swap in AlphaFoldEngine / OmegaFoldEngine later by
# implementing the same protocol.
# """

# from __future__ import annotations

# from pathlib import Path
# from typing import Protocol

# from protein_swarm.config import FoldingConfig, DEFAULT_FOLDING
# from protein_swarm.folding.scoring import compute_objective_score, score_diversity, score_repeat_penalty
# from protein_swarm.folding.structure_utils import generate_dummy_pdb
# from protein_swarm.schemas import FoldResult, ObjectiveSpec


# class FoldEngine(Protocol):
#     """Interface every folding backend must satisfy."""

#     def fold_and_score(
#         self,
#         sequence: str,
#         objective: ObjectiveSpec,
#         output_dir: str | Path,
#         iteration: int,
#     ) -> FoldResult: ...


# class DummyFoldEngine:
#     """Heuristic-only engine for rapid local iteration without GPU."""

#     def __init__(self, config: FoldingConfig | None = None) -> None:
#         self._config = config or DEFAULT_FOLDING

#     def fold_and_score(
#         self,
#         sequence: str,
#         objective: ObjectiveSpec,
#         output_dir: str | Path,
#         iteration: int,
#     ) -> FoldResult:
#         output_dir = Path(output_dir)
#         pdb_path = generate_dummy_pdb(sequence, output_dir / f"iter_{iteration}.pdb")

#         energy = self._heuristic_energy(sequence)
#         obj_score = compute_objective_score(sequence, objective)

#         cfg = self._config
#         combined = cfg.energy_weight * energy + cfg.objective_weight * obj_score

#         return FoldResult(
#             pdb_path=pdb_path,
#             energy=round(energy, 6),
#             objective_score=round(obj_score, 6),
#             combined_score=round(combined, 6),
#         )

#     def _heuristic_energy(self, sequence: str) -> float:
#         """Pseudo-energy: higher is better (inverted from physics convention for simplicity)."""
#         diversity = score_diversity(sequence)
#         repeat_pen = score_repeat_penalty(sequence)
#         cfg = self._config
#         return diversity * (1.0 + cfg.diversity_bonus) + repeat_pen * (1.0 - cfg.repeat_penalty)


"""
Pluggable folding engine interface + MVP heuristic implementation.

The FoldEngine protocol defines the contract. DummyFoldEngine fulfils it with
fast heuristic scoring. Swap in AlphaFoldEngine / OmegaFoldEngine / ESMFoldEngine
later by implementing the same protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from protein_swarm.config import FoldingConfig, DEFAULT_FOLDING
from protein_swarm.folding.scoring import (
    compute_objective_score,
    score_diversity,
    score_repeat_penalty,
)
from protein_swarm.folding.structure_utils import (
    generate_dummy_pdb,
    sanitize_sequence,
    write_pdb_text,
)
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
        sequence = sanitize_sequence(sequence)

        pdb_path = generate_dummy_pdb(sequence, output_dir / f"iter_{iteration}.pdb")

        energy = self._heuristic_energy(sequence)
        obj_score = compute_objective_score(sequence, objective)

        cfg = self._config
        combined = cfg.w_physics * energy + cfg.w_objective * obj_score

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


class ESMFoldEngine:
    """
    ESMFold-backed engine.

    - Produces a real PDB using ESMFold (via transformers).
    - Uses mean pLDDT as a proxy for "energy" (higher confidence => better).
    - Keeps your objective scoring unchanged.
    """

    def __init__(
        self,
        config: FoldingConfig | None = None,
        model_name: str = "facebook/esmfold_v1",
        device: str | None = None,
        use_fp16_if_cuda: bool = True,
    ) -> None:
        self._config = config or DEFAULT_FOLDING
        self._model_name = model_name
        self._device_override = device
        self._use_fp16_if_cuda = use_fp16_if_cuda

        self._model = None
        self._tokenizer = None
        self._torch = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding

        self._torch = torch

        tok = AutoTokenizer.from_pretrained(self._model_name)
        model = EsmForProteinFolding.from_pretrained(self._model_name)

        # Pick device
        if self._device_override:
            device = self._device_override
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = model.to(device)
        model.eval()

        # Optional: fp16 on GPU for speed/memory
        if device.startswith("cuda") and self._use_fp16_if_cuda:
            model = model.half()

        self._tokenizer = tok
        self._model = model
        self._device = device

    def fold_and_score(
        self,
        sequence: str,
        objective: ObjectiveSpec,
        output_dir: str | Path,
        iteration: int,
    ) -> FoldResult:
        self._lazy_load()

        output_dir = Path(output_dir)
        sequence = sanitize_sequence(sequence)

        # 1) Fold with ESMFold
        pdb_text, mean_plddt = self._predict_pdb_and_confidence(sequence)

        # 2) Write PDB
        pdb_path = write_pdb_text(pdb_text, output_dir / f"iter_{iteration}.pdb")

        # 3) Score objective (same as before)
        obj_score = compute_objective_score(sequence, objective)

        # 4) Energy proxy: map confidence [0..100] -> [0..1]
        # Higher is better (consistent with DummyFoldEngine's "higher energy is better" convention)
        energy = float(mean_plddt) / 100.0

        cfg = self._config
        combined = cfg.w_physics * energy + cfg.w_objective * obj_score

        return FoldResult(
            pdb_path=pdb_path,
            energy=round(energy, 6),
            objective_score=round(obj_score, 6),
            combined_score=round(combined, 6),
        )

    def _predict_pdb_and_confidence(self, sequence: str) -> tuple[str, float]:
        """
        Returns:
          pdb_text: PDB as text
          mean_plddt: average confidence [0..100]
        """
        torch = self._torch
        assert torch is not None
        assert self._model is not None
        assert self._tokenizer is not None

        device = self._device

        # Tokenize sequence
        inputs = self._tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

            # plddt is typically shape [L] or [1, L]
            plddt = getattr(outputs, "plddt", None)
            if plddt is None:
                # If not available, fall back to a neutral confidence.
                mean_plddt = 50.0
            else:
                p = plddt.detach().float()
                mean_plddt = float(p.mean().item())

            # Many versions expose a convenience method to get pdb:
            # Some provide outputs as atom37 coords; transformers provides model.output_to_pdb(...)
            if hasattr(self._model, "output_to_pdb"):
                pdbs = self._model.output_to_pdb(outputs)
                pdb_text = pdbs[0] if isinstance(pdbs, list) else str(pdbs)
            else:
                # Last-resort fallback: try "outputs.pdb" field if present.
                maybe_pdb = getattr(outputs, "pdb", None)
                if maybe_pdb is None:
                    raise RuntimeError(
                        "Unable to extract PDB from ESMFold outputs. "
                        "Your transformers version may not support output_to_pdb()."
                    )
                pdb_text = str(maybe_pdb)

        return pdb_text, mean_plddt

class ESMFoldRemoteEngine:
    """
    Calls ESMFold on Modal GPU and returns FoldResult.
    The orchestrator runs locally; only folding happens remotely.
    """

    def __init__(self, config: FoldingConfig | None = None) -> None:
        self._config = config or DEFAULT_FOLDING

        # Import here so local CPU-only dev doesn't require modal at import time
        from protein_swarm.modal_app.functions import run_esmfold

        # Modal function handle
        self._run_esmfold = run_esmfold

    def fold_and_score(
        self,
        sequence: str,
        objective: ObjectiveSpec,
        output_dir: str | Path,
        iteration: int,
    ) -> FoldResult:
        output_dir = Path(output_dir)
        sequence = sanitize_sequence(sequence)

        # Remote GPU fold
        result = self._run_esmfold.remote(sequence)
        pdb_text = result["pdb"]
        mean_plddt = float(result["mean_plddt"])

        pdb_path = write_pdb_text(pdb_text, output_dir / f"iter_{iteration}.pdb")

        obj_score = compute_objective_score(sequence, objective)

        # Energy proxy: confidence normalized to [0..1]
        energy = mean_plddt / 100.0

        cfg = self._config
        combined = cfg.w_physics * energy + cfg.w_objective * obj_score

        return FoldResult(
            pdb_path=pdb_path,
            energy=round(energy, 6),
            objective_score=round(obj_score, 6),
            combined_score=round(combined, 6),
        )