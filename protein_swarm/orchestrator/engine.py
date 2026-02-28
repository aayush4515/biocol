"""
Main orchestration engine — runs the design loop locally and dispatches
agent work to Modal (or runs locally in non-Modal mode).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from protein_swarm.agents.constraint_guard import validate_proposals
from protein_swarm.agents.memory_curator import curate_memory
from protein_swarm.agents.objective_compiler import compile_objective
from protein_swarm.agents.residue_agent import run_residue_agent_local
from protein_swarm.config import SwarmConfig, FoldingConfig, MemoryConfig
from protein_swarm.folding.fold_engine import DummyFoldEngine, FoldEngine
from protein_swarm.memory.memory_store import MemoryStore
from protein_swarm.orchestrator.decision import should_accept, should_stop
from protein_swarm.orchestrator.mutation_merge import merge_mutations
from protein_swarm.schemas import (
    AgentInput,
    DesignResult,
    FoldResult,
    IterationRecord,
    MutationProposal,
    ObjectiveSpec,
)
from protein_swarm.utils.logging import (
    log_debug,
    log_early_stop,
    log_final_result,
    log_iteration_header,
    log_mutation_summary,
    log_proposals_table,
    log_score_delta,
)


def _rosetta_to_physics_score(
    rosetta_total: float,
    norm_target: float,
    norm_scale: float,
) -> float:
    """Convert Rosetta energy (lower=better) to a 0-1 score (higher=better)."""
    return 1.0 / (1.0 + math.exp((rosetta_total - norm_target) / norm_scale))


class DesignEngine:
    """Orchestrates the full multi-agent design loop."""

    def __init__(
        self,
        swarm_config: SwarmConfig | None = None,
        folding_config: FoldingConfig | None = None,
        memory_config: MemoryConfig | None = None,
        fold_engine: FoldEngine | None = None,
    ) -> None:
        self._cfg = swarm_config or SwarmConfig()
        self._fold_cfg = folding_config or FoldingConfig()
        self._mem_cfg = memory_config or MemoryConfig()
        self._fold_engine: FoldEngine = fold_engine or self._make_fold_engine()

    def _make_fold_engine(self) -> FoldEngine:
        backend = self._cfg.fold_backend

        if backend == "dummy":
            return DummyFoldEngine(self._fold_cfg)

        if backend == "esmfold-local":
            from protein_swarm.folding.fold_engine import ESMFoldEngine
            return ESMFoldEngine(self._fold_cfg)

        raise ValueError(f"Unknown fold_backend='{backend}'")

    # ── throttling helpers ───────────────────────────────────────────────

    def _effective_mutation_rate(self, consecutive_rejects: int) -> float:
        """Decay mutation rate after repeated rejections."""
        cfg = self._cfg
        if consecutive_rejects < cfg.reject_throttle_after:
            return cfg.mutation_rate
        rate = cfg.mutation_rate * (cfg.reject_mutation_decay ** consecutive_rejects)
        return max(rate, cfg.min_mutation_rate)

    def _effective_conf_threshold(self, consecutive_rejects: int) -> float:
        """Raise confidence bar after repeated rejections."""
        cfg = self._cfg
        if consecutive_rejects < cfg.reject_throttle_after:
            return cfg.confidence_threshold
        threshold = cfg.confidence_threshold + cfg.reject_conf_bump * consecutive_rejects
        return min(threshold, 0.99)

    # ── main loop ────────────────────────────────────────────────────────

    def run(self, sequence: str, objective_text: str) -> DesignResult:
        """Execute the full design loop and return the result."""
        objective = compile_objective(
            objective_text,
            use_llm=self._cfg.use_llm_agents,
            llm_config=self._cfg.llm if self._cfg.use_llm_agents else None,
        )
        memory = MemoryStore(len(sequence), self._mem_cfg)
        output_dir = Path(self._cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        current_seq = sequence
        best_score = float("-inf")
        score_history: list[float] = []
        history: list[IterationRecord] = []
        consecutive_rejects = 0

        initial_fold = self._fold(current_seq, objective, output_dir, -1)
        best_score = initial_fold.combined_score
        score_history.append(best_score)

        for iteration in range(self._cfg.max_iterations):
            log_iteration_header(iteration, self._cfg.max_iterations)

            eff_mutation_rate = self._effective_mutation_rate(consecutive_rejects)
            eff_conf_threshold = self._effective_conf_threshold(consecutive_rejects)

            if self._cfg.debug:
                log_debug(
                    f"Throttle: consecutive_rejects={consecutive_rejects}  "
                    f"eff_mutation_rate={eff_mutation_rate:.4f}  "
                    f"eff_conf_threshold={eff_conf_threshold:.4f}"
                )

            proposals = self._run_agents(
                current_seq, memory, objective, iteration,
                mutation_rate_override=eff_mutation_rate,
            )

            proposals = validate_proposals(proposals, objective, len(current_seq))

            if self._cfg.debug:
                log_proposals_table([p.model_dump() for p in proposals])

            candidate_seq, applied = merge_mutations(
                current_seq, proposals, eff_conf_threshold,
            )

            log_mutation_summary(len(current_seq), len(applied))

            fold_result = self._fold(candidate_seq, objective, output_dir, iteration)

            accepted = should_accept(fold_result.combined_score, best_score)
            score_delta = fold_result.combined_score - best_score
            log_score_delta(fold_result.combined_score, best_score, accepted)

            record = IterationRecord(
                iteration=iteration,
                sequence_before=current_seq,
                sequence_after=candidate_seq,
                mutations=applied,
                fold_result=fold_result,
                accepted=accepted,
                score_delta=score_delta,
            )
            history.append(record)

            if accepted:
                current_seq = candidate_seq
                best_score = fold_result.combined_score
                memory.record_success(applied)
                consecutive_rejects = 0
            else:
                memory.record_failure(applied)
                consecutive_rejects += 1

            score_history.append(best_score)

            curate_memory(memory, iteration)

            stop, reason = should_stop(iteration, score_history, self._cfg)
            if stop:
                log_early_stop(reason)
                break

        final_fold = self._fold(current_seq, objective, output_dir, self._cfg.max_iterations)
        log_final_result(current_seq, best_score, final_fold.pdb_path)

        result = DesignResult(
            initial_sequence=sequence,
            final_sequence=current_seq,
            objective=objective_text,
            best_score=best_score,
            total_iterations=len(history),
            history=history,
            final_pdb_path=final_fold.pdb_path,
        )

        self._save_artefacts(result, output_dir)
        return result

    # ── agent dispatch ───────────────────────────────────────────────────

    def _run_agents(
        self,
        sequence: str,
        memory: MemoryStore,
        objective: ObjectiveSpec,
        iteration: int,
        *,
        mutation_rate_override: float | None = None,
    ) -> list[MutationProposal]:
        if self._cfg.modal_parallel:
            return self._run_agents_modal(
                sequence, memory, objective, iteration,
                mutation_rate_override=mutation_rate_override,
            )
        return self._run_agents_local(
            sequence, memory, objective,
            mutation_rate_override=mutation_rate_override,
        )

    def _run_agents_local(
        self,
        sequence: str,
        memory: MemoryStore,
        objective: ObjectiveSpec,
        *,
        mutation_rate_override: float | None = None,
    ) -> list[MutationProposal]:
        proposals: list[MutationProposal] = []
        for pos in range(len(sequence)):
            agent_input = self._build_agent_input(
                sequence, pos, memory, objective,
                mutation_rate_override=mutation_rate_override,
            )
            proposal = run_residue_agent_local(agent_input)
            proposals.append(proposal)
        return proposals

    def _run_agents_modal(
        self,
        sequence: str,
        memory: MemoryStore,
        objective: ObjectiveSpec,
        iteration: int,
        *,
        mutation_rate_override: float | None = None,
    ) -> list[MutationProposal]:
        import modal

        agent_fn = modal.Function.from_name("protein-swarm", "run_residue_agent_remote")

        inputs: list[AgentInput] = []
        for pos in range(len(sequence)):
            inputs.append(self._build_agent_input(
                sequence, pos, memory, objective,
                mutation_rate_override=mutation_rate_override,
            ))

        input_dicts = [inp.model_dump() for inp in inputs]

        if self._cfg.debug:
            log_debug(f"Dispatching {len(inputs)} agents to Modal (iteration {iteration})")

        try:
            results = list(agent_fn.map(input_dicts))
        except Exception as e:
            raise RuntimeError(
                "Failed to call Modal agents. Make sure the app is deployed:\n"
                "  modal deploy protein_swarm/modal_app/functions.py\n"
                "Or run fully local with --no-modal"
            ) from e
        return [MutationProposal(**r) for r in results]

    def _build_agent_input(
        self,
        sequence: str,
        position: int,
        memory: MemoryStore,
        objective: ObjectiveSpec,
        *,
        mutation_rate_override: float | None = None,
    ) -> AgentInput:
        llm = self._cfg.llm
        return AgentInput(
            sequence=sequence,
            position=position,
            neighbourhood_window=self._cfg.neighbourhood_window,
            memory_summary=memory.get_summary_for_position(position),
            objective=objective,
            mutation_rate=mutation_rate_override if mutation_rate_override is not None else self._cfg.mutation_rate,
            random_seed=self._cfg.random_seed,
            use_llm=self._cfg.use_llm_agents,
            llm_provider=llm.provider,
            llm_model=llm.model,
            llm_api_key=llm.resolve_api_key() if self._cfg.use_llm_agents else None,
            llm_temperature=llm.temperature,
            llm_max_tokens=llm.max_tokens,
            llm_max_retries=llm.max_retries,
        )

    # ── folding / scoring ────────────────────────────────────────────────

    def _fold(
        self,
        sequence: str,
        objective: ObjectiveSpec,
        output_dir: Path,
        iteration: int,
    ) -> FoldResult:
        """Run the fold engine locally or on Modal depending on config."""
        if self._cfg.modal_fold:
            return self._fold_modal(sequence, objective, iteration)
        return self._fold_local(sequence, objective, output_dir, iteration)

    def _fold_local(
        self,
        sequence: str,
        objective: ObjectiveSpec,
        output_dir: Path,
        iteration: int,
    ) -> FoldResult:
        """Fold locally and optionally run Rosetta scoring."""
        result = self._fold_engine.fold_and_score(sequence, objective, output_dir, iteration)

        if not self._fold_cfg.use_rosetta:
            return result

        return self._rescore_with_rosetta(
            pdb_path=result.pdb_path,
            sequence=sequence,
            objective=objective,
            mean_plddt=result.energy * 100.0,
        )

    def _fold_modal(
        self,
        sequence: str,
        objective: ObjectiveSpec,
        iteration: int,
    ) -> FoldResult:
        """Fold on Modal (ESMFold GPU) then score locally with Rosetta + heuristics."""
        import modal

        from protein_swarm.folding.scoring import compute_objective_score
        from protein_swarm.folding.structure_utils import sanitize_sequence, write_pdb_text

        sequence = sanitize_sequence(sequence)

        remote_backend = getattr(self._cfg, "remote_fold_backend", "esmfold")

        if remote_backend == "esmfold":
            fn_name = "run_esmfold"
        elif remote_backend == "dummy":
            fn_name = "run_fold_dummy_remote"
        else:
            raise ValueError(f"Unknown remote_fold_backend='{remote_backend}'")

        fold_fn = modal.Function.from_name("protein-swarm", fn_name)

        if self._cfg.debug:
            log_debug(f"Folding on Modal backend='{remote_backend}' (iteration {iteration})")

        try:
            if remote_backend == "esmfold":
                result = fold_fn.remote(sequence)
                pdb_text = result["pdb"]
                mean_plddt = float(result["mean_plddt"])

                output_dir = Path(self._cfg.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                pdb_path = write_pdb_text(pdb_text, output_dir / f"iter_{iteration}.pdb")

                return self._rescore_with_rosetta(
                    pdb_path=pdb_path,
                    sequence=sequence,
                    objective=objective,
                    mean_plddt=mean_plddt,
                )

            objective_dict = objective.model_dump()
            result_dict = fold_fn.remote(sequence, objective_dict, iteration)
            return FoldResult(**result_dict)

        except Exception as e:
            raise RuntimeError(
                f"Failed to call Modal fold function '{fn_name}' ({type(e).__name__}: {e}).\n"
                "Make sure the app is deployed: modal deploy protein_swarm/modal_app/functions.py\n"
                "Or remove --modal-fold to fold locally."
            ) from e

    def _rescore_with_rosetta(
        self,
        pdb_path: str,
        sequence: str,
        objective: ObjectiveSpec,
        mean_plddt: float,
    ) -> FoldResult:
        """Compute the 3-component combined score."""
        from protein_swarm.folding.scoring import compute_objective_score

        cfg = self._fold_cfg
        confidence_score = mean_plddt / 100.0
        obj_score = compute_objective_score(sequence, objective)

        if cfg.use_rosetta:
            from protein_swarm.folding.rosetta_energy import score_pdb_with_pyrosetta

            rosetta_result = score_pdb_with_pyrosetta(
                pdb_path,
                relax=cfg.rosetta_relax,
                relax_cycles=cfg.rosetta_relax_cycles,
            )
            rosetta_total = rosetta_result["rosetta_total_score"]
            physics_score = _rosetta_to_physics_score(
                rosetta_total, cfg.rosetta_norm_target, cfg.rosetta_norm_scale,
            )

            if rosetta_result["relaxed_pdb_path"]:
                pdb_path = rosetta_result["relaxed_pdb_path"]

            if self._cfg.debug:
                log_debug(
                    f"  pLDDT={mean_plddt:.1f}  confidence={confidence_score:.4f}  "
                    f"rosetta={rosetta_total:.1f}  physics={physics_score:.4f}  "
                    f"objective={obj_score:.4f}"
                )
        else:
            physics_score = confidence_score
            if self._cfg.debug:
                log_debug(
                    f"  pLDDT={mean_plddt:.1f}  confidence={confidence_score:.4f}  "
                    f"objective={obj_score:.4f}  (rosetta disabled, physics=confidence)"
                )

        combined = (
            cfg.w_physics * physics_score
            + cfg.w_objective * obj_score
            + cfg.w_confidence * confidence_score
        )

        if self._cfg.debug:
            log_debug(
                f"  combined = {cfg.w_physics:.2f}*{physics_score:.4f} "
                f"+ {cfg.w_objective:.2f}*{obj_score:.4f} "
                f"+ {cfg.w_confidence:.2f}*{confidence_score:.4f} "
                f"= {combined:.4f}"
            )

        return FoldResult(
            pdb_path=pdb_path,
            energy=round(physics_score, 6),
            objective_score=round(obj_score, 6),
            combined_score=round(combined, 6),
        )

    # ── artefact persistence ─────────────────────────────────────────────

    @staticmethod
    def _save_artefacts(result: DesignResult, output_dir: Path) -> None:
        (output_dir / "final_sequence.txt").write_text(result.final_sequence)

        import shutil
        final_pdb_dest = output_dir / "final_structure.pdb"
        if Path(result.final_pdb_path).exists():
            shutil.copy2(result.final_pdb_path, final_pdb_dest)

        metrics = {
            "initial_sequence": result.initial_sequence,
            "final_sequence": result.final_sequence,
            "objective": result.objective,
            "best_score": result.best_score,
            "total_iterations": result.total_iterations,
            "final_pdb_path": str(final_pdb_dest),
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        history_data = [rec.model_dump() for rec in result.history]
        (output_dir / "history.json").write_text(json.dumps(history_data, indent=2, default=str))
