"""
Main orchestration engine — runs the design loop locally and dispatches
agent work to Modal (or runs locally in non-Modal mode).
"""

from __future__ import annotations

import json
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
        self._fold_engine: FoldEngine = fold_engine or DummyFoldEngine(self._fold_cfg)

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

        initial_fold = self._fold_engine.fold_and_score(current_seq, objective, output_dir, -1)
        best_score = initial_fold.combined_score
        score_history.append(best_score)

        for iteration in range(self._cfg.max_iterations):
            log_iteration_header(iteration, self._cfg.max_iterations)

            proposals = self._run_agents(current_seq, memory, objective, iteration)

            proposals = validate_proposals(proposals, objective, len(current_seq))

            if self._cfg.debug:
                log_proposals_table([p.model_dump() for p in proposals])

            candidate_seq, applied = merge_mutations(
                current_seq, proposals, self._cfg.confidence_threshold
            )

            log_mutation_summary(len(current_seq), len(applied))

            fold_result = self._fold_engine.fold_and_score(
                candidate_seq, objective, output_dir, iteration
            )

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
            else:
                memory.record_failure(applied)

            score_history.append(best_score)

            curate_memory(memory, iteration)

            stop, reason = should_stop(iteration, score_history, self._cfg)
            if stop:
                log_early_stop(reason)
                break

        final_fold = self._fold_engine.fold_and_score(
            current_seq, objective, output_dir, self._cfg.max_iterations
        )
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

    def _run_agents(
        self,
        sequence: str,
        memory: MemoryStore,
        objective: ObjectiveSpec,
        iteration: int,
    ) -> list[MutationProposal]:
        """Dispatch residue agents — locally or via Modal."""
        if self._cfg.modal_parallel:
            return self._run_agents_modal(sequence, memory, objective, iteration)
        return self._run_agents_local(sequence, memory, objective)

    def _run_agents_local(
        self,
        sequence: str,
        memory: MemoryStore,
        objective: ObjectiveSpec,
    ) -> list[MutationProposal]:
        proposals: list[MutationProposal] = []
        for pos in range(len(sequence)):
            agent_input = self._build_agent_input(sequence, pos, memory, objective)
            proposal = run_residue_agent_local(agent_input)
            proposals.append(proposal)
        return proposals

    def _run_agents_modal(
        self,
        sequence: str,
        memory: MemoryStore,
        objective: ObjectiveSpec,
        iteration: int,
    ) -> list[MutationProposal]:
        """Fan out to Modal for true parallelism."""
        from protein_swarm.modal_app.functions import run_residue_agent_remote

        inputs: list[AgentInput] = []
        for pos in range(len(sequence)):
            inputs.append(self._build_agent_input(sequence, pos, memory, objective))

        input_dicts = [inp.model_dump() for inp in inputs]

        if self._cfg.debug:
            log_debug(f"Dispatching {len(inputs)} agents to Modal (iteration {iteration})")

        results = list(run_residue_agent_remote.map(input_dicts))
        return [MutationProposal(**r) for r in results]

    def _build_agent_input(
        self,
        sequence: str,
        position: int,
        memory: MemoryStore,
        objective: ObjectiveSpec,
    ) -> AgentInput:
        llm = self._cfg.llm
        return AgentInput(
            sequence=sequence,
            position=position,
            neighbourhood_window=self._cfg.neighbourhood_window,
            memory_summary=memory.get_summary_for_position(position),
            objective=objective,
            mutation_rate=self._cfg.mutation_rate,
            random_seed=self._cfg.random_seed,
            use_llm=self._cfg.use_llm_agents,
            llm_provider=llm.provider,
            llm_model=llm.model,
            llm_api_key=llm.resolve_api_key() if self._cfg.use_llm_agents else None,
            llm_temperature=llm.temperature,
            llm_max_tokens=llm.max_tokens,
            llm_max_retries=llm.max_retries,
        )

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
