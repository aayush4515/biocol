"""
Residue Agent — the core per-position mutation proposer.

Each agent examines its local neighbourhood, consults memory, and proposes
a single residue substitution (or no-op).

Two execution paths:
  1. Heuristic (default) — fast, deterministic, no API calls
  2. LLM-backed — sends a structured prompt to an LLM and parses JSON response
Toggle via AgentInput.use_llm.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from protein_swarm.schemas import AgentInput, MutationProposal, ObjectiveSpec, PositionMemorySummary
from protein_swarm.utils.constants import (
    AMINO_ACIDS,
    HELIX_FAVORING,
    SHEET_FAVORING,
    HYDROPHOBIC,
    POLAR,
)


def run_residue_agent_local(agent_input: AgentInput) -> MutationProposal:
    """Execute the residue agent logic for a single position.

    Routes to the LLM path when agent_input.use_llm is True, otherwise
    falls back to the deterministic heuristic path.
    """
    if agent_input.use_llm:
        return _run_llm_agent(agent_input)
    return _run_heuristic_agent(agent_input)


def _run_llm_agent(agent_input: AgentInput) -> MutationProposal:
    """LLM-backed agent: build prompt, call model, return validated proposal."""
    from protein_swarm.agents.llm_client import call_llm_for_mutation

    prompt = build_agent_prompt(agent_input)
    pos = agent_input.position
    current = agent_input.sequence[pos]

    return call_llm_for_mutation(
        prompt,
        api_key=agent_input.llm_api_key or "",
        provider=agent_input.llm_provider,
        model=agent_input.llm_model,
        temperature=agent_input.llm_temperature,
        max_tokens=agent_input.llm_max_tokens,
        max_retries=agent_input.llm_max_retries,
        position=pos,
        current_residue=current,
    )


def _run_heuristic_agent(agent_input: AgentInput) -> MutationProposal:
    """Original heuristic-based agent with deterministic scoring."""
    seq = agent_input.sequence
    pos = agent_input.position
    window = agent_input.neighbourhood_window
    mem = agent_input.memory_summary
    obj = agent_input.objective
    mutation_rate = agent_input.mutation_rate

    rng = _build_rng(agent_input.random_seed, pos)

    current = seq[pos]

    mutation_prob = mutation_rate * (mem.mutation_bias if mem else 1.0)
    if rng.random() > mutation_prob:
        return _no_op(pos, current, "Below mutation probability threshold")

    neighbourhood = _extract_neighbourhood(seq, pos, window)

    candidates = _rank_candidates(current, neighbourhood, obj, mem)
    if not candidates:
        return _no_op(pos, current, "No viable candidates")

    proposed, confidence, reason = candidates[0]

    if mem and proposed in mem.rejected_residues:
        confidence *= 0.7
        reason += " (previously rejected — confidence dampened)"

    return MutationProposal(
        position=pos,
        current_residue=current,
        proposed_residue=proposed,
        confidence=round(confidence, 4),
        reason=reason,
    )


# ── LLM prompt builder ────────────────────────────────────────────────────────

def build_agent_prompt(agent_input: AgentInput) -> str:
    """Generate a prompt string suitable for an LLM-based agent.

    Not called in the heuristic path; exists so the interface is ready for
    LLM integration without refactoring.
    """
    seq = agent_input.sequence
    pos = agent_input.position
    w = agent_input.neighbourhood_window
    start = max(0, pos - w)
    end = min(len(seq), pos + w + 1)
    local = seq[start:end]

    mem_block = ""
    if agent_input.memory_summary:
        m = agent_input.memory_summary
        mem_block = (
            f"Memory — successes: {m.success_count}, failures: {m.failure_count}, "
            f"accepted: {m.accepted_residues}, rejected: {m.rejected_residues}"
        )

    obj_block = ""
    if agent_input.objective:
        obj_block = f"Objective: {agent_input.objective.raw_text}"

    return (
        f"You are a protein residue design agent.\n"
        f"Full sequence: {seq}\n"
        f"Your position: {pos} (residue '{seq[pos]}')\n"
        f"Local window [{start}:{end}]: {local}\n"
        f"{mem_block}\n"
        f"{obj_block}\n\n"
        f"Propose a single amino acid substitution at position {pos}.\n"
        f"Respond with JSON: {{\"position\": int, \"proposed_residue\": str, "
        f"\"confidence\": float, \"reason\": str}}"
    )


# ── Heuristic helpers ─────────────────────────────────────────────────────────

def _build_rng(seed: int | None, position: int) -> random.Random:
    if seed is not None:
        combined = int(hashlib.sha256(f"{seed}-{position}".encode()).hexdigest(), 16)
        return random.Random(combined)
    return random.Random()


def _extract_neighbourhood(seq: str, pos: int, window: int) -> str:
    start = max(0, pos - window)
    end = min(len(seq), pos + window + 1)
    return seq[start:end]


def _no_op(pos: int, current: str, reason: str) -> MutationProposal:
    return MutationProposal(
        position=pos,
        current_residue=current,
        proposed_residue=current,
        confidence=0.0,
        reason=reason,
    )


def _rank_candidates(
    current: str,
    neighbourhood: str,
    objective: ObjectiveSpec | None,
    memory: PositionMemorySummary | None,
) -> list[tuple[str, float, str]]:
    """Score every possible substitution and return sorted (residue, confidence, reason)."""
    scored: list[tuple[str, float, str]] = []

    for aa in AMINO_ACIDS:
        if aa == current:
            continue
        score = 0.0
        reasons: list[str] = []

        if objective and objective.favour_helix and aa in HELIX_FAVORING:
            score += 0.4
            reasons.append("helix-favoring")

        if objective and objective.favour_sheet and aa in SHEET_FAVORING:
            score += 0.35
            reasons.append("sheet-favoring")

        if objective and objective.favour_stability:
            if current in POLAR and aa in HYDROPHOBIC:
                score += 0.25
                reasons.append("hydrophobic core packing")
            if current in HYDROPHOBIC and aa in HYDROPHOBIC:
                score += 0.15
                reasons.append("conserved hydrophobicity")

        if aa in neighbourhood:
            score -= 0.2
            reasons.append("already in neighbourhood — penalised")

        if memory:
            if aa in memory.accepted_residues:
                score += 0.2
                reasons.append("previously accepted")
            if aa in memory.rejected_residues:
                score -= 0.15
                reasons.append("previously rejected")

        if score > 0:
            scored.append((aa, min(score, 1.0), "; ".join(reasons)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
