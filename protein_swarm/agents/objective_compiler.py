"""
Compiles a natural-language design objective into a structured ObjectiveSpec.

Two paths:
  1. LLM-backed  — sends the raw text to an LLM and parses structured JSON
  2. Heuristic   — keyword matching fallback (also used if LLM fails)
"""

from __future__ import annotations

from protein_swarm.config import LLMConfig
from protein_swarm.schemas import ObjectiveSpec

_HELIX_KEYWORDS = {"helix", "helical", "alpha-helix", "alpha helix"}
_SHEET_KEYWORDS = {"sheet", "beta-sheet", "beta sheet", "strand"}
_STABILITY_KEYWORDS = {"stable", "stability", "thermostable", "robust"}
_DIVERSITY_KEYWORDS = {"diverse", "diversity", "varied", "heterogeneous"}


def compile_objective(
    raw_text: str,
    *,
    use_llm: bool = False,
    llm_config: LLMConfig | None = None,
) -> ObjectiveSpec:
    """Parse free-text objective into a machine-readable spec.

    When use_llm is True and llm_config is provided, delegates to the LLM.
    Falls back to the heuristic parser on failure or when LLM is disabled.
    """
    if use_llm and llm_config is not None:
        if llm_config.provider == "modal_local":
            return _compile_objective_heuristic(raw_text)
        return _compile_objective_llm(raw_text, llm_config)
    return _compile_objective_heuristic(raw_text)


def _compile_objective_llm(raw_text: str, llm_config: LLMConfig) -> ObjectiveSpec:
    """LLM-powered objective parsing with automatic fallback."""
    from protein_swarm.agents.llm_client import call_llm_for_objective

    return call_llm_for_objective(
        raw_text,
        api_key=llm_config.resolve_api_key(),
        provider=llm_config.provider,
        model=llm_config.model,
        temperature=0.4,
        max_tokens=llm_config.max_tokens,
        max_retries=llm_config.max_retries,
    )


def _compile_objective_heuristic(raw_text: str) -> ObjectiveSpec:
    """Keyword-based fallback parser. Also used when LLM parsing fails."""
    lower = raw_text.lower()
    tokens = set(lower.split())

    favour_helix = bool(tokens & _HELIX_KEYWORDS) or "helix" in lower
    favour_sheet = bool(tokens & _SHEET_KEYWORDS) or "sheet" in lower
    favour_stability = bool(tokens & _STABILITY_KEYWORDS) or "stable" in lower
    favour_diversity = bool(tokens & _DIVERSITY_KEYWORDS) or "divers" in lower

    constraints: list[str] = []
    if "no cysteine" in lower or "avoid cysteine" in lower:
        constraints.append("avoid_cysteine")
    if "no proline" in lower or "avoid proline" in lower:
        constraints.append("avoid_proline")

    avoid_residues: list[str] = []
    if "avoid_cysteine" in constraints:
        avoid_residues.append("C")
    if "avoid_proline" in constraints:
        avoid_residues.append("P")

    return ObjectiveSpec(
        raw_text=raw_text,
        favour_helix=favour_helix,
        favour_sheet=favour_sheet,
        favour_stability=favour_stability,
        favour_diversity=favour_diversity,
        custom_constraints=constraints,
        avoid_residues=avoid_residues,
    )
