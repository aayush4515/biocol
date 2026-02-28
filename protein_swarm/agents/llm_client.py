"""
LLM client for residue agents and objective compilation.

Thin wrapper around the OpenAI SDK that handles:
- Prompt dispatch with system/user message separation
- Structured JSON output parsing with schema validation
- Retries on malformed responses
- Provider abstraction (OpenAI today, Anthropic/Together ready)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from protein_swarm.schemas import MutationProposal, ObjectiveSpec

logger = logging.getLogger(__name__)

_RESIDUE_SYSTEM_PROMPT = (
    "You are an expert protein design agent. You propose single amino acid "
    "substitutions to optimise a protein sequence toward a given objective.\n\n"
    "Rules:\n"
    "- Only propose one of the 20 standard amino acids: A C D E F G H I K L M N P Q R S T V W Y\n"
    "- Your confidence (0.0–1.0) should reflect how strongly the substitution "
    "serves the objective given the local context and memory.\n"
    "- If no beneficial mutation exists, propose the current residue with confidence 0.0.\n\n"
    "Respond with ONLY a JSON object (no markdown fences):\n"
    '{"position": <int>, "proposed_residue": "<char>", "confidence": <float>, "reason": "<str>"}'
)

_OBJECTIVE_SYSTEM_PROMPT = (
    "You are a protein design objective parser. Given a natural-language design "
    "goal, extract structured parameters.\n\n"
    "Respond with ONLY a JSON object (no markdown fences) matching this schema:\n"
    "{\n"
    '  "favour_helix": <bool>,\n'
    '  "favour_sheet": <bool>,\n'
    '  "favour_stability": <bool>,\n'
    '  "favour_diversity": <bool>,\n'
    '  "custom_constraints": [<str>, ...],\n'
    '  "target_properties": [<str>, ...],\n'
    '  "avoid_residues": [<str>, ...],\n'
    '  "structural_motifs": [<str>, ...],\n'
    '  "free_text_reasoning": "<str>"\n'
    "}\n\n"
    "custom_constraints should use tokens like: avoid_cysteine, avoid_proline, "
    "disulfide_required, max_hydrophobic_stretch_5, etc.\n"
    "target_properties examples: thermostable, soluble, membrane-spanning, protease-resistant.\n"
    "avoid_residues: single-letter amino acid codes to globally avoid.\n"
    "structural_motifs examples: coiled-coil, beta-barrel, zinc-finger, transmembrane-helix.\n"
    "free_text_reasoning: 1-2 sentence explanation of your interpretation."
)


def _build_openai_client(api_key: str, provider: str) -> OpenAI:
    """Construct an OpenAI-compatible client for the given provider."""
    base_urls = {
        "openai": None,
        "together": "https://api.together.xyz/v1",
    }
    base_url = base_urls.get(provider)
    return OpenAI(api_key=api_key, base_url=base_url)


def _call_llm_raw(
    system_prompt: str,
    user_prompt: str,
    *,
    api_key: str,
    provider: str = "openai",
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """Send a chat completion request and return the raw response text."""
    client = _build_openai_client(api_key, provider)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    return content.strip()


def _parse_json(raw: str) -> dict[str, Any]:
    """Extract JSON from LLM response, stripping markdown fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def call_llm_for_mutation(
    user_prompt: str,
    *,
    api_key: str,
    provider: str = "openai",
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 512,
    max_retries: int = 2,
    position: int,
    current_residue: str,
) -> MutationProposal:
    """Call the LLM to propose a residue mutation, with retry on parse failure."""
    last_error: Exception | None = None

    if provider == "modal_local":
        from protein_swarm.llm.providers import ModalLLMProvider

        for attempt in range(1 + max_retries):
            try:
                data = ModalLLMProvider().propose(_RESIDUE_SYSTEM_PROMPT, user_prompt)
                data["position"] = position
                if "current_residue" not in data:
                    data["current_residue"] = current_residue
                return MutationProposal(**data)
            except Exception as e:
                last_error = e
                logger.warning("Modal LLM mutation attempt %d failed: %s", attempt + 1, e)
        logger.error("Modal LLM mutation failed after %d attempts, returning no-op", 1 + max_retries)
        return MutationProposal(
            position=position,
            current_residue=current_residue,
            proposed_residue=current_residue,
            confidence=0.0,
            reason=f"Modal LLM failure after {1 + max_retries} attempts: {last_error}",
        )

    for attempt in range(1 + max_retries):
        try:
            raw = _call_llm_raw(
                _RESIDUE_SYSTEM_PROMPT,
                user_prompt,
                api_key=api_key,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            data = _parse_json(raw)

            data["position"] = position
            if "current_residue" not in data:
                data["current_residue"] = current_residue

            return MutationProposal(**data)

        except Exception as e:
            last_error = e
            logger.warning("LLM mutation parse attempt %d failed: %s", attempt + 1, e)

    logger.error("LLM mutation failed after %d attempts, returning no-op", 1 + max_retries)
    return MutationProposal(
        position=position,
        current_residue=current_residue,
        proposed_residue=current_residue,
        confidence=0.0,
        reason=f"LLM parse failure after {1 + max_retries} attempts: {last_error}",
    )


def call_llm_for_objective(
    raw_text: str,
    *,
    api_key: str,
    provider: str = "openai",
    model: str = "gpt-4o",
    temperature: float = 0.4,
    max_tokens: int = 512,
    max_retries: int = 2,
) -> ObjectiveSpec:
    """Call the LLM to parse a design objective into a structured spec."""
    last_error: Exception | None = None

    for attempt in range(1 + max_retries):
        try:
            raw = _call_llm_raw(
                _OBJECTIVE_SYSTEM_PROMPT,
                f"Design objective: {raw_text}",
                api_key=api_key,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            data = _parse_json(raw)
            data["raw_text"] = raw_text
            return ObjectiveSpec(**data)

        except Exception as e:
            last_error = e
            logger.warning("LLM objective parse attempt %d failed: %s", attempt + 1, e)

    logger.error(
        "LLM objective parsing failed after %d attempts, falling back to keyword parser",
        1 + max_retries,
    )
    from protein_swarm.agents.objective_compiler import _compile_objective_heuristic
    return _compile_objective_heuristic(raw_text)
