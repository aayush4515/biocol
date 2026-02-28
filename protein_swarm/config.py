"""
Central configuration for the Protein Swarm Design Engine.

All tunable parameters live here. Nothing is hardcoded in business logic.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for LLM-backed agents and objective parsing."""

    provider: str = Field(default="openai", description="LLM provider: openai | anthropic | together")
    model: str = Field(default="gpt-4o", description="Model identifier")
    api_key: str | None = Field(default=None, description="API key; falls back to OPENAI_API_KEY env var")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=64)
    max_retries: int = Field(default=2, ge=0, description="Retries on malformed LLM JSON responses")

    def resolve_api_key(self) -> str:
        """Return the effective API key, falling back to env vars."""
        if self.api_key:
            return self.api_key
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "together": "TOGETHER_API_KEY",
        }
        env_var = env_map.get(self.provider, "OPENAI_API_KEY")
        key = os.environ.get(env_var, "")
        if not key:
            raise ValueError(
                f"No API key provided and ${env_var} is not set. "
                f"Pass --llm-api-key or export {env_var}."
            )
        return key


class SwarmConfig(BaseModel):
    """Top-level configuration for a single design run."""

    use_llm_agents: bool = Field(default=False, description="Use LLM-based agents instead of heuristics")
    llm: LLMConfig = Field(default_factory=LLMConfig)

    max_iterations: int = Field(default=50, ge=1, description="Maximum optimisation iterations")
    mutation_rate: float = Field(default=0.3, ge=0.0, le=1.0, description="Base probability of mutating any position")
    neighbourhood_window: int = Field(default=3, ge=1, description="±positions visible to each residue agent")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum agent confidence to accept a proposal")
    plateau_window: int = Field(default=5, ge=2, description="Consecutive non-improving iterations before early stop")
    plateau_tolerance: float = Field(default=1e-4, ge=0.0, description="Minimum score delta to count as improvement")
    random_seed: int | None = Field(default=None, description="Reproducibility seed; None for non-deterministic runs")
    debug: bool = Field(default=False, description="Enable verbose debug logging")
    modal_parallel: bool = Field(default=True, description="Use Modal for distributed agent execution")
    output_dir: str = Field(default="outputs", description="Directory for artefacts")


class FoldingConfig(BaseModel):
    """Configuration for the folding / scoring subsystem."""

    energy_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    objective_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    diversity_bonus: float = Field(default=0.1, ge=0.0)
    helix_bonus: float = Field(default=0.15, ge=0.0)
    repeat_penalty: float = Field(default=0.2, ge=0.0)


class MemoryConfig(BaseModel):
    """Configuration for the shared memory system."""

    decay_factor: float = Field(default=0.95, ge=0.0, le=1.0, description="Exponential decay applied to old records")
    success_boost: float = Field(default=1.5, ge=1.0, description="Multiplier for success-influenced mutation probability")
    failure_dampen: float = Field(default=0.5, ge=0.0, le=1.0, description="Multiplier for failure-influenced mutation probability")


DEFAULT_SWARM = SwarmConfig()
DEFAULT_FOLDING = FoldingConfig()
DEFAULT_MEMORY = MemoryConfig()
