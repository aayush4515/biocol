"""
Central configuration for the Protein Swarm Design Engine.

All tunable parameters live here. Nothing is hardcoded in business logic.
"""

from __future__ import annotations

import logging
import os

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class LLMConfig(BaseModel):
    """Configuration for LLM-backed agents and objective parsing."""

    provider: str = Field(default="openai", description="LLM provider: openai | anthropic | together")
    model: str = Field(default="gpt-4o-mini", description="Model identifier")
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
    modal_fold: bool = Field(default=False, description="Run folding engine on Modal (remote). False = fold locally.")
    modal_fold_gpu: bool = Field(default=False, description="Use GPU worker for folding on Modal (requires fold_image)")
    fold_backend: str = Field(default="dummy", description="Local folding backend: 'dummy' | 'esmfold-local'")
    remote_fold_backend: str = Field(default="esmfold", description="Remote (Modal) folding backend when modal_fold=True: 'dummy' | 'esmfold'")
    output_dir: str = Field(default="outputs", description="Directory for artefacts")


class FoldingConfig(BaseModel):
    """Configuration for the folding / scoring subsystem.

    Scoring weights (w_physics, w_objective, w_confidence) control how the
    final combined_score is computed:

        combined = w_physics * physics_score
                 + w_objective * objective_score
                 + w_confidence * confidence_score

    - physics_score:    Rosetta energy mapped to [0,1] via logistic sigmoid.
                        Falls back to confidence_score when Rosetta is disabled.
    - objective_score:  Heuristic score from sequence-level objective analysis.
    - confidence_score: ESMFold mean pLDDT normalised to [0,1].

    Rosetta normalisation:  The raw Rosetta total_score (lower = better) is
    converted to a 0-1 "goodness" value using:
        physics_score = 1 / (1 + exp((score - norm_target) / norm_scale))
    where norm_target is the "good" energy centre and norm_scale controls the
    steepness of the sigmoid.
    """

    # ── scoring weights ──────────────────────────────────────────────────
    w_physics: float = Field(default=0.55, ge=0.0, description="Weight for physics-based (Rosetta) score")
    w_objective: float = Field(default=0.35, ge=0.0, description="Weight for heuristic objective score")
    w_confidence: float = Field(default=0.10, ge=0.0, description="Weight for ESMFold pLDDT confidence")

    # ── rosetta ──────────────────────────────────────────────────────────
    use_rosetta: bool = Field(default=True, description="Run local PyRosetta scoring on predicted PDBs")
    rosetta_relax: bool = Field(default=False, description="Run FastRelax before scoring (slower, better energy)")
    rosetta_relax_cycles: int = Field(default=1, ge=1, description="Number of FastRelax cycles when relax is enabled")

    rosetta_norm_target: float = Field(
        default=-200.0,
        description="Sigmoid centre for Rosetta score normalisation (typical 'good' energy)",
    )
    rosetta_norm_scale: float = Field(
        default=50.0, gt=0.0,
        description="Sigmoid scale — smaller = steeper transition around norm_target",
    )

    # ── legacy heuristic knobs (used by DummyFoldEngine / objective scorer) ──
    diversity_bonus: float = Field(default=0.1, ge=0.0)
    helix_bonus: float = Field(default=0.15, ge=0.0)
    repeat_penalty: float = Field(default=0.2, ge=0.0)

    @model_validator(mode="after")
    def _normalise_weights(self) -> "FoldingConfig":
        total = self.w_physics + self.w_objective + self.w_confidence
        if total <= 0:
            raise ValueError("Scoring weights must sum to a positive value")
        if abs(total - 1.0) > 1e-6:
            logger.debug(
                "Scoring weights sum to %.6f — auto-normalising to 1.0", total,
            )
            self.w_physics /= total
            self.w_objective /= total
            self.w_confidence /= total
        return self


class MemoryConfig(BaseModel):
    """Configuration for the shared memory system."""

    decay_factor: float = Field(default=0.95, ge=0.0, le=1.0, description="Exponential decay applied to old records")
    success_boost: float = Field(default=1.5, ge=1.0, description="Multiplier for success-influenced mutation probability")
    failure_dampen: float = Field(default=0.5, ge=0.0, le=1.0, description="Multiplier for failure-influenced mutation probability")


DEFAULT_SWARM = SwarmConfig()
DEFAULT_FOLDING = FoldingConfig()
DEFAULT_MEMORY = MemoryConfig()
