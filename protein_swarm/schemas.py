"""
Pydantic schemas for all data flowing through the system.

Every inter-module boundary uses a typed schema — no raw dicts in production paths.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Agent I/O ──────────────────────────────────────────────────────────────────

class AgentInput(BaseModel):
    """Payload sent to each residue agent."""

    sequence: str
    position: int
    neighbourhood_window: int = 3
    memory_summary: PositionMemorySummary | None = None
    objective: ObjectiveSpec | None = None
    mutation_rate: float = 0.3
    random_seed: int | None = None

    use_llm: bool = False
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    llm_api_key: str | None = None
    llm_temperature: float = 0.7
    llm_max_tokens: int = 512
    llm_max_retries: int = 2


class MutationProposal(BaseModel):
    """Single residue-level mutation proposed by an agent."""

    position: int
    current_residue: str
    proposed_residue: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


# ── Objective ──────────────────────────────────────────────────────────────────

class ObjectiveSpec(BaseModel):
    """Parsed design objective that agents can consume."""

    raw_text: str
    favour_helix: bool = False
    favour_sheet: bool = False
    favour_stability: bool = True
    favour_diversity: bool = False
    custom_constraints: list[str] = Field(default_factory=list)
    target_properties: list[str] = Field(default_factory=list, description="e.g. thermostable, soluble, membrane-spanning")
    avoid_residues: list[str] = Field(default_factory=list, description="Single-letter AA codes to globally avoid")
    structural_motifs: list[str] = Field(default_factory=list, description="e.g. coiled-coil, beta-barrel, zinc-finger")
    free_text_reasoning: str = Field(default="", description="LLM chain-of-thought explaining the parsed objective")


# ── Folding / Scoring ─────────────────────────────────────────────────────────

class FoldResult(BaseModel):
    """Output of a folding + scoring pass."""

    pdb_path: str
    energy: float
    objective_score: float
    combined_score: float


# ── Memory ─────────────────────────────────────────────────────────────────────

class PositionMemorySummary(BaseModel):
    """Per-position memory snapshot sent to agents."""

    position: int
    success_count: int = 0
    failure_count: int = 0
    accepted_residues: list[str] = Field(default_factory=list)
    rejected_residues: list[str] = Field(default_factory=list)
    mutation_bias: float = Field(default=1.0, description="Multiplier on mutation probability")


# ── Iteration Record ──────────────────────────────────────────────────────────

class IterationRecord(BaseModel):
    """Full snapshot of one optimisation iteration."""

    iteration: int
    sequence_before: str
    sequence_after: str
    mutations: list[MutationProposal]
    fold_result: FoldResult
    accepted: bool
    score_delta: float


class DesignResult(BaseModel):
    """Final output of a complete design run."""

    initial_sequence: str
    final_sequence: str
    objective: str
    best_score: float
    total_iterations: int
    history: list[IterationRecord]
    final_pdb_path: str


# Forward-reference resolution (PositionMemorySummary referenced before definition)
AgentInput.model_rebuild()
