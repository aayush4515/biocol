"""
Pydantic schemas for all data flowing through the system.

Every inter-module boundary uses a typed schema — no raw dicts in production paths.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Structure Context (computed from PDB per position) ─────────────────────────

class SpatialNeighbor(BaseModel):
    """A residue that is spatially close (within Å cutoff) to the query position."""
    position: int
    residue: str
    distance: float


class StructureContext(BaseModel):
    """Per-position structural context derived from the latest accepted PDB."""

    secondary_structure: str = Field(default="UNKNOWN", description="DSSP label: H/E/C/T/S/G/B/I or UNKNOWN")
    linear_neighbors_n: str = Field(default="", description="N-terminal linear neighbors (up to 2 residues)")
    linear_neighbors_c: str = Field(default="", description="C-terminal linear neighbors (up to 2 residues)")
    spatial_neighbors: list[SpatialNeighbor] = Field(default_factory=list, description="Residues within distance cutoff")
    contact_density: int = Field(default=0, description="Number of Cα contacts within cutoff")
    avg_local_distance: float = Field(default=0.0, description="Mean Cα distance to ±window residues (compactness)")
    std_local_distance: float = Field(default=0.0, description="Std of Cα distances (flexibility proxy)")
    region_guess: str = Field(default="unknown", description="Heuristic: buried/surface/helical-like/loop-like")


# ── Memory Schemas ─────────────────────────────────────────────────────────────

class PositionMutationEvent(BaseModel):
    """A single mutation event at a specific position, stored in memory."""
    iteration: int
    position: int
    from_res: str
    to_res: str
    accepted: bool
    reason: str = ""
    combined_score: float | None = None
    objective_score: float | None = None
    physics_score: float | None = None
    rosetta_total_score: float | None = None
    design_goal_score: float | None = None
    num_mutations_in_iteration: int | None = None


class GlobalMemoryStats(BaseModel):
    """Aggregate stats across all iterations, computed once per iteration."""
    total_iterations: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    acceptance_rate: float = 0.0
    recent_acceptance_rate: float = 0.0
    energy_trend: str = Field(default="unknown", description="improving / flat / worsening / unknown")
    recent_scores: list[float] = Field(default_factory=list, description="Last K combined scores (accepted only)")


# ── Goal Evaluation ────────────────────────────────────────────────────────────

class GoalEvaluation(BaseModel):
    """Design goal evaluation summary for prompt context."""
    goal_score: float = Field(default=0.0, description="Goal achievement score 0-100")
    rating: str = Field(default="UNKNOWN", description="POOR / OK / GOOD")
    key_aspects: dict[str, float] = Field(default_factory=dict, description="Sub-scores")
    recommendations: list[str] = Field(default_factory=list)


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

    # ── paper-style context (populated by orchestrator) ──────────────────
    structure_context: StructureContext | None = None
    global_memory_stats: GlobalMemoryStats | None = None
    position_history: list[PositionMutationEvent] = Field(default_factory=list)
    neighborhood_history: list[PositionMutationEvent] = Field(default_factory=list)
    goal_evaluation: GoalEvaluation | None = None
    iteration: int = 0
    dump_prompt: bool = False


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
    target_properties: list[str] = Field(default_factory=list)
    avoid_residues: list[str] = Field(default_factory=list)
    structural_motifs: list[str] = Field(default_factory=list)
    free_text_reasoning: str = Field(default="")


# ── Folding / Scoring ─────────────────────────────────────────────────────────

class FoldResult(BaseModel):
    """Output of a folding + scoring pass."""

    pdb_path: str
    energy: float
    objective_score: float
    combined_score: float
    rosetta_total_score: float | None = None


# ── Memory ─────────────────────────────────────────────────────────────────────

class PositionMemorySummary(BaseModel):
    """Per-position memory snapshot sent to agents."""

    position: int
    success_count: int = 0
    failure_count: int = 0
    accepted_residues: list[str] = Field(default_factory=list)
    rejected_residues: list[str] = Field(default_factory=list)
    mutation_bias: float = Field(default=1.0)


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


# Forward-reference resolution
AgentInput.model_rebuild()
