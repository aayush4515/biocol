"""
Modal remote functions.

Each function is a thin wrapper that deserialises inputs, calls the
pure-Python logic, and returns serialisable output.  Modal concerns
(images, secrets, GPU) stay here — business logic stays in the library.

Two fold paths:
  - run_fold_dummy_remote : CPU, DummyFoldEngine (default, no GPU cost)
  - run_fold_gpu_remote   : GPU, placeholder for ESMFold / AlphaFold
"""

from __future__ import annotations

from protein_swarm.modal_app.app import app, agent_image, fold_image
# Register GPU LLM function (modal_local) so it is deployed with the app.
from protein_swarm.modal_app import llm_inference  # noqa: F401

# ── Residue Agent (CPU, with optional LLM API key via Modal secret) ──────────

@app.function(image=agent_image, timeout=120)
def run_residue_agent_remote(agent_input_dict: dict) -> dict:
    """Execute a single residue agent on Modal infrastructure.

    The LLM API key travels inside agent_input_dict.llm_api_key (set by the
    orchestrator from CLI / env).  For extra isolation you can also create a
    Modal secret ('modal secret create openai-secret OPENAI_API_KEY=sk-...')
    and add  secrets=[modal.Secret.from_name("openai-secret")]  here.
    """
    from protein_swarm.schemas import AgentInput
    from protein_swarm.agents.residue_agent import run_residue_agent_local

    agent_input = AgentInput(**agent_input_dict)
    proposal = run_residue_agent_local(agent_input)
    return proposal.model_dump()


# ── Dummy Fold (CPU, lightweight — the default path) ─────────────────────────

@app.function(image=agent_image, timeout=300)
def run_fold_dummy_remote(
    sequence: str,
    objective_dict: dict,
    iteration: int,
) -> dict:
    """Run the DummyFoldEngine on Modal (CPU only, no GPU cost)."""
    from protein_swarm.schemas import ObjectiveSpec
    from protein_swarm.folding.fold_engine import DummyFoldEngine

    objective = ObjectiveSpec(**objective_dict)
    engine = DummyFoldEngine()
    result = engine.fold_and_score(sequence, objective, "/tmp/fold_output", iteration)
    return result.model_dump()


# ── GPU Fold (A10G, ready for ESMFold / real backends) ────────────────────────

@app.function(image=fold_image, gpu="A10G", timeout=600)
def run_fold_gpu_remote(
    sequence: str,
    objective_dict: dict,
    iteration: int,
) -> dict:
    """Run the folding engine on a GPU worker.

    Currently falls back to DummyFoldEngine.  To activate real folding:
      1. Implement an ESMFoldEngine in folding/esmfold_engine.py
      2. Replace the DummyFoldEngine import below
    """
    from protein_swarm.schemas import ObjectiveSpec
    from protein_swarm.folding.fold_engine import DummyFoldEngine

    objective = ObjectiveSpec(**objective_dict)
    engine = DummyFoldEngine()
    result = engine.fold_and_score(sequence, objective, "/tmp/fold_output", iteration)
    return result.model_dump()
