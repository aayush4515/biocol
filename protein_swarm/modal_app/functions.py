"""
Modal remote functions.

Each function here is a thin wrapper that deserialises inputs, calls the
pure-Python logic, and returns serialisable output.  This keeps Modal
concerns isolated from business logic.
"""

from __future__ import annotations

from protein_swarm.modal_app.app import app, swarm_image


@app.function(image=swarm_image, timeout=120)
def run_residue_agent_remote(agent_input_dict: dict) -> dict:
    """Execute a single residue agent on Modal infrastructure.

    Accepts and returns plain dicts so Modal's serialisation is trivial.
    Deserialisation into Pydantic models happens inside the container.
    """
    from protein_swarm.schemas import AgentInput
    from protein_swarm.agents.residue_agent import run_residue_agent_local

    agent_input = AgentInput(**agent_input_dict)
    proposal = run_residue_agent_local(agent_input)
    return proposal.model_dump()


@app.function(image=swarm_image, timeout=300)
def run_fold_remote(sequence: str, objective_dict: dict, iteration: int) -> dict:
    """Run the folding engine on Modal (future: GPU-accelerated).

    Currently uses the DummyFoldEngine; swap for AlphaFold by changing the
    engine instantiation here.
    """
    from protein_swarm.schemas import ObjectiveSpec
    from protein_swarm.folding.fold_engine import DummyFoldEngine

    objective = ObjectiveSpec(**objective_dict)
    engine = DummyFoldEngine()
    result = engine.fold_and_score(sequence, objective, "/tmp/fold_output", iteration)
    return result.model_dump()
