"""
Modal application definition and container images.

This module owns the single `modal.App` instance and the container images
used by all remote functions:

  - agent_image: lightweight CPU image for residue agents (heuristic or LLM)
  - fold_image:  GPU-capable image for structure prediction (ESMFold, etc.)

The fold_image is defined but only consumed when a GPU folding backend is
enabled.  The DummyFoldEngine runs on agent_image to avoid pulling a heavy
GPU base image during development.
"""

from __future__ import annotations

import modal

app = modal.App("protein-swarm")

# ── CPU image for residue agents + lightweight folding (DummyFoldEngine) ──────

agent_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pydantic>=2.9.0",
        "numpy>=2.1.0",
        "openai>=1.40.0",
    )
    .add_local_python_source("protein_swarm")
)

# ── GPU image for real structure prediction (ESMFold / OmegaFold) ─────────────
# Not used in the current DummyFoldEngine path — kept ready for when a real
# folding backend is wired in.

fold_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.01-py3")
    .pip_install(
        "fair-esm",
        "biopython",
        "pydantic>=2.9.0",
        "numpy>=2.1.0",
    )
    .add_local_python_source("protein_swarm")
)
