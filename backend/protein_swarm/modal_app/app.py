"""
Modal application definition and container images.

This module owns the single `modal.App` instance and the container images
used by all remote functions:

  - agent_image: lightweight CPU image for residue agents (heuristic or LLM)
  - fold_image:  GPU-capable image for structure prediction (ESMFold, etc.)
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
    .add_local_python_source("backend")
)

# ── GPU image for ESMFold structure prediction ────────────────────────────────

fold_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.41.0",
        "accelerate>=0.33.0",
        "safetensors>=0.4.3",
        "pydantic>=2.9.0",
        "numpy>=2.1.0",
        "biopython>=1.81",
    )
    .add_local_python_source("backend")
)
