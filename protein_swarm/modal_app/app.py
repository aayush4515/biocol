"""
Modal application definition and container images.

This module owns the single `modal.App` instance and the container images
used by all remote functions:

  - agent_image: lightweight CPU image for residue agents (heuristic or LLM)
  - llm_image:   GPU image for open-source LLM inference (modal_local)
  - fold_image:  GPU image for structure prediction (ESMFold, etc.)
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

# ── GPU image for open-source LLM inference (modal_local, no API) ──────────────

llm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "pydantic>=2.9.0",
    )
    .run_commands(
        # Pre-download tokenizer + model weights into the image at build time.
        # This avoids hitting the HF Hub on first inference.
        "python -c \"import torch; from transformers import AutoModelForCausalLM, AutoTokenizer; "
        "model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'; "
        "AutoTokenizer.from_pretrained(model_name); "
        "AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)\""
    )
    .add_local_python_source("protein_swarm")
)

# ── GPU image for real structure prediction (ESMFold / OmegaFold) ─────────────

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
