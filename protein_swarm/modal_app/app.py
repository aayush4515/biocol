"""
Modal application definition and shared image.

This module owns the single `modal.App` instance and the container image
used by all remote functions.
"""

from __future__ import annotations

import modal

app = modal.App("protein-swarm")

swarm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pydantic>=2.9.0",
        "numpy>=2.1.0",
        "openai>=1.40.0",
    )
    .add_local_python_source("protein_swarm")
)
