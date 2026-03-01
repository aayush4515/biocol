"""
Memory curation agent — decides when to apply decay and prunes stale entries.

Runs once per iteration after the accept/reject decision.
"""

from __future__ import annotations

from protein_swarm.memory.memory_store import MemoryStore


def curate_memory(memory: MemoryStore, iteration: int, decay_interval: int = 10) -> None:
    """Apply periodic decay so ancient history doesn't dominate decisions."""
    if iteration > 0 and iteration % decay_interval == 0:
        memory.apply_decay()
