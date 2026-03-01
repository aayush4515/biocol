"""
Local PyRosetta scoring module.

Scores PDB structures using the Rosetta full-atom energy function.
Optionally runs FastRelax for energy minimisation before scoring.

This module runs LOCALLY only — never on Modal.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PYROSETTA_INITIALISED = False


def init_pyrosetta_once(extra_flags: str | None = None) -> None:
    """Initialise PyRosetta exactly once (idempotent).

    Uses quiet flags by default.  Pass extra_flags for debugging
    (e.g. "-out:level 300").
    """
    global _PYROSETTA_INITIALISED
    if _PYROSETTA_INITIALISED:
        return

    try:
        import pyrosetta
    except ImportError as e:
        raise RuntimeError(
            "PyRosetta not available locally. "
            "Install it or run with --no-rosetta."
        ) from e

    flags = "-mute all"
    if extra_flags:
        flags = f"{flags} {extra_flags}"

    pyrosetta.init(flags, silent=True)
    _PYROSETTA_INITIALISED = True
    logger.debug("PyRosetta initialised with flags: %s", flags)


def score_pdb_with_pyrosetta(
    pdb_path: str,
    relax: bool = False,
    relax_cycles: int = 1,
) -> dict:
    """Score a PDB file using Rosetta's full-atom score function.

    Args:
        pdb_path: Path to the PDB file to score.
        relax: If True, run FastRelax before scoring.
        relax_cycles: Number of FastRelax cycles (only used when relax=True).

    Returns:
        {
            "rosetta_total_score": float,   # total Rosetta energy (lower is better)
            "relaxed_pdb_path": str | None, # path to relaxed PDB if relax=True
        }
    """
    init_pyrosetta_once()

    import pyrosetta

    pose = pyrosetta.pose_from_pdb(pdb_path)
    sfxn = pyrosetta.get_fa_scorefxn()

    relaxed_pdb_path: str | None = None

    if relax:
        from pyrosetta.rosetta.protocols.relax import FastRelax

        relax_mover = FastRelax()
        relax_mover.set_scorefxn(sfxn)
        relax_mover.max_iter(relax_cycles * 200)

        for _ in range(relax_cycles):
            relax_mover.apply(pose)

        stem = Path(pdb_path).stem
        relaxed_path = Path(pdb_path).parent / f"{stem}.relaxed.pdb"
        pose.dump_pdb(str(relaxed_path))
        relaxed_pdb_path = str(relaxed_path)
        logger.debug("Relaxed PDB written to %s", relaxed_pdb_path)

    total_score = sfxn(pose)
    logger.debug("Rosetta total_score = %.2f (relax=%s)", total_score, relax)

    return {
        "rosetta_total_score": total_score,
        "relaxed_pdb_path": relaxed_pdb_path,
    }
