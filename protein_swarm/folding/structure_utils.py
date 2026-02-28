"""
Utilities for generating PDB structure files.

Generates a synthetic alpha-carbon trace for MVP scoring.  A real backend
(AlphaFold, OmegaFold, ESMFold) replaces only this module.
"""

from __future__ import annotations

import math
from pathlib import Path

from protein_swarm.utils.constants import THREE_LETTER, PDB_ATOM_FORMAT


def generate_dummy_pdb(sequence: str, output_path: str | Path) -> str:
    """Write a fake alpha-carbon backbone PDB for the given sequence.

    Places C-alpha atoms along an ideal alpha-helix backbone with 3.8 A rise
    and 100 degree rotation per residue.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rise_per_residue = 1.5  # angstroms along helix axis
    radius = 2.3
    angle_per_residue = math.radians(100)

    lines: list[str] = []
    for i, aa in enumerate(sequence):
        theta = i * angle_per_residue
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        z = i * rise_per_residue

        line = PDB_ATOM_FORMAT.format(
            serial=i + 1,
            name="CA",
            alt=" ",
            resname=THREE_LETTER.get(aa, "UNK"),
            chain="A",
            resseq=i + 1,
            icode=" ",
            x=x,
            y=y,
            z=z,
            occ=1.00,
            temp=0.00,
            element="C",
            charge="  ",
        )
        lines.append(line)

    lines.append("END\n")
    output_path.write_text("".join(lines))
    return str(output_path)
