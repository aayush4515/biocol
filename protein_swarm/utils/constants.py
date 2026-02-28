"""
Bioinformatics constants used across the system.
"""

from __future__ import annotations

AMINO_ACIDS: list[str] = list("ACDEFGHIKLMNPQRSTVWY")

HYDROPHOBIC: set[str] = {"A", "V", "I", "L", "M", "F", "W", "P"}
POLAR: set[str] = {"S", "T", "Y", "H", "C", "N", "Q"}
CHARGED_POS: set[str] = {"K", "R"}
CHARGED_NEG: set[str] = {"D", "E"}
AROMATIC: set[str] = {"F", "W", "Y", "H"}

HELIX_FAVORING: set[str] = {"A", "E", "L", "M", "Q", "K", "R", "H"}
SHEET_FAVORING: set[str] = {"V", "I", "Y", "F", "W", "T"}
COIL_FAVORING: set[str] = {"G", "P", "D", "N", "S"}

AA_TO_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

PDB_ATOM_FORMAT: str = (
    "ATOM  {serial:>5d}  {name:<4s}{alt:1s}{resname:>3s} {chain:1s}{resseq:>4d}{icode:1s}"
    "   {x:>8.3f}{y:>8.3f}{z:>8.3f}{occ:>6.2f}{temp:>6.2f}          {element:>2s}{charge:>2s}\n"
)

THREE_LETTER: dict[str, str] = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}
