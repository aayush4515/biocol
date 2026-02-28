"""
PDB structure analysis: distance matrix, DSSP, spatial neighbors, compactness.

All computations run locally on the orchestrator before dispatching agents.
Outputs are lightweight JSON-serializable summaries (no numpy arrays cross-boundary).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np

from protein_swarm.schemas import SpatialNeighbor, StructureContext

logger = logging.getLogger(__name__)

_THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


# ── Distance matrix from PDB ────────────────────────────────────────────────

def compute_distance_matrix_from_pdb(
    pdb_path: str,
) -> tuple[list[tuple[int, str]], np.ndarray]:
    """Parse Cα atoms from a PDB and compute pairwise Euclidean distances.

    Returns:
        positions: list of (residue_index_0based, single_letter_aa)
        dist_matrix: NxN float array of pairwise Cα distances in Å
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_path)
    model = structure[0]

    ca_coords: list[np.ndarray] = []
    positions: list[tuple[int, str]] = []

    for chain in model:
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue
            if "CA" not in res:
                continue
            aa = _THREE_TO_ONE.get(res.get_resname(), "X")
            coord = res["CA"].get_vector().get_array()
            positions.append((len(positions), aa))
            ca_coords.append(np.array(coord, dtype=np.float64))

    n = len(ca_coords)
    if n == 0:
        return positions, np.zeros((0, 0), dtype=np.float64)

    coords = np.stack(ca_coords)  # (N, 3)
    diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 3)
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))  # (N, N)

    return positions, dist_matrix


# ── Spatial neighbors ────────────────────────────────────────────────────────

def spatial_neighbors_for_position(
    positions: list[tuple[int, str]],
    dist_matrix: np.ndarray,
    idx: int,
    cutoff: float = 8.0,
    top_k: int | None = 8,
) -> list[SpatialNeighbor]:
    """Return residues within `cutoff` Å of position `idx`, sorted by distance."""
    n = dist_matrix.shape[0]
    if idx < 0 or idx >= n:
        return []

    neighbors: list[SpatialNeighbor] = []
    for j in range(n):
        if j == idx:
            continue
        d = float(dist_matrix[idx, j])
        if d <= cutoff:
            _, aa = positions[j]
            neighbors.append(SpatialNeighbor(position=j, residue=aa, distance=round(d, 2)))

    neighbors.sort(key=lambda x: x.distance)
    if top_k is not None:
        neighbors = neighbors[:top_k]
    return neighbors


# ── Structural summaries (per-position) ──────────────────────────────────────

def compute_structural_summaries(
    dist_matrix: np.ndarray,
    idx: int,
    window: int = 5,
    cutoff: float = 8.0,
) -> tuple[float, float, int]:
    """Compute compactness/flexibility proxies for a position.

    Returns (avg_local_distance, std_local_distance, contact_density).
    """
    n = dist_matrix.shape[0]
    if n == 0 or idx < 0 or idx >= n:
        return 0.0, 0.0, 0

    start = max(0, idx - window)
    end = min(n, idx + window + 1)
    local_dists = [float(dist_matrix[idx, j]) for j in range(start, end) if j != idx]

    avg_dist = float(np.mean(local_dists)) if local_dists else 0.0
    std_dist = float(np.std(local_dists)) if local_dists else 0.0

    contact_density = int(np.sum(dist_matrix[idx] <= cutoff)) - 1  # exclude self

    return round(avg_dist, 2), round(std_dist, 2), max(contact_density, 0)


def _guess_region(
    ss_label: str,
    contact_density: int,
    avg_dist: float,
) -> str:
    """Heuristic region classification from structural stats."""
    if ss_label in ("H", "G", "I"):
        return "helical"
    if ss_label in ("E", "B"):
        return "sheet/strand"
    if contact_density >= 12:
        return "buried"
    if contact_density <= 5:
        return "surface-exposed"
    if avg_dist > 10.0:
        return "loop-like"
    return "intermediate"


# ── DSSP secondary structure ─────────────────────────────────────────────────

def dssp_secondary_structure(pdb_path: str) -> dict[int, str]:
    """Run DSSP on a PDB and return {0-based_residue_index -> DSSP label}.

    DSSP labels: H=helix, E=strand, C/T/S/G/B/I=coil variants.
    Returns empty dict (with warning) if DSSP binary or Bio.PDB.DSSP unavailable.
    """
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.DSSP import DSSP
    except ImportError:
        logger.warning("BioPython DSSP module not available — skipping secondary structure")
        return {}

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("struct", pdb_path)
        model = structure[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dssp = DSSP(model, pdb_path, dssp="mkdssp")

        ss_map: dict[int, str] = {}
        for i, key in enumerate(dssp.keys()):
            ss_label = dssp[key][2]  # secondary structure label
            ss_map[i] = ss_label if ss_label != "-" else "C"
        return ss_map

    except FileNotFoundError:
        logger.warning("mkdssp binary not found — install DSSP for secondary structure. Continuing without it.")
        return {}
    except Exception as e:
        logger.warning("DSSP failed (%s: %s) — continuing without secondary structure", type(e).__name__, e)
        return {}


# ── Build full StructureContext for one position ─────────────────────────────

def build_structure_context(
    sequence: str,
    position: int,
    pdb_path: str | None,
    *,
    dist_matrix: np.ndarray | None = None,
    ca_positions: list[tuple[int, str]] | None = None,
    dssp_map: dict[int, str] | None = None,
    cutoff: float = 8.0,
    neighbor_window: int = 5,
    spatial_top_k: int = 8,
) -> StructureContext:
    """Build a complete StructureContext for a single position.

    The caller should pre-compute dist_matrix/dssp_map once per iteration
    and pass them here to avoid redundant PDB parsing.
    """
    n = len(sequence)
    pos = position

    # linear neighbors
    n_start = max(0, pos - 2)
    c_end = min(n, pos + 3)
    lin_n = sequence[n_start:pos]
    lin_c = sequence[pos + 1:c_end]

    ss_label = "UNKNOWN"
    avg_dist = 0.0
    std_dist = 0.0
    contact_density = 0
    spatial = []

    if dist_matrix is not None and ca_positions is not None and dist_matrix.size > 0:
        idx = min(pos, dist_matrix.shape[0] - 1)
        avg_dist, std_dist, contact_density = compute_structural_summaries(
            dist_matrix, idx, window=neighbor_window, cutoff=cutoff,
        )
        spatial = spatial_neighbors_for_position(
            ca_positions, dist_matrix, idx, cutoff=cutoff, top_k=spatial_top_k,
        )

    if dssp_map is not None:
        ss_label = dssp_map.get(pos, "UNKNOWN")

    region = _guess_region(ss_label, contact_density, avg_dist)

    return StructureContext(
        secondary_structure=ss_label,
        linear_neighbors_n=lin_n,
        linear_neighbors_c=lin_c,
        spatial_neighbors=spatial,
        contact_density=contact_density,
        avg_local_distance=avg_dist,
        std_local_distance=std_dist,
        region_guess=region,
    )
