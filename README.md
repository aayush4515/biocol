# Protein Swarm Design Engine

Swarm-based multi-agent protein sequence optimisation using ESMFold (Modal GPU) and PyRosetta (local) scoring.

## Setup

### 1. Install local dependencies

```bash
pip install -r protein_swarm/requirements.txt
```

For Rosetta scoring, install PyRosetta separately (requires a license):
```bash
# See https://www.pyrosetta.org/downloads for installation
pip install pyrosetta-installer
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
```

### 2. Deploy Modal functions

```bash
modal token set          # one-time auth
modal deploy protein_swarm/modal_app/functions.py
```

## Usage

### Full pipeline (ESMFold on Modal + local Rosetta scoring)

```bash
python -m protein_swarm.main design \
  -s "VVVVVVVVVVVVVVVVVVVV" \
  -o "introduce flexibility through turn-promoting residues" \
  --use-llm \
  --modal-fold --remote-fold-backend esmfold \
  --use-rosetta \
  --max-iterations 10 --debug
```

### Without Rosetta (confidence-only scoring)

```bash
python -m protein_swarm.main design \
  -s "VVVVVVVVVVVVVVVVVVVV" \
  -o "Design a stable helix-rich protein" \
  --modal-fold --remote-fold-backend esmfold \
  --no-rosetta \
  --max-iterations 5
```

### Fully local (no Modal, no Rosetta)

```bash
python -m protein_swarm.main design \
  -s "ACDEFGHIKLMNPQRSTVWY" \
  -o "maximise diversity" \
  --no-modal --no-rosetta \
  --max-iterations 10
```

### Scoring weights

The combined score is: `w_physics * physics + w_objective * objective + w_confidence * confidence`

```bash
--w-physics 0.55 --w-objective 0.35 --w-confidence 0.10
```

Weights are auto-normalised to sum to 1.0.

## Architecture

```
CLI (main.py)
  └─ DesignEngine (orchestrator/engine.py)
       ├─ Agents: residue_agent × N  [Modal parallel or local]
       ├─ Folding: ESMFold            [Modal GPU or local]
       ├─ Scoring: PyRosetta          [local only, optional]
       ├─ Scoring: objective heuristic [local]
       └─ Memory: shared memory store  [local]
```
