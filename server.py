"""
Root entrypoint for Render/Vercel: sets up paths and exposes the dashboard FastAPI app.
The app is defined in protein_swarm/dashboard/server.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from protein_swarm.dashboard.server import app

__all__ = ["app"]