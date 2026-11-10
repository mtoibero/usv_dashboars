"""Streamlit dashboard for the IAcquaBot USV simulator.

The package is split into GUI-independent modules (``config``, ``engine``,
``metrics``, ``figures``, ``session``) and a thin Streamlit front end
(``app.py``).  Only ``app.py`` imports Streamlit.

The simulator modules (``controllers``, ``usv_model``, ``run_simulation``...)
live in the project root as top-level modules, so the root is added to
``sys.path`` here.  This keeps the existing project layout unchanged.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
