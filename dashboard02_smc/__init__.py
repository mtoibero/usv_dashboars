"""Sliding-mode-control dashboard (dashboard02_smc) for the IAcquaBot USV.

GUI-independent modules (``config``, ``engine``, ``metrics``, ``figures``,
``session``) plus a thin Streamlit front end (``app.py``).  Only ``app.py``
imports Streamlit.

The simulator modules (``controllers``, ``usv_model``, ``disturbances``...)
are top-level modules in the project root, so the root is added to
``sys.path`` here.  The existing project layout is not modified.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
