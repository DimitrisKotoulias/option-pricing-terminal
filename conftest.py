"""Pytest configuration shared by the whole repo.

``app/`` is a Streamlit entry point, not a package: it has no ``__init__.py`` and
its modules import each other flat (``from _data import ...``), which only
resolves because ``streamlit run`` puts the script's directory on ``sys.path``.
That is the official Streamlit layout, but it leaves the dashboard -- a quarter
of the codebase -- unimportable from pytest. Replicating that one line here lets
the app modules be imported and smoke-tested like anything else.
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
