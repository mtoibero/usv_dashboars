"""Launch the dashboard with ``python -m dashboard [streamlit options]``."""

import sys
from pathlib import Path


def main():
    from streamlit.web import cli as streamlit_cli

    app = Path(__file__).resolve().with_name("app.py")
    sys.argv = ["streamlit", "run", str(app), *sys.argv[1:]]
    sys.exit(streamlit_cli.main())


if __name__ == "__main__":
    main()
