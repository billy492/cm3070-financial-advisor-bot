"""Thin launcher that starts the Streamlit prototype UI.

Running ``python scripts/run_app.py`` execs ``streamlit run streamlit_app.py``
from the code root so the non-technical web interface can be launched without
remembering the Streamlit invocation. Any extra command-line arguments are
forwarded verbatim to Streamlit (e.g. ``--server.port 8502``).

Examples:
    ::

        python scripts/run_app.py
        python scripts/run_app.py --server.port 8502 --server.headless true

This module shells out to the ``streamlit`` console script; it does not import
Streamlit itself, so importing this file is cheap and side-effect free.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Code root = parent of the scripts/ directory = where streamlit_app.py lives.
_CODE_ROOT: Path = Path(__file__).resolve().parent.parent
_APP_PATH: Path = _CODE_ROOT / "streamlit_app.py"


def main(argv: list[str] | None = None) -> int:
    """Launch the Streamlit app, forwarding any extra arguments.

    Args:
        argv: Optional argument vector to forward to Streamlit (defaults to
            ``sys.argv[1:]``).

    Returns:
        The Streamlit subprocess exit code (``0`` on clean exit), or ``1`` if
        the app file or the Streamlit console script cannot be found.
    """
    extra = list(sys.argv[1:] if argv is None else argv)

    if not _APP_PATH.exists():  # defensive: give a clear message, don't crash
        print(
            f"ERROR: cannot find Streamlit app at {_APP_PATH}.",
            file=sys.stderr,
        )
        return 1

    cmd = [sys.executable, "-m", "streamlit", "run", str(_APP_PATH), *extra]
    print(f"Launching: {' '.join(cmd)}")

    try:
        completed = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print(
            "ERROR: Streamlit is not installed. Install it with "
            "'pip install streamlit'.",
            file=sys.stderr,
        )
        return 1
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
