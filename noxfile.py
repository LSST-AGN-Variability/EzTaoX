"""Nox sessions for testing EzTaoX locally against each supported Python version.

Each session installs the package with its ``dev`` extra into a fresh virtual
environment, the same way CI does, so a passing local run predicts a passing
CI run. Nox skips any Python version it cannot find; see TESTING.md.
"""

import nox

PYTHON_VERSIONS = ["3.12", "3.13", "3.14"]

# Use uv to build session environments when it is installed; it is much faster.
nox.options.default_venv_backend = "uv|virtualenv"
nox.options.sessions = ["tests"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run pytest. Arguments after ``--`` on the nox command line go to pytest."""
    session.install("-e", ".[dev]")
    session.run("pytest", *session.posargs)
