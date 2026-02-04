from nox_poetry import session

PYTHON_VERSIONS = ["3.10", "3.11", "3.12"]


@session(python=PYTHON_VERSIONS)
def tests(session):  # noqa: D103
    tmp_path = session.poetry.export_requirements()
    session.run(
        "poetry",
        "export",
        "--without-hashes",
        "--with",
        "test",
        "-o",
        f"{tmp_path}",
    )
    session.install("-r", f"{tmp_path}")
    session.install(".")
    session.run("pytest", "tests/")
