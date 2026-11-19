# Testing EzTaoX

There are two ways to run the tests locally:

- **pytest in your Poetry environment.** Fast. Use it while you work.
- **nox.** Runs the suite once per supported Python version (3.12, 3.13, 3.14), each in a fresh environment, the same way CI does. Use it before you push.

## Set up

EzTaoX uses [Poetry](https://python-poetry.org/) to manage its development environment. Install the package together with its `dev` extra, which holds every testing and development tool, including pytest and nox:

```bash
poetry install --extras dev
```

The project needs Python 3.12, 3.13 or 3.14. If Poetry picks an unsupported interpreter, point it at a supported one:

```bash
poetry env use python3.12
```

## Quick runs with pytest

Run the whole suite:

```bash
poetry run pytest
```

Run one file, or only the tests whose names match a keyword:

```bash
poetry run pytest tests/eztaox/test_models_quasisep.py
```

```bash
poetry run pytest tests/eztaox/test_kernels_quasisep.py -k carma
```

Measure coverage. The HTML report is written to `htmlcov/index.html`:

```bash
poetry run pytest --cov=eztaox --cov-report=html
```

A few things to know:

- **JAX compilation cache.** The tests store compiled JAX functions in `.jax_cache/` at the repository root, which makes later runs much faster. Pass `--jax_cache=<dir>` to use a different folder.
- **Slow tests.** `tests/eztaox/test_fitter.py` fits many simulated light curves and takes the longest. Leave it out with `--ignore=tests/eztaox/test_fitter.py` or run it on its own.
- **Experimental tests.** `tests/eztaox/_test_laguerre.py` starts with an underscore so pytest skips it. It covers the hidden, experimental Laguerre kernel.

## Every supported Python version with nox

Run the full matrix:

```bash
poetry run nox
```

Run a single Python version:

```bash
poetry run nox -s tests-3.13
```

Pass arguments to pytest after `--`:

```bash
poetry run nox -s tests-3.12 -- tests/eztaox/test_ts_utils.py -k merge
```

Reuse the session environments from the previous run instead of rebuilding them:

```bash
poetry run nox -r
```

Each session installs the package with `pip install -e '.[dev]'`, exactly as CI does. It resolves the newest compatible versions of every dependency, not the pinned versions in `poetry.lock`. A passing nox run therefore predicts a passing CI run. When [uv](https://docs.astral.sh/uv/) is installed, nox uses it to build the session environments, which is much faster.

### Install the Python versions you want to test

**Nox only tests the Python versions it can find on your machine.** For every version that is missing, nox skips that session and prints a line like this:

```text
nox > Session tests-3.14 skipped: Python interpreter 3.14 not found.
```

A skipped session is not a failure, so a run can look green while testing fewer versions than you expect. Check the summary at the end of the run.

Install the missing versions with pyenv or uv.

**pyenv.** Install all three versions, then make them available in this repository. The `.python-version` file that `pyenv local` writes is git-ignored.

```bash
for v in 3.12 3.13 3.14; do pyenv install --skip-existing "$v"; done
```

```bash
pyenv local 3.14 3.13 3.12
```

**uv.** Install all three versions. Recent uv releases also put `python3.12`, `python3.13` and `python3.14` into `~/.local/bin`, which must be on your `PATH`.

```bash
uv python install 3.12 3.13 3.14
```

With uv installed, nox also downloads any missing version through uv on its own, so you rarely see a skipped session. To turn that off and test only the versions already installed, add `--download-python never`.

Interpreters from conda environments also work, as long as `python3.12`, `python3.13` and `python3.14` are on your `PATH` when you run nox.

## What CI runs

| Workflow | When | What it checks |
| --- | --- | --- |
| Unit test and code coverage | Every push and pull request to `main` | The full suite on Python 3.12, 3.13 and 3.14, with coverage uploaded to Codecov. A second job runs the suite on Python 3.12 with the lowest allowed versions of the runtime dependencies. |
| Unit test smoke test | Daily | The full suite on Python 3.12, 3.13 and 3.14 against the latest dependency releases. |

Locally, a pre-push hook runs the full suite with coverage every time you `git push`. Commits only run the fast formatting and lint hooks. Install both hook types once with:

```bash
poetry run pre-commit install
```

The test hook runs the `python` on your `PATH`. If that Python can't run pytest, the hook reports "Passed" without running any tests, so activate the project environment with `source .venv/bin/activate` before you push. To push without running the tests once, use `git push --no-verify`.
