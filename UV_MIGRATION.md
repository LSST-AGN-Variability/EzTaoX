# Proposal: switch local development from Poetry to uv

Status: proposal only. Nothing in the repository changes until this plan is approved.
This file is removed by the pull request that carries out the migration.

## Why

- CI already installs with uv (`uv pip install -e .[dev]`, and `uv pip compile` for
  the lowest-versions job), and `noxfile.py` already prefers the uv backend. Local
  development is the only place Poetry is still used.
- uv also manages Python interpreters, which TESTING.md already recommends for running
  nox across 3.12, 3.13 and 3.14. One tool would then cover environments, interpreters
  and multi-version testing.
- Poetry plays no part in building or publishing. The build backend is setuptools with
  setuptools_scm, and the `[tool.poetry]` table is unused.

## What stays the same

- **Dependency layout.** Runtime dependencies stay in `[project].dependencies`. All
  development tools stay in the single `dev` extra under
  `[project.optional-dependencies]`. CI and nox install with `pip install -e '.[dev]'`,
  which needs an extra, so the list does not move to a PEP 735 `[dependency-groups]`
  table.
- **Build backend.** setuptools with setuptools_scm.
- **CI workflows.** All of them already use uv or plain pip and need no edits.
- **`noxfile.py`.** Already uses `uv|virtualenv`.
- **`.setup_dev.sh`.** Template-managed and pip-based. Leaving it alone avoids conflicts
  on the next copier update.
- **`.jax_cache`.** Untouched.

## Proposed changes

### pyproject.toml

- Delete the unused `[tool.poetry]` table (`version = "0.1.0"`, which setuptools_scm
  overrides).
- Reword the comment above the `dev` extra:

  ```toml
  # Development and testing dependencies, kept in one list that both uv and CI read.
  #   Local setup:      uv sync --extra dev
  #   Add a package:    uv add --optional dev <package>
  #   CI installs with: uv pip install -e '.[dev]'  (keep the single quotes on a mac)
  ```

### .gitignore

- Replace the `poetry.lock` entry with `uv.lock`, keeping today's policy: the lock
  pins only the local environment, while CI and nox resolve the newest compatible
  versions.

### .pre-commit-config.yaml

- Comment only. The black hook's note about "the project's Poetry .venv" becomes
  "the project's uv .venv". No hook behavior changes.

### README.md, TESTING.md, CLAUDE.md

Command mapping:

| Poetry | uv |
| --- | --- |
| `poetry install --extras dev` | `uv sync --extra dev` |
| `poetry run <cmd>` | `uv run <cmd>` |
| `poetry add --optional dev <pkg>` | `uv add --optional dev <pkg>` |
| `poetry env use python3.12` | `uv python pin 3.12`, then `uv sync --extra dev` |
| `poetry.lock` | `uv.lock` |

Other edits:

- Add a line on installing uv itself, since it cannot come from the `dev` extra:
  `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- TESTING.md: the opening bullet becomes "pytest in your uv environment". The nox
  section no longer needs "when uv is installed", because uv is now required.
- CLAUDE.md: rewrite the package-management paragraph to say "Use uv to manage
  packages" and to name `uv.lock` instead of `poetry.lock`.

## Open question: commit `uv.lock`?

| Option | Effect |
| --- | --- |
| Keep it git-ignored (proposed default) | Same as today. Each developer's lock is local. CI and nox keep testing the newest compatible versions. |
| Commit it | Everyone gets the same local environment. Dependabot's `pip` ecosystem would switch to `uv` so it keeps the lock current, and `uv sync --locked` could be added to CI. |

## Migrating an existing checkout

```bash
brew install uv
uv sync --extra dev
uv run pre-commit install
```

`uv sync` reuses the `.venv` folder. If the existing Poetry-built `.venv` was created
from a conda interpreter, uv may recreate it on a uv-managed Python. Delete
`poetry.lock` afterwards.

## Verification for the migration PR

```bash
uv run pytest tests/eztaox/test_ts_utils.py
uv run pre-commit run --all-files
uv run nox -s tests-3.12 -- tests/eztaox/test_ts_utils.py
```

The migration commit touches only docs and configuration comments, so its subject can
carry `[skip-test]`.
