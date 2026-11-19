# Drop the `eztao` and `celerite` test dependencies

<!-- Draft PR body. Branch suggestion: `drop-eztao-deps` off `main`. Open as a draft PR. -->

## Change Description

Closes #63. Supersedes the stop-gap comment from #64.

`eztao` and `celerite` are only used by two test files:

- `tests/eztaox/test_kernel_stats.py` imports `drw_acf/drw_sf/drw_psd` and
  `carma_acf/carma_sf/carma_psd` from `eztao.carma` and compares them with
  `eztaox.kernel_stat2`.
- `tests/eztaox/test_kernels_quasisep.py::test_carma` builds five
  `eztao.carma.CARMA_term` kernels, computes their log-likelihood with
  `celerite.GP`, and compares with the `quasisep.CARMA` log-probability.

No notebook, script, benchmark, or source module imports either package.

The original motivation (eztao pinned `numpy<2`) no longer applies: eztao 0.5.1
requires `numpy>=2`. The remaining reasons to drop them:

- eztao pins `scipy<1.15` and `matplotlib<3.10` on Python <3.13, so every dev
  environment and the CI test jobs are held back (the local env currently resolves
  to scipy 1.14.1 and matplotlib 3.9.4). This also blocks Dependabot updates.
- eztao pulls in `numba` and `emcee`, which nothing else here uses.
- celerite is a compiled C++ extension, which is the slowest and most fragile
  part of `pip install -e '.[dev]'` on new Python versions.
- Both are used as *reference implementations*, so they belong in a one-off
  fixture generation step rather than in every test run.

## Solution Description

Three parts, all confined to `tests/`, `tests/data/`, `pyproject.toml`, and docs.

### 1. Freeze the eztao / celerite reference values into a fixture

Follow the existing `tests/data/unit_test_lc.npz` + `generate_data.py` pattern.

- Add `tests/data/generate_reference_stats.py`. It imports `eztao.carma` and
  `celerite`, evaluates them on exactly the inputs the tests use, and writes
  `tests/data/reference_stats.npz`:
  - DRW: `tau=100.0`, `amp=0.1`, `ts=linspace(0, 1000, 100)`,
    `fs=logspace(-5, 5, 100)` -> `drw_acf`, `drw_sf`, `drw_psd`.
  - CARMA(2,0), (2,1), (3,0): the six `(ar, ma)` pairs already in
    `test_kernel_stats.py`, on their `ts` / `fs` grids -> `acf`, `sf`, `psd`
    arrays per pair.
  - CARMA log-likelihoods: the five `CARMA_term` kernels in `test_carma`,
    evaluated with `celerite.GP` on the seeded `data` fixture
    (`default_rng(84930)`, `x = sort(uniform(-3, 3, 50))`, `y = sin(x)`,
    `yerr = sqrt(0.1)`) -> one scalar per kernel.
- The script header documents how to regenerate: run it in a throwaway env with
  `pip install eztao celerite`. It is never imported by pytest.
- Tests load the `.npz` through a `conftest.py` fixture and `assert_allclose`
  against it. Assertion structure in the three CARMA stats tests stays as is;
  only the `eztao_*` callables are replaced by arrays from the fixture.

### 2. Keep an analytic cross-check that does not depend on frozen numbers

So the suite still verifies correctness if `kernel_stat2` or the CARMA state-space
code is edited later:

- `test_drw`: inline the closed forms instead of loading them:
  `acf = exp(-t/tau)`, `sf = sqrt(2) * amp * sqrt(1 - exp(-t/tau))`,
  `psd = 2 * sqrt(2) * amp**2 * tau / (1 + (2*pi*f*tau)**2)` (match eztao's
  normalisation; confirm against the frozen values once while writing the test).
- `test_carma`: add a dense `tinygp` GP built from the closed-form ACF,
  `tinygp.kernels.Custom(lambda a, b: carma_acf(abs(a - b), alpha, beta))`,
  with the same `x` and `diag=0.1`, and compare its `log_probability` with the
  quasisep `CARMA` GP. This checks the state-space solver against the ACF through
  an independent O(N^3) path, the same strategy the other kernel tests already use
  against dense tinygp kernels.
- CARMA stats tests: add the identities `sf(t) == 2 * (acf(0) - acf(t))` and a
  PSD check against the quasisep kernel's `.power(f)`, so ACF, SF, and PSD are
  tied together without reimplementing eztao's formulas.

### 3. Remove the dependencies and update docs

- Delete the `celerite>=0.4` and `eztao>=0.5` lines from the `dev` extra in
  `pyproject.toml`.
- `CLAUDE.md`: replace the sentence saying correctness tests compare against
  `eztao`/`celerite` with one that points at the frozen fixture and the dense
  tinygp cross-checks.
- `TESTING.md`: add a short section on `tests/data/reference_stats.npz` and how to
  regenerate it.
- `Changelog.rst`: entry for the next release.

### Verification

- `poetry run pytest tests/eztaox/test_kernel_stats.py tests/eztaox/test_kernels_quasisep.py`
  passes with `eztao` and `celerite` uninstalled (`poetry run pip uninstall eztao celerite`).
- Full `poetry run pytest` and `poetry run nox` pass.
- CI `test-lowest-versions` job passes: it resolves `scipy` / `matplotlib` to new
  lower bounds once the eztao pins are gone, so watch for new lowest-version
  failures there.

### Trade-off

Part 1 alone turns a cross-implementation test into a regression test against a
snapshot. Part 2 is what keeps the suite meaningful going forward, so both land
in this PR.

## Code Quality
- [ ] I have read the Contribution Guide and agree to the Code of Conduct
- [ ] My code follows the code style of this project
- [ ] My code builds (or compiles) cleanly without any errors or warnings
- [ ] My code contains relevant comments and necessary documentation

🤖 Generated with [Claude Code](https://claude.com/claude-code)
