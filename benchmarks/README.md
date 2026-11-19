# Benchmarks

This directory contains files that will be run via continuous testing either
nightly or after committing code to a pull request. 

The runtime and/or memory usage of the functions defined in these files will be
tracked and reported to give you a sense of the overall performance of your code.

You are encouraged to add, update, or remove benchmark functions to suit the needs
of your project.

For more information, see the documentation here: https://lincc-ppt.readthedocs.io/en/latest/practices/ci_benchmarking.html

## Benchmarks on pull requests

Benchmarks don't run on every pull request. To compare a pull request against `main`,
add the `benchmark` label to it. The benchmarks run right away and again on every later
push while the label stays, and a bot comment shows the comparison table. Remove the
label to stop.

You can also start the "Run benchmarks for PR" workflow by hand from the Actions tab.
That run shows the comparison table on its summary page instead of posting a comment.

Benchmarks still run on every push to `main` and nightly. The push-to-`main` run
benchmarks only the pushed commit, which keeps it to a few minutes. The nightly run
compares `main` against the previous night, then fills in a sample of commits after
`afe10654` (April 2026) so the dashboard shows history; older commits predate the
current fitter API and the benchmark suite, so they are left out. Both runs publish
their results to the `benchmarks` branch and the dashboard on GitHub Pages.
