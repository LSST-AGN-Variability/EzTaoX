"""Compare fit quality across several Optax optimizers."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import numpy as np
import numpyro
import optax
from joblib import Parallel, delayed
from numpyro import distributions as dist
from scipy.stats import median_abs_deviation as mad

from eztaox.fitter import random_search
from eztaox.kernels.quasisep import Exp
from eztaox.models import MultiVarModel

matplotlib.use("Agg")

from matplotlib import pyplot as plt

BASEKEY_SEED = 10
FIT_BANDS = jnp.array([0, 1])
TRUE_PARAMS = {
    "log_kernel_param": jnp.array([jnp.log(50.0), jnp.log(0.2)]),
    "lag": jnp.array(2.0),
}
DEFAULT_LEARNING_RATES = (1e-3, 1e-2)
DEFAULT_LR_OPT_STEP = 1_000
DEFAULT_LBFGS_OPT_STEP = 100
DEFAULT_STOPPING_MAX_OPT_STEP = 2_000
DEFAULT_STOPPING_TOL = 1e-2
DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "unit_test_lc.npz"
)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent / "_results" / "optimizer_fit_comparison.md"
)
DEFAULT_PLOT_PATH = (
    Path(__file__).resolve().parent / "_results" / "optimizer_fit_distributions.png"
)
LR_OPTIMIZER_FACTORIES: tuple[
    tuple[str, Callable[[float], optax.GradientTransformation], bool], ...
] = (
    ("adam", optax.adam, False),
    ("rmsprop", optax.rmsprop, False),
    ("sgd", optax.sgd, False),
    ("lion", optax.lion, False),
)
STATEFUL_OPTIMIZER_FACTORIES: tuple[
    tuple[str, Callable[[], optax.GradientTransformation], bool], ...
] = (
    ("lbfgs", optax.lbfgs, True),
)


def initSampler():  # noqa: N802
    """Sample the initial multiband DRW parameters."""
    log_drw_scale = numpyro.sample(
        "drw_scale", dist.Uniform(jnp.log(0.1), jnp.log(10000))
    )
    log_drw_sigma = numpyro.sample("drw_sigma", dist.Uniform(jnp.log(0.01), jnp.log(2)))
    log_kernel_param = jnp.stack([log_drw_scale, log_drw_sigma])
    numpyro.deterministic("log_kernel_param", log_kernel_param)

    log_amp_scale = numpyro.sample("log_amp_scale", dist.Uniform(-2, 2))
    mean = numpyro.sample("mean", dist.Normal(loc=0.0, scale=0.1))
    lag = numpyro.sample("lag", dist.Uniform(-10.0, 10.0))

    return {
        "log_kernel_param": log_kernel_param,
        "log_amp_scale": log_amp_scale,
        "mean": mean,
        "lag": lag,
    }


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare fit quality across several Optax optimizers."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the benchmark light-curve dataset.",
    )
    parser.add_argument(
        "--max-curves",
        type=int,
        default=100,
        help="Maximum number of benchmark light curves to evaluate.",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=2_000,
        help="Number of random-search parameter draws per light curve.",
    )
    parser.add_argument(
        "--n-best",
        type=int,
        default=5,
        help="Number of best random-search draws retained for local optimization.",
    )
    parser.add_argument(
        "--lr-opt-step",
        type=int,
        default=DEFAULT_LR_OPT_STEP,
        help="Number of optimizer steps for learning-rate-based optimizers.",
    )
    parser.add_argument(
        "--lbfgs-opt-step",
        type=int,
        default=DEFAULT_LBFGS_OPT_STEP,
        help="Number of optimizer steps for L-BFGS.",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=list(DEFAULT_LEARNING_RATES),
        help="Learning rates to benchmark for optimizers that require one.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the Markdown table.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Path to save the optimizer error-distribution plot.",
    )
    return parser.parse_args()


def build_optimizer_specs(
    learning_rates: list[float],
    lr_opt_step: int,
    lbfgs_opt_step: int,
) -> list[dict[str, object]]:
    """Construct the optimizer configurations to benchmark."""
    specs: list[dict[str, object]] = []
    for learning_rate in learning_rates:
        for name, factory, use_state_grad in LR_OPTIMIZER_FACTORIES:
            if name == "sgd" and not np.isclose(learning_rate, 1e-3):
                continue
            specs.append(
                {
                    "optimizer": name,
                    "optimizer_obj": factory(learning_rate),
                    "use_state_grad": use_state_grad,
                    "learning_rate": learning_rate,
                    "n_opt_step": lr_opt_step,
                    "use_stopping": False,
                    "max_opt_step": None,
                    "tol": None,
                }
            )
            specs.append(
                {
                    "optimizer": f"{name}_stopped",
                    "optimizer_obj": factory(learning_rate),
                    "use_state_grad": use_state_grad,
                    "learning_rate": learning_rate,
                    "n_opt_step": lr_opt_step,
                    "use_stopping": True,
                    "max_opt_step": DEFAULT_STOPPING_MAX_OPT_STEP,
                    "tol": DEFAULT_STOPPING_TOL,
                }
            )
    for name, factory, use_state_grad in STATEFUL_OPTIMIZER_FACTORIES:
        specs.append(
            {
                "optimizer": name,
                "optimizer_obj": factory(),
                "use_state_grad": use_state_grad,
                "learning_rate": None,
                "n_opt_step": lbfgs_opt_step,
                "use_stopping": False,
                "max_opt_step": None,
                "tol": None,
            }
        )
        specs.append(
            {
                "optimizer": f"{name}_stopped",
                "optimizer_obj": factory(),
                "use_state_grad": use_state_grad,
                "learning_rate": None,
                "n_opt_step": lbfgs_opt_step,
                "use_stopping": True,
                "max_opt_step": DEFAULT_STOPPING_MAX_OPT_STEP,
                "tol": DEFAULT_STOPPING_TOL,
            }
        )
    return specs


def load_dataset(
    data_path: Path, max_curves: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the benchmark light curves and apply the max-curve limit."""
    with np.load(data_path) as data:
        ts = data["ts"][:max_curves]
        bands = data["bands"][:max_curves]
        ys = data["ys"][:max_curves]
    return ts, bands, ys


def fit_single_light_curve(
    t: np.ndarray,
    band: np.ndarray,
    y: np.ndarray,
    key_index: int,
    optimizer: optax.GradientTransformation,
    n_sample: int,
    n_best: int,
    n_opt_step: int,
    use_value_and_grad_from_state: bool,
    max_opt_step: int | None,
    tol: float | None,
) -> dict[str, jax.Array]:
    """Fit one light curve and return the best-fit parameters."""
    band_mask = np.isin(band, np.asarray(FIT_BANDS))
    filtered_t = t[band_mask]
    filtered_band = band[band_mask]
    filtered_y = y[band_mask]
    yerr = jnp.ones_like(filtered_y) * 1e-6

    model = MultiVarModel(
        (filtered_t, filtered_band),
        filtered_y,
        yerr,
        Exp(scale=100.0, sigma=0.1),
        nBand=len(FIT_BANDS),
        zero_mean=True,
        has_lag=True,
    )
    fit_key = jr.fold_in(jr.PRNGKey(BASEKEY_SEED), key_index)
    best_params, log_likelihood = random_search(
        model,
        initSampler,
        fit_key,
        n_sample,
        n_best,
        batch_size=1000,
        optimizer=optimizer,
        n_opt_step=n_opt_step,
        max_opt_step=max_opt_step,
        tol=tol,
        use_value_and_grad_from_state=use_value_and_grad_from_state,
        clear_cache_after_opt=True,
    )
    if jnp.ndim(log_likelihood) != 0:
        raise ValueError("Expected scalar log-likelihood from random_search.")
    return best_params


def summarise_optimizer(
    optimizer_name: str,
    optimizer: optax.GradientTransformation,
    use_value_and_grad_from_state: bool,
    learning_rate: float | None,
    ts: np.ndarray,
    bands: np.ndarray,
    ys: np.ndarray,
    n_sample: int,
    n_best: int,
    n_opt_step: int,
    use_stopping: bool,
    max_opt_step: int | None,
    tol: float | None,
) -> tuple[dict[str, float | int | str | bool], dict[str, np.ndarray]]:
    """Run the benchmark for one optimizer and summarise the recovery metrics."""
    start_time = perf_counter()
    best_params = Parallel(n_jobs=-1)(
        delayed(fit_single_light_curve)(
            ts[i],
            bands[i],
            ys[i],
            i,
            optimizer,
            n_sample,
            n_best,
            n_opt_step,
            use_value_and_grad_from_state,
            max_opt_step,
            tol,
        )
        for i in range(len(ts))
    )
    fit_seconds = perf_counter() - start_time

    best_param_all = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *best_params)

    tau_diff = np.asarray(
        best_param_all["log_kernel_param"][:, 0] - TRUE_PARAMS["log_kernel_param"][0]
    )
    amp_diff = np.asarray(
        best_param_all["log_kernel_param"][:, 1] - TRUE_PARAMS["log_kernel_param"][1]
    )
    lag_diff = np.asarray(best_param_all["lag"] - TRUE_PARAMS["lag"])

    tau_stats = compute_diff_stats(tau_diff)
    amp_stats = compute_diff_stats(amp_diff)
    lag_stats = compute_diff_stats(lag_diff)

    summary = {
        "optimizer": optimizer_name,
        "learning_rate": learning_rate if learning_rate is not None else "default",
        "use_state_grad": use_value_and_grad_from_state,
        "use_stopping": use_stopping,
        "n_runs": len(ts),
        "n_opt_step": n_opt_step,
        "max_opt_step": max_opt_step if max_opt_step is not None else "-",
        "tol": tol if tol is not None else "-",
        "fit_seconds": fit_seconds,
        "tau_mae": tau_stats["mae"],
        "tau_bias": tau_stats["bias"],
        "tau_mad": tau_stats["mad"],
        "amp_mae": amp_stats["mae"],
        "amp_bias": amp_stats["bias"],
        "amp_mad": amp_stats["mad"],
        "lag_mae": lag_stats["mae"],
        "lag_bias": lag_stats["bias"],
        "lag_mad": lag_stats["mad"],
    }
    diffs = {
        "tau_diff": tau_diff,
        "amp_diff": amp_diff,
        "lag_diff": lag_diff,
    }
    return summary, diffs


def compute_diff_stats(diff: np.ndarray) -> dict[str, float]:
    """Compute the same summary statistics used in the fitter test."""
    return {
        "mae": float(np.mean(np.abs(diff))),
        "bias": float(np.mean(diff)),
        "mad": float(mad(diff, scale="normal")),
    }


def format_value(value: object) -> str:
    """Format values for a Markdown table."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def make_optimizer_label(summary: dict[str, float | int | str | bool]) -> str:
    """Build a compact optimizer label for tables and plots."""
    suffix = " + stop" if summary["use_stopping"] else ""
    return "{} ({}{})".format(summary["optimizer"], summary["learning_rate"], suffix)


def render_markdown_table(rows: list[dict[str, float | int | str | bool]]) -> str:
    """Render the optimizer comparison as a Markdown table."""
    if not rows:
        raise ValueError("No optimizer results were produced.")

    columns = [
        "optimizer",
        "learning_rate",
        "use_state_grad",
        "use_stopping",
        "n_runs",
        "n_opt_step",
        "max_opt_step",
        "tol",
        "fit_seconds",
        "tau_mae",
        "tau_bias",
        "tau_mad",
        "amp_mae",
        "amp_bias",
        "amp_mad",
        "lag_mae",
        "lag_bias",
        "lag_mad",
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(format_value(row[column]) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def save_diff_distribution_plot(
    rows: list[dict[str, float | int | str | bool]],
    diffs_by_optimizer: list[dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    """Save grouped error-distribution plots for tau, amplitude, and lag."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [make_optimizer_label(row) for row in rows]
    metrics = [
        ("tau_diff", "Tau Error Distribution"),
        ("amp_diff", "Amplitude Error Distribution"),
        ("lag_diff", "Lag Error Distribution"),
    ]

    fig, axes = plt.subplots(
        1, 3, figsize=(max(12, 1.2 * len(labels)), 6), sharey=False
    )
    for ax, (metric_key, title) in zip(axes, metrics, strict=False):
        series = [diffs[metric_key] for diffs in diffs_by_optimizer]
        violin = ax.violinplot(series, showmeans=True, showmedians=False)
        for body in violin["bodies"]:
            body.set_alpha(0.6)
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_title(title)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Recovered - true")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the optimizer comparison and print the Markdown table."""
    args = parse_args()
    ts, bands, ys = load_dataset(args.data_path, args.max_curves)
    optimizer_specs = build_optimizer_specs(
        args.learning_rates, args.lr_opt_step, args.lbfgs_opt_step
    )

    summaries: list[dict[str, float | int | str | bool]] = []
    diffs_by_optimizer: list[dict[str, np.ndarray]] = []
    for spec in optimizer_specs:
        summary, diffs = summarise_optimizer(
            spec["optimizer"],
            spec["optimizer_obj"],
            spec["use_state_grad"],
            spec["learning_rate"],
            ts,
            bands,
            ys,
            args.n_sample,
            args.n_best,
            spec["n_opt_step"],
            spec["use_stopping"],
            spec["max_opt_step"],
            spec["tol"],
        )
        summaries.append(summary)
        diffs_by_optimizer.append(diffs)

    markdown_table = render_markdown_table(summaries)
    print(markdown_table)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown_table + "\n", encoding="utf-8")

    if args.plot_output is not None:
        save_diff_distribution_plot(summaries, diffs_by_optimizer, args.plot_output)


if __name__ == "__main__":
    main()
