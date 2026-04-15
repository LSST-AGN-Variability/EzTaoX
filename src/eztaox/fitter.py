"""This module contains the fitter functions that fits a model to data."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import optax
from numpyro.handlers import seed
from tinygp.helpers import JAXArray

from eztaox.models import MultiVarModel, UniVarModel

DEFAULT_ADAM_OPTIMIZER = optax.adam(1e-2)


def _make_loss(model: UniVarModel | MultiVarModel) -> Callable:
    @jax.jit
    def loss(params) -> JAXArray:
        return -model.log_prob(params)

    return loss


@partial(jax.jit, static_argnames=("solver", "loss"))
def _optimizer_step(
    params: dict[str, JAXArray],
    opt_state: optax.OptState,
    solver: optax.GradientTransformationExtraArgs,
    loss: Callable,
) -> tuple[dict[str, JAXArray], optax.OptState, JAXArray, dict[str, JAXArray]]:
    val, grad = jax.value_and_grad(loss)(params)
    updates, opt_state = solver.update(
        grad, opt_state, params, value=val, grad=grad, value_fn=loss
    )
    params = optax.apply_updates(params, updates)
    return params, opt_state, val, grad


@partial(jax.jit, static_argnames=("solver", "loss"))
def _optimizer_step_from_state(
    params: dict[str, JAXArray],
    opt_state: optax.OptState,
    solver: optax.GradientTransformationExtraArgs,
    loss: Callable,
) -> tuple[dict[str, JAXArray], optax.OptState, JAXArray, dict[str, JAXArray]]:
    val, grad = optax.value_and_grad_from_state(loss)(params, state=opt_state)
    updates, opt_state = solver.update(
        grad, opt_state, params, value=val, grad=grad, value_fn=loss
    )
    params = optax.apply_updates(params, updates)
    return params, opt_state, val, grad


def _sample_top_params(
    initSampler: Callable,
    prng_key: jax.random.PRNGKey,
    nSample: int,
    nBest: int,
    loss: Callable,
    batch_size: int,
) -> list[dict[str, JAXArray]]:
    # init samples
    init_keys = jax.random.split(prng_key, int(nSample))
    batched_samples = jax.vmap(lambda k: seed(initSampler, rng_seed=k)())(init_keys)

    # batched loss
    losses = jax.lax.map(loss, batched_samples, batch_size=batch_size)

    # select top nBest
    loss_idx = jnp.argsort(losses)
    top_params = {}
    for p in batched_samples:
        top_params[p] = batched_samples[p][loss_idx[:nBest]]

    # convert from pytree to list of pytrees
    return [
        dict(zip(top_params.keys(), values, strict=False))
        for values in zip(*top_params.values(), strict=False)
    ]


def random_search(
    model: UniVarModel | MultiVarModel,
    initSampler: Callable,
    prng_key: jax.random.PRNGKey,
    nSample: int,
    nBest: int,
    *,
    batch_size: int = 1000,
    optimizer: optax.GradientTransformation = DEFAULT_ADAM_OPTIMIZER,
    n_opt_step: int = 1000,
    max_opt_step: int | None = None,
    tol: float | None = None,
    use_value_and_grad_from_state: bool = False,
    clear_cache_after_opt: bool = False,
) -> tuple[dict[str, JAXArray], JAXArray]:
    """Fit a model using random search plus local optimization.

    Args:
        model (UniVarModel | MultiVarModel): EzTaoX Light curve model.
        initSampler (Callable): Function to sample initial parameters.
        prng_key (jax.random.PRNGKey): Random number generator key.
        nSample (int): Number of random samples to draw.
        nBest (int): Number of best samples (selected based on their likelihod values)
            to keep for optimization.
        batch_size (int, optional): The batch size used in evaluating likehood of
            randomly drawn samples. Defaults to 1000.
        optimizer (optax.GradientTransformation, optional): Optimizer used in local
            optimization. Defaults to optax.adam(1e-2).
        n_opt_step (int, optional): Number of optimization steps per retained sample.
            Defaults to 1000 for the default adam optimizer.
        max_opt_step (int | None, optional): Maximum number of optimization steps when
            using the tolerance-based stopping criterion. Defaults to None.
        tol (float | None, optional): Gradient-norm tolerance for early stopping.
            This criterion is only used when max_opt_step is also provided. Defaults
            to None.
        use_value_and_grad_from_state (bool, optional): Whether to reuse value and
            gradients from the optimizer state when available. This is useful for
            Optax optimizers such as L-BFGS. Defaults to False.
        clear_cache_after_opt (bool, optional): Clear JAX caches after opt.
            Defaults to False.

    Returns:
        tuple[dict[str, JAXArray], JAXArray]: Best parameters and their log likelihood.
    """
    # first do random search to get good initial parameters
    loss = _make_loss(model)
    list_of_params = _sample_top_params(
        initSampler, prng_key, nSample, nBest, loss, batch_size
    )

    # then do local optimization
    solver = optax.with_extra_args_support(optimizer)
    step_fn = (
        _optimizer_step_from_state if use_value_and_grad_from_state else _optimizer_step
    )

    log_prob, param = [], []
    for item in list_of_params:
        params = item
        opt_state = solver.init(params)
        if max_opt_step is not None and tol is not None:
            for _ in range(max_opt_step):
                params, opt_state, val, grad = step_fn(params, opt_state, solver, loss)
                if optax.tree.norm(grad) < tol:
                    break
        else:
            for _ in range(n_opt_step):
                params, opt_state, val, grad = step_fn(params, opt_state, solver, loss)
        final_loss = loss(params)
        log_prob.append(-final_loss)
        param.append(params)

    best_param = param[jnp.argmax(jnp.asarray(log_prob))]

    if clear_cache_after_opt:
        jax.clear_caches()

    return best_param, max(log_prob)


def simple_optimizer(
    model: UniVarModel | MultiVarModel,
    initSample: dict[str, JAXArray],
    *,
    optimizer: optax.GradientTransformation = DEFAULT_ADAM_OPTIMIZER,
    n_step: int = 3000,
    use_value_and_grad_from_state: bool = False,
) -> tuple[
    dict[str, JAXArray], tuple[dict[str, JAXArray], JAXArray, dict[str, JAXArray]]
]:
    """Fit a model using a simple optimizer.

    Args:
        model (UniVarModel | MultiVarModel): EzTaoX Light curve model.
        initSample (dict[str, JAXArray]): The initial guess of parameters.
        optimizer (optax.GradientTransformation): Optimizer to use.
        n_step (int): Number of optimization steps.
        use_value_and_grad_from_state (bool, optional): Whether to reuse value and
            gradients from the optimizer state when available. This is useful for
            Optax optimizers such as L-BFGS. Defaults to False.

    Returns:
        tuple[dict[str, JAXArray], tuple[dict[str, JAXArray], JAXArray,
        dict[str, JAXArray]]]:
        Best parameters, (parameter history, loss history, gradient history).
    """

    loss = _make_loss(model)

    param_hist, loss_hist, grad_hist = [], [], []
    params = initSample.copy()
    solver = optax.with_extra_args_support(optimizer)
    opt_state = solver.init(params)
    step_fn = (
        _optimizer_step_from_state if use_value_and_grad_from_state else _optimizer_step
    )
    for _ in range(n_step):
        param_hist.append(params)
        params, opt_state, val, grad = step_fn(params, opt_state, solver, loss)
        loss_hist.append(val)
        grad_hist.append(grad)

    param_hist = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *param_hist)
    grad_hist = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *grad_hist)
    return params, (param_hist, jnp.asarray(loss_hist), grad_hist)
