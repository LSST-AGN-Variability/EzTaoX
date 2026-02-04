"""
This module contains the fitter functions that fits a model to data.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jaxopt
import optax
from numpyro.handlers import seed
from tinygp.helpers import JAXArray

from eztaox.models import MultiVarModel, UniVarModel


def random_search(
    model: UniVarModel | MultiVarModel,
    initSampler: Callable,
    prng_key: jax.random.PRNGKey,
    nSample: int,
    nBest: int,
    jaxoptMethod: str = "SLSQP",
    batch_size: int = 1000,
) -> tuple[dict[str, JAXArray], JAXArray]:
    """Fit a model using random search plus optimization.

    Args:
        model (UniVarModel | MultiVarModel): EzTaoX Light curve model.
        initSampler (Callable): Function to sample initial parameters.
        prng_key (jax.random.PRNGKey): Random number generator key.
        nSample (int): Number of random samples to draw.
        nBest (int): Number of best samples (selected based on their likelihod values)
            to keep for optimization.
        jaxoptMethod (str, optional): Optimization algorithm. Defaults to "SLSQP".
        batch_size (int, optional): The batch size used in evaluating likehood of
            randomly drawn samples. Defaults to 1000.

    Returns:
        tuple[dict[str, JAXArray], JAXArray]: Best parameters and their log likelihood.
    """

    # define loss
    @jax.jit
    def loss(params) -> JAXArray:
        return -model.log_prob(params)

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
    list_of_params = [
        dict(zip(top_params.keys(), values, strict=False))
        for values in zip(*top_params.values(), strict=False)
    ]

    # jaxopt optimize
    opt = jaxopt.ScipyMinimize(fun=loss, method=jaxoptMethod)
    log_prob, param = [], []
    for item in list_of_params:
        soln = opt.run(item)
        log_prob.append(-soln.state.fun_val)
        param.append(soln.params)
    best_param = param[jnp.argmax(jnp.asarray(log_prob))]

    return best_param, max(log_prob)


def simpleOptimizer(  # noqa: N802
    model: UniVarModel | MultiVarModel,
    optimizer: optax.GradientTransformation,
    initSample: dict[str, JAXArray],
    nStep: int,
) -> tuple[
    dict[str, JAXArray], tuple[dict[str, JAXArray], JAXArray, dict[str, JAXArray]]
]:
    """Fit a model using a simple optimizer.

    Args:
        model (UniVarModel | MultiVarModel): EzTaoX Light curve model.
        optimizer (optax.GradientTransformation): Optimizer to use.
        initSample (dict[str, JAXArray]): The initial guess of parameters.
        nStep (int): Number of optimization steps.

    Returns:
        tuple[dict, tuple[dict, JAXArray, dict]]: Best parameters, (parameter history,
            loss history, gradient history).
    """

    @jax.jit
    def loss(params) -> JAXArray:
        return -model.log_prob(params)

    param_hist, loss_hist, grad_hist = [], [], []
    params = initSample.copy()
    opt_state = optimizer.init(params)
    for _ in range(nStep):
        # compute loss, grad for current param & save to hist
        val, grad = jax.value_and_grad(loss)(params)
        param_hist.append(params)
        loss_hist.append(val)
        grad_hist.append(grad)

        # update
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

    param_hist = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *param_hist)
    grad_hist = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *grad_hist)
    return params, (param_hist, jnp.asarray(loss_hist), grad_hist)
