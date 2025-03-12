"""
This module contains the fitter functions that fits a model to data.
"""
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jaxopt
import optax
from tinygp.helpers import JAXArray

from eztaox.models import MultiVarModel, UniVarModel


def fit(
    model: UniVarModel | MultiVarModel,
    optimizer: optax.GradientTransformation,
    initSampler: Callable,
    prng_key: jax.random.PRNGKey,
    nSample: int,
    nIter: int,
    nBest: int,
    jaxoptMethod: str = "SLSQP",
    batch_size: int = 1000,
) -> tuple[dict[str, JAXArray], JAXArray]:
    # define loss
    @jax.jit
    def loss(params) -> JAXArray:
        return -model.log_prob(params)

    def single_loop(params) -> tuple[dict[str, JAXArray], JAXArray]:
        opt_state = optimizer.init(params)
        for _ in range(nIter):
            grads = jax.grad(loss)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

        return params, loss(params)

    # init samples
    initSamples = initSampler(prng_key, nSample)

    # adam loops
    opt_params, losses = jax.lax.map(single_loop, initSamples, batch_size=batch_size)

    # select top nBest
    loss_idx = jnp.argsort(losses)
    top_params = {}
    for p in opt_params:
        top_params[p] = opt_params[p][loss_idx[:nBest]]

    # convert from pytree to list of pytrees
    list_of_parmas = [
        dict(zip(top_params.keys(), values)) for values in zip(*top_params.values())
    ]

    # jaxopt optimize
    opt = jaxopt.ScipyMinimize(fun=loss, method=jaxoptMethod)
    log_prob, param = [], []
    for item in list_of_parmas:
        soln = opt.run(item)
        # if soln.state.success is np.True_:
        log_prob.append(-soln.state.fun_val)
        param.append(soln.params)
    best_param = param[jnp.argmax(jnp.asarray(log_prob))]

    return best_param, max(log_prob)
