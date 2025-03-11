# %% [markdown]
# ## Fitting DHO (Single-Band)
# **Note:** 
# - The CARMA autoregressive parameter index follows the covention of Kelly+14, which is different from the one used in Moreno+19.
# - You need to import CARMA kernel from EzTaoX.kernels. The one from tinygp doesn't work (I will fix it later)

# %%
import warnings

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
from eztao.carma import CARMA_term, DHO_term, DRW_term
from eztao.ts import addNoise, gpSimFull, gpSimRand
from eztaox.fitter import fit
from eztaox.initializers import CARMAInit, DRWInit, UniformInit
from eztaox.kernels import CARMA
from eztaox.models import MultiVarModel, UniVarModel
from eztaox.utils import formatlc
from numpyro.infer import ESS, MCMC, NUTS, SA, BarkerMH
from eztaox.kernel_utils import carma_sf, carma_psd
from functools import partial

# from tinygp import kernels

warnings.filterwarnings(
    "ignore", category=RuntimeWarning
)  # Ignore ArviZ underflow warnings

jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ### 1. Simulate LC

# %%
# note that here alpha parameter index are defined in this cell following the convention used in Moreno+19, which is the opposite to the one used in Kelly+14.
alphas = {"g": [0.06, 0.0001], "i": [0.06, 0.0001]}
betas = {"g": [0.005, 0.2], "i": [0.005, 0.2]}
snrs = {"g": 10, "i": 10}
sampling_seeds = {"g": 2, "i": 5}
noise_seeds = {"g": 111, "i": 2}

ts, ys, yerrs = {}, {}, {}
ys_noisy = {}
seed = 10
for band in "gi":
    DHO_kernel = DHO_term(*np.log(alphas[band]), *np.log(betas[band]))
    t, y, yerr = gpSimRand(
        DHO_kernel,
        snrs[band],
        365 * 10,
        200,
        full_N=100_000,
        lc_seed=seed,
    )

    # add to dict
    ts[band] = t
    ys[band] = y
    yerrs[band] = yerr
    ys_noisy[band] = addNoise(ys[band], yerrs[band], seed=noise_seeds[band] + seed)

for b in "g":
    plt.errorbar(ts[b][::1], ys_noisy[b][::1], yerrs[b][::1], fmt=".")
plt.xlabel('Time')
plt.ylabel('Flux')

# %% [markdown]
# ### 2. Single Band Fit
# #### 2.1 Best Fit

# %% [markdown]
# To fit DHO, you need to initiate a CARMA kernel by calling `CARMA.init`, which takes two arrays as arguments: alpha and beta. As noted above, the alpha parameter index should follow the convention used in Kelly+14.

# %%
# define params
zero_mean = False
has_jitter = True
p = 2
test_params = {"log_kernel_param": jnp.log(np.array([0.1, 1.1, 1.0, 3.0]))}

# define model
k = CARMA.init(
    jnp.exp(test_params["log_kernel_param"][:p]),
    jnp.exp(test_params["log_kernel_param"][p:]),
)
m = UniVarModel(
    ts["g"], ys_noisy["g"], yerrs["g"], k, zero_mean=zero_mean, has_jitter=has_jitter
)
m

# %% [markdown]
# The `CARMAInit` function takes four input:
# - CARMA p order
# - CARMA q order
# - alpha parameters
# - beta parameter

# %%
# for CARMA kernel, the learning_rate needs to be larger
optimizer = optax.adam(learning_rate=1)

def initSampler2(key, nSample):
    # split keys
    subkeys = jax.random.split(key, 10)

    # uniform sampler
    meanSampler = UniformInit(1, [-0.3, 0.3])
    logJitterSampler = UniformInit(1, [-20, -5])

    # kernel init
    kernelSampler = CARMAInit(2, 1, [-16.0, 10.0], [-10.0, 2.0])

    return {
        "log_kernel_param": kernelSampler(subkeys[0], nSample),
        "mean": meanSampler(subkeys[2], nSample),
        "log_jitter": logJitterSampler(subkeys[4], nSample),
    }


initSample2 = initSampler2(jax.random.PRNGKey(10), 1)
initSample2

# %%
prng_key = jax.random.PRNGKey(1)
nInitSample = 10_000
nIter = 3
nBest = 5
jaxoptMethod = "SLSQP"

bestP, logProb = fit(
    m, optimizer, initSampler2, prng_key, nInitSample, nIter, nBest, batch_size=1000
)
bestP, logProb

# %% [markdown]
# #### 2.2 MCMC

# %%
params1 = bestP
prior_sigma = 2

def numpyro_model(t, yerr, y=None):
    # kernel param
    flat_normal = dist.Normal(
        params1["log_kernel_param"],
        jnp.array([prior_sigma, prior_sigma, prior_sigma, prior_sigma]),
    )
    diag_normal = dist.Independent(flat_normal, 1)
    log_kernel_param = numpyro.sample("log_kernel_param", diag_normal)

    # log jitter
    log_jitter = numpyro.sample("log_jitter", dist.Normal(params1["log_jitter"], 2.0))
    mean = numpyro.sample("mean", dist.Normal(0.0, 0.1))

    # kernel
    k = CARMA.init(jnp.exp(log_kernel_param)[:p], jnp.exp(log_kernel_param)[p:])

    m = UniVarModel(
        t,
        y,
        yerr,
        k,
        zero_mean=zero_mean,
        has_jitter=has_jitter,
    )

    sample_params = {
        "log_kernel_param": log_kernel_param,
        "log_jitter": log_jitter,
        "mean": mean,
    }
    m.sample(sample_params)

# %%
# %%time
nuts_kernel = NUTS(
    numpyro_model,
    dense_mass=True,
    target_accept_prob=0.9,
    # adapt_step_size=True,
)

mcmc = MCMC(
    nuts_kernel,
    num_warmup=500,
    num_samples=1000,
    num_chains=1,
    progress_bar=True,
    # chain_method="vectorized",
)

seed = 10
test_band = "g"
mcmc.run(
    jax.random.PRNGKey(seed),
    jnp.asarray(ts[test_band]),
    jnp.asarray(yerrs[test_band]),
    y=jnp.asarray(ys_noisy[test_band]),
)
data = az.from_numpyro(mcmc)
mcmc.print_summary()

# %%
az.plot_pair(data, var_names=["log_kernel_param"])
az.plot_posterior(data, var_names=["log_kernel_param", "mean"])
az.plot_trace(data, var_names=["log_kernel_param", "mean"])

# %% [markdown]
# ### 3. Visualize LC

# %%
sim_t = jnp.linspace(0, 3650, 1000)
sim_y, sim_yerr = m.pred(bestP, sim_t)

# %%
plt.errorbar(ts["g"], ys_noisy["g"], yerrs["g"], fmt="k.", label='Input')
plt.plot(sim_t, sim_y, label='Pred LC')
plt.legend()
plt.xlabel('time')
plt.ylabel('flux')

# %% [markdown]
# ### 4. Visualize PSD/SF

# %%
from eztao.carma import carma_psd as eztao_carma_psd
from eztao.carma import carma_sf as eztao_carma_sf

# %% [markdown]
# #### 4.1 PSD

# %%
# get mcmc sample
samples = mcmc.get_samples()

f = jnp.logspace(-4, -1)
# best fit PSD

best_dho_param = jnp.exp(bestP['log_kernel_param'])
best_psd = carma_psd(f, best_dho_param[:p], best_dho_param[p:])

# MCMC psd; warning: high memory usage
mcmc_dho_params = jnp.exp(samples['log_kernel_param']).T
mcmc_dho_params = mcmc_dho_params[:, ::5] # select a sub_sample
mcmc_psds = jax.vmap(partial(carma_psd, f))(mcmc_dho_params[:p].T, mcmc_dho_params[p:].T)

mcmc_psd_gmean = 10**(jnp.log10(mcmc_psds).mean(axis=0))
mcmc_psd_p16 = jnp.percentile(mcmc_psds, 16, axis=0)
mcmc_psd_p84 = jnp.percentile(mcmc_psds, 84, axis=0)

# true psd
true_psd = eztao_carma_psd(alphas['g'], betas['g'])

# %%
plt.loglog(f, true_psd(f), label='True', color='k')
plt.loglog(f, mcmc_psd_gmean, label='MCMC Mean', color='tab:orange')
plt.fill_between(f, mcmc_psd_p16, mcmc_psd_p84, color='tab:orange', alpha=0.2)
plt.loglog(f, best_psd, label='MLE')
plt.legend()
plt.xlabel(r'Frequeny/$2\pi$')
plt.ylabel('PSD')

# %% [markdown]
# #### 4.2 SF

# %%
# get mcmc sample
# samples = mcmc.get_samples()

t = np.logspace(-1, 4)
# best fit PSD

best_sf = carma_sf(t, best_dho_param[:p], best_dho_param[p:])

# MCMC psd; warning: high memory usage
mcmc_sfs = jax.vmap(partial(carma_sf, t))(mcmc_dho_params[:p].T, mcmc_dho_params[p:].T)
mcmc_sf_gmean = 10**(jnp.log10(mcmc_sfs).mean(axis=0))
mcmc_sf_p16 = jnp.percentile(mcmc_sfs, 16, axis=0)
mcmc_sf_p84 = jnp.percentile(mcmc_sfs, 84, axis=0)

# true sf
true_sf = eztao_carma_sf(np.array(alphas['g']), np.array(betas['g']))

# %%
plt.loglog(t, true_sf(t), label='True', color='k')
plt.loglog(t, mcmc_sf_gmean, label='MCMC Mean', color='tab:orange')
plt.fill_between(t, mcmc_sf_p16, mcmc_sf_p84, color='tab:orange', alpha=0.2)
plt.loglog(t, best_sf, label='MLE')
plt.legend()
plt.xlabel(r'Time')
plt.ylabel('SF')

# %%
