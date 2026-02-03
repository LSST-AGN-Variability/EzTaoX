# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from eztao.carma import DRW_term
from eztao.ts import addNoise, gpSimRand

mpl.rcParams.update(
    {
        "text.usetex": True,
        "axes.labelsize": 20,
        "figure.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.constrained_layout.wspace": 0,
        "figure.constrained_layout.hspace": 0,
        "figure.constrained_layout.h_pad": 0,
        "figure.constrained_layout.w_pad": 0,
        "axes.linewidth": 1.3,
    }
)

import jax
import jax.numpy as jnp

# you should always set this
jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Damped Random Walk (DRW) Fitting

# %% [markdown]
# ### 1. Light Curve Simulation

# %%
amps = {"g": 0.35}
taus = {"g": 100}
snrs = {"g": 5}  # ratio of the DRW amplitude to the median error bar
sampling_seeds = {"g": 2}  # seed for random sampling
noise_seeds = {"g": 11}  # seed for mocking observational noise

ts, ys, yerrs = {}, {}, {}
ys_noisy = {}
seed = 1
for band in "g":
    DRW_kernel = DRW_term(np.log(amps[band]), np.log(taus[band]))
    t, y, yerr = gpSimRand(
        DRW_kernel,
        snrs[band],
        365 * 10,  # 10 year LC
        100,
        lc_seed=seed,
        downsample_seed=sampling_seeds[band],
    )

    # add to dict
    ts[band] = t
    ys[band] = y
    yerrs[band] = yerr
    # add simulated photometric noise
    ys_noisy[band] = addNoise(ys[band], yerrs[band], seed=noise_seeds[band] + seed)

for b in "g":
    plt.errorbar(
        ts[b][::1], ys_noisy[b][::1], yerrs[b][::1], fmt=".", label=f"{b}-band"
    )

plt.xlabel("Time (day)")
plt.ylabel("Flux (mag)")
plt.legend(fontsize=15)

# %% [markdown]
# ### 2. Fitting
# Here, we demonstrate how to use the `UniVarModel` for fitting single-band light curves.

# %%
import numpyro
import numpyro.distributions as dist
from eztaox.fitter import random_search
from eztaox.kernels.quasisep import Exp
from eztaox.models import UniVarModel
from numpyro.handlers import seed as numpyro_seed

# %% [markdown]
# #### 2.1 Initialize Light Curve Model

# %%
zero_mean = False

# initialize a GP kernel, note the initial parameters are not used in the fitting
k = Exp(scale=100.0, sigma=1.0)
m = UniVarModel(ts["g"], ys_noisy["g"], yerrs["g"], k, zero_mean=zero_mean)
m


# %% [markdown]
# #### 2.2 Define InitSampler


# %%
def initSampler():
    # GP kernel param
    log_drw_scale = numpyro.sample(
        "drw_scale", dist.Uniform(jnp.log(0.01), jnp.log(1000))
    )
    log_drw_sigma = numpyro.sample(
        "drw_sigma", dist.Uniform(jnp.log(0.01), jnp.log(10))
    )
    log_kernel_param = jnp.stack([log_drw_scale, log_drw_sigma])
    numpyro.deterministic("log_kernel_param", log_kernel_param)

    # mean
    mean = numpyro.sample("mean", dist.Uniform(low=-0.2, high=0.2))

    sample_params = {"log_kernel_param": log_kernel_param, "mean": mean}
    return sample_params


# %%
# generate a random initial guess
sample_key = jax.random.PRNGKey(1)
prior_sample = numpyro_seed(initSampler, rng_seed=sample_key)()
prior_sample

# %% [markdown]
# #### 2.3 MLE Fitting

# %%
# %%time
model = m
sampler = initSampler
fit_key = jax.random.PRNGKey(1)
nSample = 10_000
nBest = 10  # it seems like this number needs to be high

bestP, ll = random_search(model, initSampler, fit_key, nSample, nBest)
bestP

# %%
print("True DRW Params (in natual log):")
print(np.log(np.hstack([taus["g"], amps["g"]])))
print("MLE DHO Params (in natual log):")
print(bestP["log_kernel_param"])

# %% [markdown]
# ### 3. MCMC

# %%
import arviz as az
from numpyro.infer import MCMC, NUTS, init_to_median


# %%
def numpyro_model(t, yerr, y=None):
    # GP kernel param
    log_drw_scale = numpyro.sample(
        "log_drw_scale", dist.Uniform(jnp.log(0.01), jnp.log(1000))
    )
    log_drw_sigma = numpyro.sample(
        "log_drw_sigma", dist.Uniform(jnp.log(0.01), jnp.log(10))
    )
    log_kernel_param = jnp.stack([log_drw_scale, log_drw_sigma])
    numpyro.deterministic("log_kernel_param", log_kernel_param)

    # mean: use a normal prior for better convergence
    mean = numpyro.sample("mean", dist.Normal(0.0, 0.1))

    sample_params = {"log_kernel_param": log_kernel_param, "mean": mean}

    # the following is different from the initSampler
    zero_mean = False

    k = Exp(scale=100.0, sigma=1.0)  # init params for k are not used
    m = UniVarModel(ts["g"], ys_noisy["g"], yerrs["g"], k, zero_mean=zero_mean)
    m.sample(sample_params)


# %%
# %%time
nuts_kernel = NUTS(
    numpyro_model,
    dense_mass=True,
    target_accept_prob=0.9,
    init_strategy=init_to_median,
)

mcmc = MCMC(
    nuts_kernel,
    num_warmup=1000,
    num_samples=5000,
    num_chains=1,
    # progress_bar=False,
)

mcmc_seed = 0
mcmc.run(jax.random.PRNGKey(mcmc_seed), ts["g"], yerrs["g"], y=ys_noisy["g"])
data = az.from_numpyro(mcmc)
mcmc.print_summary()

# %% [markdown]
# #### Visualize Chains, Posterior Distributions

# %%
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# %%
az.plot_trace(data, var_names=["log_drw_scale", "log_drw_sigma", "mean"])
plt.subplots_adjust(hspace=0.4)

# %%
az.plot_pair(data, var_names=["log_drw_scale", "log_drw_sigma", "mean"])

# %% [markdown]
# ### 4. Second-order Statistics

# %%
from eztaox.kernel_stat2 import gpStat2

ts = np.logspace(0, 4)
fs = np.logspace(-4, 0)

# %%
# get MCMC samples
flatPost = data.posterior.stack(sample=["chain", "draw"])
log_drw_draws = flatPost["log_kernel_param"].values.T

# %%
# create second-order stat object
drw_k = Exp(scale=taus["g"], sigma=amps["g"])
gpStat2_drw = gpStat2(drw_k)

# %% [markdown]
# #### 4.1 Structure Function

# %%
# compute sf for MCMC draws
mcmc_sf = jax.vmap(gpStat2_drw.sf, in_axes=(None, 0))(ts, jnp.exp(log_drw_draws))

# %%
## plot
# ture SF
plt.loglog(ts, gpStat2_drw.sf(ts), c="k", label="True SF", zorder=100, lw=2)
plt.legend(fontsize=15)
# MCMC SFs
for sf in mcmc_sf[::50]:
    plt.loglog(ts, sf, c="tab:green", alpha=0.15)

plt.xlabel("Time")
plt.ylabel("SF")

# %% [markdown]
# #### 4.1 Power Spectral Density (PSD)

# %%
# compute sf for MCMC draws
mcmc_psd = jax.vmap(gpStat2_drw.psd, in_axes=(None, 0))(fs, jnp.exp(log_drw_draws))

# %%
## plot
# ture PSD
plt.loglog(fs, gpStat2_drw.psd(fs), c="k", label="True PSD", zorder=100, lw=2)
plt.legend(fontsize=15)

# MCMC PSDs
for psd in mcmc_psd[::50]:
    plt.loglog(fs, psd, c="tab:green", alpha=0.15)

plt.xlabel("Frequency")
plt.ylabel("PSD")
