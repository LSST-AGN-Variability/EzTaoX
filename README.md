# EzTao*X*

[![DOI](https://zenodo.org/badge/928658755.svg)](https://doi.org/10.5281/zenodo.17467662)
[![PyPI](https://img.shields.io/pypi/v/eztaox?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/eztaox/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/eztaox.svg?color=blue&logo=condaforge&logoColor=white)](https://anaconda.org/conda-forge/eztaox) 

[![Read the Docs](https://img.shields.io/readthedocs/eztaox)](https://eztaox.readthedocs.io/en/latest/index.html)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/LSST-AGN-Variability/EzTaoX/smoke-test.yml)](https://github.com/LSST-AGN-Variability/EzTaoX/actions/workflows/smoke-test.yml)
[![codecov](https://codecov.io/gh/LSST-AGN-Variability/EzTaoX/branch/main/graph/badge.svg)](https://codecov.io/gh/LSST-AGN-Variability/EzTaoX)
[![benchmarks](https://img.shields.io/github/actions/workflow/status/LSST-AGN-Variability/EzTaoX/asv-main.yml?label=benchmarks)](https://LSST-AGN-Variability.github.io/EzTaoX/)

<!--TODO: the mybinder badge-->

`EzTaoX` is a flexible framework for multi-wavelength and multi-survey AGN light-curve modeling using Gaussian Processes (GPs).  Built on top of `tinygp`---a scalable GP library in `JAX`---`EzTaoX` is fast, modular, and integrates seamlessly with the `JAX` ecosystem for statistical inference and modern machine learning.

> #### `EzTaoX` is under active development, breaking API changes are expected.

### Installation
```
pip install eztaox
```
#### Dependencies
##### Supports Python 3.10, 3.11, 3.12
```
"jax (<=0.4.31)",
"jaxlib (<=0.4.31)",
"tinygp (>=0.3.0,<0.4.0)",
"jaxopt (>=0.8.3,<0.9.0)",
"optax (>=0.2.4,<0.3.0)",
"numpyro (>=0.17.0,<0.20.0)",
```

### Documentation
Please see our [readthedocs](https://eztaox.readthedocs.io/)

### Citation
If you find `EzTaoX` useful for your research, please consider citing the following paper [arXiv:2511.21479](https://arxiv.org/abs/2511.21479):
```
@ARTICLE{Yu2025,
       author = {{Yu}, Weixiang and {Ruan}, John J. and {Burke}, Colin J. and {Assef}, Roberto J. and {Ananna}, Tonima T. and {Bauer}, Franz E. and {De Cicco}, Demetra and {Horne}, Keith and {Hern{\'a}ndez-Garc{\'\i}a}, Lorena and {Ili{\'c}}, Dragana and {Kova{\v{c}}evi{\'c}}, Andjelka B. and {Marculewicz}, Marcin and {Panda}, Swayamtrupta and {Ricci}, Claudio and {Richards}, Gordon T. and {Riffel}, Rogemar A. and {Schneider}, Donald P. and {S{\'a}nchez-S{\'a}ez}, Paula and {Satheesh Sheeba}, Sarath and {Tombesi}, Francesco and {Temple}, Matthew J. and {Vogeley}, Michael S. and {Yoon}, Ilsang and {Zou}, Fan},
        title = "{Scalable and Robust Multiband Modeling of AGN Light Curves in Rubin-LSST}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics of Galaxies, Instrumentation and Methods for Astrophysics},
         year = 2025,
        month = nov,
          eid = {arXiv:2511.21479},
        pages = {arXiv:2511.21479},
          doi = {10.48550/arXiv.2511.21479},
archivePrefix = {arXiv},
       eprint = {2511.21479},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv251121479Y},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

### Acknowledgment
`EzTaoX` is built on top of (and inspired by) [`tinygp`](https://github.com/dfm/tinygp)---a general purpose GP modeling framework written in `JAX`. For more general GP modeling tasks, experienced users can directly explore `tinygp`.
