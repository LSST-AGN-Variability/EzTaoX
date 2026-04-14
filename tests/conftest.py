import os

import jax
import numpy as np
import pytest

DATA_DIR_NAME = "data"
TEST_DIR = os.path.dirname(__file__)


@pytest.fixture
def test_data() -> dict[str, np.ndarray]:
    test_data_dir = os.path.join(TEST_DIR, DATA_DIR_NAME)

    # load test data
    data = np.load(test_data_dir + "/unit_test_lc.npz")
    return data


@pytest.fixture
def basekey_seed() -> int:
    return 10


def pytest_addoption(parser):
    cache_dir = os.path.join(os.getcwd(), ".jax_cache")
    parser.addoption("--jax_cache", action="store", default=cache_dir)


def pytest_configure(config):
    cache_dir = config.getoption("--jax_cache")
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
