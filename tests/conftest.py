import os

import jax
import pytest

DATA_DIR_NAME = "data"
TEST_DIR = os.path.dirname(__file__)


@pytest.fixture
def test_data_dir() -> str:
    return os.path.join(TEST_DIR, DATA_DIR_NAME)


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
