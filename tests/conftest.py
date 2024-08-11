"""PyTest fixture and test helper definitions."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING
from typing import Any

# isort: off
# Hack around https://github.com/pydata/xarray/issues/7259 which also affects pyglotaran <= 0.7.0
import numpy  # noqa: F401
import netCDF4  # noqa: F401
# isort: on

from dataclasses import replace

import matplotlib.pyplot as plt
import pytest
from glotaran.optimization.optimize import optimize
from glotaran.testing.simulated_data.parallel_spectral_decay import SCHEME as SCHEME_PAR
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME as SCHEME_SEQ
from ruamel.yaml import YAML

from pyglotaran_extras.config.config import CONFIG_FILE_STEM
from pyglotaran_extras.config.config import Config
from pyglotaran_extras.io.setup_case_study import get_script_dir
from tests import TEST_DATA

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def monkeypatch_all(monkeypatch: pytest.MonkeyPatch, name: str, value: Any):
    """Context to monkeypatch all usages across modules."""
    with monkeypatch.context() as m:
        for module_name, module in sys.modules.items():
            if module_name.startswith("pyglotaran_extras") and hasattr(module, name):
                m.setattr(module, name, value)
        yield


def generator_is_exhausted(generator: Generator) -> bool:
    """Check if ``generator`` is exhausted.

    Parameters
    ----------
    generator : Generator
        Generator to check.

    Returns
    -------
    bool
    """
    is_empty = object()
    return next(generator, is_empty) is is_empty


def wrapped_get_script_dir() -> Path:
    """Test function for calls to get_script_dir used inside of other functions."""
    return get_script_dir(nesting=1)


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close all figures after each test function."""
    plt.close()


@pytest.fixture(scope="session")
def result_parallel_spectral_decay():
    """Test result from ``glotaran.testing.simulated_data.parallel_spectral_decay``."""
    scheme = replace(SCHEME_PAR, maximum_number_function_evaluations=1)
    return optimize(scheme)


@pytest.fixture(scope="session")
def result_sequential_spectral_decay():
    """Test result from ``glotaran.testing.simulated_data.sequential_spectral_decay``."""
    scheme = replace(SCHEME_SEQ, maximum_number_function_evaluations=1)
    return optimize(scheme)


@pytest.fixture
def mock_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Mock ``pathlib.Path.home`` to return ``tmp_path/"home"``."""

    mock_home_path = tmp_path / "home"
    mock_home_path.mkdir()

    with monkeypatch.context() as m:
        m.setattr(Path, "home", lambda: mock_home_path)
        MockPath = Path  # noqa: N806
        with monkeypatch_all(monkeypatch, "Path", MockPath):
            yield mock_home_path


@pytest.fixture
def test_config_file() -> Path:
    """Path to the test config file."""
    return TEST_DATA / "config/pygta_config.yml"


@pytest.fixture
def test_config_values(test_config_file: Path) -> dict[str, Any]:
    """Read test config into dict."""
    return YAML().load(test_config_file)


@pytest.fixture
def mock_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, test_config_file: Path
) -> Generator[tuple[Config, dict[str, Any]], None, None]:
    """Mock config with test config and empty function registry."""
    dest_path = tmp_path / f"{CONFIG_FILE_STEM}.yml"
    copyfile(test_config_file, dest_path)
    config = Config()
    config.load(dest_path)
    mock_registry: dict[str, Any] = {}
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    with (
        monkeypatch_all(monkeypatch, "CONFIG", config),
        monkeypatch_all(monkeypatch, "SCRIPT_DIR", project_dir),
        monkeypatch_all(monkeypatch, "__PlotFunctionRegistry", mock_registry),
    ):
        yield config, mock_registry
