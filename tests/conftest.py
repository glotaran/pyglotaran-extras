# isort: off
# Hack around https://github.com/pydata/xarray/issues/7259 which also affects pyglotaran <= 0.7.0
import numpy  # noqa
import netCDF4  # noqa

# isort: on

from dataclasses import replace

import pytest
from glotaran.optimization.optimize import optimize
from glotaran.testing.simulated_data.parallel_spectral_decay import SCHEME as scheme_par
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME as scheme_seq

from pyglotaran_extras.io.setup_case_study import get_script_dir


def wrapped_get_script_dir():
    """Testfunction for calls to get_script_dir used inside of other functions."""
    return get_script_dir(nesting=1)


@pytest.fixture(scope="session")
def result_parallel_spectral_decay():
    """Test result from ``glotaran.testing.simulated_data.parallel_spectral_decay``."""
    scheme = replace(scheme_par, maximum_number_function_evaluations=1)
    yield optimize(scheme)


@pytest.fixture(scope="session")
def result_sequential_spectral_decay():
    """Test result from ``glotaran.testing.simulated_data.sequential_spectral_decay``."""
    scheme = replace(scheme_seq, maximum_number_function_evaluations=1)
    yield optimize(scheme)
