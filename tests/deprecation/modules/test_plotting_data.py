from pathlib import Path

import pytest

from pyglotaran_extras.deprecation.deprecation_utils import PyglotaranExtrasApiDeprecationWarning


def test_io_boilerplate():
    """Importing from ``pyglotaran_extras.plotting.data`` raises deprecation warning."""
    with pytest.warns(PyglotaranExtrasApiDeprecationWarning) as record:
        from pyglotaran_extras.plotting.data import plot_data_overview  # noqa:F401

        assert len(record) == 1
        assert Path(record[0].filename) == Path(__file__)
        assert "'pyglotaran_extras.plotting.data'" in record[0].message.args[0]
