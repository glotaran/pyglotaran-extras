"""Tests for ``pyglotaran_extras.io.load_data``."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import xarray as xr
from glotaran.io import load_result

from pyglotaran_extras.io.load_data import load_data

if TYPE_CHECKING:
    from _pytest.recwarn import WarningsRecorder
    from glotaran.project.result import Result

MULTI_DATASET_WARING = (
    "Result contains multiple datasets, auto selecting 'dataset_1'.\n"
    "Pass the dataset set you want to plot (e.g. result.data['dataset_1']) , "
    "to deactivate this Warning.\nPossible dataset names are: ['dataset_1', 'dataset_2']"
)


def run_load_data_test(result: xr.Dataset, compare: xr.Dataset | None = None):
    """Factored out test runner function for ``test_load_data``."""
    assert isinstance(result, xr.Dataset)
    assert hasattr(result, "data")
    if compare is not None:
        assert result.equals(compare)


def test_load_data(
    result_sequential_spectral_decay: Result, tmp_path: Path, recwarn: WarningsRecorder
):
    """All input_type permutations result in a ``xr.Dataset``."""
    compare_dataset = result_sequential_spectral_decay.data["dataset_1"]

    from_result = load_data(result_sequential_spectral_decay)

    run_load_data_test(from_result, compare_dataset)

    from_dataset = load_data(compare_dataset)

    run_load_data_test(from_dataset, compare_dataset)

    result_sequential_spectral_decay.save(tmp_path / "result.yml")

    from_file = load_data(tmp_path / "dataset_1.nc")

    run_load_data_test(from_file, compare_dataset)

    data_array = xr.DataArray([[1, 2], [3, 4]])
    from_data_array = load_data(data_array)

    run_load_data_test(from_data_array)
    assert data_array.equals(from_data_array.data)

    # No warning til now
    assert len(recwarn) == 0

    # Ensure not to mutate original fixture
    result_multi_dataset = load_result(tmp_path / "result.yml")
    result_multi_dataset.data["dataset_2"] = xr.Dataset({"foo": [1]})

    from_result_multi_dataset = load_data(result_multi_dataset)

    run_load_data_test(from_result_multi_dataset, compare_dataset)

    assert len(recwarn) == 1

    assert recwarn[0].category == UserWarning
    assert recwarn[0].message.args[0] == MULTI_DATASET_WARING
    assert Path(recwarn[0].filename) == Path(__file__)

    def wrapped_call(result: Result):
        return load_data(result, _stacklevel=3)

    result_wrapped_call = wrapped_call(result_multi_dataset)

    run_load_data_test(result_wrapped_call, compare_dataset)

    assert len(recwarn) == 2

    assert recwarn[1].category == UserWarning
    assert recwarn[1].message.args[0] == MULTI_DATASET_WARING
    assert Path(recwarn[1].filename) == Path(__file__)

    with pytest.raises(TypeError) as excinfo:
        load_data([1, 2])

    assert str(excinfo.value) == (
        "Result needs to be of type typing.Union[xarray.core.dataset.Dataset, "
        "xarray.core.dataarray.DataArray, str, pathlib.Path], but was [1, 2]."
    )
