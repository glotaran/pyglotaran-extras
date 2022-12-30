"""Tests for ``pyglotaran_extras.inspect.a_matrix``."""

from typing import Any

import pytest
import xarray as xr
from glotaran.project import Result
from tests import TEST_DATA

from pyglotaran_extras.inspect.a_matrix import a_matrix_to_html_table
from pyglotaran_extras.inspect.a_matrix import show_a_matrixes


@pytest.mark.parametrize(
    "kwargs, compare_file_suffix",
    (
        ({}, "default"),
        ({"normalize_initial_concentration": True}, "normalized"),
        ({"decimal_places": 2}, "decimal_2"),
    ),
)
def test_a_matrix_to_html_table(
    result_parallel_spectral_decay: Result, kwargs: dict[str, Any], compare_file_suffix: str
):
    """Same string as in test file except final newline added by editors."""
    expected = (
        TEST_DATA / f"a_matrix/a_matrix_to_html_table_{compare_file_suffix}.md"
    ).read_text()
    assert a_matrix_to_html_table(
        result_parallel_spectral_decay.data["dataset_1"].a_matrix_megacomplex_parallel_decay,
        "megacomplex_parallel_decay",
        **kwargs,
    ) == expected.rstrip("\n")


@pytest.mark.parametrize(
    "kwargs, compare_file_suffix",
    (
        ({}, "default"),
        ({"normalize_initial_concentration": True}, "normalized"),
        ({"decimal_places": 2}, "decimal_2"),
        ({"expanded_datasets": ("dataset_2",)}, "expanded_dataset_2"),
        ({"heading_offset": 0}, "heading_offset_0"),
    ),
)
def test_show_a_matrixes(
    result_parallel_spectral_decay: Result,
    result_sequential_spectral_decay: Result,
    kwargs: dict[str, Any],
    compare_file_suffix: str,
):
    """Same string as in test file except final newline added by editors."""
    expected = (TEST_DATA / f"a_matrix/show_a_matrixes_{compare_file_suffix}.md").read_text()

    result = result_parallel_spectral_decay
    result.data["dataset_2"] = result_sequential_spectral_decay.data["dataset_1"]
    # dummy data for filtering based on a-matrix size
    single_entry_data = result_sequential_spectral_decay.data[
        "dataset_1"
    ].a_matrix_megacomplex_sequential_decay[:1, :1]
    single_entry_data = single_entry_data.rename(
        {
            name: name.replace("megacomplex_sequential_decay", "single_entry")
            for name in single_entry_data.coords
        }
    )
    result.data["single_entry_a_matrix"] = xr.Dataset({"a_matrix_single_entry": single_entry_data})

    assert str(show_a_matrixes(result, **kwargs)) == expected.rstrip("\n")
