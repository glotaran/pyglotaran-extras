"""Tests for ``pyglotaran_extras.inspect.a_matrix``."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pytest
import xarray as xr

from pyglotaran_extras.inspect.a_matrix import a_matrix_to_html_table
from pyglotaran_extras.inspect.a_matrix import BASE_LEGEND_LABEL_MAP
from pyglotaran_extras.inspect.a_matrix import show_a_matrixes
from tests import TEST_DATA

if TYPE_CHECKING:
    from glotaran.project import Result


@pytest.mark.parametrize(
    ("kwargs", "compare_file_suffix"),
    [
        ({}, "default"),
        ({"normalize_initial_concentration": True}, "normalized"),
        ({"decimal_places": 2}, "decimal_2"),
    ],
)
def test_a_matrix_to_html_table(
    result_parallel_spectral_decay: Result, kwargs: dict[str, Any], compare_file_suffix: str
):
    """Same string as in test file except final newline added by editors."""
    expected = (TEST_DATA / f"a_matrix/a_matrix_to_html_table_{compare_file_suffix}.md").read_text(
        encoding="utf8"
    )
    assert a_matrix_to_html_table(
        result_parallel_spectral_decay.data["dataset_1"].a_matrix_megacomplex_parallel_decay,
        "megacomplex_parallel_decay",
        **kwargs,
    ) == expected.rstrip("\n")


@pytest.mark.parametrize(
    ("kwargs", "compare_file_suffix"),
    [
        ({}, "default"),
        ({"normalize_initial_concentration": True}, "normalized"),
        ({"decimal_places": 2}, "decimal_2"),
        ({"expanded_datasets": ("dataset_2",)}, "expanded_dataset_2"),
        ({"heading_offset": 0}, "heading_offset_0"),
    ],
)
def test_show_a_matrixes(
    result_parallel_spectral_decay: Result,
    result_sequential_spectral_decay: Result,
    kwargs: dict[str, Any],
    compare_file_suffix: str,
):
    """Same string as in test file except final newline added by editors."""
    expected = (TEST_DATA / f"a_matrix/show_a_matrixes_{compare_file_suffix}.md").read_text(
        encoding="utf8"
    )

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


def test_show_a_matrixes_multiple_a_matrixes_in_dataset(
    result_sequential_spectral_decay: Result,
):
    """Add two new lines in front of headings except for the first in a dataset."""
    expected = (
        TEST_DATA / "a_matrix/show_a_matrixes_multiple_a_matrixes_in_dataset.md"
    ).read_text(encoding="utf8")

    single_entry_data = result_sequential_spectral_decay.data[
        "dataset_1"
    ].a_matrix_megacomplex_sequential_decay[:1, :1]

    a_matrix_one = single_entry_data.rename(
        {
            name: name.replace("megacomplex_sequential_decay", "megacomplex_one")
            for name in single_entry_data.coords
        }
    )
    a_matrix_two = single_entry_data.rename(
        {
            name: name.replace("megacomplex_sequential_decay", "megacomplex_two")
            for name in single_entry_data.coords
        }
    )
    dummy_dataset = xr.Dataset(
        {"a_matrix_megacomplex_one": a_matrix_one, "a_matrix_megacomplex_two": a_matrix_two}
    )
    assert str(show_a_matrixes(dummy_dataset)) == expected.rstrip("\n")


def test_a_matrix_to_html_table_scientific_decimal_places_and_empty_threshold():
    """Render scientific notation with custom precision and hide tiny values."""
    a_matrix = xr.DataArray(
        np.array([[-6.942e-4, 5.0e-12]], dtype=np.float64),
        dims=["lifetime_index", "species_megacomplex_test"],
        coords={
            "species_megacomplex_test": ("species_megacomplex_test", ["S1", "S2"]),
            "initial_concentration_megacomplex_test": (
                "species_megacomplex_test",
                [1.0, 1.0],
            ),
            "lifetime_megacomplex_test": ("lifetime_index", [1.0]),
        },
    )

    rendered = a_matrix_to_html_table(
        a_matrix,
        "megacomplex_test",
        decimal_places=3,
        scientific_decimal_places=1,
        empty_cell_threshold=1e-10,
    )

    assert "-6.9e-4" in rendered
    assert "-6.942e-04" not in rendered
    assert "5.0e-12" not in rendered


def test_a_matrix_to_html_table_species_label_map_default_and_custom():
    """Replace species labels with mapped legend labels when requested."""
    a_matrix = xr.DataArray(
        np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        dims=["lifetime_index", "species_megacomplex_test"],
        coords={
            "species_megacomplex_test": (
                "species_megacomplex_test",
                ["s1", "s2", "unmapped"],
            ),
            "initial_concentration_megacomplex_test": (
                "species_megacomplex_test",
                [1.0, 1.0, 1.0],
            ),
            "lifetime_megacomplex_test": ("lifetime_index", [1.0]),
        },
    )

    rendered_default = a_matrix_to_html_table(a_matrix, "megacomplex_test")
    assert f"{BASE_LEGEND_LABEL_MAP['s1']}<br>" in rendered_default
    assert f"{BASE_LEGEND_LABEL_MAP['s2']}<br>" in rendered_default
    assert "unmapped<br>" in rendered_default

    rendered_custom = a_matrix_to_html_table(
        a_matrix,
        "megacomplex_test",
        species_label_map={"s1": "custom-1", "s2": "custom-2"},
    )
    assert "custom-1<br>" in rendered_custom
    assert "custom-2<br>" in rendered_custom
    assert "unmapped<br>" in rendered_custom


def test_show_a_matrixes_defaults_scientific_decimal_places_to_decimal_minus_one():
    """Use one fewer decimal place for scientific notation when not explicitly configured."""
    a_matrix = xr.DataArray(
        np.array([[1.14e-3]], dtype=np.float64),
        dims=["lifetime_index", "species_megacomplex_test"],
        coords={
            "species_megacomplex_test": ("species_megacomplex_test", ["s1"]),
            "initial_concentration_megacomplex_test": ("species_megacomplex_test", [1.14e-3]),
            "lifetime_megacomplex_test": ("lifetime_index", [1.0]),
        },
    )
    dataset = xr.Dataset({"a_matrix_megacomplex_test": a_matrix})

    rendered = str(show_a_matrixes(dataset, decimal_places=2))

    assert "1.1e-3" in rendered
    assert "1.14e-3" not in rendered


def test_show_a_matrixes_decimal_places_scientific_argument():
    """Allow explicitly controlling scientific notation decimals with new argument."""
    a_matrix = xr.DataArray(
        np.array([[1.14e-3]], dtype=np.float64),
        dims=["lifetime_index", "species_megacomplex_test"],
        coords={
            "species_megacomplex_test": ("species_megacomplex_test", ["s1"]),
            "initial_concentration_megacomplex_test": ("species_megacomplex_test", [1.14e-3]),
            "lifetime_megacomplex_test": ("lifetime_index", [1.0]),
        },
    )
    dataset = xr.Dataset({"a_matrix_megacomplex_test": a_matrix})

    rendered = str(show_a_matrixes(dataset, decimal_places=2, decimal_places_scientific=2))

    assert "1.14e-3" in rendered


def test_show_a_matrixes_scientific_decimal_places_conflict_raises():
    """Reject conflicting scientific precision values from new and legacy argument names."""
    a_matrix = xr.DataArray(
        np.array([[1.14e-3]], dtype=np.float64),
        dims=["lifetime_index", "species_megacomplex_test"],
        coords={
            "species_megacomplex_test": ("species_megacomplex_test", ["s1"]),
            "initial_concentration_megacomplex_test": ("species_megacomplex_test", [1.14e-3]),
            "lifetime_megacomplex_test": ("lifetime_index", [1.0]),
        },
    )
    dataset = xr.Dataset({"a_matrix_megacomplex_test": a_matrix})

    with pytest.raises(ValueError, match="decimal_places_scientific"):
        show_a_matrixes(
            dataset,
            decimal_places=2,
            decimal_places_scientific=2,
            scientific_decimal_places=1,
        )
