"""Module containing a-matrix render functionality."""

from __future__ import annotations

import numpy as np
import xarray as xr
from glotaran.utils.ipython import MarkdownStr
from tabulate import tabulate

from pyglotaran_extras.inspect.utils import pretty_format_numerical
from pyglotaran_extras.inspect.utils import pretty_format_numerical_iterable
from pyglotaran_extras.inspect.utils import wrap_in_details_tag
from pyglotaran_extras.io.utils import result_dataset_mapping
from pyglotaran_extras.types import ResultLike


def a_matrix_to_html_table(
    a_matrix: xr.DataArray,
    megacomplex_suffix: str,
    *,
    normalize_initial_concentration: bool = False,
    decimal_places: int = 3,
) -> str:
    """Create HTML multi header table from a-matrix.

    Parameters
    ----------
    a_matrix: xr.DataArray
        DataArray containing the a-matrix values and coordinates.
    megacomplex_suffix: str
        Megacomplex suffix used for the a-matrix data variable and coordinate names.
    normalize_initial_concentration: bool
        Whether or not to normalize the initial concentration. Defaults to False.
    decimal_places: int
        Decimal places to display. Defaults to 3.

    Returns
    -------
    str
        Multi header HTML table representing the a-matrix.
    """
    species = a_matrix.coords[f"species_{megacomplex_suffix}"].values
    # Crete a copy so normalization does not mutate the original values
    initial_concentration = np.array(
        a_matrix.coords[f"initial_concentration_{megacomplex_suffix}"].values
    )
    lifetime = a_matrix.coords[f"lifetime_{megacomplex_suffix}"].values

    if normalize_initial_concentration is True:
        initial_concentration /= initial_concentration.sum()

    header = (
        ["species<br>initial concentration<br>lifetime↓"]
        + [
            f"{sp}<br>{pretty_format_numerical(ic,decimal_places)}<br>&nbsp;"
            for sp, ic in zip(species, initial_concentration)
        ]
        + ["Sum"]
    )

    data = [
        pretty_format_numerical_iterable(
            (lifetime, *amps, amps.sum()), decimal_places=decimal_places
        )
        for lifetime, amps in zip(lifetime, a_matrix.values)
    ]
    data.append(
        pretty_format_numerical_iterable(
            ("Sum", *a_matrix.values.sum(axis=0), a_matrix.values.sum()),
            decimal_places=decimal_places,
        )
    )

    return (
        tabulate(
            data, headers=header, showindex=False, tablefmt="unsafehtml", disable_numparse=True
        )
        .replace(" 0 ", "   ")
        .replace(" 0<", "  <")
        .replace(">0 ", ">  ")
    )


def show_a_matrixes(
    result: ResultLike,
    *,
    normalize_initial_concentration: bool = False,
    decimal_places: int = 3,
    a_matrix_min_size: int | None = None,
    expanded_datasets: tuple[str, ...] = (),
    heading_offset: int = 2,
) -> MarkdownStr:
    """Show all a-matrixes of a result grouped by dataset and megacomplex name.

    Each dataset is wrapped in a HTML details tag which is by default collapsed.

    Parameters
    ----------
    result: ResultLike
        Result or result dataset.
    normalize_initial_concentration: bool
        Whether or not to normalize the initial concentration. Defaults to False.
    decimal_places: int
        Decimal places to display. Defaults to 3.
    a_matrix_min_size: int
        Defaults to None.
    expanded_datasets: tuple[str, ...]
        Names of dataset to expand the details view for. Defaults to empty tuple () which means no
        dataset is expanded.
    heading_offset: int
        Number of heading level to offset the headings. Defaults to 2 which means that the
        first/top most heading is h3.

    Returns
    -------
    MarkdownStr
        Markdown representation of the a-matrixes used in the optimization.
    """
    heading_prefix = heading_offset * "#"
    output_str = f"#{heading_prefix} A-Matrixes\n"

    result_map = result_dataset_mapping(result)

    for dataset_name in result_map:
        a_matrix_names = list(
            filter(
                lambda var_name: var_name.startswith("a_matrix_"),
                result_map[dataset_name].data_vars,
            )
        )

        if not a_matrix_names:
            continue

        details_content = ""
        header_newline_prefix = ""

        for a_matrix_name in a_matrix_names:
            mc_suffix = a_matrix_name.replace("a_matrix_", "")

            a_matrix = result_map[dataset_name][a_matrix_name]

            if a_matrix_min_size is not None and max(a_matrix.shape) < a_matrix_min_size:
                continue

            details_content += f"{header_newline_prefix}###{heading_prefix} {mc_suffix}:\n\n"

            details_content += a_matrix_to_html_table(
                a_matrix,
                mc_suffix,
                normalize_initial_concentration=normalize_initial_concentration,
                decimal_places=decimal_places,
            )
            header_newline_prefix = "\n\n"

        if details_content != "":
            output_str += wrap_in_details_tag(
                details_content,
                summary_content=dataset_name,
                summary_heading_level=2 + heading_offset,
                is_open=dataset_name in expanded_datasets,
            )

    return MarkdownStr(output_str)
