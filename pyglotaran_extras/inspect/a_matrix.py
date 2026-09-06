"""Module containing a-matrix render functionality."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Mapping

import numpy as np
from glotaran.utils.ipython import MarkdownStr
from tabulate import tabulate

from pyglotaran_extras.inspect.utils import wrap_in_details_tag
from pyglotaran_extras.io.utils import result_dataset_mapping

if TYPE_CHECKING:
    import xarray as xr

    from pyglotaran_extras.types import ResultLike


BASE_LEGEND_LABEL_MAP: dict[str, str] = {
    "s1": "PI",
    "s2": "rPI",
    "s8": "rLI~PI",
    "s3": "LI~PI",
    "s11": "LII~PI",
    "s10": "LII~PII",
    "s00": "b~PII",
    "s12": "PII",
    "s13": "RP1PII",
    "s14": "RP2PII",
    "s4": "LI~PII",
    "s9": "rLI~PII",
    "s0": "b~PII~LI",
    "s5": "PII~LI",
    "s6": "RP1PII",
    "s7": "RP2PII",
}


def _format_scientific_exponent(value: str) -> str:
    """Normalize scientific exponent by removing leading zeros.

    Parameters
    ----------
    value : str
        String representation of a number in scientific notation.

    Returns
    -------
    str
        Scientific notation with a normalized exponent such as ``e-4``.
    """
    return re.sub(r"e([+-])0(\d)", r"e\1\2", value)


def _pretty_format_a_matrix_value(
    value: float | int,
    *,
    decimal_places: int,
    scientific_decimal_places: int | None,
    empty_cell_threshold: float | None,
) -> str:
    """Pretty-format a-matrix values with optional scientific precision and blank threshold.

    Parameters
    ----------
    value : float | int
        Value to format.
    decimal_places : int
        Decimal places for non-scientific representation.
    scientific_decimal_places : int | None
        Decimal places for scientific notation values.
    empty_cell_threshold : float | None
        Values with absolute magnitude below this threshold are rendered as empty strings.

    Returns
    -------
    str
        Formatted string representation for table rendering.
    """
    if not np.isfinite(value):
        return str(value)

    if empty_cell_threshold is not None and abs(value) < empty_cell_threshold:
        return ""

    if abs(value - int(value)) <= np.finfo(np.float64).eps:
        return str(int(value))

    abs_value = abs(value)
    if abs_value < 10 ** (-decimal_places):
        sci_places = (
            decimal_places if scientific_decimal_places is None else scientific_decimal_places
        )
        return _format_scientific_exponent(f"{value:.{sci_places}e}")
    if abs_value < 10 ** (decimal_places):
        return f"{value:.{decimal_places}f}"
    return f"{value:.0f}"


def _pretty_format_lifetime_value(value: float | int) -> str:
    """Pretty-format lifetime values using adaptive precision by magnitude.

    Parameters
    ----------
    value : float | int
        Lifetime value in ps.

    Returns
    -------
    str
        Lifetime formatted as 2 decimals below 10 ps, 1 decimal from 10-100 ps,
        and 0 decimals above 100 ps.
    """
    if not np.isfinite(value):
        return str(value)

    if abs(value - int(value)) <= np.finfo(np.float64).eps:
        return str(int(value))

    abs_value = abs(value)
    if abs_value < 10:
        return f"{value:.2f}"
    if abs_value < 100:
        return f"{value:.1f}"
    return f"{value:.0f}"


def _pretty_format_a_matrix_iterable(
    values: tuple[str | float, ...],
    *,
    decimal_places: int,
    scientific_decimal_places: int | None,
    empty_cell_threshold: float | None,
) -> list[str]:
    """Pretty-format values in a row of the rendered a-matrix table.

    Parameters
    ----------
    values : tuple[str | float, ...]
        Row values to format.
    decimal_places : int
        Decimal places for non-scientific representation.
    scientific_decimal_places : int | None
        Decimal places for scientific notation values.
    empty_cell_threshold : float | None
        Values with absolute magnitude below this threshold are rendered as empty strings.

    Returns
    -------
    list[str]
        Formatted row values.
    """
    out: list[str] = []
    for value in values:
        if isinstance(value, str):
            out.append(value)
        else:
            out.append(
                _pretty_format_a_matrix_value(
                    value,
                    decimal_places=decimal_places,
                    scientific_decimal_places=scientific_decimal_places,
                    empty_cell_threshold=empty_cell_threshold,
                )
            )
    return out


def _resolve_scientific_decimal_places(
    *,
    decimal_places: int,
    decimal_places_scientific: int | None,
    scientific_decimal_places: int | None,
    default_offset: int,
) -> int:
    """Resolve scientific decimal places from preferred and legacy arguments."""
    if (
        decimal_places_scientific is not None
        and scientific_decimal_places is not None
        and decimal_places_scientific != scientific_decimal_places
    ):
        msg = (
            "Received both decimal_places_scientific and scientific_decimal_places with "
            "different values. Please use only one of them."
        )
        raise ValueError(msg)

    if decimal_places_scientific is not None:
        return decimal_places_scientific
    if scientific_decimal_places is not None:
        return scientific_decimal_places
    return max(decimal_places + default_offset, 0)


def a_matrix_to_html_table(
    a_matrix: xr.DataArray,
    megacomplex_suffix: str,
    *,
    species_label_map: Mapping[str, str] | None = None,
    normalize_initial_concentration: bool = False,
    decimal_places: int = 3,
    decimal_places_scientific: int | None = None,
    scientific_decimal_places: int | None = None,
    empty_cell_threshold: float | None = None,
) -> str:
    """Create HTML multi header table from a-matrix.

    Parameters
    ----------
    a_matrix : xr.DataArray
        DataArray containing the a-matrix values and coordinates.
    megacomplex_suffix : str
        Megacomplex suffix used for the a-matrix data variable and coordinate names.
    species_label_map : Mapping[str, str] | None
        Mapping used to replace species labels in the table header. If None,
        ``BASE_LEGEND_LABEL_MAP`` is used.
    normalize_initial_concentration : bool
        Whether or not to normalize the initial concentration. Defaults to False.
    decimal_places : int
        Decimal places to display. Defaults to 3.
    decimal_places_scientific : int | None
        Decimal places used for scientific notation values. Preferred over the legacy
        ``scientific_decimal_places`` argument. If None and legacy argument is also None,
        uses ``decimal_places``. Defaults to None.
    scientific_decimal_places : int | None
        Legacy alias for ``decimal_places_scientific``. If None, uses ``decimal_places`` unless
        ``decimal_places_scientific`` is set. Defaults to None.
    empty_cell_threshold : float | None
        Hide values with absolute magnitude below this threshold by rendering them as empty cells.
        Defaults to None.

    Returns
    -------
    str
        Multi header HTML table representing the a-matrix.
    """
    resolved_scientific_decimal_places = _resolve_scientific_decimal_places(
        decimal_places=decimal_places,
        decimal_places_scientific=decimal_places_scientific,
        scientific_decimal_places=scientific_decimal_places,
        default_offset=0,
    )

    resolved_species_label_map = BASE_LEGEND_LABEL_MAP if species_label_map is None else species_label_map
    species = a_matrix.coords[f"species_{megacomplex_suffix}"].to_numpy()
    # Crete a copy so normalization does not mutate the original values
    initial_concentration = np.array(
        a_matrix.coords[f"initial_concentration_{megacomplex_suffix}"].to_numpy()
    )
    lifetime = a_matrix.coords[f"lifetime_{megacomplex_suffix}"].to_numpy()

    if normalize_initial_concentration is True:
        initial_concentration /= initial_concentration.sum()

    header = ["species<br>initial concentration<br>lifetime↓"]
    for sp, ic in zip(species, initial_concentration, strict=True):
        formatted_initial_concentration = _pretty_format_a_matrix_value(
            ic,
            decimal_places=decimal_places,
            scientific_decimal_places=resolved_scientific_decimal_places,
            empty_cell_threshold=empty_cell_threshold,
        )
        header.append(
            f"{resolved_species_label_map.get(str(sp), str(sp))}<br>"
            f"{formatted_initial_concentration}<br>&nbsp;"
        )
    header.append("Sum")

    data = []
    for lifetime_value, amps in zip(lifetime, a_matrix.values, strict=True):
        row = _pretty_format_a_matrix_iterable(
            (lifetime_value, *amps, amps.sum()),
            decimal_places=decimal_places,
            scientific_decimal_places=resolved_scientific_decimal_places,
            empty_cell_threshold=empty_cell_threshold,
        )
        row[0] = _pretty_format_lifetime_value(lifetime_value)
        data.append(row)
    data.append(
        _pretty_format_a_matrix_iterable(
            ("Sum", *a_matrix.to_numpy().sum(axis=0), a_matrix.to_numpy().sum()),
            decimal_places=decimal_places,
            scientific_decimal_places=resolved_scientific_decimal_places,
            empty_cell_threshold=empty_cell_threshold,
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
    species_label_map: Mapping[str, str] | None = None,
    normalize_initial_concentration: bool = False,
    decimal_places: int = 3,
    decimal_places_scientific: int | None = None,
    scientific_decimal_places: int | None = None,
    empty_cell_threshold: float | None = None,
    a_matrix_min_size: int | None = None,
    expanded_datasets: tuple[str, ...] = (),
    heading_offset: int = 2,
) -> MarkdownStr:
    """Show all a-matrixes of a result grouped by dataset and megacomplex name.

    Each dataset is wrapped in a HTML details tag which is by default collapsed.

    Parameters
    ----------
    result : ResultLike
        Result or result dataset.
    species_label_map : Mapping[str, str] | None
        Mapping used to replace species labels in table headers. If None,
        ``BASE_LEGEND_LABEL_MAP`` is used.
    normalize_initial_concentration : bool
        Whether or not to normalize the initial concentration. Defaults to False.
    decimal_places : int
        Decimal places to display. Defaults to 3.
    decimal_places_scientific : int | None
        Decimal places used for scientific notation values. Preferred over the legacy
        ``scientific_decimal_places`` argument. If None and legacy argument is also None,
        uses ``max(decimal_places - 1, 0)``. Defaults to None.
    scientific_decimal_places : int | None
        Legacy alias for ``decimal_places_scientific``. If None, uses
        ``max(decimal_places - 1, 0)`` unless ``decimal_places_scientific`` is set.
        Defaults to None.
    empty_cell_threshold : float | None
        Hide values with absolute magnitude below this threshold by rendering them as empty cells.
        Defaults to None.
    a_matrix_min_size : int | None
        Defaults to None.
    expanded_datasets : tuple[str, ...]
        Names of dataset to expand the details view for. Defaults to empty tuple () which means no
        dataset is expanded.
    heading_offset : int
        Number of heading level to offset the headings. Defaults to 2 which means that the
        first/top most heading is h3.

    Returns
    -------
    MarkdownStr
        Markdown representation of the a-matrixes used in the optimization.
    """
    resolved_scientific_decimal_places = _resolve_scientific_decimal_places(
        decimal_places=decimal_places,
        decimal_places_scientific=decimal_places_scientific,
        scientific_decimal_places=scientific_decimal_places,
        default_offset=-1,
    )

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
                species_label_map=species_label_map,
                normalize_initial_concentration=normalize_initial_concentration,
                decimal_places=decimal_places,
                scientific_decimal_places=resolved_scientific_decimal_places,
                empty_cell_threshold=empty_cell_threshold,
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
