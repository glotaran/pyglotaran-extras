"""Inspection utility module."""

from __future__ import annotations

from collections.abc import Generator
from collections.abc import Iterable

import numpy as np


def wrap_in_details_tag(
    details_content: str,
    *,
    summary_content: str | None = None,
    summary_heading_level: int | None = None,
    is_open: bool = False,
) -> str:
    """Wrap ``details_content`` in a html details tag and add summary if ``summary_content`` set.

    Parameters
    ----------
    details_content: str
        Markdown string that should be displayed when the details are expanded.
    summary_content: str | None
        Summary test that should be displayed. Defaults to None so the summary is ``Details``.
    summary_heading_level: int | None
        Level of the heading wrapping the ``summary`` if it is not None. Defaults to None.
    is_open: bool
        Whether or not the details tag should be initially opened. Defaults to False.

    Returns
    -------
    str
    """
    out_str = f'\n<details {"open" if is_open else ""}>\n'
    if summary_content is not None:
        out_str += "<summary>\n"
        if summary_heading_level is None:
            out_str += f"{summary_content}\n"
        else:
            # Ref.:
            # https://css-tricks.com/two-issues-styling-the-details-element-and-how-to-solve-them/
            out_str += f'<h{summary_heading_level} style="display:inline;">\n'
            out_str += f"{summary_content}\n"
            out_str += f"</h{summary_heading_level}>\n"

        out_str += "</summary>\n"

    out_str += f"\n{details_content}\n<br>\n</details>"
    return out_str


def pretty_format_numerical(value: float | int, decimal_places: int = 1) -> str:
    """Format value with with at most ``decimal_places`` decimal places.

    Used to format values like the t-value.

    TODO: remove after raise pyglotaran dependency to 0.7.0
    Forward port of https://github.com/glotaran/pyglotaran/pull/1192

    Parameters
    ----------
    value: float | int
        Numerical value to format.
    decimal_places: int
        Decimal places to display. Defaults to 1.

    Returns
    -------
    str
        Pretty formatted version of the value.
    """
    # Bool returned by numpy do not support the ``is`` comparison (not same singleton as in python)
    # Ref: https://stackoverflow.com/a/37744300/3990615
    if not np.isfinite(value):
        return str(value)
    if abs(value - int(value)) <= np.finfo(np.float64).eps:
        return str(int(value))
    abs_value = abs(value)
    if abs_value < 10 ** (-decimal_places):
        format_instruction = f".{decimal_places}e"
    elif abs_value < 10 ** (decimal_places):
        format_instruction = f".{decimal_places}f"
    else:
        format_instruction = ".0f"
    return f"{value:{format_instruction}}"


def pretty_format_numerical_iterable(
    input_values: Iterable[str | float], decimal_places: int | None = 3
) -> Generator[str | float, None, None]:
    """Pretty format numerical values in an iterable of numerical values or strings.

    Parameters
    ----------
    input_values: Iterable[str  |  float]
        Values that should be formatted.
    decimal_places: int | None
        Number of decimal places a value should have, if None the original value will be used.
        Defaults to 3.

    See Also
    --------
    pretty_format_numerical

    Yields
    ------
    str | float
        Formatted string or initial value if ``decimal_places`` is None.
    """
    for val in input_values:
        if decimal_places is None or isinstance(val, str):
            yield val
        else:
            yield pretty_format_numerical(val, decimal_places=decimal_places)
