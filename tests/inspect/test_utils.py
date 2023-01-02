"""Tests for ``pyglotaran_extras.inspect.utils``."""
from __future__ import annotations

from collections.abc import Iterable
from textwrap import dedent

import numpy as np
import pytest

from pyglotaran_extras.inspect.utils import pretty_format_numerical
from pyglotaran_extras.inspect.utils import pretty_format_numerical_iterable
from pyglotaran_extras.inspect.utils import wrap_in_details_tag


@pytest.mark.parametrize(
    "details_content, summary_content, summary_heading_level,is_open,expected",
    (
        pytest.param(
            "FOO",
            None,
            None,
            False,
            dedent(
                """
                <details >

                FOO
                <br>
                </details>"""
            ),
            id="defaults",
        ),
        pytest.param(
            "FOO",
            None,
            5,
            False,
            dedent(
                """
                <details >

                FOO
                <br>
                </details>"""
            ),
            id="heading no summary",
        ),
        pytest.param(
            "FOO",
            None,
            None,
            True,
            dedent(
                """
                <details open>

                FOO
                <br>
                </details>"""
            ),
            id="defaults open details",
        ),
        pytest.param(
            "FOO",
            "Bar",
            None,
            False,
            dedent(
                """
                <details >
                <summary>
                Bar
                </summary>

                FOO
                <br>
                </details>"""
            ),
            id="defaults with simple summary",
        ),
        pytest.param(
            "FOO",
            "Bar",
            5,
            False,
            dedent(
                """
                <details >
                <summary>
                <h5 style="display:inline;">
                Bar
                </h5>
                </summary>

                FOO
                <br>
                </details>"""
            ),
            id="defaults with heading summary",
        ),
    ),
)
def test_wrap_in_details_tag(
    details_content: str,
    summary_content: str | None,
    summary_heading_level: int | None,
    is_open: bool,
    expected: str,
):
    """Correct details."""
    assert (
        wrap_in_details_tag(
            details_content,
            summary_content=summary_content,
            summary_heading_level=summary_heading_level,
            is_open=is_open,
        )
        == expected
    )


@pytest.mark.parametrize(
    "value, decimal_places, expected",
    (
        (0.00000001, 1, "1.0e-08"),
        (-0.00000001, 1, "-1.0e-08"),
        (0.1, 1, "0.1"),
        (1.7, 1, "1.7"),
        (10, 1, "10"),
        (1.0000000000000002, 10, "1"),
        (-1.0000000000000002, 10, "-1"),
        (10, 10, "10"),
        (-10, 10, "-10"),
        (0.00000001, 8, "0.00000001"),
        (-0.00000001, 8, "-0.00000001"),
        (0.009, 2, "9.00e-03"),
        (-0.009, 2, "-9.00e-03"),
        (0.01, 2, "0.01"),
        (12.3, 2, "12.30"),
        (np.nan, 1, "nan"),
        (np.inf, 1, "inf"),
        (-np.inf, 1, "-inf"),
    ),
)
def test_pretty_format_numerical(value: float, decimal_places: int, expected: str):
    """Pretty format values depending on decimal_places to show.

    TODO: remove after raise pyglotaran dependency to 0.7.0
    Forward port of https://github.com/glotaran/pyglotaran/pull/1192 tests
    """
    result = pretty_format_numerical(value, decimal_places)

    assert result == expected


@pytest.mark.parametrize(
    "decimal_places, expected",
    (
        (None, ["Foo", 1, 0.009, -1.0000000000000002, np.nan, np.inf]),
        (2, ["Foo", "1", "9.00e-03", "-1", "nan", "inf"]),
    ),
)
def test_pretty_format_numerical_iterable(decimal_places: int, expected: Iterable[str | float]):
    """Values correct formatted"""
    values = ("Foo", 1, 0.009, -1.0000000000000002, np.nan, np.inf)
    assert list(pretty_format_numerical_iterable(values, decimal_places)) == expected
