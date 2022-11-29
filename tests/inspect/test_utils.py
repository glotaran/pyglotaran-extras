"""Tests for ``pyglotaran_extras.inspect.utils``."""
from __future__ import annotations

from textwrap import dedent

import pytest

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
