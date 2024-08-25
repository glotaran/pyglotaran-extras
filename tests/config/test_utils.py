"""Tests for ``pyglotaran_extras.config.utils``."""

from __future__ import annotations

from textwrap import dedent
from typing import Any

from IPython.core.formatters import format_display_data
from pydantic import BaseModel

from pyglotaran_extras.config.utils import add_yaml_repr
from pyglotaran_extras.config.utils import to_yaml_str


class UtilTestClass(BaseModel):
    """Class with test data by default."""

    str_attr: str = "str_val"
    int_attr: int = 2
    dict_attr: dict[str, Any] = {"key1": 1, "key2": None}
    list_attr: list[Any] = ["str", 1, None]


EXPECTED_YAML_STR = dedent(
    """\
    str_attr: str_val
    int_attr: 2
    dict_attr:
      key1: 1
      key2: null
    list_attr:
      - str
      - 1
      - null
    """
)


def test_to_yaml_str():
    """Created yaml str has expected format."""
    test_instance = UtilTestClass()
    assert to_yaml_str(test_instance) == EXPECTED_YAML_STR


def test_add_yaml_repr():
    """Added methods behave as expected when converting to string and rendering in ipython."""
    test_instance = add_yaml_repr(UtilTestClass)()
    assert str(test_instance) == EXPECTED_YAML_STR

    rendered_result = format_display_data(test_instance)[0]

    assert "text/markdown" in rendered_result
    assert rendered_result["text/markdown"] == f"```yaml\n{EXPECTED_YAML_STR}\n```"
    assert rendered_result["text/plain"] == repr(test_instance)
