"""Module containing config utilities."""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

from ruamel.yaml import YAML

if TYPE_CHECKING:
    from pyglotaran_extras.types import SupportsModelDump


def to_yaml_str(self: SupportsModelDump) -> str:
    """Create yaml string from dumped model.

    Parameters
    ----------
    self : SupportsModelDump
        Instance of a class that supports ``model_dump``.

    Returns
    -------
    str

    See Also
    --------
    add_yaml_repr
    """
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    buffer = StringIO()
    yaml.dump(self.model_dump(), buffer)
    buffer.seek(0)
    return buffer.read()


def add_yaml_repr(cls: type[SupportsModelDump]) -> type[SupportsModelDump]:
    """Add yaml ``__str__`` and ``_repr_markdown_`` methods to class that supports ``model_dump``.

    Parameters
    ----------
    cls : type[SupportsModelDump]
        Class to add the methods to.

    Returns
    -------
    type[SupportsModelDump]
    """
    cls.__str__ = to_yaml_str  # type:ignore[method-assign]
    cls._repr_markdown_ = lambda self: f"```yaml\n{self}\n```"  # type:ignore[attr-defined]

    return cls
