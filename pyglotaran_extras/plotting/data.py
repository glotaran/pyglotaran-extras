"""Deprecated module."""

from pyglotaran_extras.deprecation import warn_deprecated
from pyglotaran_extras.plotting.plot_data import plot_data_overview

__all__ = ["plot_data_overview"]

warn_deprecated(
    deprecated_qual_name_usage="pyglotaran_extras.plotting.data",
    new_qual_name_usage="pyglotaran_extras.plotting.plot_data",
    to_be_removed_in_version="0.7.0",
    stacklevel=3,
)
